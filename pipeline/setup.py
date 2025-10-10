"""Copyright (2025) Tilde Research Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List, Dict, Optional
import os, gc
from pipeline.vault import HookUploader, LocalHookUploader
from accelerate import dispatch_model, infer_auto_device_map
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(
    transformer_config: dict,
    hooks: List[str] = None,
    decode_enabled: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Set up and configure a transformer model and tokenizer for activation extraction.

    This function handles model loading, dtype configuration, model truncation based on hooks,
    and device mapping for efficient memory usage.

    Args:
        transformer_config: Configuration dictionary containing:
            - model_name: Name/path of the HuggingFace model
            - dtype: Model precision ("float16", "bfloat16", or "float32")
        hooks: List of activation hook names, used to determine model truncation

    Returns:
        tuple: (model, tokenizer)
            - model: Configured AutoModelForCausalLM instance
            - tokenizer: Associated AutoTokenizer instance

    Example:
        ```python
        config = {
            "model_name": "gpt2",
            "dtype": "float16"
        }
        model, tokenizer = setup_model_and_tokenizer(config, hooks=["models.layers.24.mlp.post"])
        ```
    """
    tokenizer = AutoTokenizer.from_pretrained(transformer_config["model_name"])

    # Determine dtype
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(
        transformer_config.get("dtype", "float16"), torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        transformer_config["model_name"],
        torch_dtype=dtype,
        device_map="cpu",
        cache_dir=transformer_config.get("cache_dir", None),
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        **transformer_config.get("kwargs", {}),
    )

    should_truncate = hooks and not decode_enabled

    if should_truncate:

        # Find the highest layer number from hooks
        layer_numbers = [int(hook.split(".")[2]) for hook in hooks if hook.startswith("models.layers")]
        max_layer = max(layer_numbers)

        # Print model size before truncation
        total_params_before = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters before truncation: {total_params_before:,}")

        # Truncate the model
        num_layers = len(model.model.layers)
        removed_layers = model.model.layers[max_layer + 1 :]
        model.model.layers = model.model.layers[: max_layer + 1]

        del removed_layers
        del model.lm_head
        torch.cuda.empty_cache()
        gc.collect()

        model.lm_head = torch.nn.Identity()

        # Print model size after truncation
        total_params_after = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters after truncation: {total_params_after:,}")
        logger.info(
            f"Removed {num_layers - (max_layer + 1)} layers, keeping first {max_layer + 1} layers"
        )
        logger.info(
            f"Memory saved: {(total_params_before - total_params_after) * dtype.itemsize / (1024**3):.2f} GB"
        )
    else:
        if hooks and decode_enabled:
            logger.info("Decode enabled, using all layers. Hooks provided.")
        else:
            logger.info("No hooks provided, using all layers")
            hooks = [f"models.layers.{i}.mlp.post" for i in range(model.config.num_hidden_layers)]

    decoder_cls = model.model.layers[0].__class__.__name__

    num_gpus = torch.cuda.device_count()

    device_map = infer_auto_device_map(
        model,
        max_memory={
            i: transformer_config.get("max_per_device_memory", "55GB") for i in range(num_gpus)
        },
        no_split_module_classes=[decoder_cls],
    )

    model = dispatch_model(model, device_map=device_map)

    torch.cuda.empty_cache()
    gc.collect()

    return model, tokenizer, hooks


def maybe_add_mlp_attn_hooks(model: AutoModelForCausalLM, hooks: List[str] = None) -> List[str]:
    """Add MLP and attention hooks to the model.

    Args:
        model: The transformer model
        hooks: List of hook names to check for mlp/attn hooks

    Returns:
        List[str]: Potentially expanded list of hooks including new mlp/attn hooks
    """
    if not hooks:
        return hooks

    pytorch_hooks = []
    hook_activations = {}

    for hook in list(hooks):
        layer_idx = int(hook.split(".")[2])

        if "mlp.pre" in hook:
            def get_mlp_pre_hook(layer_idx):
                def pre_hook(module, input):
                    if input and isinstance(input, tuple) and len(input) > 0:
                        hook_activations[f"models.layers.{layer_idx}.mlp.pre"] = input[0]
                return pre_hook

            pytorch_hook = get_mlp_pre_hook(layer_idx)
            model.model.layers[layer_idx].mlp.register_forward_pre_hook(pytorch_hook)
            pytorch_hooks.append(pytorch_hook)
            logger.info(f"Added MLP pre hook for layer {layer_idx}")

        if "mlp.post" in hook:
            def get_mlp_post_hook(layer_idx):
                def post_hook(module, input, output):
                    # MLP output is a direct tensor
                    hook_activations[f"models.layers.{layer_idx}.mlp.post"] = output
                return post_hook

            pytorch_hook = get_mlp_post_hook(layer_idx)
            model.model.layers[layer_idx].mlp.register_forward_hook(pytorch_hook)
            pytorch_hooks.append(pytorch_hook)
            logger.info(f"Added MLP post hook for layer {layer_idx}")

        if "self_attn.pre" in hook:
            def get_attn_pre_hook(layer_idx):
                def pre_hook(module, input):
                    if input and isinstance(input, tuple) and len(input) > 0:
                        hook_activations[f"models.layers.{layer_idx}.self_attn.pre"] = input[0]
                return pre_hook

            pytorch_hook = get_attn_pre_hook(layer_idx)
            model.model.layers[layer_idx].self_attn.register_forward_pre_hook(pytorch_hook)
            pytorch_hooks.append(pytorch_hook)
            logger.info(f"Added attention pre hook for layer {layer_idx}")

        if "self_attn.post" in hook:
            def get_attn_post_hook(layer_idx):
                def post_hook(module, input, output):
                    # Attention output is a tuple, we want the first element
                    if isinstance(output, tuple) and len(output) > 0:
                        hook_activations[f"models.layers.{layer_idx}.self_attn.post"] = output[0]
                return post_hook

            pytorch_hook = get_attn_post_hook(layer_idx)
            model.model.layers[layer_idx].self_attn.register_forward_hook(pytorch_hook)
            pytorch_hooks.append(pytorch_hook)
            logger.info(f"Added attention post hook for layer {layer_idx}")

    return pytorch_hooks, hook_activations


def setup_uploaders(
    run_name: str, hooks: List[str], batches_per_upload: int, bucket_name: str, decode_enabled: bool = False,
    storage_backend: str = "s3", local_output_dir: str = "activations"
) -> Tuple[Dict[str, HookUploader], Optional[dict[str, HookUploader]]]:
    """Create S3 uploaders for storing activation data from each hook.

    Args:
        run_name: Unique identifier for this run
        hooks: List of hook names requiring uploaders
        batches_per_upload: Number of batches to accumulate before upload
        bucket_name: S3 bucket name for storage
        decode_enabled: whether to provide secondary uploaders for decode-stage activations
        storage_backend:
        local_output_dir:

    Returns:
        dict: Mapping of hook names to their respective uploaders

    Example:
        ```python
        uploaders, decode_uploaders = setup_uploaders(
            "experiment_1",
            hooks=["models.layers.0.mlp.post"],
            batches_per_upload=10,
            decode_enabled=True,
        )
        ```
    """
    backend = storage_backend.lower()
    if backend not in {"s3", "disk"}:
        raise ValueError(f"Unsupported storage backend: {backend}")

    uploaders = {}
    decode_uploaders = {} if decode_enabled else None

    local_root = None
    if backend == "disk":
        local_root = Path(local_output_dir)
        local_root.mkdir(parents=True, exist_ok=True)

    for hook in hooks:
        base_prefix = f"{run_name}/{hook}"

        if backend == "disk":
            uploaders[hook] = LocalHookUploader(
                output_root=local_root,
                prefix_path=f"{base_prefix}/prefill",
                batches_per_upload=batches_per_upload,
            )
            if decode_uploaders is not None:
                decode_uploaders[hook] = LocalHookUploader(
                    output_root=local_root,
                    prefix_path=f"{base_prefix}/decode",
                    batches_per_upload=batches_per_upload,
                )
            continue

        uploaders[hook] = HookUploader.from_credentials(
            access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            prefix_path=f"{base_prefix}/prefill",
            batches_per_upload=batches_per_upload,
            bucket_name=bucket_name,
        )
        if decode_uploaders is not None:
            decode_uploaders[hook] = HookUploader.from_credentials(
                access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                prefix_path=f"{base_prefix}/decode",
                batches_per_upload=batches_per_upload,
                bucket_name=bucket_name,
            )

    return uploaders, decode_uploaders


def calculate_machine_params(
    machine_index: int, total_batches: int, num_runs: int
) -> Tuple[int, int]:
    """Calculate batch distribution for distributed processing.

    Determines how many batches each machine should process and their starting points.

    Args:
        machine_index: Index of the current machine (0 to num_runs-1)
        total_batches: Total number of batches to process
        num_runs: Total number of parallel runs

    Returns:
        tuple: (batches_per_machine, start_batch)
            - batches_per_machine: Number of batches for this machine
            - start_batch: Starting batch index

    Note:
        The returned start_batch should be added to any global start_batch parameter.

    Example:
        ```python
        batches, start = calculate_machine_params(0, 1000, 4)
        # Returns (250, 0) for first machine
        ```
    """
    batches_per_machine = total_batches // num_runs
    start_batch = machine_index * batches_per_machine
    end_batch = start_batch + batches_per_machine - 1

    if num_runs > 1:
        logger.info(
            f"Run {machine_index + 1}/{num_runs} processing batches {start_batch} to {end_batch} "
            f"({batches_per_machine} out of {total_batches} total batches)"
        )
    else:
        logger.info(
            f"Single run processing all batches from {start_batch} to {end_batch} "
            f"({total_batches} total batches)"
        )

    return batches_per_machine, start_batch


def display_job_stats(model, hooks, config, batches_per_machine):
    """Display job statistics including space usage and token count.

    Calculates and presents a box containing key statistics about
    the current job, including model dimensions, token counts, and estimated storage
    requirements.

    Args:
        model: The transformer model
        hooks: List of activation hooks
        config: Configuration object
        batches_per_machine: Number of batches to be processed by this machine
    """
    # Calculate space usage statistics
    d_model = model.config.hidden_size
    n_hooks = len(hooks)
    dtype_size = {"float16": 2, "bfloat16": 2, "float32": 4}.get(
        config.transformer_config.get("dtype", "float16"), 2
    )  # Size in bytes
    n_batches = batches_per_machine
    batch_size = config.data_config["batch_size"]
    seq_length = config.data_config["seq_length"]

    total_tokens = n_batches * batch_size * seq_length
    total_space_bytes = d_model * n_hooks * dtype_size * total_tokens

    # Convert to human-readable format
    space_gb = total_space_bytes / (1024**3)

    # Create beautiful stats display
    logger.info("┌─" + "─" * 60 + "─┐")
    logger.info("│ " + "Job Statistics".center(60) + " │")
    logger.info("├─" + "─" * 60 + "─┤")
    logger.info(f"│ Model Hidden Dimension: {d_model:,}".ljust(62) + "│")
    logger.info(f"│ Number of Hooks: {n_hooks}".ljust(62) + "│")
    logger.info(
        f"│ Data Type: {config.transformer_config.get('dtype', 'float16')} ({dtype_size} bytes)".ljust(
            62
        )
        + "│"
    )
    logger.info(f"│ Batch Size: {batch_size}".ljust(62) + "│")
    logger.info(f"│ Sequence Length: {seq_length}".ljust(62) + "│")
    logger.info(f"│ Number of Batches: {n_batches:,}".ljust(62) + "│")
    logger.info(f"│ Total Tokens: {total_tokens:,}".ljust(62) + "│")
    logger.info(f"│ Estimated Storage Required: {space_gb:.2f} GB".ljust(62) + "│")
    logger.info("└─" + "─" * 60 + "─┘")
