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
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.generation.utils import top_k_top_p_filtering
from typing import Dict, Optional
from pipeline.data.dataloader import DataLoader
from pipeline.config import Config
from pipeline.vault import HookUploader
from s3.utils import create_s3_client
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)


def generate_activations(
    model: AutoModelForCausalLM,
    loader: DataLoader,
    config: Config,
    uploaders: Dict[str, HookUploader],
    hook_activations: Dict[str, Dict[str, torch.Tensor]] = None,
    decode_uploaders: Optional[Dict[str, HookUploader]] = None,
) -> None:
    """
    Main activation generation loop.

    Args:
        model: The transformer model (already on correct device)
        loader: DataLoader instance
        config: Configuration object
        uploaders: Dictionary mapping hook names to their uploaders
        hook_activations: Cache populated by PyTorch hooks
        decode_uploaders: Dictionary mapping hook names to their decode-stage uploaders
    """
    # Set d_model in config
    config.d_model = model.config.hidden_size

    # Create S3 client for config saving
    s3_client = create_s3_client()

    # Load existing config and stats if available
    existing_config = Config.load_from_s3(s3_client, config.data_config["bucket_name"])
    if existing_config:
        logger.info(f"Resuming run {config.run_name} from {existing_config.total_tokens} tokens")
        config.total_tokens = existing_config.total_tokens
        config.n_total_files = existing_config.n_total_files
        config.batches_processed = existing_config.batches_processed

        # Skip tokens based on existing total tokens with an offset
        tokens_to_skip = existing_config.total_tokens + 3000
        loader.skip_tokens(tokens_to_skip)

    # Initialize statistics tracking
    hooks = config.upload_config["hooks"]
    means = {hook: torch.zeros(model.config.hidden_size, device=model.device) for hook in hooks}

    # M2 stores sum of squared differences from the mean (for Welford's algorithm)
    M2s = {hook: torch.zeros(model.config.hidden_size, device=model.device) for hook in hooks}
    counts = {hook: 0 for hook in hooks}  # Track number of samples per dimension
    norm_sums = {hook: torch.zeros(1, device=model.device) for hook in hooks}  # Track sum of norms
    norm_counts = {hook: 0 for hook in hooks}  # Track count for norms

    # Load existing statistics if available
    for hook in hooks:
        stats = Config.load_hook_statistics(
            s3_client, config.run_name, hook, config.data_config["bucket_name"]
        )
        if stats:
            logger.info(f"Loading existing statistics for {hook}")
            means[hook] = torch.tensor(stats["mean"], device=model.device)
            if "M2" in stats:
                M2s[hook] = torch.tensor(stats["M2"], device=model.device)
            else:
                # If M2 not available, approximate from std (for backward compatibility)
                std = torch.tensor(stats["std"], device=model.device)
                M2s[hook] = std * std * config.batches_processed

            counts[hook] = config.batches_processed
            norm_sums[hook] = stats.get("norm", 0.0) * config.batches_processed
            norm_counts[hook] = config.batches_processed

    # Prepare for activation collection
    layers = {hook: int(hook.split(".")[2]) for hook in hooks}

    # Initialize batches processed from config
    batches_processed = config.batches_processed

    # Calculate tokens to skip based on batches processed
    tokens_to_skip = (
        config.batches_processed
        * config.data_config["batch_size"]
        * config.data_config["seq_length"]
    )
    loader.skip_tokens(tokens_to_skip)

    decode_enabled = (
        decode_uploaders is not None and config.decode_config.get("enable", False)
    )
    if config.decode_config.get("enable", False) and decode_uploaders is None:
        raise RuntimeError("Decode activations requested but decode uploaders were not provided")

    max_new_tokens = int(config.decode_config.get("max_new_tokens", 1024)) if decode_enabled else 0
    temperature = float(config.decode_config.get("temperature", 0.7))
    top_p = float(config.decode_config.get("top_p", 0.95))
    stop_on_eos = bool(config.decode_config.get("stop_on_eos", True))
    eos_token_id = model.config.eos_token_id
    decode_tokens_written = False

    def sample_next_token(logits):
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        filtered_logits = top_k_top_p_filtering(
            logits / max(temperature, 1e-5),
            top_k=0,
            top_p=top_p,
        )
        probs = torch.softmax(filtered_logits, dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def extract_activations(hidden_states, token_tensor):
        activations = {}
        for hook in hooks:
            if hook_activations and hook in hook_activations:
                payload = hook_activations[hook]
            else:
                layer_idx = int(hook.split(".")[2])
                if "pre" in hook:
                    payload = hidden_states[layer_idx]
                else:
                    payload = hidden_states[layer_idx + 1]

            activations[hook] = {
                "states": payload,
                "input_ids": token_tensor,
            }

        if hook_activations:
            hook_activations.clear()
        return activations

    def queue_stage(cleaned_map, stage_uploaders, group_uuid):
        if not stage_uploaders:
            return False
        
        any_uploaded = False
        for hook in hooks:
            if hook not in cleaned_map:
                continue

            states = cleaned_map[hook]["states"]
            input_ids = cleaned_map[hook]["input_ids"]
            if states.numel() == 0:
                continue

            N, T = states.shape[0], states.shape[1]
            total_tokens = N * T
            counts[hook] += total_tokens

            delta = states.mean(dim=(0, 1)) - means[hook]
            means[hook] += delta * (total_tokens / counts[hook])

            delta2 = states.mean(dim=(0, 1)) - means[hook]
            M2s[hook] += total_tokens * delta * delta2

            norm_sums[hook] += torch.norm(states, dim=2).sum().item()
            norm_counts[hook] += total_tokens

            cpu_activations = {
                "states": states.to(device="cpu", non_blocking=True),
                "input_ids": input_ids.to(device="cpu", non_blocking=True),
            }
            file_id = stage_uploaders[hook].append(cpu_activations, group_uuid)
            if file_id:
                any_uploaded = True

        return any_uploaded

    # Main loop
    model.eval()
    with torch.no_grad():
        total_batches = (
            loader.batches_per_machine
            if loader.batches_per_machine is not None
            else config.data_config["n_batches"]
        )
        pbar = tqdm(total=total_batches)
        pbar.update(batches_processed)

        # Generate a new UUID for each batch group
        current_group_uuid = str(uuid4())

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= config.data_config["n_batches"]:
                break

            # Move batch to model's device
            batch = {k: v.to(device=model.device) for k, v in batch.items()}
            attention_mask = batch.get("attention_mask")
            use_cache = decode_enabled and max_new_tokens > 0

            # Forward pass
            outputs = model(
                **batch, output_hidden_states=True, use_cache=use_cache
            )
            past_key_values = outputs.past_key_values if use_cache else None

            # Extract activations
            activations = extract_activations(outputs.hidden_states, batch["input_ids"])

            try:
                # Clean special tokens from activations (e.g. BOS)
                cleaned_activations = {}
                for hook in hooks:
                    cleaned_input_ids, cleaned_states = loader.clean_batch(
                        activations[hook]["input_ids"], activations[hook]["states"]
                    )
                    cleaned_activations[hook] = {
                        "states": cleaned_states,
                        "input_ids": cleaned_input_ids,
                    }
            except Exception as e:
                logger.warning(f"SKIPPING BATCH {batch_idx} due to error: {e}")
                continue

            # Update total tokens
            config.total_tokens += (
                config.data_config["batch_size"] * config.data_config["seq_length"]
            )

            # Compute statistics and move to CPU
            if uploaders:
                any_file_uploaded = queue_stage(cleaned_activations, uploaders, current_group_uuid)

                if any_file_uploaded:
                    config.n_total_files += 1
                    # Generate new UUID for next group since we just uploaded
                    current_group_uuid = str(uuid4())

            # Update batches processed
            batches_processed += 1

            # Save config periodically only from machine index 0
            if (
                config.machine_index == 0
                and batches_processed % config.upload_config["batches_per_upload"] == 0
            ):
                # Update config with the current state
                config.batches_processed = batches_processed

                # Save config in a non-blocking way
                config.save_to_s3(s3_client, blocking=False)

                # Save statistics
                if uploaders:
                    for hook in hooks:
                        # Extract final statistics from running calculations
                        mean = means[hook].cpu()

                        # Calculate standard deviation from M2
                        variance = M2s[hook] / counts[hook]
                        std = torch.sqrt(variance).cpu()

                        # Calculate average norm
                        norm = norm_sums[hook] / norm_counts[hook] if norm_counts[hook] > 0 else 0.0

                        # Also save M2 for future resumption
                        uploaders[hook].save_stats(mean, std, norm, M2=M2s[hook].cpu())

                        if decode_uploaders:
                            decode_uploaders[hook].save_stats(
                                mean, std, norm, M2=M2s[hook].cpu()
                            )

            if decode_enabled and max_new_tokens > 0 and past_key_values is not None:
                logits = outputs.logits[:, -1, :]

                if temperature <= 0:
                    next_tokens = torch.argmax(logits, dim=-1)
                else:
                    next_tokens = sample_next_token(logits)
                
                next_input_ids = next_tokens.unsqueeze(-1)
                finished = torch.zeros(
                    next_tokens.shape[0], dtype=torch.bool, device=model.device
                )
                eos_tensor = (
                    torch.full_like(next_tokens, eos_token_id)
                    if eos_token_id is not None else None
                )

                decode_attention_mask = attention_mask

                for decode_step in range(max_new_tokens):
                    decode_kwargs = {
                        "input_ids": next_input_ids,
                        "use_cache": True,
                        "output_hidden_states": True,
                        "past_key_values": past_key_values
                    }
                    if decode_attention_mask is not None:
                        ones = torch.ones_like(next_input_ids, device=model.device)
                        decode_attention_mask = torch.cat(
                            [decode_attention_mask, ones], dim=-1
                        )
                        decode_kwargs["attention_mask"] = decode_attention_mask

                    decode_outputs = model(**decode_kwargs)
                    past_key_values = decode_outputs.past_key_values

                    decode_activations = extract_activations(decode_outputs.hidden_states, next_input_ids)

                    active_mask = ~finished
                    if active_mask.any():
                        cleaned_decode = {}
                        for hook in hooks:
                            states = decode_activations[hook]["states"][active_mask]
                            tokens = next_input_ids[active_mask]
                            cleaned_decode[hook] = {
                                "states": states,
                                "input_ids": tokens,
                            }
                        active_tokens = int(active_mask.sum().item())
                        config.total_tokens += active_tokens

                        decode_group_uuid = str(uuid4())
                        if queue_stage(cleaned_decode, decode_uploaders, decode_group_uuid):
                            config.n_total_files += 1
                            decode_tokens_written = True

                    logits = decode_outputs.logits[:, -1, :]

                    if temperature <= 0:
                        next_tokens = torch.argmax(logits, dim=-1)
                    else:
                        next_tokens = sample_next_token(logits)

                    if eos_tensor is not None:
                        finished = finished | (next_tokens == eos_tensor)
                        next_tokens = torch.where(finished, eos_tensor, next_tokens)
                    
                    next_input_ids = next_tokens.unsqueeze(-1)

                    if stop_on_eos and finished.all():
                        break

            pbar.update(1)

        pbar.close()

        # Save final config and statistics only from machine index 0
        if config.machine_index == 0:
            config.batches_processed = batches_processed
            config.save_to_s3(s3_client, blocking=True)  # Block on final save

            if uploaders:
                for hook in hooks:
                    # Extract final statistics from running calculations
                    mean = means[hook].cpu()

                    # Calculate standard deviation from M2
                    variance = M2s[hook] / counts[hook]
                    std = torch.sqrt(variance).cpu()

                    # Calculate average norm
                    norm = norm_sums[hook] / norm_counts[hook] if norm_counts[hook] > 0 else 0.0

                    # Also save M2 for future resumption
                    uploaders[hook].save_stats(mean, std, norm, M2=M2s[hook].cpu())
                    uploaders[hook].finalize()

                    if decode_uploaders:
                        if decode_tokens_written:
                            decode_uploaders[hook].save_stats(
                                mean, std, norm, M2=M2s[hook].cpu()
                            )
                        decode_uploaders[hook].finalize()
