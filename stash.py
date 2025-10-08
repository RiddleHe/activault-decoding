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

from pipeline.data.dataset import load_dataset_by_key
from pipeline.data.dataloader import DataLoader
from pipeline.config import Config
from pipeline.setup import (
    setup_model_and_tokenizer,
    setup_uploaders,
    calculate_machine_params,
    display_job_stats,
    maybe_add_mlp_attn_hooks,
)
from pipeline.generate import generate_activations
import logging
import time

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [Worker-%(process)d] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # Load config from command line args or default
    config = Config.from_args()

    # Calculate machine-specific parameters
    batches_per_machine, start_batch_skip = calculate_machine_params(
        config.machine_index, config.data_config["n_batches"], config.num_runs
    )

    # Add the global start_batch to the machine-specific start_batch
    start_batch_skip += config.data_config["start_batch"]

    # Load dataset
    dataset = load_dataset_by_key(config.data_config["data_key"])

    # Setup model and tokenizer with hooks for truncation
    model, tokenizer, hooks = setup_model_and_tokenizer(
        config.transformer_config, hooks=config.upload_config.get("hooks")
    )
    logger.info(f"Model loaded: {model}")

    added_hooks, added_hook_activations = maybe_add_mlp_attn_hooks(model, hooks)

    config.upload_config["hooks"] = hooks

    # Display job statistics
    display_job_stats(model, hooks, config, batches_per_machine)

    # Setup uploaders (one per hook)
    decode_enabled = config.decode_config.get("enable", False)
    uploaders, decode_uploaders = setup_uploaders(
        run_name=config.run_name,
        hooks=hooks,
        batches_per_upload=config.upload_config["batches_per_upload"],
        bucket_name=config.data_config["bucket_name"],
        decode_enabled=decode_enabled,
        storage_backend=config.upload_config.get("storage_backend", "s3"),
        local_output_dir=config.upload_config.get("local_output_dir", "activations"),
    )
    logger.info(f"Uploaders loaded: {uploaders}")

    # Create dataloader
    loader = DataLoader(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=config.data_config["seq_length"],
        batch_size=config.data_config["batch_size"],
        start_batch_skip=start_batch_skip,
        batches_per_machine=batches_per_machine,
        dataset_key=config.data_config["data_key"],
        skip_cache=config.data_config["skip_cache"],
        clean_added_tokens=config.data_config["clean_added_tokens"],
    )

    # Run generation
    generate_activations(model, loader, config, uploaders, added_hook_activations, decode_uploaders=decode_uploaders)

    # just in case...
    logger.info(
        "DONE generating activations. Waiting 5 minutes before exiting, in case there are any remaining uploads."
    )
    time.sleep(5 * 60)


if __name__ == "__main__":
    main()
