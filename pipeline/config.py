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

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import argparse
import json
import boto3
import threading
import re


@dataclass
class Config:
    """Configuration manager for the ActiVault pipeline.

    This class handles all configuration aspects of the pipeline, including loading from YAML,
    command line arguments, and S3 integration. It supports both synchronous and asynchronous
    saving to S3.

    Attributes:
        run_name: Unique identifier for this pipeline run (format: model_family.model_size.dataset[.template])
        transformer_config: Configuration settings for the transformer model
        data_config: Settings for data loading and processing
        upload_config: Configuration for upload behavior and hooks
        num_runs: Number of parallel runs to execute
        total_tokens: Counter for processed tokens
        d_model: Model dimension if applicable
        n_total_files: Counter for processed files
        batches_processed: Counter for processed batches
        machine_index: Index for distributed processing

    Example:
        ```python
        # Load from YAML
        config = Config.from_yaml("configs/default.yaml")

        # Load from command line
        config = Config.from_args()

        # Save to S3
        config.save_to_s3(s3_client)
        ```
    """

    run_name: str
    transformer_config: Dict[str, Any]
    data_config: Dict[str, Any]
    upload_config: Dict[str, Any]
    decode_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "enable": False,
            "max_new_tokens": 1024,
            "temperature": 1.0,
            "top_p": 0.95,
            "stop_on_eos": True,
            "log_dir": "decode_logs",
        }
    )
    num_runs: int
    total_tokens: int = 0
    d_model: Optional[int] = None
    n_total_files: int = 0
    batches_processed: int = 0
    _save_thread: Optional[threading.Thread] = None
    machine_index: int = 0
    _config_key: Optional[str] = None  # Track the config file path once created

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary format for serialization.

        Returns:
            dict: Configuration in dictionary format
        """
        return {
            "run_name": self.run_name,
            "transformer_config": self.transformer_config,
            "data_config": self.data_config,
            "upload_config": self.upload_config,
            "decode_config": self.decode_config,
            "total_tokens": self.total_tokens,
            "d_model": self.d_model,
            "n_total_files": self.n_total_files,
            "batches_processed": self.batches_processed,
            "num_runs": self.num_runs,
        }

    @classmethod
    def from_yaml(cls, path: str = "configs/default.yaml") -> "Config":
        """Create configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Config: New configuration instance
        """
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        config_dict.setdefault(
            "decode_config", {
                "enable": False,
                "max_new_tokens": 1024,
                "temperature": 1.0,
                "top_p": 0.95,
                "stop_on_eos": True,
                "log_dir": "decode_logs",
            },
        )

        upload_config = config_dict.setdefault("upload_config", {})
        upload_config.setdefault("storage_backend", "s3")
        upload_config.setdefault("local_output_dir", "activations")

        return cls(**config_dict)

    @classmethod
    def from_args(cls) -> "Config":
        """Create configuration from command line arguments.

        Supports --config and --machine arguments.

        Returns:
            Config: New configuration instance
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config", type=str, default="configs/llama3.3_70b.yaml", help="Path to config file"
        )
        parser.add_argument(
            "--machine", type=int, default=0, help="Index of the machine for distributed processing"
        )
        args = parser.parse_args()
        config = cls.from_yaml(args.config)
        config.machine_index = args.machine
        return config

    def _get_next_config_number(self, s3_client: boto3.client) -> int:
        """Find the next available config number by checking existing configs."""
        try:
            # List all objects in the run directory
            response = s3_client.list_objects_v2(
                Bucket=self.data_config["bucket_name"], Prefix=f"{self.run_name}/cfg"
            )

            # Find all numbered configs (cfg1.json, cfg2.json, etc)
            numbers = [0]  # 0 represents the base cfg.json
            pattern = re.compile(r"cfg(\d+)\.json$")

            if "Contents" in response:
                for obj in response["Contents"]:
                    match = pattern.search(obj["Key"])
                    if match:
                        numbers.append(int(match.group(1)))

            return max(numbers) + 1
        except Exception:
            return 1

    def _get_latest_config_key(self, s3_client: boto3.client) -> str:
        """Find the latest config file (highest number)."""
        try:
            response = s3_client.list_objects_v2(
                Bucket=self.data_config["bucket_name"], Prefix=f"{self.run_name}/cfg"
            )

            latest_num = -1
            latest_key = f"{self.run_name}/cfg.json"
            pattern = re.compile(r"cfg(\d+)\.json$")

            if "Contents" in response:
                for obj in response["Contents"]:
                    match = pattern.search(obj["Key"])
                    if match:
                        num = int(match.group(1))
                        if num > latest_num:
                            latest_num = num
                            latest_key = obj["Key"]
                    elif obj["Key"].endswith("/cfg.json") and latest_num == -1:
                        latest_key = obj["Key"]

            return latest_key
        except Exception:
            return f"{self.run_name}/cfg.json"

    def _save_config_thread(self, s3_client: boto3.client) -> None:
        """Thread target for saving config to S3."""
        config_dict = self.to_dict()

        # Only determine the config file name if we haven't already
        if self._config_key is None:
            self._config_key = f"{self.run_name}/cfg.json"

        s3_client.put_object(
            Body=json.dumps(config_dict),
            Bucket=self.data_config["bucket_name"],
            Key=self._config_key,
        )

    def save_to_s3(self, s3_client: boto3.client, blocking: bool = False) -> None:
        """Save configuration to S3.

        Args:
            s3_client: Boto3 S3 client
            blocking: If True, wait for previous save to complete before starting new one

        Note:
            For a new run, the first save will create a new config file.
            Subsequent saves will overwrite the same file.
        """
        # Wait for previous save to complete if requested
        if blocking and self._save_thread is not None:
            self._save_thread.join()

        # Start new save thread
        self._save_thread = threading.Thread(target=self._save_config_thread, args=(s3_client,))
        self._save_thread.start()

        # If blocking, wait for this save to complete
        if blocking:
            self._save_thread.join()

    @classmethod
    def load_from_s3(cls, s3_client: boto3.client, bucket_name: str) -> Optional["Config"]:
        """Load existing configuration from S3.

        Args:
            s3_client: Boto3 S3 client
            run_name: Name of the run to load
            bucket_name: S3 bucket name

        Returns:
            Optional[Config]: Loaded configuration or None if not found
        """
        try:
            # Create temporary config to access methods
            temp_config = cls(
                run_name="temp", transformer_config={}, data_config={}, upload_config={}, num_runs=1
            )
            temp_config.data_config["bucket_name"] = bucket_name

            # Get the latest config key
            key = temp_config._get_latest_config_key(s3_client)

            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            config_dict = json.loads(response["Body"].read())

            config_dict.setdefault(
                "decode_config", {
                    "enable": False,
                    "max_new_tokens": 1024,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "stop_on_eos": True,
                    "log_dir": "decode_logs",
                },
            )

            # Convert from old format to new
            if "max_length" in config_dict:
                config_dict["seq_length"] = config_dict.pop("max_length")
            if "batches_per_cache" in config_dict:
                config_dict["batches_per_upload"] = config_dict.pop("batches_per_cache")

            # Move n_batches to data_config if it exists at root level
            if "n_batches" in config_dict:
                if "data_config" not in config_dict:
                    config_dict["data_config"] = {}
                config_dict["data_config"]["n_batches"] = config_dict.pop("n_batches")

            # Add missing fields with defaults
            config_dict.setdefault("num_runs", 1)  # Default to 1 if missing

            upload_config = config_dict.setdefault("upload_config", {})
            upload_config.setdefault("storage_backend", "s3")
            upload_config.setdefault("local_output_dir", "activations")

            # Remove thread field if present
            config_dict.pop("_save_thread", None)

            return cls(**config_dict)
        except s3_client.exceptions.NoSuchKey:
            return None

    @staticmethod
    def load_hook_statistics(
        s3_client: boto3.client, run_name: str, hook: str, bucket_name: str
    ) -> Optional[Dict[str, Any]]:
        """Load existing statistics for a hook from S3."""
        try:
            response = s3_client.get_object(
                Bucket=bucket_name, Key=f"{run_name}/{hook}/statistics.json"
            )
            return json.loads(response["Body"].read())
        except s3_client.exceptions.NoSuchKey:
            return None
