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

import json
import threading
import queue
import os
import torch
from uuid import uuid4
import time
import multiprocessing as mp
import os
import io
import atexit
from s3.utils import create_s3_client
import math
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _metadata_path(prefix_path: str) -> str:
    """Generate the metadata file path for a given prefix path.

    Args:
        prefix_path: Base path for the hook's data

    Returns:
        str: Path in format "{prefix_path}/metadata.json"
    """
    return f"{prefix_path}/metadata.json"


def _statistics_path(prefix_path: str) -> str:
    """Generate the statistics file path for a given prefix path.

    Args:
        prefix_path: Base path for the hook's data

    Returns:
        str: Path in format "{prefix_path}/statistics.json"
    """
    return f"{prefix_path}/statistics.json"


def _strip_run_name(prefix_path):
    """Strip the run name from the prefix path."""
    return prefix_path.split("/", 1)[1]


class HookUploader:
    """Asynchronous uploader for neural network activation data to S3-compatible storage.

    This class manages the collection, batching, and uploading of activation data from
    neural network hooks. It uses a multi-process architecture with threaded workers
    to handle concurrent uploads efficiently.

    Architecture:
        - Main Process: Collects activations and queues them for upload
        - Upload Process: Manages upload threads and data transfer
        - Upload Threads: Handle actual S3 upload operations

    Attributes:
        batches_per_upload (int): Number of batches to accumulate before upload
        prefix_path (str): Base path for storing data in S3
        bucket_name (str): S3 bucket name
        num_upload_threads (int): Number of concurrent upload threads
        pending_uploads (mp.Value): Counter for pending upload operations
        upload_queue_size (mp.Value): Size of the upload queue
        running_mean (Optional[torch.Tensor]): Running mean of activations
        running_std (Optional[torch.Tensor]): Running standard deviation
        running_norm (float): Running norm of activations
        n_batches_processed (int): Total number of processed batches

    Example:
        ```python
        uploader = HookUploader.from_credentials(
            access_key_id="your_key",
            secret="your_secret",
            prefix_path="run_name/hook_name",
            batches_per_upload=32
        )

        # Add activations
        uploader.append({
            'states': activation_tensor,
            'input_ids': input_ids_tensor
        })

        # Finalize and cleanup
        uploader.finalize()
        ```
    """

    @classmethod
    def from_credentials(cls, access_key_id: str, secret: str, *args, **kwargs) -> "HookUploader":
        """Create an HookUploader instance using storage credentials.

        Args:
            access_key_id: Storage access key ID
            secret: Storage secret access key
            *args: Additional positional arguments for HookUploader
            **kwargs: Additional keyword arguments for HookUploader

        Returns:
            HookUploader: Configured uploader instance

        Raises:
            Exception: If bucket verification fails
        """
        s3_client = create_s3_client(access_key_id, secret)
        logger.debug("Created S3 client with endpoint: %s", s3_client._endpoint)
        return cls(s3_client, *args, **kwargs)

    def __init__(
        self,
        s3_client,
        prefix_path: str,
        batches_per_upload: int = 32,
        bucket_name: str = "renes-bucket",
        num_upload_threads: int = 1,
    ):
        """Initialize the HookUploader."""
        self.batches_per_upload = batches_per_upload
        self.prefix_path = prefix_path
        self._in_mem = []  # Back to being a simple list
        self.current_group_uuid = None  # Track current UUID
        self.s3_client = s3_client
        self.metadata = None
        self.bucket_name = bucket_name
        self.num_upload_threads = num_upload_threads
        self.upload_attempt_count = 0
        self.pending_uploads = 0

        logger.debug("Initializing HookUploader with bucket: %s", self.bucket_name)

        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.debug("Successfully verified bucket %s exists", self.bucket_name)
        except Exception as e:
            logger.error("Error verifying bucket %s: %s", self.bucket_name, str(e), exc_info=True)
            raise

        self.running_mean = None
        self.running_std = None
        self.running_norm = 0.0
        self.n_batches_processed = 0

        self.mp_upload_queue = mp.Queue(2)
        self.stop_event = mp.Event()

        self.upload_process = mp.Process(target=self._upload_worker)
        self.upload_process.start()

        atexit.register(self.cleanup)

    def cleanup(self) -> None:
        """Clean up resources and ensure all uploads are complete.

        Waits for pending uploads to complete (with timeout) and terminates
        the upload process. Automatically called on program exit.
        """
        logger.info("Cleaning up HookUploader for %s...", _strip_run_name(self.prefix_path))
        self.stop_event.set()
        if self.upload_process.is_alive():
            minutes_to_wait = 10
            self.upload_process.join(timeout=minutes_to_wait * 60)
            if self.upload_process.is_alive():
                logger.warning(
                    "HookUploader for %s upload process is still alive. TERMINATING...",
                    _strip_run_name(self.prefix_path),
                )
                self.upload_process.terminate()
        logger.debug("HookUploader for %s cleanup complete.", _strip_run_name(self.prefix_path))

    def _upload_worker(self) -> None:
        """Worker process for handling S3 uploads.

        Manages a pool of upload threads and coordinates data transfer from
        the multiprocessing queue to the thread queue.

        Note:
            Runs in a separate process and manages its own thread pool.
        """
        upload_queue = queue.Queue(2)
        thread_stop_event = threading.Event()
        logger.debug("%s PID: %s", _strip_run_name(self.prefix_path), os.getpid())

        threads = []
        for _ in range(self.num_upload_threads):
            t = threading.Thread(target=self._upload_thread, args=(upload_queue, thread_stop_event))
            t.start()
            threads.append(t)

        last_log_time = time.time()
        log_interval = 10  # Log every 10 seconds

        while True:
            if self.stop_event.is_set() and self.mp_upload_queue.empty():
                logger.debug(
                    "stop_event reached, mp_upload_queue size at moment: %d",
                    self.mp_upload_queue.qsize(),
                )
                break

            try:
                item = self.mp_upload_queue.get(timeout=10)
                self.pending_uploads += 1
                upload_queue.put(item)

                # Log queue sizes periodically
                current_time = time.time()
                if current_time - last_log_time > log_interval:
                    logger.debug(
                        "[%s QUEUE STATUS] MP Queue: %d, Thread Queue: %d, Pending Uploads: %d",
                        _strip_run_name(self.prefix_path),
                        self.mp_upload_queue.qsize(),
                        upload_queue.qsize(),
                        self.pending_uploads,
                    )
                    last_log_time = current_time

            except mp.queues.Empty:
                continue
            except Exception as e:
                logger.error("Exception in _upload_worker: %s", str(e), exc_info=True)
                raise e

        thread_stop_event.set()

        for t in threads:
            t.join()

    def _upload_thread(self, upload_queue: queue.Queue, stop_event: threading.Event) -> None:
        """Thread for uploading data to S3.

        Args:
            upload_queue: Queue containing data to upload
            stop_event: Event signaling thread should stop

        Note:
            Runs in upload worker process and handles actual S3 uploads.
        """
        while True:
            if stop_event.is_set() and upload_queue.empty():
                logger.debug(
                    "stop_event reached, upload_queue size at moment: %d", upload_queue.qsize()
                )
                break

            try:
                activations, group_uuid = upload_queue.get(timeout=1)
                self._save(activations, group_uuid)
                self.pending_uploads -= 1  # Decrement pending uploads after successful save
                upload_queue.task_done()
            except queue.Empty:
                if stop_event.is_set():
                    continue
                time.sleep(0.25)
                continue
            except Exception as e:
                logger.error("Exception in _upload_thread: %s", str(e), exc_info=True)
                raise e

    def append(self, activations: dict, group_uuid: str) -> Optional[str]:
        """Append activations to the cache, queueing for S3 upload when batch is full.

        Args:
            activations: Dictionary containing activation data
            group_uuid: UUID for this batch group (shared across layers)

        Returns:
            Optional[str]: Upload ID if batch was queued, None otherwise
        """
        # Update current group UUID
        self.current_group_uuid = group_uuid

        if self.metadata is None:
            self.metadata = self._get_metadata(activations, self.batches_per_upload)
            self._save_metadata()
        else:
            if not self._validate_activations(activations):
                return None

        self._in_mem.append(activations)

        if len(self._in_mem) == self.batches_per_upload:
            return self._queue_save_in_mem()

        return None

    def _queue_save_in_mem(self) -> str:
        """Queue the in-memory activations for S3 upload.

        Returns:
            str: The group UUID used for this upload
        """
        # Combine all states and input_ids from the group
        combined_states = torch.cat([item["states"] for item in self._in_mem])
        combined_input_ids = torch.cat([item["input_ids"] for item in self._in_mem])

        states_bytes = combined_states.numel() * combined_states.element_size()
        input_ids_bytes = combined_input_ids.numel() * combined_input_ids.element_size()

        combined_dict = {
            "states": combined_states,
            "input_ids": combined_input_ids,
            "tensor_bytes": states_bytes + input_ids_bytes,
        }

        self.mp_upload_queue.put((combined_dict, self.current_group_uuid))
        self._in_mem = []
        return self.current_group_uuid

    def _save(self, activations_dict: dict, group_uuid: str) -> None:
        """Save the activations dictionary to S3 using multipart upload.

        Args:
            activations_dict: Dictionary containing activation data
            group_uuid: UUID for this batch group

        Raises:
            Exception: If upload fails

        Note:
            Uses multipart upload for large files and handles failures.
        """

        filename = self._filename(group_uuid)
        logger.debug("Starting upload of %s", filename)

        serialization_start = time.time()
        buffer = io.BytesIO()
        torch.save(activations_dict, buffer)
        buffer.seek(0)
        tensor_bytes = buffer.getvalue()
        serialization_end = time.time()
        serialization_time = serialization_end - serialization_start

        # Start multipart upload
        upload_start = time.time()
        multipart_upload = self.s3_client.create_multipart_upload(
            Bucket=self.bucket_name, Key=filename, ContentType="application/octet-stream"
        )
        upload_id = multipart_upload["UploadId"]

        try:
            # Upload parts
            part_size = 100 * 1024 * 1024  # 100 MB
            num_parts = math.ceil(len(tensor_bytes) / part_size)
            parts = []

            for part_number in range(1, num_parts + 1):
                start_byte = (part_number - 1) * part_size
                end_byte = min(part_number * part_size, len(tensor_bytes))

                part_response = self.s3_client.upload_part(
                    Bucket=self.bucket_name,
                    Key=filename,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=tensor_bytes[start_byte:end_byte],
                )
                parts.append({"PartNumber": part_number, "ETag": part_response["ETag"]})

            # Complete multipart upload
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=filename,
                MultipartUpload={"Parts": parts},
                UploadId=upload_id,
            )

        except Exception as e:
            logger.error("Upload failed: %s", str(e))
            # Abort the multipart upload on any error
            self.s3_client.abort_multipart_upload(
                Bucket=self.bucket_name, Key=filename, UploadId=upload_id
            )
            raise e

        upload_end = time.time()
        upload_time = upload_end - upload_start
        total_time = upload_end - serialization_start

        # Save metadata
        self.metadata = self._get_metadata(
            activations_dict, self.batches_per_upload, len(tensor_bytes)
        )
        self.metadata["bytes_per_file"] = len(tensor_bytes)
        self._save_metadata()

        logger.debug(
            "Successfully saved %s to S3! PID: %d, Serialization time: %.2fs, S3 upload time: %.2fs, Total time: %.2fs",
            _strip_run_name(filename),
            os.getpid(),
            serialization_time,
            upload_time,
            total_time,
        )

    def finalize(self) -> None:
        """Finalize the cache, ensuring all data is saved and processes are stopped."""
        if self.metadata is None:
            raise ValueError("Cannot finalize cache without any data")

        # Handle any remaining batches
        if len(self._in_mem) > 0:
            if len(self._in_mem) == self.batches_per_upload:
                logger.debug("Queueing save for final full batch group")
                self._queue_save_in_mem()
            else:
                logger.warning("Discarding %d incomplete batches", len(self._in_mem))
                self._in_mem = []

        # Wait for all pending uploads to complete
        logger.info(
            "Waiting for %d pending uploads to complete for %s...",
            self.pending_uploads,
            self.prefix_path,
        )
        wait_start = time.time()
        while self.pending_uploads > 0:
            if time.time() - wait_start > 3600:  # 1 hour timeout
                logger.warning(
                    "Timeout waiting for uploads to complete. %d uploads still pending.",
                    self.pending_uploads,
                )
                break
            time.sleep(5)

        self.cleanup()

    def _validate_activations(self, activations: dict) -> bool:
        """Validate the shape and dtype of the activations against the metadata.

        Args:
            activations: Dictionary containing activation data

        Returns:
            bool: True if validation passes, False otherwise
        """
        seq_len = self.metadata["sequence_length"] # decode and prefill are using different loaders
        d_in = self.metadata["d_in"]
        states = activations["states"]
        input_ids = activations["input_ids"]

        if states.shape[1] != seq_len or states.shape[2] != d_in:
            logger.warning(f"NOT SAVING: expected seq_len={seq_len} d_in={d_in}, got {states.shape[1:]}")
            return False 

        if seq_len == 1: # decode have varying batch size, so avoid the check
            if input_ids.shape[0] != states.shape[0] or input_ids.shape[1] != 1:
                logger.warning(f"NOT SAVING: expected decode input_ids=({states.shape[0]}, 1), got {input_ids.shape}")
                return False

        else:
            expected_batch = self.metadata["batch_size"]
            if states.shape[0] != expected_batch or input_ids.shape != (expected_batch, seq_len):
                logger.warning(f"NOT SAVING: expected batch={expected_batch} input_ids=({expected_batch},{seq_len}), got {states.shape[0]}, {input_ids.shape}")
                return False

        if str(states.dtype) != self.metadata["dtype"]:
            logger.warning(f"NOT SAVING: expected dtype={self.metadata['dtype']}, got {states.dtype}")
            return False

        return True

    def _save_metadata(self) -> None:
        """Save the metadata to S3.

        Stores metadata about activation shapes, types, and batch sizes.
        """
        logger.debug("Saving metadata for %s", _strip_run_name(self.prefix_path))
        self.s3_client.put_object(
            Body=json.dumps(self.metadata),
            Bucket=self.bucket_name,
            Key=_metadata_path(self.prefix_path),
        )

    def _filename(self, group_uuid: str) -> str:
        """Generate the filename for a given group UUID.

        Args:
            group_uuid: UUID for this batch group

        Returns:
            str: Full path including attempt number
        """
        slurm_job = os.getenv("SLURM_JOB_ID", 0)
        self.upload_attempt_count += 1
        return f"{self.prefix_path}/{group_uuid}--{self.upload_attempt_count}_{slurm_job}.saved.pt"

    def _get_metadata(
        self, activations: dict, batches_per_upload: int, bytes_per_file: Optional[int] = None
    ) -> dict:
        """Create metadata dictionary from activation data.

        Args:
            activations: Dictionary containing activation data
            batches_per_upload: Number of batches per upload file
            bytes_per_file: Optional size of file in bytes

        Returns:
            dict: Metadata including shapes, types, and sizes
        """
        metadata = {
            "batch_size": activations["states"].shape[0],
            "sequence_length": activations["states"].shape[1],
            "dtype": str(activations["states"].dtype),
            "d_in": activations["states"].shape[2],
            "batches_per_file": batches_per_upload,
            "shape": list(activations["states"].shape),
            "input_ids_shape": list(activations["input_ids"].shape),
        }
        if bytes_per_file is not None:
            metadata["bytes_per_file"] = bytes_per_file
        return metadata

    def save_stats(self, mean: torch.Tensor, std: torch.Tensor, norm: float, M2: torch.Tensor):
        """
        Save statistics for the hook.

        Args:
            mean: Mean tensor
            std: Standard deviation tensor
            norm: Optional average L2 norm
            M2: Optional M2 value from Welford's algorithm for resumable stats
        """
        stats = {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

        if norm is not None:
            stats["norm"] = float(norm)  # Ensure norm is a Python float

        if M2 is not None:
            stats["M2"] = M2.tolist()  # Convert M2 tensor to list

        # Save stats to S3 directly using the client
        self.s3_client.put_object(
            Body=json.dumps(stats),
            Bucket=self.bucket_name,
            Key=_statistics_path(self.prefix_path),
        )
