from pathlib import Path
import queue, threading, logging

logger = logging.getLogger(__name__)

class LocalHookUploader:
    def __init__(self, output_root, prefix_path, batches_per_upload=32):
        self.output_root = Path(output_root)
        self.prefix_path = self.output_root / Path(prefix_path)
        self.prefix_path.mkdir(parents=True, exist_ok=True)

        self.batches_per_upload = batches_per_upload
        self._in_mem = []
        self.current_group_uuid = None
        self.metadata = None

        self.pending_uploads = 0
        self.upload_attempt_count = 0

        self._queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._writer_loop, daemon=True)
        self._worker.start()

    def append(self, activations, group_uuid):
        self.current_group_uuid = group_uuid

        if self.metadata is None:
            self.metadata = self._get_metadata(activations, self.batches_per_upload)
            self._save_metadata()
        elif not self._validate_activations(activations):
            return None

        self._in_mem.append(activations)
        if len(self._in_mem) == self.batches_per_upload:
            return self._queue_save_in_mem()
        return None

    def save_stats(self, mean, std, norm, M2):
        stats = {"mean": mean.tolist(), "std": std.tolist()}
        if norm is not None:
            stats["norm"] = float(norm)
        if M2 is not None:
            stats["M2"] = M2.tolist()

        stats_path = self.prefix_path / "statistics.json"
        stats_path.write_text(json.dumps(stats))

    def finalize(self):
        if self.metadata is None:
            raise ValueError("Cannot finalize cache without any data")

        if self._in_mem:
            logger.debug(f"Flushing {len(self._in_mem)} residual batches for {self.prefix_path}")
            self._flush_in_mem(force=True)
            
        logger.info(f"Waiting for {self.pending_uploads} pending disk writes for {self.prefix_path}")
        wait_start = time.time()
        while self.pending_uploads > 0:
            try:
                self._queue.join()
                break
            except KeyboardInterrupt:
                break
            if time.time() - wait_start > 3600:
                logger.warning(f"Timeout waiting for disk writes to complete for {self.prefix_path} ({self.pending_uploads} pending)")
                break
        
        self.cleanup()

    def cleanup(self):
        self._stop_event.set()
        self._queue.put(None)
        self._worker.join(timeout=30)

    def _queue_save_in_mem(self):
        return self._flush_in_mem(force=False)

    def _flush_in_mem(self, force):
        if not self._in_mem:
            return None
        if not force and len(self._in_mem) != self.batches_per_upload:
            return None

        combined_states = torch.cat([item["states"] for item in self._in_mem])
        combined_input_ids = torch.cat([item["input_ids"] for item in self._in_mem])
        combined = {
            "states": combined_states,
            "input_ids": combined_input_ids,
        }
        group_uuid = self.current_group_uuid or str(uuid4())
        self.pending_uploads += 1
        self._queue.put((combined, group_uuid))
        self._in_mem = []
        self.current_group_uuid = group_uuid
        return group_uuid

    def _writer_loop(self):
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                self._queue.task_done()
                break

            activations, group_uuid = item
            try:
                self._save(activations, group_uuid)
            except Exception:
                logger.exception(f"Failed to write activations for {self.prefix_path}")
            finally:
                self.pending_uploads -= 1
                self._queue.task_done()

    def _save(self, activations_dict, group_uuid):
        filename = self._filename(group_uuid)
        file_path = self.prefix_path / filename

        torch.save(activations_dict, file_path)
        bytes_per_file = (
            activations_dict["states"].numel() * activations_dict["states"].element_size() +
            activations_dict["input_ids"].numel() * activations_dict["input_ids"].element_size()
        )

        if self.metadata is not None and "bytes_per_file" not in self.metadata:
            self.metadata["bytes_per_file"] = bytes_per_file
            self._save_metadata()

    def _validate_activations(self, activations):
        seq_len = self.metadata["sequence_length"] # decode and prefill are using different loaders
        d_in = self.metadata["d_in"]
        states = activations["states"]
        input_ids = activations["input_ids"]

        if states.shape[1] != seq_len or states.shape[2] != d_in:
            logger.warning(f"NOT SAVING: expected seq_len={seq_len} d_in={d_in}, got {states.shape[1:]}")
            return False 

        if seq_len == 1:
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

    def _save_metadata(self):
        if self.metadata is None:
            return

        metadata_path = self.prefix_path / "metadata.json"
        metadata_path.write_text(json.dumps(self.metadata))

    @staticmethod
    def _get_metadata(activations, batches_per_upload):
        states = activations["states"]
        input_ids = activations["input_ids"]
        return {
            "batch_size": states.shape[0],
            "sequence_length": states.shape[1],
            "dtype": str(states.dtype),
            "d_in": states.shape[2],
            "batches_per_file": batches_per_upload,
            "shape": list(states.shape),
            "input_ids_shape": list(input_ids.shape),
        }