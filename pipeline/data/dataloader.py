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

from typing import Iterator, List, Dict
import torch
import numpy as np
from transformers import PreTrainedTokenizer, DataCollatorWithFlattening
from datasets import IterableDataset
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# If your samples are very short and you are cleaning tokens, you may need to increase this.
# Alternatively, you can reduce this for higher efficiency.
MAX_N_PACKED = 12


class DataLoader:
    """
    A streaming dataloader that packs clean token sequences for activation collection.
    Removes redundant/anomalous tokens (special tokens, chat templates, etc.) from the dataset.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        batch_size: int,
        start_batch_skip: int = 0,
        batches_per_machine: int = None,
        dataset_key: str = None,
        skip_cache: bool = False,
        clean_added_tokens: bool = True,
        clean_default_system_prompt: bool = True,
        add_generation_prompt: bool = False,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.batches_per_machine = batches_per_machine
        self.apply_chat_template = dataset_key.endswith(":CHAT_TEMPLATE")
        self.skip_cache = skip_cache
        self.clean_added_tokens = clean_added_tokens
        self.clean_default_system_prompt = clean_default_system_prompt
        self.add_generation_prompt = self.apply_chat_template and add_generation_prompt
        logger.info(f"NOTICE: Cleaning default system prompt: {clean_default_system_prompt}")
        # Initialize dataset iterator
        self.dataset_iter = iter(self.dataset)

        # Setup invalid tokens list
        self.invalid_tokens = [
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.bos_token_id,
            -100,
            128000,  # Chat template token
        ]

        if hasattr(self.tokenizer, "added_tokens_decoder") and self.clean_added_tokens:
            self.invalid_tokens.extend(self.tokenizer.added_tokens_decoder.keys())

        # Add buffer to account for token replacement
        self.buffer_size = 0
        if self.clean_added_tokens:
            self.buffer_size += 20 * MAX_N_PACKED

        # Chat-specific setup
        self.generation_prompt_tokens = []
        if self.apply_chat_template and self.clean_default_system_prompt:
            # Get the start string pattern by applying template to a dummy message
            dummy_msg = [{"role": "user", "content": "test"}]
            template_text = tokenizer.apply_chat_template(dummy_msg, tokenize=False)

            pos = template_text.find("test")
            self.begin_str = template_text[:pos]

            # Tokenize string to remove
            self.tokenized_system_prompt = tokenizer(
                self.begin_str, truncation=False, return_attention_mask=False
            )["input_ids"][1:]

            self.buffer_size += len(self.tokenized_system_prompt) * MAX_N_PACKED

            if self.add_generation_prompt:
                template_with_prompt = tokenizer.apply_chat_template(
                    dummy_msg, tokenize=False, add_generation_prompt=True
                )
                prompt_suffix = template_with_prompt[len(template_text):]
                if prompt_suffix:
                    tokenized_prompt = tokenizer(
                        prompt_suffix, truncation=False, return_attention_mask=False
                    )["input_ids"][1:]
                    self.generation_prompt_tokens = tokenized_prompt
                    self.buffer_size += len(self.generation_prompt_tokens) * MAX_N_PACKED

        self.base_max_length = max_length
        self.max_length = max_length + self.buffer_size
        logger.info(f"Buffer size: {self.buffer_size}")

        # Initialize state
        self.current_batch_tokens = 0
        self.current_texts = []
        self.cached_text = None

        # Setup collator
        self.collator = DataCollatorWithFlattening(return_position_ids=True, separator_id=-100)

        # Initial skip based on start batch
        start_tokens = 25 * start_batch_skip * batch_size * max_length
        self.skip_tokens(start_tokens)

    def skip_tokens(self, num_tokens_to_skip: int) -> None:
        """Skip the specified number of tokens."""
        tokens_so_far = 0
        n_seqs = 0

        # Heuristic -> average seq_len = 256
        if num_tokens_to_skip > 1e7:
            base_token_volume = 64
            self.dataset = self.dataset.skip((num_tokens_to_skip - 1e7) // base_token_volume)
            tokens_so_far = num_tokens_to_skip - 1e7
            n_seqs = (num_tokens_to_skip - 1e7) // base_token_volume

        pbar = tqdm(total=num_tokens_to_skip, desc="Skipping tokens")
        pbar.update(tokens_so_far)

        while tokens_so_far < num_tokens_to_skip:
            text_batch = []
            for _ in range(64):
                try:
                    text = next(self.dataset_iter)["text"]
                    if self.apply_chat_template:
                        text_batch.append(
                            [self._convert_chat_format(text[i]) for i in range(len(text))]
                        )
                    else:
                        text_batch.append(text)
                except Exception as e:
                    if isinstance(e, StopIteration):
                        break
                    else:
                        logger.error(f"Error skipping tokens: {e}")

            if not text_batch:
                break

            if self.apply_chat_template:
                chat_kwargs = {"tokenize": False}
                if self.add_generation_prompt:
                    chat_kwargs["add_generation_prompt"] = True
                text_batch = self.tokenizer.apply_chat_template(text_batch, **chat_kwargs)

            tokenized = self.tokenizer(text_batch, truncation=False, return_attention_mask=False)

            for input_ids in tokenized["input_ids"]:
                n_tokens = len(self._clean_sequence(input_ids)[0])
                tokens_so_far += n_tokens
                n_seqs += 1
                pbar.update(n_tokens)

                if tokens_so_far >= num_tokens_to_skip:
                    break

        pbar.close()
        logger.info(f"Skipped {tokens_so_far} tokens from {n_seqs} sequences")

    @staticmethod
    def _convert_chat_format(msg: Dict[str, str]) -> Dict[str, str]:
        """Unify chat message formats."""
        try:
            # If already in correct format, validate fields
            if "role" in msg:
                if not isinstance(msg.get("content"), str):
                    # Default to empty string if content is None or invalid
                    msg["content"] = ""
                if not isinstance(msg.get("role"), str):
                    # Default to user if role is None or invalid
                    msg["role"] = "user"
                return msg
            
            # Convert from old format
            if not isinstance(msg.get("from"), str) or not isinstance(msg.get("value"), str):
                # Default to empty message if fields are invalid
                return {"role": "user", "content": ""}
            
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            role = role_map.get(msg["from"], "user")  # Default to user if unknown role
            return {"role": role, "content": msg["value"]}
        except Exception as e:
            logger.warning(f"Error converting chat format for message: {msg}. Error: {e}")
            return {"role": "user", "content": ""}  # Return safe default

    def _clean_sequence(self, input_ids: torch.Tensor) -> tuple[List[int], List[bool]]:
        """Clean a sequence by removing invalid tokens."""
        ids = input_ids.cpu().numpy() if torch.is_tensor(input_ids) else np.array(input_ids)
        length = len(ids)

        # Initialize mask array
        mask = np.ones(length, dtype=bool)

        # Remove invalid tokens
        invalid_mask = np.isin(ids, self.invalid_tokens)
        mask &= ~invalid_mask

        if self.clean_default_system_prompt:
            # Remove both sys prompt
            for sys_str in [self.tokenized_system_prompt]:
                cutoff_len = len(sys_str)
                if cutoff_len > 0 and len(ids) >= cutoff_len:
                    windows = np.lib.stride_tricks.sliding_window_view(ids, cutoff_len)
                    matches = np.all(windows == sys_str, axis=1)
                    match_positions = np.where(matches)[0]
                    for pos in match_positions:
                        mask[pos : pos + cutoff_len] = False

        if self.add_generation_prompt:
            cutoff_len = len(self.generation_prompt_tokens)
            if cutoff_len > 0 and len(ids) >= cutoff_len:
                windows = np.lib.stride_tricks.sliding_window_view(ids, cutoff_len)
                matches = np.all(windows == self.generation_prompt_tokens, axis=1)
                match_positions = np.where(matches)[0]
                for pos in match_positions:
                    mask[pos : pos + cutoff_len] = False

        return ids[mask].tolist(), mask.tolist()

    def clean_batch(
        self, input_ids: torch.Tensor, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Clean a batch of sequences by removing invalid tokens.

        Args:
            input_ids: tensor of shape [batch_size, seq_len]
            states: tensor of shape [batch_size, seq_len, hidden_dim]
        Returns:
            cleaned input_ids and states
        """
        batch_size, seq_len = input_ids.shape
        true_input_ids = []
        true_states = []

        for i in range(batch_size):
            valid_ids, valid_mask = self._clean_sequence(input_ids[i])
            valid_positions = torch.where(torch.tensor(valid_mask))[0]

            if len(valid_positions) > 0:
                true_input_ids.append(
                    input_ids[i][valid_positions].contiguous()[: self.base_max_length]
                )
                true_states.append(states[i][valid_positions].contiguous()[: self.base_max_length])
        try:
            return torch.stack(true_input_ids), torch.stack(true_states)
        except Exception as e:
            logger.error(
                f"Error cleaning batch. \nThis is likely due to too small a buffer. Consider increasing MAX_N_PACKED in dataloader.py"
            )
            raise e

    def __len__(self) -> int:
        return len(self.dataset) // (self.batch_size * self.max_length)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        self.processed_batches = 0
        return self

    @staticmethod
    def _fix_chat_sequence(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Ensure chat messages follow the required alternating pattern."""
        if not messages:
            return [{"role": "user", "content": ""}]
        
        fixed = []
        # Handle optional system message
        if messages[0]["role"] == "system":
            fixed.append(messages[0])
            messages = messages[1:]
        
        # Start with user if no messages or last was assistant
        if not messages or (fixed and fixed[-1]["role"] == "assistant") or messages[0]["role"] == "assistant":
            fixed.append({"role": "user", "content": ""})
        
        # Process remaining messages ensuring alternation
        for msg in messages:
            # Skip if would violate alternation
            if fixed and msg["role"] == fixed[-1]["role"]:
                continue
            # Add dummy message if needed to maintain alternation
            if fixed and fixed[-1]["role"] not in ["user", "assistant"]:
                fixed.append({"role": "user" if msg["role"] == "assistant" else "assistant", "content": ""})
            fixed.append(msg)
        
        # Ensure we end with assistant response
        if not fixed or fixed[-1]["role"] == "user":
            fixed.append({"role": "assistant", "content": ""})
        
        return fixed

    def __next__(self) -> Dict[str, torch.Tensor]:
        if (
            self.batches_per_machine is not None
            and self.processed_batches >= self.batches_per_machine
        ):
            raise StopIteration

        # Initialize sequence buffer
        seq_buffer = []

        # Keep collecting sequences until we have batch_size or hit StopIteration
        while len(seq_buffer) < self.batch_size:
            try:
                # Get next sequence
                if self.cached_text and not self.apply_chat_template and not self.skip_cache:
                    sample = self.cached_text
                    self.cached_text = None
                else:
                    sample = next(self.dataset_iter)["text"]

                if self.apply_chat_template:
                    messages = [self._convert_chat_format(s) for s in sample]
                    messages = self._fix_chat_sequence(messages)
                    chat_kwargs = {"tokenize": False}
                    if self.add_generation_prompt:
                        chat_kwargs["add_generation_prompt"] = True
                    sample = self.tokenizer.apply_chat_template(messages, **chat_kwargs)

                tokenized = self.tokenizer(sample, truncation=False, return_attention_mask=False)

                n_tokens = len(tokenized["input_ids"])
                self.current_texts.append(tokenized)

                space_left = self.max_length - self.current_batch_tokens
                if n_tokens > space_left:
                    if n_tokens > self.max_length * 2:
                        self.cached_text = self.tokenizer.decode(
                            self.current_texts[-1]["input_ids"][space_left:]
                        )
                    self.current_texts[-1]["input_ids"] = self.current_texts[-1]["input_ids"][
                        :space_left
                    ]

                    # Add packed sequence to buffer
                    texts = self.current_texts
                    self.current_texts = []
                    self.current_batch_tokens = 0
                    seq_buffer.append(self.collator(texts))

                else:
                    self.current_batch_tokens += n_tokens
                    if self.current_batch_tokens >= self.max_length:
                        # Add packed sequence to buffer
                        texts = self.current_texts
                        self.current_texts = []
                        self.current_batch_tokens = 0
                        seq_buffer.append(self.collator(texts))

            except StopIteration:
                if self.current_texts:
                    # Add final packed sequence to buffer
                    seq_buffer.append(self.collator(self.current_texts))
                    self.current_texts = []
                    self.current_batch_tokens = 0
                if not seq_buffer:
                    raise StopIteration
                break

        # Stack all sequences in the buffer into a batch
        self.processed_batches += 1
        return {
            "input_ids": torch.cat([seq["input_ids"] for seq in seq_buffer], dim=0),
            "position_ids": torch.cat([seq["position_ids"] for seq in seq_buffer], dim=0),
        }
