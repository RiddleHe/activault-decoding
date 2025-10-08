# Activault 

A fork of the [Activault repo](https://github.com/tilde-research/activault) for high-throughput model activation generation & storage.

Built upon features in the original repo, this version enables activation generation for both prefill and decode and can parallelize storage on both S3 and local disk.

## Install
```
# python=3.10+ and CUDA-compatible PyTorch
pip install -r requirements
```

## Run job

1. Create a config file in `configs/`
2. Launch: 

    `python stash.py --config configs/[config file].yaml`
- Prefill activations go to `[local_output_dir]/[run_name]/[hook]/prefill`.
- Decode activations (when enabled) go to `[local_output_dir]/[run_name]/[hook]/decode`.
- Prompt/response transcripts land in `decode_logs/[run_name]`.
- Stats & resumes are stored under the same root.

### Config file

Fill the following fields in the config file:

```
run_name:
transformer_config:
    model_name:
    dtype: bfloat16
    cache_dir: /model-cache
decode_config:
    enable: true
    max_new_tokens: 256
    temperature: 0.7
    top_p: 0.95
    stop_on_eos: true
    log_dir: decode_logs
data_config:
    data_key: # must exist in datasets.json
    seq_length: 2048
    batch_size: 2
    n_batches: 5000
    start_batch: 0
    skip_cache: false
    clean_added_tokens: true
upload_config:
    hooks:
      - models.layer.15.self_attn.post
      - mdoels.layer.19.mlp.post
    batches_per_upload: 4
    storage_backend: disk
    local_output_dir: /tmp/activations
```