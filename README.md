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