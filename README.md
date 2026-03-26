# gpu-occupy

`gpu-occupy` is a small Rust CLI for reserving NVIDIA GPU memory on Linux.

It is designed for the practical case where you want a card to stay visibly in use by holding VRAM, without needing a heavy training or inference process.

## Features

- Interactive terminal UI when launched as `gpu-occupy`
- Multi-select one or more GPUs
- Treats the entered memory size as a total target and splits it evenly across selected GPUs
- Direct CLI subcommands for scripting: `list`, `hold`, `install`
- Clean release on `Ctrl+C`

## Requirements

- Linux
- NVIDIA GPU
- `nvidia-smi` available in `PATH`
- CUDA driver library available as `libcuda.so`
- Rust toolchain if building from source

## Build

```bash
cargo build --release
```

## Install

Build and install a local command:

```bash
cargo build --release
./target/release/gpu-occupy install
```

If `~/.local/bin` is in your `PATH`, you can then run:

```bash
gpu-occupy
```

## Interactive Usage

Run:

```bash
gpu-occupy
```

Menu actions:

- `Hold GPU memory`
- `List GPUs`
- `Install gpu-occupy into ~/.local/bin`
- `Exit`

The hold flow asks for:

- target GPU list, with multi-select support
- memory size in GiB

If you choose multiple GPUs, the entered memory is split evenly across them.
Example: `10GiB` across 2 GPUs becomes `5GiB` per GPU.

Interactive mode uses these defaults internally:

- reserve: `1GiB`
- chunk size: `256MiB`
- status interval: `5s`

## Direct CLI Usage

List GPUs:

```bash
gpu-occupy list
```

Hold 20 GiB on GPU 2:

```bash
gpu-occupy hold --gpu 2 --memory 20GiB
```

Auto-pick the emptiest GPU that can satisfy the request:

```bash
gpu-occupy hold --memory 20GiB
```

Override reserve and status interval:

```bash
gpu-occupy hold --gpu 2 --memory 20GiB --reserve 2GiB --status-interval 2
```

## Notes

- Supported size units include `MiB`, `GiB`, `MB`, `GB`
- Actual observed memory usage in `nvidia-smi` can be slightly larger than the requested amount because CUDA context and allocator overhead also consume VRAM
- If you omit `--gpu`, the tool auto-selects a single GPU with enough free memory
- If you select multiple GPUs in interactive mode, the total target is evenly distributed

## Privacy

- The tool does not send data over the network
- The tool does not read model files or datasets
- The tool only queries local GPU state via `nvidia-smi` and allocates VRAM through the CUDA driver API

## License

Apache-2.0. See [LICENSE](LICENSE).
