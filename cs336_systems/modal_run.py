import modal

app = modal.App("benchmarking")

image = (
    modal.Image.debian_slim()
    .add_local_dir("./cs336-basics", "/.uv/cs336-basics", copy=True)
    .uv_sync()
    .add_local_python_source("cs336_systems")
)

@app.function(
    image=image,
    gpu="h100"
)
def run_benchmark():
    import torch
    from cs336_systems.benchmarking import benchmark
    device = torch.device("cuda")
    print("---no torch.compile---")
    for operation in ["forward", "forward_and_backward", "full_training_step"]:
        benchmark(
            vocab_size=10_000,
            context_length=512,
            d_model=1280,
            num_layers=36,
            num_heads=20,
            d_ff=5120,
            rope_theta=10_000.0,
            n_warmup_steps=5,
            n_timed_steps=10,
            torch_compile=False,
            device=device,
            batch_size=4,
            timed_operation=operation,
        )
    print("---use torch.compile---")
    for operation in ["forward", "forward_and_backward", "full_training_step"]:
        benchmark(
            vocab_size=10_000,
            context_length=512,
            d_model=1280,
            num_layers=36,
            num_heads=20,
            d_ff=5120,
            rope_theta=10_000.0,
            n_warmup_steps=5,
            n_timed_steps=10,
            torch_compile=True,
            device=device,
            batch_size=4,
            timed_operation=operation,
        )


@app.local_entrypoint()
def main():
    run_benchmark.remote()
