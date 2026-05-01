from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.optimizer import AdamW
import timeit
import torch
from typing import Literal

def benchmark(
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        n_warmup_steps: int,
        n_timed_steps: int,
        device: torch.device,
        torch_compile: bool = True,
        batch_size: int = 4,
        seed: int = 42,
        timed_operation: Literal["forward", "forward_and_backward", "full_training_step"] = "forward",
):
    model: BasicsTransformerLM = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta=rope_theta)
    model.to(device=device)
    print(f"running on {device}")
    if torch_compile:
        model: BasicsTransformerLM = torch.compile(model)
    torch.manual_seed(seed=seed)
    data = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length+1), device=device)
    input_data = data[:, :-1]
    targets = data[:, 1:]
    assert timed_operation in ["forward", "forward_and_backward", "full_training_step"]
    print(f"warming up for timing {timed_operation}")
    if timed_operation == "forward":
        for _ in range(n_warmup_steps):
            model(input_data)
        print("warmed up")
        start = timeit.default_timer()
        for _ in range(n_timed_steps):
            model(input_data)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        end = timeit.default_timer()
    elif timed_operation == "forward_and_backward":
        for _ in range(n_warmup_steps):
            logits = model(input_data)
            loss = cross_entropy(logits, targets)
            loss.backward()
        print("warmed up")
        start = timeit.default_timer()
        for _ in range(n_timed_steps):
            logits = model(input_data)
            loss = cross_entropy(logits, targets)
            loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        end = timeit.default_timer()
    else:
        optimizer = AdamW(model.parameters())
        for _ in range(n_warmup_steps):
            optimizer.zero_grad()
            logits = model(input_data)
            loss = cross_entropy(logits, targets)
            loss.backward()
            clip_gradient(model.parameters(), max_norm=1.0)
            optimizer.step()
        print("warmed up")
        start = timeit.default_timer()
        for _ in range(n_timed_steps):
            optimizer.zero_grad()
            logits = model(input_data)
            loss = cross_entropy(logits, targets)
            loss.backward()
            clip_gradient(model.parameters(), max_norm=1.0)
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        end = timeit.default_timer()

    print(f"timed {timed_operation} for {n_timed_steps} steps with {n_warmup_steps} warmup steps, total time taken is {end-start} sec")


if __name__ == "__main__":
    device = torch.device("cuda")
    benchmark(
        vocab_size=10_000,
        context_length=512,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        rope_theta=10_000.0,
        n_warmup_steps=5,
        n_timed_steps=10,
        torch_compile=False,
        device=device,
        batch_size=4,
        timed_operation="full_training_step"
    )



