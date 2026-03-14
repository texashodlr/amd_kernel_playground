# sweep_ref.py
import time
import torch
import reference

CASES = [
    (4, 1024),
    (4, 2048),
    (4, 4096),
    (4, 8192),
    (16, 1024),
    (16, 2048),
    (16, 4096),
    (16, 8192),
    (32, 1024),
    (32, 2048),
    (32, 4096),
    (32, 8192),
    (64, 1024),
    (64, 2048),
    (64, 4096),
    (64, 8192),
    (128, 1024),
    (128, 2048),
    (128 4096),
    (128, 8192),
    (256, 1024),
    (256, 2048),
    (256, 4096),
    (256, 8192),
]

def run_case(bs, kv, tp=8, warmup=10, iters=50):
    data = reference.generate_input(batchsize=bs, qseqlen=1, kvseqlen=kv, tp=tp, seed=1234)

    for _ in range(warmup):
        _ = reference.ref_kernel(data)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = reference.ref_kernel(data)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1e6 / iters  # microseconds

if __name__ == "__main__":
    assert torch.cuda.is_available()

    reference.Q_DTYPE = "fp8"
    reference.KV_DTYPE = "fp8"

    for bs, kv in CASES:
        us = run_case(bs, kv)
        print(f"bs={bs:3d} kv={kv:4d}  latency={us:8.2f} us")