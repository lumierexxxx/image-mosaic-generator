import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from mosaic_core import (
    preprocess,
    compute_vectorized,
    generate,
    build_mosaic,
    benchmark_grid_sizes,
)

GRID_SIZES = [16, 32, 64]
N_REPEATS = 5

def benchmark_grid_computation(img_path: str):
    pil_img = Image.open(img_path)
    results = benchmark_grid_sizes(pil_img, grid_sizes=GRID_SIZES, n_repeats=N_REPEATS)
    return {
        "grid_size": [r["grid_size"] for r in results],
        "vectorized_ms": [r["vectorized_ms"] for r in results],
        "loop_ms": [r["loop_ms"] for r in results],
    }


def benchmark_full_pipeline(img_path: str):
    pil_img = Image.open(img_path)
    full_times = []

    for gs in GRID_SIZES:
        img_array = preprocess(pil_img, gs)
        t0 = time.perf_counter()
        grid_colors = compute_vectorized(img_array, gs)
        tiles = generate("cross", levels=6, tile_px=24)
        build_mosaic(grid_colors, tiles, tile_px=24)
        full_times.append((time.perf_counter() - t0) * 1000)

    return full_times


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py path/to/test_image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    grid_results = benchmark_grid_computation(img_path)
    full_times = benchmark_full_pipeline(img_path)

    print(f"\n=== Grid Color Computation Time (ms, avg of {N_REPEATS} runs) ===")
    print(f"{'Grid Size':>10} | {'Vectorized (ms)':>16} | {'Loop (ms)':>12} | {'Speedup':>8}")
    for gs, v, l in zip(grid_results["grid_size"], grid_results["vectorized_ms"], grid_results["loop_ms"]):
        speedup = l / v if v > 0 else float("inf")
        print(f"{gs:>10} | {v:>16.3f} | {l:>12.3f} | {speedup:>7.1f}x")

    print("\n=== Full Pipeline Time (Vectorized, ms) ===")
    for gs, t in zip(GRID_SIZES, full_times):
        print(f"Grid Size {gs}: {t:.2f} ms")

    # Draw
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(grid_results["grid_size"], grid_results["vectorized_ms"], "o-", label="Vectorized (NumPy)")
    axes[0].plot(grid_results["grid_size"], grid_results["loop_ms"], "s-", label="Nested Loop")
    axes[0].set_xlabel("Grid Size")
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_title("Grid Color Computation: Vectorized vs Loop")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(GRID_SIZES, full_times, "o-", color="green")
    axes[1].set_xlabel("Grid Size")
    axes[1].set_ylabel("Time (ms)")
    axes[1].set_title("Full Pipeline Time vs Grid Size")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=150)
    print("\nSaved chart to benchmark_results.png")

if __name__ == "__main__":
    main()