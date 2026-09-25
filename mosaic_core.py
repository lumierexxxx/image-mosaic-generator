import time
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from skimage.metrics import mean_squared_error, structural_similarity


# Step 1

def preprocess(pil_image: Image.Image, grid_size: int, cell_px: int = 16) -> np.ndarray:
    target_size = grid_size * cell_px
    img = pil_image.convert("RGB")

    w, h = img.size
    scale = target_size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    img = img.crop((left, top, left + target_size, top + target_size))

    return np.array(img, dtype=np.uint8)


# Step 2

def compute_vectorized(img_array: np.ndarray, grid_size: int) -> np.ndarray:
    h, w, c = img_array.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    reshaped = img_array.reshape(grid_size, cell_h, grid_size, cell_w, c)
    grid_colors = reshaped.mean(axis=(1, 3))
    return grid_colors.astype(np.uint8)


def compute_loop(img_array: np.ndarray, grid_size: int) -> np.ndarray:
    h, w, c = img_array.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    grid_colors = np.zeros((grid_size, grid_size, c), dtype=np.float64)

    for i in range(grid_size):
        for j in range(grid_size):
            cell = img_array[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            grid_colors[i, j] = cell.reshape(-1, c).mean(axis=0)

    return grid_colors.astype(np.uint8)


# Step 3

def generate(style: str = "solid", levels: int = 6, tile_px: int = 24):
    values = np.linspace(0, 255, levels).astype(np.uint8)
    tiles = {}

    yy, xx = np.mgrid[0:tile_px, 0:tile_px]

    for r in values:
        for g in values:
            for b in values:
                base_color = np.array([r, g, b], dtype=np.float64)
                tile = np.zeros((tile_px, tile_px, 3), dtype=np.uint8)

                if style == "solid":
                    tile[:, :] = base_color

                elif style == "gradient":
                    diag = (xx + yy) / (2 * (tile_px - 1))
                    shade = 0.7 + 0.3 * diag  # 0.7~1.0
                    tile = np.clip(base_color[None, None, :] * shade[:, :, None], 0, 255).astype(np.uint8)

                elif style == "cross":
                    tile[:, :] = (base_color * 0.75).astype(np.uint8)
                    inset = max(1, tile_px // 6)
                    tile[inset:-inset, inset:-inset] = base_color.astype(np.uint8)

                else:
                    raise ValueError(f"unknown: {style}")

                tiles[(int(r), int(g), int(b))] = tile

    return tiles


# Step 4

def build_mosaic(grid_colors: np.ndarray, tiles: dict, tile_px: int) -> np.ndarray:
    keys = np.array(list(tiles.keys()))
    tree = cKDTree(keys)

    grid_size = grid_colors.shape[0]
    flat_colors = grid_colors.reshape(-1, 3)
    _, nearest_idx = tree.query(flat_colors)

    mosaic = np.zeros((grid_size * tile_px, grid_size * tile_px, 3), dtype=np.uint8)

    for n in range(flat_colors.shape[0]):
        i, j = divmod(n, grid_size)
        tile = tiles[tuple(keys[nearest_idx[n]])]
        mosaic[i * tile_px:(i + 1) * tile_px, j * tile_px:(j + 1) * tile_px] = tile

    return mosaic


def add_grid(img_array: np.ndarray, grid_size: int, line_color=(255, 0, 0)) -> np.ndarray:
    overlay = img_array.copy()
    h, w, _ = overlay.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    for i in range(grid_size + 1):
        y = min(i * cell_h, h - 1)
        overlay[y:y + 1, :] = line_color
    for j in range(grid_size + 1):
        x = min(j * cell_w, w - 1)
        overlay[:, x:x + 1] = line_color

    return overlay


# Step 5

def compute_metrics(original: np.ndarray, mosaic: np.ndarray):
    if original.shape != mosaic.shape:
        orig_img = Image.fromarray(original).resize((mosaic.shape[1], mosaic.shape[0]), Image.LANCZOS)
        original = np.array(orig_img)

    mse_val = mean_squared_error(original, mosaic)
    ssim_val = structural_similarity(original, mosaic, channel_axis=2)
    return float(mse_val), float(ssim_val)


# Step 6

def benchmark_grid_sizes(pil_image: Image.Image, grid_sizes=(16, 32, 64), n_repeats: int = 3):
    results = []
    for gs in grid_sizes:
        img_array = preprocess(pil_image, gs)

        t_vec = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            compute_vectorized(img_array, gs)
            t_vec.append(time.perf_counter() - t0)

        t_loop = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            compute_loop(img_array, gs)
            t_loop.append(time.perf_counter() - t0)

        vec_ms = float(np.mean(t_vec) * 1000)
        loop_ms = float(np.mean(t_loop) * 1000)
        results.append({
            "grid_size": gs,
            "vectorized_ms": vec_ms,
            "loop_ms": loop_ms,
            "speedup": loop_ms / vec_ms if vec_ms > 0 else float("inf"),
        })
    return results


def format_benchmark_report(results: list) -> str:
    lines = [
        "**Step 6 — Performance: vectorized vs. nested loop**",
        "",
        "| Grid Size | Vectorized (ms) | Loop (ms) | Speedup |",
        "|---|---|---|---|",
    ]
    for r in results:
        gs = r["grid_size"]
        lines.append(f"| {gs}x{gs} | {r['vectorized_ms']:.3f} | {r['loop_ms']:.3f} | {r['speedup']:.1f}x |")

    smallest, largest = results[0], results[-1]
    cell_factor = (largest["grid_size"] / smallest["grid_size"]) ** 2
    loop_growth = largest["loop_ms"] / smallest["loop_ms"]
    vec_growth = largest["vectorized_ms"] / smallest["vectorized_ms"]
    speedups = [r["speedup"] for r in results]

    lines.append("")
    lines.append(
        f"Going from a {smallest['grid_size']}x{smallest['grid_size']} to a "
        f"{largest['grid_size']}x{largest['grid_size']} grid means {cell_factor:.0f}x more cells to process. "
        f"The nested-loop time grew {loop_growth:.1f}x over that range, while the vectorized time grew only "
        f"{vec_growth:.1f}x, because NumPy processes the whole grid as one batched array operation instead of "
        f"looping over cells in Python. Vectorized stayed {min(speedups):.1f}x-{max(speedups):.1f}x faster "
        f"across the tested sizes."
    )
    return "\n".join(lines)


def run_pipeline(pil_image: Image.Image, grid_size: int, tile_style: str,
                  cell_px: int = 16, tile_px: int = 24):
    img_array = preprocess(pil_image, grid_size, cell_px)

    t0 = time.perf_counter()
    grid_colors = compute_vectorized(img_array, grid_size)
    grid_time = time.perf_counter() - t0

    tiles = generate(tile_style, levels=6, tile_px=tile_px)
    mosaic = build_mosaic(grid_colors, tiles, tile_px)

    mse_val, ssim_val = compute_metrics(img_array, mosaic)
    grid_overlay = add_grid(img_array, grid_size)

    return img_array, grid_overlay, mosaic, grid_time, mse_val, ssim_val