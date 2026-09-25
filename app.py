import gradio as gr
from mosaic_core import run_pipeline, benchmark_grid_sizes, format_benchmark_report


def process(image, grid_size, tile_style):
    if image is None:
        return None, None, None, ""

    grid_size = int(grid_size)
    original, grid_overlay, mosaic, grid_time, mse_val, ssim_val = run_pipeline(
        image, grid_size=grid_size, tile_style=tile_style
    )

    bench_results = benchmark_grid_sizes(image, grid_sizes=(16, 32, 64), n_repeats=3)
    report = format_benchmark_report(bench_results)

    return original, grid_overlay, mosaic, report


with gr.Blocks(title="Interactive Image Mosaic Generator") as demo:
 
    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Upload the Image")
            grid_size_in = gr.Slider(8, 64, value=32, step=8, label="Grid Size")
            tile_style_in = gr.Dropdown(
                choices=["solid", "gradient", "cross"], value="cross", label="Tile Style"
            )
            run_btn = gr.Button("Generate!", variant="primary")
 
        with gr.Column(scale=2):
            with gr.Row():
                original_out = gr.Image(label="preprocess")
                grid_out = gr.Image(label="grided")
            mosaic_out = gr.Image(label="mosaic-rized")
            benchmark_out = gr.Markdown()

    run_btn.click(
        fn=process,
        inputs=[image_in, grid_size_in, tile_style_in],
        outputs=[original_out, grid_out, mosaic_out, benchmark_out],
    )
 
    gr.Examples(
        examples=[
            "examples/_FIO0399.jpg.webp",
            "examples/_FIO0422.jpg.webp",
            "examples/_FIO0443.jpg.webp",
        ],
        inputs=[image_in],
        label="example",
    )
 
if __name__ == "__main__":
    demo.launch()