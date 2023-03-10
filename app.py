import gradio as gr
import torch

from gradio_depth_pred import create_demo as create_depth_pred_demo
from gradio_im_to_3d import create_demo as create_im_to_3d_demo
from gradio_pano_to_3d import create_demo as create_pano_to_3d_demo


css = """
#img-display-container {
    max-height: 50vh;
    }
#img-display-input {
    max-height: 40vh;
    }
#img-display-output {
    max-height: 40vh;
    }
    
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(DEVICE).eval()

title = "# ZoeDepth"
description = """Official demo for **ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth**.

ZoeDepth is a deep learning model for metric depth estimation from a single image.

Please refer to our [paper](https://arxiv.org/abs/2302.12288) or [github](https://github.com/isl-org/ZoeDepth) for more details."""

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Tab("Depth Prediction"):
        create_depth_pred_demo(model)
    with gr.Tab("Image to 3D"):
        create_im_to_3d_demo(model)
    with gr.Tab("360 Panorama to 3D"):
        create_pano_to_3d_demo(model)

    gr.HTML('''<br><br><br><center>You can duplicate this Space to skip the queue:<a href="https://huggingface.co/spaces/shariqfarooq/ZoeDepth?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a><br>
        <p><img src="https://visitor-badge.glitch.me/badge?page_id=shariqfarooq.zoedepth_demo_hf" alt="visitors"></p></center>''')

if __name__ == '__main__':
    demo.queue().launch()