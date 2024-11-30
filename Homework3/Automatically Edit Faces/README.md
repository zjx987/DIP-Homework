# Automatically Edit Faces

 
## Installation

To install requirements:

```setup
conda create -n stylegan3 python==3.10 -y
conda activate DIPwithPyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install face-alignment
```

## Running

run:

```
python gradio.py
```
Click the effect(smile,close_eyes,close_mouth,enlarge_eyes,face_slimming) you want to achieve, then click Start.
<img src="pics/result_1.png" alt="alt text" width="400">

## Results 

### Pix2Pix:
<img src="pics/result_1.png" alt="alt text" width="400">
<img src="pics/result_2.png" alt="alt text" width="400">
<img src="pics/result_3.png" alt="alt text" width="400">
<img src="pics/result_4.png" alt="alt text" width="400">
<img src="pics/result_5.png" alt="alt text" width="400">

## Acknowledgement
>📋 Thanks for the code proposed by
>[Paper: Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/).
