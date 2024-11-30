# Automatically Edit Faces

 
## Installation

To install requirements:

```setup
conda create --name DIPwithPyTorch python=3.8
conda activate DIPwithPyTorch
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install gradio
```
## Datasets

Download the datasets using the following script.`cityscapes`: 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com/).

```
bash download_dataset.sh cityscapes
```

## Running

To run Pix2Pix, run:

```
python train.py
```

## Results 

### Pix2Pix:
<img src="pics/result_1.png" alt="alt text" width="400">
<img src="pics/result_2.png" alt="alt text" width="400">
<img src="pics/result_3.png" alt="alt text" width="400">
<img src="pics/result_4.png" alt="alt text" width="400">
<img src="pics/result_5.png" alt="alt text" width="400">

## Acknowledgement
>📋 Thanks for the algorithms proposed by
>[Paper: Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/).
