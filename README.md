<p align="center">
    <img src="logo.svg" height="300" title="My ML Repository Logo">
</p>
 
<p align="center">
 <img src="https://img.shields.io/github/last-commit/FarnoushRJ/MLProject_Pix2Pix/master?color=green&style=for-the-badge">
 <img src="https://img.shields.io/static/v1?label=python&message=3.6.9&color=red&style=for-the-badge">
 <img src="https://img.shields.io/github/repo-size/FarnoushRJ/MLProject_Pix2Pix?color=yellow&style=for-the-badge">
</p>

 ***

This is an unofficial PyTorch implementation of the paper [Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/).

If you find this code useful, please star the repository.

## Getting Started

### Installation
- Clone this repository
```
git clone "https://github.com/FarnoushRJ/MLProject_Pix2Pix.git"
```
- Install the requirements

  **Other Requirements**
   * Pillow 7.0.0
   * numpy 1.18.4
   * matplotlib 3.2.1
   * barbar 0.2.1
   * torch 1.5.0
   * torchvision 0.6.0


### Data
  * Facades and Maps datasets can be downloaded from [this link](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/).
  
**Data Directory Structure**
```
|__ DATASET_ROOT
    |__ train       
    |__ test
    |__ val     
```

### How to train

```python
cd train/
python train.py --args
```

## Training Loss Curves
The models is trained for 200 epochs on both Facades and Maps datasets. 

<center>
<div>
  <table>
    <tr>
      <td><img src="plots/facades_loss.png"/></td>
      <td><img src="plots/maps_AtoB_loss.png"/></td>
      <td><img src="plots/maps_BtoA_loss.png"/></td>
    </tr>
    <tr>
      <td align="center"><em>Facades Training Loss</em></td>
      <td align="center"><em>Maps(AtoB) Training Loss</em></td>
      <td align="center"><em>Maps(BtoA) Training Loss</em></td>
    </tr>
  </table>
</div>
</center>

## Qualitative Results
### Facades Dataset

  <p align="center">
    <img src="results/Facades_1.png"/>
    </br>
    <img src="results/Facades_2.png"/>
    </br>
    <img src="results/Facades_3.png"/>
    </br>
    <em>Input, Fake Target, Real Target</em>
  </p>

### Maps Dataset

<p align="center">
    <img src="results/Maps_AtoB_1.png"/>
    </br>
    <img src="results/Maps_AtoB_2.png"/>
    </br>
    <img src="results/Maps_AtoB_3.png"/>
    </br>
    <em>Input, Fake Target, Real Target (AtoB)</em>
</p>
</br>
<p align="center">
    <img src="results/Maps_BtoA_1.png"/></td></tr>
    </br>
    <img src="results/Maps_BtoA_2.png"/></td></tr>
    </br>
    <img src="results/Maps_BtoA_3.png"/></td></tr>
    </br>
    <em>Input, Fake Target, Real Target (BtoA)</em>
</p>

## TODO List
* Models
  - [ ] Modified Model for deblurring, denoising and Inpainting 

## References
- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [pytorch-pix2pix](https://github.com/znxlwm/pytorch-pix2pix)

## License
[![License: MIT](https://img.shields.io/github/license/FarnoushRJ/MLAlgorithms?color=blueviolet&style=for-the-badge)](https://opensource.org/licenses/MIT)
