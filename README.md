# Image-to-Image Translation with Conditional Adversarial Nets

This is an unofficial PyTorch implementation of the paper [Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/).


## Qualitative Results
### Facades Dataset

<center>
  <table>
    <tr><td><img src="results/Facades_1.png"/></td></tr>
    <tr><td><img src="results/Facades_2.png"/></td></tr>
    <tr><td><img src="results/Facades_3.png"/></td></tr>
    <tr><td align="center"><em>Input, Fake Target, Real Target</em></td></tr>
  </table>
</center>

### Maps Dataset

<center>
<div>
  <table>
    <tr><td><img src="results/Maps_AtoB_1.png"/></td></tr>
    <tr><td><img src="results/Maps_AtoB_2.png"/></td></tr>
    <tr><td><img src="results/Maps_AtoB_3.png"/></td></tr>
    <tr><td align="center"><em>Input, Fake Target, Real Target (AtoB)</em></td></tr>
  </table>
  <table>
    <tr><td><img src="results/Maps_BtoA_1.png"/></td></tr>
    <tr><td><img src="results/Maps_BtoA_2.png"/></td></tr>
    <tr><td><img src="results/Maps_BtoA_3.png"/></td></tr>
    <tr><td align="center"><em>Input, Fake Target, Real Target (BtoA)</em></td></tr>
  </table>
</div>
</center>

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

## TODO List
* Models
  - [ ] Modified Model for deblurring, denoising and Inpainting 

## Acknowledgement
- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [pytorch-pix2pix](https://github.com/znxlwm/pytorch-pix2pix)

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
