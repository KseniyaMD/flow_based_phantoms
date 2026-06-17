# Sequence-Conditioned Flow-based Models for Digital Phantom Generation in MRI

<div align="center">
    <span>Kseniya Belousova<sup>1</sup></span>&emsp;
    <span>Zilya Badrieva<sup>1</sup></span>&emsp;
    <span>Ekaterina Brui<sup>1</sup></span>&emsp;
    <span>Walid Al-Haidri<sup>1</sup></span>
</div>
<div align="center">
    <sup>1</sup>ITMO University
</div>
<br><br>


<br><br>

## Description

This repository contains the official implementation of the paper **"Sequence-Conditioned Flow-based Models for Digital Phantom Generation in MRI"**.
MICCAI 2026

### Overview
Digital phantoms are crucial in MRI for pulse sequence optimization, artifact correction, and generating physically grounded synthetic data via augmentation. This work introduces a self-supervised, flow-based deep learning framework to estimate quantitative $T_1$, $T_2$, and $PD$ maps directly from conventional weighted brain MR images. 

### Key Features
* **Sequence-Aware Conditioning:** Explicitly incorporates MRI pulse sequence parameters — repetition time ($TR$), echo time ($TE$), and echo spacing ($ES$) — to improve model robustness to contrast variations.
* **No Ground Truth Required:** Operates in a self-supervised manner, eliminating the need for real ground-truth quantitative maps during training.
* **High Anatomical Fidelity:** Utilizes a **Sequence-Aware Affine Network (SA)** within a Hierarchy Flow architecture to preserve fine structural details without introducing checkerboard artifacts.
* **Multi-Contrast Synthesis:** Enables the simulation of various unseen pulse sequences (e.g., MPRAGE, GRE, FLAIR) derived from the generated digital phantoms.

### Performance
The proposed **SA** strategy achieves high structural similarity between original and synthesized images on real-world datasets, outperforming standard baseline methods with MS-SSIM scores of **0.99** ($T_1$), **0.95** ($T_2$), and **0.98** ($PD$).


## Experiments
Modify config files:
```yaml
#change the rootA and rootB for train and test respectively
  train:
    rootA: '{YOUR PATH TO WEIGHTED MR IMAGES}'
    rootB: '{YOUR PATH TO SAMPLE QUANTITATIVE MAPS}'

  test:
    rootA: '{YOUR PATH TO WEIGHTED MR IMAGES}
    rootB: '{YOUR PATH TO SAMPLE QUANTITATIVE MAPS}
```

**Training**
```Shell

python train_all.py --config configs/config_diff_synth_formula.yaml
```

**Test**
```Shell

python test_all.py --config configs/config_diff_synth_formula.yaml --load_path {PATH TO CHECKPOINT}
```

## Acknowledgements

Our work is built upon the methods and from the paper ["Hierarchy Flow For High-Fidelity Image-to-Image Translation"](https://arxiv.org/abs/2308.06909). 
We sincerely thank the authors for their foundational research.
