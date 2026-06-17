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




This repository contains the **official implementation** of the following paper:

> **Sequence-Conditioned Flow-based Models for Digital Phantom Generation in MRI**<br>
> MICCAI 2026

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
