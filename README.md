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


This repository contains the **official implementation** of the following paper:

> **Sequence-Conditioned Flow-based Models for Digital Phantom Generation in MRI**<br>
> MICCAI 2026
> 

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

Our work is built upon the methods and insights from the paper ["Hierarchy Flow For High-Fidelity Image-to-Image Translation"](https://arxiv.org/abs/2308.06909). We sincerely thank the authors for their foundational research.
