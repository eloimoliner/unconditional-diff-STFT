Unconditional synthesis of music using a diffusion model in the STFT domain.

This repository is the result of a set of experiments training a diffusion model to unconditionally synthesize piano music. The model, based on the complex-spectrogram domain shows superior performance to previous works using time-domain models as the backbone of a diffusion model. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eloimoliner/unconditional-diff-STFT/blob/main/colab/demo.ipynb)
## Implementation details

## References

This implementation is based on / inspired by:

- [Diffusion model for gramophone noise synthesis PyTorch repo](https://github.com/eloimoliner/gramophone_noise_synth).
- [DiffWave unoffifial PyTorch repo](https://github.com/lmnt-com/diffwave).
- [CRASH PyTorch repo](https://github.com/simonrouard/CRASH).
