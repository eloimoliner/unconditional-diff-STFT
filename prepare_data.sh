#!/bin/bash
#Downlad audio examples
#wget https://github.com/eloimoliner/bwe_historical_recordings/releases/download/v0.0-alpha/audio_examples.zip
#unzip audio_examples.zip -d audio_examples

#Download checkpoints BEHM-GAN
#wget https://github.com/eloimoliner/bwe_historical_recordings/releases/download/v0.0-alpha/checkpoint_piano
wget https://github.com/eloimoliner/unconditional-diff-STFT/releases/download/weights_piano/weights_piano_uncond_synth.pt
mv weights_piano_uncond_synth.pt  experiments/piano
#wget https://github.com/eloimoliner/bwe_historical_recordings/releases/download/v0.0-alpha/checkpoint_strings
#mv weights_strings_uncond_synth.pt  experiments/strings

