"""
Train a diffusion model on images.

"""
import soundfile as sf
import os
import logging
from tqdm import tqdm
import torch
import numpy as npp
import hydra

import dataset_loader

from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from torch.utils.data import DataLoader
import numpy as np

from getters import get_sde
from unet_STFT import Unet2d

import scipy.signal

def run(args):

    args = OmegaConf.structured(OmegaConf.to_yaml(args))
    #OmegaConf.set_struct(conf, True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    dirname = os.path.dirname(__file__)
    path_experiment = os.path.join(dirname, str(args.model_dir))

    if not os.path.exists(path_experiment):
        os.makedirs(path_experiment)
    args.model_dir=path_experiment

    model_dir = os.path.join(path_experiment, args.inference.checkpoint) #hardcoded for now
        

    model=Unet2d(args).to(device)

    state_dict= torch.load(model_dir, map_location=device)

    if hasattr(model, 'module') and isinstance(model.module, nn.Module):
        model.module.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict['model'])

    torch.backends.cudnn.benchmark = True

    sde = get_sde(args.sde_type, args.sde_kwargs)

    sampler=SDESampling_context(model, sde)

    segment_size=args.audio_len
    overlapsize=int(args.audio_len/4)

    numchunks=args.inference.num_sample_chunks
    
    T=args.inference.T

    pointer=0

    for i in tqdm(range(numchunks)):
        if i==0:
            if args.inference.stereo:
                contextL=torch.zeros((1,segment_size)).to(device)
                contextR=torch.zeros((1,segment_size)).to(device)
                mask=torch.ones((1,segment_size)).to(device)
            else:
                context=torch.zeros((1,segment_size)).to(device)
                mask=torch.ones((1,segment_size)).to(device)
        else:
            if args.inference.stereo:
                mask=torch.cat((torch.zeros((1,overlapsize)),torch.ones((1,segment_size-overlapsize))),dim=1).to(device)
                contextL=torch.cat((predL[:,segment_size-overlapsize::],torch.zeros((1,segment_size-overlapsize)).to(device)),dim=1).to(device)
                contextR=torch.cat((predR[:,segment_size-overlapsize::],torch.zeros((1,segment_size-overlapsize)).to(device)),dim=1).to(device)

            else:
                mask=torch.cat((torch.zeros((1,overlapsize)),torch.ones((1,segment_size-overlapsize))),dim=1).to(device)
                context=torch.cat((pred[:,segment_size-overlapsize::],torch.zeros((1,segment_size-overlapsize)).to(device)),dim=1).to(device)
        
        if args.inference.stereo:
            predL, predR=sampler.predict(contextL, contextR, mask, T, stereo=True, stereo_split=0.05)
            pred_2=torch.stack((predL.squeeze(0), predR.squeeze(0)), dim=1)
        else:
            pred=sampler.predict(context, mask, T)
            pred_2=pred.squeeze(0)

        if i==0:
            bwe_data=pred_2
        else:
            bwe_data=torch.cat((bwe_data,pred_2[overlapsize::]),dim=0)

        pointer=pointer+segment_size-overlapsize


    bwe_data=bwe_data.cpu().numpy()

    wav_output_name=os.path.join(path_experiment, "unconditional.wav")
    sf.write(wav_output_name, bwe_data, 22050)


class SDESampling:
    """
    DDPM-like discretization of the SDE as in https://arxiv.org/abs/2107.00630
    This is the most precise discretization
    """

    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def create_schedules(self, nb_steps):
        t_schedule = torch.arange(0, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        sigma_schedule = self.sde.sigma(t_schedule)
        m_schedule = self.sde.mean(t_schedule)

        return sigma_schedule, m_schedule

    def predict(
        self,
        audio,
        nb_steps
    ):

        with torch.no_grad():

            sigma, m = self.create_schedules(nb_steps)

            for n in range(nb_steps - 1, 0, -1):
                # begins at t = 1 (n = nb_steps - 1)
                # stops at t = 2/nb_steps (n=1)

                audio = m[n-1] / m[n] * audio + (m[n] / m[n-1] * (sigma[n-1])**2 / sigma[n] - m[n-1] / m[n] * sigma[n]) * \
                    self.model(audio, sigma[n])

                if n > 0:  # everytime
                    noise = torch.randn_like(audio)
                    audio += sigma[n-1]*(1 - (sigma[n-1]*m[n] /
                                              (sigma[n]*m[n-1]))**2)**0.5 * noise

            # The noise level is now sigma(1/nb_steps) = sigma[0]
            # Jump step
            audio = (audio - sigma[0] * self.model(audio,
                                                   sigma[0])) / m[0]

        return audio
class SDESampling_context:
    """
    DDPM-like discretization of the SDE as in https://arxiv.org/abs/2107.00630
    Using context, stereo...
    """

    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def create_schedules(self, nb_steps, stereo_split):
        t_schedule = torch.arange(0, nb_steps + 1) / nb_steps
        t_schedule = (self.sde.t_max - self.sde.t_min) * \
            t_schedule + self.sde.t_min
        split= (self.sde.t_max - self.sde.t_min) * \
           stereo_split + self.sde.t_min
        split=int(split*nb_steps)
        sigma_schedule = self.sde.sigma(t_schedule)
        m_schedule = self.sde.mean(t_schedule)

        return sigma_schedule, m_schedule, split

    def predict(
        self,
        contextL,
        contextR,
        mask,
        nb_steps,
        stereo=False,
        stereo_split=0.05
    ):

        with torch.no_grad():

            sigma, m ,stereo_split  = self.create_schedules(nb_steps, stereo_split)

            #map audio to latent space 

            #start sampling from trunc
            context=(contextL+contextR)/2
            context_noisy = m[nb_steps-1] * context + sigma[nb_steps-1] * torch.randn_like(context)
            audio=context_noisy

            for n in range(nb_steps - 1, 0, -1):
                # begins at t = 1 (n = nb_steps - 1)
                # stops at t = 2/nb_steps (n=1)
                print(1,n)
                #map context to latent space

                audio = m[n-1] / m[n] * audio + (m[n] / m[n-1] * (sigma[n-1])**2 / sigma[n] - m[n-1] / m[n] * sigma[n]) * \
                    self.model(audio, sigma[n])

                if n > 0:  # everytime
                    noise = torch.randn_like(audio)
                    audio += sigma[n-1]*(1 - (sigma[n-1]*m[n] /
                                              (sigma[n]*m[n-1]))**2)**0.5 * noise
                #map to latent space
                context_noisy = m[n-1] * context + sigma[n-1] * torch.randn_like(context)

                #combine context and no context
                audio=(1-mask)*context_noisy+mask*audio
                if stereo and n==stereo_split:
                    audio_stereo=torch.clone(audio)
                    context=contextL

            # The noise level is now sigma(1/nb_steps) = sigma[0]
            # Jump step
            audio = (audio - sigma[0] * self.model(audio,
                                                   sigma[0])) / m[0]
        
            audio=(1-mask)*context+mask*audio

            if stereo:
                audio_left=audio
                audio=audio_stereo
                context=contextR
                for n in range(stereo_split - 1, 0, -1):
            
                    print(1,n)
                    #map context to latent space
    
                    audio = m[n-1] / m[n] * audio + (m[n] / m[n-1] * (sigma[n-1])**2 / sigma[n] - m[n-1] / m[n] * sigma[n]) * \
                        self.model(audio, sigma[n])
    
                    if n > 0:  # everytime
                        noise = torch.randn_like(audio)
                        audio += sigma[n-1]*(1 - (sigma[n-1]*m[n] /
                                                  (sigma[n]*m[n-1]))**2)**0.5 * noise
                    #map to latent space
                    context_noisy = m[n-1] * context + sigma[n-1] * torch.randn_like(context)
    
                    #combine context and no context
                    audio=(1-mask)*context_noisy+mask*audio
    
                # The noise level is now sigma(1/nb_steps) = sigma[0]
                # Jump step
                audio = (audio - sigma[0] * self.model(audio,
                                                       sigma[0])) / m[0]
            
                audio=(1-mask)*context+mask*audio
                audio_right=audio
                return audio_left, audio_right
            else:    
                return audio

def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    run(args)

@hydra.main(config_path="conf", config_name="conf")
def main(args):
    #try:
        _main(args)
    #except Exception:
        #logger.exception("Some error happened")
    #    os._exit(1)

if __name__ == "__main__":
    main()
