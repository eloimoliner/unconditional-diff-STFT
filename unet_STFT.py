import torch.nn as nn
import math
import numpy as np
import math as m
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm
#import torchaudio
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import utils


class DenseBlock(nn.Module):
    '''
    [B, T, F, N] => [B, T, F, N] 
    DenseNet Block consisting of "num_layers" densely connected convolutional layers
    '''
    def __init__(self, num_layers,N0, N, ksize):
        '''
        num_layers:     number of densely connected conv. layers
        N:              Number of filters (same in each layer) 
        ksize:          Kernel size (same in each layer) 
        '''
        super(DenseBlock, self).__init__()

        self.H=nn.ModuleList()
        self.num_layers=num_layers

        for i in range(num_layers):
            if i==0:   
                Nin=N0
            else:
                Nin=N0+i*N
             
            self.H.append(nn.Sequential(
                                weight_norm(nn.Conv2d(Nin,N,
                                      kernel_size=ksize,
                                      stride=1,
                                      padding='same',
                                      padding_mode='reflect',
                                      )),
                                nn.ELU()        ))

    def forward(self, x):
        x_ = self.H[0](x)
        if self.num_layers>1:
            for h in self.H[1:]:
                x = torch.cat((x_, x), 1)
                #x_=tf.pad(x, self.padding_modes_1, mode='SYMMETRIC')
                x_ = h(x)  
                #add elu here

        return x_


class FinalBlock(nn.Module):
    '''
    [B, T, F, N] => [B, T, F, 2] 
    Final block. Basiforwardy, a 3x3 conv. layer to map the output features to the output complex spectrogram.

    '''
    def __init__(self, N0):
        super(FinalBlock, self).__init__()
        ksize=(3,3)
        self.conv2=weight_norm(nn.Conv2d(N0,out_channels=2,
                      kernel_size=ksize,
                      stride=1, 
                      padding='same',
                      padding_mode='reflect'))


    def forward(self, inputs ):

        pred=self.conv2(inputs)

        return pred



class AddFreqEncoding(nn.Module):
    '''
    [B, T, F, 2] => [B, T, F, 12]  
    Generates frequency positional embeddings and concatenates them as 10 extra channels
    This function is optimized for F=1025
    '''
    def __init__(self, f_dim):
        super(AddFreqEncoding, self).__init__()
        pi=torch.pi
        self.f_dim=f_dim #f_dim is fixed
        n=torch.arange(start=0,end=f_dim)/(f_dim-1)
        # n=n.type(torch.FloatTensor)
        coss=torch.cos(pi*n)
        f_channel = torch.unsqueeze(coss, -1) #(1025,1)
        self.fembeddings= f_channel
        
        for k in range(1,10):   
            coss=torch.cos(2**k*pi*n)
            f_channel = torch.unsqueeze(coss, -1) #(1025,1)
            self.fembeddings=torch.cat((self.fembeddings,f_channel),-1) #(1025,10)

        self.fembeddings=nn.Parameter(self.fembeddings)
        #self.register_buffer('fembeddings_const', self.fembeddings)

    

    def forward(self, input_tensor):

        batch_size_tensor = input_tensor.shape[0]  # get batch size
        time_dim = input_tensor.shape[2]  # get time dimension

        fembeddings_2 = torch.broadcast_to(self.fembeddings, [batch_size_tensor, time_dim, self.f_dim, 10])
        fembeddings_2=fembeddings_2.permute(0,3,1,2)
    
        
        return torch.cat((input_tensor,fembeddings_2),1)  #(batch,12,427,1025)


class Decoder(nn.Module):
    '''
    [B, T, F, N] , skip connections => [B, T, F, N]  
    Decoder side of the U-Net subnetwork.
    '''
    def __init__(self, Ns, Ss, args):
        super(Decoder, self).__init__()

        self.Ns=Ns
        self.Ss=Ss
        self.args=args
        self.depth=self.args.unet2d.depth
        self.attention_layers=nn.ModuleList()
        self.attention_index=self.args.unet2d.attention.attention_indexes

        self.dblocks=nn.ModuleList()
        for i in range(self.depth):
            self.dblocks.append(D_Block(layer_idx=i,N0=self.Ns[i+1] ,N=self.Ns[i], S=self.Ss[i],num_tfc=self.args.unet2d.num_tfc, ksize=tuple(self.args.unet2d.ksize_decoder)))
            if i in self.attention_index:
                
                self.attention_layers.append(AttentionBlock(self.Ns[i], args)) #substittute None per shape, it is necessary for the positioal encoding
            else:
                self.attention_layers.append(None)


    def forward(self,inputs, contracting_layers):
        x=inputs
        for i in range(self.depth,0,-1):
                
            x=self.dblocks[i-1](x, contracting_layers[i-1])
            if self.args.unet2d.use_attention:
                if (i-1) in self.attention_index:
                    x=self.attention_layers[i-1](x) 
        return x 

class Film(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_layer = nn.Linear(512, 2 * output_dim)

    def forward(self, sigma_encoding):
        sigma_encoding = self.output_layer(sigma_encoding)
        sigma_encoding = sigma_encoding.unsqueeze(-1)
        sigma_encoding = sigma_encoding.unsqueeze(-1) #we need a secnond unsqueeze because our data is 2d [B,C,1,1]
        gamma, beta = torch.chunk(sigma_encoding, 2, dim=1)
        return gamma, beta

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        args,
        x_len=None,
        num_heads=4,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.args=args
        self.channels = channels
        num_heads=self.args.unet2d.attention.num_heads
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        #self.norm = normalization(channels)
        self.qkv = nn.Conv1d( channels, channels * 3, 1)
        # split qkv before split heads
        self.attention = QKVAttention(self.num_heads)
        #I'm not using positional encoding for now, but I believe it will be useful
        self.Posencoding= PositionalEncoding2D(channels)

        self.proj_out = nn.Conv1d( channels, channels, 1)

    #def forward(self, x):
    #    return checkpoint(self._forward, (x,), self.parameters(), True)

    def forward(self, x):
        b, c,f,t  = x.shape
        x=x.permute(0,2,3,1)
        self.Posencoding(x) 
        x=x.permute(0,3,1,2)
        x = x.reshape(b, c, -1)
        qkv = self.qkv(x)
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c,f, t)

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc

class PositionalEncoding_2d(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x=x.permute(2,0,1)
        x = x + self.pe[:x.size(0)]
        x=x.permute(1,2,0)
        return self.dropout(x)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)
class Encoder(nn.Module):

    '''
    [B, T, F, N] => skip connections , [B, T, F, N_4]  
    Encoder side of the U-Net subnetwork.
    '''
    def __init__(self,N0, Ns, Ss, args):
        super(Encoder, self).__init__()
        self.Ns=Ns
        self.Ss=Ss
        self.args=args
        self.depth=args.unet2d.depth

        self.contracting_layers = {}

        self.eblocks=nn.ModuleList()
        self.attention_index=self.args.unet2d.attention.attention_indexes

        
        self.attention_layers=nn.ModuleList()
        self.film=nn.ModuleList()
        for i in range(self.depth):
            if i==0:
                Nin=N0
            else:
                Nin=self.Ns[i]
            if self.args.unet2d.use_attention:
                if i in self.attention_index:
                    
                    self.attention_layers.append(AttentionBlock(Nin, args)) #substittute None per shape, it is necessary for the positioal encoding
                else:
                    self.attention_layers.append(None)
                    
                
            self.film.append(Film(Nin))
            self.eblocks.append(E_Block(layer_idx=i,N0=Nin,N01=self.Ns[i],N=self.Ns[i+1],S=self.Ss[i], num_tfc=args.unet2d.num_tfc, ksize=tuple(args.unet2d.ksize_encoder)))

        if self.args.unet2d.use_attention:
            self.attention_layers.append(AttentionBlock(self.Ns[i+1],args)) #substittute None per shape, it is necessary for the positioal encoding
        self.i_block=I_Block(self.Ns[self.depth],self.Ns[self.depth],args.unet2d.num_tfc, tuple(args.unet2d.ksize_bn))

    def forward(self, inputs,sigma_encoding):
        x=inputs
        for i in range(self.depth):

            gamma, beta = self.film[i](sigma_encoding)
            #apply the modulation here
            x = gamma * x + beta

            if self.args.unet2d.use_attention:
                if i in self.attention_index:
                    x=self.attention_layers[i](x) 
            #attention here
            x, x_contract=self.eblocks[i](x)
        
            self.contracting_layers[i] = x_contract #if remove 0, correct this

        #print(i+1,x.shape)
        #attention here
        if self.args.unet2d.use_attention:
            x=self.attention_layers[self.args.unet2d.depth](x) 

        x=self.i_block(x)

        return x, self.contracting_layers

class RFF_MLP_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, 32]), requires_grad=False)
        self.MLP = nn.ModuleList([
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
        ])

    def forward(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = self._build_RFF_embedding(sigma)
        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table
class Unet2d(nn.Module):

    def __init__(self, args):
        super(Unet2d, self).__init__()
        global weight_norm
        if not(args.unet2d.use_weight_norm):
            weight_norm=lambda x: x
        self.args=args
        self.depth=args.unet2d.depth
        Nin=2 #default by now imaginary/real
        if self.args.unet2d.use_fencoding:
            self.freq_encoding=AddFreqEncoding(self.args.unet2d.f_dim)
            Nin=Nin+10 #hardcoded
        self.use_fencoding=self.args.unet2d.use_fencoding
        #Encoder
        self.Ns= self.args.unet2d.Ns
        self.Ss= [(2,2),(2,2),(2,2),(2,2),(2,2),(2,2)]
        
        #initial feature extractor
        ksize=tuple(self.args.unet2d.ksize_init)

        self.conv2d_1 = nn.Sequential(weight_norm(nn.Conv2d(Nin,self.Ns[0],
                      kernel_size=ksize,
                      padding='same',
                      padding_mode='reflect')),
                      nn.ELU())
                        
        self.encoder=Encoder(self.Ns[0],self.Ns, self.Ss, self.args)
        self.decoder=Decoder(self.Ns, self.Ss, self.args)

        self.cropconcat = CropConcatBlock()
        #self.cropadd = CropAddBlock()

        self.finalblock=FinalBlock(self.Ns[0])

        self.embedding = RFF_MLP_Block()

    def forward(self, inputs, sigma):
        #inputs:
        # audio: [B, C, T]
        #sigma: scalar
        
        sigma_encoding = self.embedding(sigma)

        inputs=inputs.squeeze(1) #need to squeeze to do the stft, I think
        
        xF =utils.do_stft(inputs, win_size=self.args.unet2d.stft.win_size, hop_size=self.args.unet2d.stft.hop_size, device=inputs.device)
        #print(xF.shape)
        #B,F,T,C

        if self.use_fencoding:
            x_w_freq=self.freq_encoding(xF)   #None, None, 1025, 12 
        else:
            x_w_freq=inputs

        #intitial feature extractor
        x=self.conv2d_1(x_w_freq) #None, None, 1025, 32

        x, contracting_layers_s1= self.encoder(x, sigma_encoding)
        #decoder

        feats_s1 =self.decoder(x, contracting_layers_s1) #None, None, 1025, 32 features

        pred_f=self.finalblock(feats_s1) 

        pred_time=utils.do_istft(pred_f, self.args.unet2d.stft.win_size, self.args.unet2d.stft.hop_size, x.device)
        pred_time=pred_time[:,0:inputs.shape[-1]]
        assert pred_time.shape==inputs.shape, "bad shapes"
        return pred_time
            
class I_Block(nn.Module):
    '''
    [B, T, F, N] => [B, T, F, N] 
    Intermediate block:
    Basiforwardy, a densenet block with a residual connection
    '''
    def __init__(self,N0,N, num_tfc,ksize, **kwargs):
        super(I_Block, self).__init__(**kwargs)

        self.tfc=DenseBlock(num_tfc,N0,N,ksize)

        self.conv2d_res= weight_norm(nn.Conv2d(N0,N,
                                      kernel_size=(1,1),
                                      stride=1,
                                      padding='same',
                                      padding_mode='reflect'))

    def forward(self,inputs):
        x=self.tfc(inputs)

        inputs_proj=self.conv2d_res(inputs)
        return torch.add(x,inputs_proj)


class E_Block(nn.Module):

    def __init__(self, layer_idx,N0,N01, N,  S, num_tfc, ksize, **kwargs):
        super(E_Block, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.N0=N0
        self.N=N
        self.S=S
        self.i_block=I_Block(N0,N01,num_tfc, ksize)

        self.conv2d_2 = nn.Sequential(weight_norm(nn.Conv2d(N01,N,
                                          kernel_size=(S[0]+2,S[1]+2),
                                          padding=(2,2),
                                          stride=S,
                                          padding_mode='reflect')),
                                      nn.ELU())


    def forward(self, inputs, training=None, **kwargs):
        x=self.i_block(inputs)
        
        x_down = self.conv2d_2(x)

        return x_down, x


class D_Block(nn.Module):

    def __init__(self, layer_idx,N0, N,  S,  num_tfc,ksize, **kwargs):
        super(D_Block, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.N=N
        self.S=S
        self.tconv_1= nn.Sequential(
                                weight_norm(nn.ConvTranspose2d(N0,N,
                                             kernel_size=(S[0]+2, S[1]+2),
                                             stride=S,
                                             padding_mode='zeros')),
                                nn.ELU())

        self.upsampling = nn.Upsample(scale_factor=S, mode="nearest")

        self.projection =weight_norm(nn.Conv2d(N0,N,
                                      kernel_size=(1,1),
                                      stride=1,
                                      padding='same',
                                      padding_mode='reflect'))
        self.cropadd=CropAddBlock()
        self.cropconcat=CropConcatBlock()

        self.i_block=I_Block(2*N,N,num_tfc, ksize)

    def forward(self, inputs, bridge, **kwargs):
        x = self.tconv_1(inputs)

        x2= self.upsampling(inputs)

        if x2.shape[-1]!=x.shape[-1]:
            x2= self.projection(x2)

        x= self.cropadd(x,x2)
        
        x=self.cropconcat(x,bridge)

        x=self.i_block(x)
        return x


class CropAddBlock(nn.Module):

    def forward(self,down_layer, x,  **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        #print(x1_shape,x2_shape)
        height_diff = (x1_shape[2] - x2_shape[2]) // 2
        width_diff = (x1_shape[3] - x2_shape[3]) // 2


        down_layer_cropped = down_layer[:,
                                        :,
                                        height_diff: (x2_shape[2] + height_diff),
                                        width_diff: (x2_shape[3] + width_diff)]
        x = torch.add(down_layer_cropped, x)
        return x

class CropConcatBlock(nn.Module):

    def forward(self, down_layer, x, **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        height_diff = (x1_shape[2] - x2_shape[2]) // 2
        width_diff = (x1_shape[3] - x2_shape[3]) // 2
        down_layer_cropped = down_layer[:,
                                        :,
                                        height_diff: (x2_shape[2] + height_diff),
                                        width_diff: (x2_shape[3] + width_diff)]
        x = torch.cat((down_layer_cropped, x),1)
        return x
