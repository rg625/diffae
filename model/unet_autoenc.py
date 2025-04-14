from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu

from .latentnet import *
from .unet import *
from choices import *


@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None

    def make_model(self):
        return BeatGANsAutoencModel(self)


class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        self.conf = conf

        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )
        self.encoder = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            model_channels=conf.model_channels,
            out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            pool=conf.enc_pool,
        ).make_model()

        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()

    def forward(self, x, t, **kwargs):
        # Pure unconditional forward pass
        if t is not None:
            t_emb = timestep_embedding(t, self.conf.model_channels)
        else:
            t_emb = None

        res = self.time_embed.forward(time_emb=t_emb)
        emb = res.time_emb

        h = x.type(self.dtype)
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=emb)  # Only t_emb
                hs[i].append(h)
                k += 1

        h = self.middle_block(h, emb=emb)

        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                try:
                    lateral = hs[-i - 1].pop()
                except IndexError:
                    lateral = None
                h = self.output_blocks[k](h, emb=emb, lateral=lateral)
                k += 1

        pred = self.out(h)
        return AutoencReturn(pred=pred)

    def guided_sample(self, x, t, x_start, guidance_scale=1.0):
        """
        Apply classifier-free guidance during sampling.
        Uses the gradient of the encoder's log-density to steer the diffusion process.
        """
        # Get the p_mean_var (mean, variance, and log variance)
        p_mean_var = self.sampler.p_mean_variance(self, x, t)

        # Compute the guidance score (∇ log p(x|z))
        z = self.encode(x_start)['cond_fn']
        score_grad = self.score(x_start, z, t)

        # Adjust the mean with the gradient (guidance)
        new_mean = p_mean_var['mean'] + guidance_scale * score_grad

        return {
            'mean': new_mean,
            'variance': p_mean_var['variance'],
            'log_variance': p_mean_var['log_variance']
        }
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        assert self.conf.is_stochastic
        print(f'sample_z.shape: {n, self.conf.enc_out_channels}')
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x, t=None):
        if t is None:
            t = torch.zeros(x.shape[0]).type_as(x)

        latent_distribution_parameters = self.encoder(x, t)

        channels = latent_distribution_parameters.size(1) // 2
        mean_z = latent_distribution_parameters[:, :channels]
        log_var_z = latent_distribution_parameters[:, channels:]

        cond = mean_z + (0.5 * log_var_z).exp() * torch.randn_like(mean_z)

        return {
            'cond_fn': cond,
            'mu': mean_z,
            'logvar': log_var_z
        }

    def log_density(self, x, z, t):
        latent_distribution_parameters = self.encoder(x, t)
        channels = latent_distribution_parameters.size(1) // 2
        mean_z = latent_distribution_parameters[:, :channels]
        log_var_z = latent_distribution_parameters[:, channels:]

        # Flatten mean_z and log_var_z for consistent shape handling
        mean_z_flat = mean_z.view(mean_z.size(0), -1)
        log_var_z_flat = log_var_z.view(log_var_z.size(0), -1)
        z_flat = z.view(z.size(0), -1)

        logdensity = -0.5 * torch.sum(torch.square(z_flat - mean_z_flat) / log_var_z_flat.exp(), dim=1)
        return logdensity

    def score(self, x, z, t):
        """
        Returns a function that computes the latent correction score from the encoder.
        
        Returns:
            latent_correction_fn: A function that computes the latent correction score.
        """

        device = x.device
        x.requires_grad = True
        ftx = self.log_density(x, z, t)
        grad_log_density = torch.autograd.grad(outputs=ftx, inputs=x,
                                                grad_outputs=torch.ones(ftx.size()).to(device),
                                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        assert grad_log_density.size() == x.size()
        return grad_log_density


class AutoencReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None


class EmbedReturn(NamedTuple):
    time_emb: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )

    def forward(self, time_emb=None, **kwargs):
        if time_emb is not None:
            time_emb = self.time_embed(time_emb)
        return EmbedReturn(time_emb=time_emb)
