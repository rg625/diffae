import copy
import json
import os
import re
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid, save_image

from config import *
from dataset import *
from dist_utils import *
from lmdb_writer import *
from metrics import *
from renderer import *


class ConditionalModel:
    def __init__(self, conf: TrainConfig):
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set seeds for reproducibility
        if conf.seed is not None:
            np.random.seed(conf.seed)
            torch.manual_seed(conf.seed)
            torch.cuda.manual_seed(conf.seed)

        self.model = conf.make_model_conf().make_model().to(self.device)
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()
        self.T_sampler = conf.make_T_sampler()

        self.optimizer = self.configure_optimizer()
        self.scheduler = self.configure_scheduler()

        self.train_data = self.conf.make_dataset()
        self.val_data = self.train_data
        self.x_T = torch.randn(conf.sample_size, 3, conf.img_size, conf.img_size).to(self.device)

        self.conds_mean = None
        self.conds_std = None

        if conf.pretrain is not None:
            self.load_pretrained(conf.pretrain.path)
        
        # Freeze UNet parameters
        for name, param in self.model.named_parameters():
            if "encoder" not in name:
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert "encoder." in name, f"Only encoder parameters should be trained, but found {name} with requires_grad {param.requires_grad}"


    def configure_optimizer(self):
        if self.conf.optimizer == OptimizerType.adam:
            return Adam(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamw:
            return AdamW(self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError("Unsupported optimizer type.")

    def configure_scheduler(self):
        if self.conf.warmup > 0:
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min(step, self.conf.warmup) / self.conf.warmup,
            )
        return None

    def load_pretrained(self, path):
        state = torch.load(path, map_location="cpu")
        print(f"Loading pretrained weights from {path}, step: {state.get('global_step', 'N/A')}")

        pretrained_dict = state["model_state_dict"]
        model_dict = self.model.state_dict()

        # Key remapping
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            new_k = k
            if k.startswith("time_embed."):
                new_k = "time_embed.time_embed" + k[len("time_embed"):]  # insert one more 'time_embed'
            new_state_dict[new_k] = v

        # Load only matching keys
        self.model.load_state_dict(new_state_dict, strict=False)

    def ema_update(self, decay):
        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.copy_(ema_param.data * decay + model_param.data * (1 - decay))

    def train(self, epochs):
        dataloader = DataLoader(
            self.train_data, batch_size=self.conf.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}/{epochs}")
            for batch_idx, batch in pbar:
                self.optimizer.zero_grad()

                imgs = batch["img"].to(self.device)  # x0
                t, weight = self.T_sampler.sample(len(imgs), self.device)  # t ~ U(0, T)

                # Sample x_t from q(x_t | x_0)
                x_t = self.sampler.q_sample(imgs, t)

                # Freeze UNet: compute ∇log p(x_t | x_0) using pretrained score model
                with torch.no_grad():
                    score_ref = self.ema_model(x_t, t)  # s_theta(x_t, t), ∇log p_t(x_t|x0)

                # Trainable encoder + conditional score model
                encoder_output = self.model.encode(x_t)  # q_phi(z | x_t)
                z = encoder_output['cond_fn']  # could be a dict or tensor depending on your encoder
                score_cond = self.model.score(x_t, z, t)  # s_theta_phi(x_t, z, t)

                # Weighting function g(t), usually sqrt of cumulative alpha prod or similar
                g_t = self.sampler.g(t) if hasattr(self.sampler, "g") else 1.0

                # Main guidance loss
                score_loss = ((score_ref.pred - score_cond) ** 2).sum(dim=(1, 2, 3))
                loss_guidance = (g_t ** 2) * score_loss.mean()

                # Optional: KL between q_phi(z | x_0) and p(z)
                if "kl" in encoder_output:
                    loss_kl = encoder_output["kl"].mean()
                else:
                    # assume q_phi(z|x) = N(mu, sigma), p(z) = N(0,1)
                    mu, logvar = encoder_output["mu"], encoder_output["logvar"]
                    loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

                loss = loss_guidance + 0.01*loss_kl#self.conf.kl_weight * loss_kl
                loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                self.ema_update(self.conf.ema_decay)

                # if batch_idx % 10 == 0:
                #     print(f"Epoch [{epoch}/{epochs}], Step [{batch_idx}], Loss: {loss.item():.4f}, Guidance: {loss_guidance.item():.4f}, KL: {loss_kl.item():.4f}")

            # Save checkpoint after each epoch
            self.save_checkpoint(epoch)

    def evaluate(self):
        self.model.eval()
        dataloader = DataLoader(
            self.val_data, batch_size=self.conf.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                imgs = batch["img"].to(self.device)
                outputs = self.model(imgs)
                # Add evaluation metrics here as needed

    def sample(self, num_samples):
        self.ema_model.eval()
        noise = torch.randn(num_samples, 3, self.conf.img_size, self.conf.img_size, device=self.device)
        samples = self.eval_sampler.sample(self.ema_model, noise=noise)
        samples = (samples + 1) / 2
        return samples

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.conf.logdir, f"last.ckpt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def render_samples(self, samples, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        grid = make_grid(samples)
        save_image(grid, os.path.join(save_dir, "sample.png"))
        print(f"Samples saved at {save_dir}")


if __name__ == "__main__":
    conf = TrainConfig()  # Update with your specific configuration
    model = ConditionalModel(conf)

    # Train the model
    model.train(epochs=1000)

    # Evaluate the model
    model.evaluate()

    # Generate samples
    samples = model.sample(num_samples=16)

    # Save generated samples
    model.render_samples(samples, save_dir="./samples")