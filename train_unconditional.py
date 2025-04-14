import copy
import json
import os
import re
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda import amp
from torchvision.utils import make_grid, save_image

from config import *
from dataset import *
from dist_utils import *
from lmdb_writer import *
from metrics import *
from renderer import *


class ModelTrainer:
    def __init__(self, conf: TrainConfig):
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reproducibility
        if conf.seed is not None:
            np.random.seed(conf.seed)
            torch.manual_seed(conf.seed)
            torch.cuda.manual_seed(conf.seed)

        # Model Initialization
        self.model = conf.make_model_conf().make_model().to(self.device)
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        # Optimizer and Scheduler
        self.optimizer = self.configure_optimizer()
        self.scheduler = self.configure_scheduler()

        # Other components
        self.sampler = conf.make_diffusion_conf().make_sampler()
        self.eval_sampler = conf.make_eval_diffusion_conf().make_sampler()
        self.T_sampler = conf.make_T_sampler()
        self.x_T = torch.randn(
            conf.sample_size, 3, conf.img_size, conf.img_size, device=self.device
        )

        # Load pretrained weights if needed
        if conf.pretrain is not None:
            self.load_pretrained(conf.pretrain.path)

        self.train_data = conf.make_dataset()
        self.val_data = self.train_data  # Change if validation data is different

        self.conds_mean = None
        self.conds_std = None

    def load_pretrained(self, path):
        state = torch.load(path, map_location="cpu")
        print(f"Loading pretrained weights from {path}, step: {state['global_step']}")
        self.model.load_state_dict(state["state_dict"], strict=False)

    def configure_optimizer(self):
        if self.conf.optimizer == OptimizerType.adam:
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.conf.lr,
                weight_decay=self.conf.weight_decay,
            )
        elif self.conf.optimizer == OptimizerType.adamw:
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.conf.lr,
                weight_decay=self.conf.weight_decay,
            )
        else:
            raise NotImplementedError("Unsupported optimizer type.")

    def configure_scheduler(self):
        if self.conf.warmup > 0:
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min(step, self.conf.warmup) / self.conf.warmup,
            )
        return None

    def ema_update(self, decay):
        for source_param, target_param in zip(
            self.model.parameters(), self.ema_model.parameters()
        ):
            target_param.data.copy_(
                target_param.data * decay + source_param.data * (1 - decay)
            )

    def train(self, epochs):
        dataloader = DataLoader(
            self.train_data,
            batch_size=self.conf.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}")
            for batch_idx, batch in pbar:
                self.optimizer.zero_grad()

                imgs = batch["img"].to(self.device)
                t, weight = self.T_sampler.sample(len(imgs), imgs.device)

                outputs = self.sampler.training_losses(
                    model=self.model,
                    x_start=imgs,
                    t=t,
                )
                loss = outputs["loss"].mean()
                loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                # EMA update
                self.ema_update(self.conf.ema_decay)

                # Update tqdm with loss info
                pbar.set_postfix(loss=loss.item())

            # Save checkpoints
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.conf.logdir, f"last.ckpt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "ema_model_state_dict": self.ema_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved at {checkpoint_path}")

    def evaluate(self):
        self.model.eval()
        dataloader = DataLoader(
            self.val_data,
            batch_size=self.conf.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                imgs = batch["img"].to(self.device)
                outputs = self.model(imgs)
                # Add evaluation logic here (e.g., FID, LPIPS, etc.)

    def sample(self, num_samples):
        self.model.eval()
        noise = torch.randn(
            num_samples,
            3,
            self.conf.img_size,
            self.conf.img_size,
            device=self.device,
        )
        samples = self.eval_sampler.sample(self.ema_model, noise=noise)
        samples = (samples + 1) / 2
        return samples

    def render_samples(self, samples, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        grid = make_grid(samples)
        save_image(grid, os.path.join(save_dir, "sample.png"))
        print(f"Samples saved at {save_dir}")


if __name__ == "__main__":
    conf = TrainConfig()  # Define your training configuration
    trainer = ModelTrainer(conf)
    trainer.train(epochs=conf.epochs)
    trainer.evaluate()
    samples = trainer.sample(num_samples=16)
    trainer.render_samples(samples, save_dir="./samples")