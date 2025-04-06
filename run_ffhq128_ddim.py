from templates import *
from templates_latent import *
from train_unconditional import train

if __name__ == '__main__':
    gpus = [0]
    conf = ffhq128_ddpm_130M()
    train(conf, gpus=gpus)

    gpus = [0]
    conf.eval_programs = ['fid10']
    train(conf, gpus=gpus, mode='eval')