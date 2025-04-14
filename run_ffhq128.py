from templates import *
from templates_latent import *
from experiment import ConditionalModel  # Import the refactored model class

if __name__ == "__main__":
    # Step 1: Train the autoencoder model
    print("Training the autoencoder model...")
    gpus = [0]
    conf = ffhq128_autoenc_130M()
    conf.gpu_ids = gpus
    autoenc_model = ConditionalModel(conf)
    autoenc_model.train(epochs=1000)  # Adjust the number of epochs as needed

    # Step 2: Infer the latents for training the latent DPM
    print("Inferring latents for training the latent DPM...")
    conf.eval_programs = ["infer"]
    autoenc_model.evaluate()  # Perform latent inference

    # Step 3: Train the latent DPM
    print("Training the latent DPM...")
    conf = ffhq128_autoenc_latent()
    conf.gpu_ids = gpus
    latent_model = ConditionalModel(conf)
    latent_model.train(epochs=1000)  # Adjust the number of epochs as needed

    # Step 4: Evaluate unconditional sampling score
    print("Evaluating unconditional sampling score...")
    conf.eval_programs = ["fid(10,10)"]
    latent_model.evaluate()  # Perform sampling score evaluation