from templates import *
from templates_latent import *
from train_unconditional import ModelTrainer  # Import the refactored trainer

if __name__ == "__main__":
    # Configuration for training
    conf = ffhq128_ddpm_130M()
    trainer = ModelTrainer(conf)  # Initialize the trainer with the configuration

    # Train the model
    trainer.train(epochs=1000)  # Adjust the number of epochs as needed

    # Configuration for evaluation
    conf.eval_programs = ["fid10"]  # Set the evaluation program
    trainer = ModelTrainer(conf)  # Reinitialize the trainer for evaluation
    trainer.evaluate()  # Evaluate the model