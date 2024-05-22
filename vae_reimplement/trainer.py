from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
import tqdm
import torch
import time
from collections import defaultdict
from PIL import Image
import wandb
from vae_reimplement.vae import VAE

# def batches_iterator(dataset_train, batch_size):
#     len_dataset = len(dataset_train)
#     num_passed = 0
#     while num_passed < len_dataset:
#         batch = dataset_train[num_passed: min(num_passed + batch_size, len_dataset)]
#         num_passed += batch_size
#         yield batch

class Trainer:
    vae: VAE

    def __init__(self, vae, datasetloaders, lr, epochs, lr_scheduler_gamma, device, logger):
        self.device = device
        self.vae = vae.to(device)
        self.lr = lr
        self.epochs = epochs
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.logger = logger
        self.datasetloaders = datasetloaders
    
    def init_train(self):
        # to train a VAE, need to sample from the model, and compute the loss based on the 
        # log likelihood of the image produced as well as the posterior's distance from the prior.
        self.optimizer = AdamW(self.vae.parameters(), lr=self.lr)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=self.lr_scheduler_gamma)

        self.dataloader_train, self.dataloader_eval = self.datasetloaders.get_dataloaders()

    def evaluate(self, epoch):
        eval_dict = dict()
        metrics = defaultdict(list)
        with torch.no_grad():
            # running the eval loop:
            self.vae.eval()
            for batch in tqdm.tqdm(self.dataloader_eval):
                labels = batch["label"].to(self.device)
                images = batch["pixel_values"].to(self.device)
                losses = self.vae.loss(images)
                for key in losses:
                    metrics[key + '_list'].append(losses[key].item())
            avg_metrics = {"mean_eval_" + metric_key.replace("_list", ""): sum(metric_list)/len(metric_list) for metric_key, metric_list in metrics.items()}
            eval_dict.update(avg_metrics)

            num_images = 4
            def make_grid(images, rows=2, cols=2):
                w, h = images[0].size
                grid = Image.new('RGB', size=(cols*w, rows*h))
                for i, image in enumerate(images):
                    grid.paste(image, box=(i%cols*w, i//cols*h))
                return grid
            
            # creating images to log:
            image_x_give_z_mu = self.vae.sample_images(num_images)
            # data_dist = torch.distributions.normal.Normal(image_x_give_z_mu, torch.exp(0.5 * image_x_give_z_sigma_log))
            # images = data_dist.sample().chunk(num_images, dim=0)
            images = [self.datasetloaders.get_inverse_transform()(im.squeeze(0)) for im in image_x_give_z_mu.chunk(num_images, dim=0)]
            # create the grid where the images will be displayed in, and then uploaded to wandb
            grid = make_grid(images)
            grid = wandb.Image(grid, caption=f"2x2 grid from {epoch=}")
            eval_dict.update({"generated_images": grid})

            # create autoencoding example for logs:
            # want the same images every time: 
            sample_data = self.datasetloaders.get_consistent_samples('eval', num_images)
            sample_images: Image = sample_data["img"] # these are pil
            sample_images_noised_tensor: torch.Tensor = sample_data['pixel_values']
            generated_images = self.vae.sample_images_from_latent(sample_images_noised_tensor.to(self.device))
            sample_images_noised_pil = [self.datasetloaders.get_inverse_transform()(im.squeeze(0)) for im in sample_images_noised_tensor.chunk(num_images, dim=0)]
            generated_images = [self.datasetloaders.get_inverse_transform()(im.squeeze(0)) for im in generated_images.chunk(num_images, dim=0)]
            sample_images_grid = make_grid(sample_images)
            sample_images_grid = wandb.Image(sample_images_grid, caption=f"eval 2x2 grid ({sample_data['label_string']}) from {epoch=}")
            sample_images_noised_grid = make_grid(sample_images_noised_pil)
            sample_images_noised_grid = wandb.Image(sample_images_noised_grid, caption=f"eval 2x2 grid ({sample_data['label_string']}) from {epoch=}")
            generated_images_grid = make_grid(generated_images)
            generated_images_grid = wandb.Image(generated_images_grid, caption=f"eval 2x2 grid ({sample_data['label_string']})from {epoch=}")
            eval_dict.update({"eval_x_images": sample_images_grid, "eval_x_given_z_images": generated_images_grid, "eval_x_noised_images": sample_images_noised_grid})
            
            # create autoencoding example for logs:
            # want the same images every time: 
            sample_data = self.datasetloaders.get_consistent_samples('train', num_images)
            sample_images: Image = sample_data["img"] # these are pil
            sample_images_noised_tensor: torch.Tensor = sample_data['pixel_values']
            generated_images = self.vae.sample_images_from_latent(sample_images_noised_tensor.to(self.device))
            sample_images_noised_pil = [self.datasetloaders.get_inverse_transform()(im.squeeze(0)) for im in sample_images_noised_tensor.chunk(num_images, dim=0)]
            generated_images = [self.datasetloaders.get_inverse_transform()(im.squeeze(0)) for im in generated_images.chunk(num_images, dim=0)]
            sample_images_grid = make_grid(sample_images)
            sample_images_grid = wandb.Image(sample_images_grid, caption=f"train 2x2 grid ({sample_data['label_string']}) from {epoch=}")
            sample_images_noised_grid = make_grid(sample_images_noised_pil)
            sample_images_noised_grid = wandb.Image(sample_images_noised_grid, caption=f"train 2x2 grid ({sample_data['label_string']}) from {epoch=}")
            generated_images_grid = make_grid(generated_images)
            generated_images_grid = wandb.Image(generated_images_grid, caption=f"train 2x2 grid ({sample_data['label_string']})from {epoch=}")
            eval_dict.update({"train_x_images": sample_images_grid, "train_x_given_z_images": generated_images_grid, "train_x_noised_images": sample_images_noised_grid})

        return eval_dict

    def train(self):
        self.init_train()
        global_steps = 0
        metrics = defaultdict(list)
        for epoch in tqdm.tqdm(range(self.epochs)):
            self.vae.train()
            
            start_time = time.time()
            for batch in tqdm.tqdm(self.dataloader_train):
                self.optimizer.zero_grad()
                labels = batch["label"].to(self.device)
                images = batch["pixel_values"].to(self.device)
                global_steps += 1
                losses = self.vae.loss(images)
                loss = losses["loss"]
                loss.backward()
                for key in losses:
                    metrics[key + '_list'].append(losses[key].item())
                self.optimizer.step()
            train_time = time.time() - start_time 

            start_time = time.time()
            log_dict = dict()
            self.lr_scheduler.step()
            avg_metrics = {"mean_" + metric_key.replace("_list", ""): sum(metric_list)/len(metric_list) for metric_key, metric_list in metrics.items()}
            log_dict.update(avg_metrics)
            # TODO: create images to log along with the log dict:
            eval_metrics = self.evaluate(epoch)
            log_dict.update(eval_metrics)
            metrics = defaultdict(list)
            print(f"{epoch=} done with {global_steps=} completed")
            eval_time = time.time() - start_time
            log_dict.update({'epoch':epoch, "steps": global_steps, "train_time": train_time, "eval_time": eval_time, "lr": self.lr_scheduler.get_last_lr()[0]})
            self.logger.log(log_dict)

        