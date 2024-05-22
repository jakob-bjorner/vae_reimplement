from torchvision.transforms import v2
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

class Cifar10Loader:
    def __init__(self, epsilon_noise, batch_size, device, num_workers):
        self.epsilon_noise = epsilon_noise
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        dataset = load_dataset("cifar10")
        class CustomAddEpsilonNoise(object):
            def __init__(self, epsilon_noise):
                epsilon_noise = epsilon_noise
            def __call__(self, sample):
                noise = torch.randn_like(sample) / 255 * epsilon_noise 
                sample = torch.clamp_(sample + noise, torch.zeros_like(sample), torch.ones_like(sample))
                return sample
        # these are wrong for cifar 10, would be closer to 
        # means = torch.tensor([0.49139968, 0.48215827, 0.44653124])
        # stds = torch.tensor([0.24703233, 0.24348505, 0.26158768])
        means = torch.tensor([0.5,0.5,0.5])
        stds = torch.tensor([0.5,0.5,0.5])
        forward_transform = v2.Compose([
            v2.ToImageTensor(),
            v2.ConvertImageDtype(dtype=torch.float32),
            # v2.Lambda(lambda image: image.permute(1,2,0)),
            CustomAddEpsilonNoise(epsilon_noise), # smoother data manifold like in diffusion.
            v2.Normalize(means, stds)
        ])
        forward_transform_no_noise = v2.Compose([
            v2.ToImageTensor(),
            v2.ConvertImageDtype(dtype=torch.float32),
            # v2.Lambda(lambda image: image.permute(1,2,0)),
            v2.Normalize(means, stds)
        ])
        ds_temp = dataset['test'].train_test_split(.5, seed=420)
        del dataset['test']
        dataset['eval'] = ds_temp['train']
        dataset['test'] = ds_temp['test']
        label_names = dataset['train'].features['label'].names
        def transform(examples):
            """transforms are best applied only once but why do we have the images represented in the PIL Image format instead of converting them to numpy arrays directly."""
            examples['pixel_values'] = [forward_transform(image) for image in examples["img"]]
            examples['label_string'] = [label_names[l] for l in examples["label"]]
            return examples
        def transform_no_noise(examples):
            """transforms are best applied only once but why do we have the images represented in the PIL Image format instead of converting them to numpy arrays directly."""
            examples['pixel_values'] = [forward_transform_no_noise(image) for image in examples["img"]]
            examples['label_string'] = [label_names[l] for l in examples["label"]]
            return examples
        dataset['train'].set_transform(transform)
        dataset['eval'].set_transform(transform_no_noise)

        inverse_transform = v2.Compose([
            v2.Normalize(-means/stds, 1/stds), # using the normalize math: (img - mean) / std = new -> new * std + mean = img = (new - (-mean/std)) / (1/std)
            v2.ToPILImage()
        ])
        def collate_fn(batch_list):
            batch_map_keys = batch_list[0].keys()
            batch_map = {k: list() for k in batch_map_keys}
            for bat in batch_list:
                for key in batch_map_keys:
                    batch_map[key].append(bat[key])
            for key in batch_map_keys:
                if isinstance(batch_map[key][0], torch.Tensor):
                    batch_map[key] = torch.stack(batch_map[key], dim=0)
                if isinstance(batch_map[key][0], int):
                    batch_map[key] = torch.LongTensor(batch_map[key])
            return batch_map
        self.dataset = dataset
        self.collate_fn = collate_fn
        dataloader_train = DataLoader(dataset['train'], batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True, pin_memory_device=device, num_workers=num_workers)
        dataloader_eval = DataLoader(dataset["eval"], batch_size, collate_fn=collate_fn, shuffle=False, pin_memory=True, pin_memory_device=device, num_workers=num_workers)
        self.dataloader_train = dataloader_train
        self.dataloader_eval = dataloader_eval
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
    def get_dataloaders(self):
        return self.dataloader_train, self.dataloader_eval
    def get_inverse_transform(self):
        return self.inverse_transform
    def get_consistent_samples(self, split_name, num_samples):
        return self.collate_fn([self.dataset[split_name][i] for i in range(num_samples)])

