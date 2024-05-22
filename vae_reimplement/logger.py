import wandb
import logging
from omegaconf import OmegaConf

class Logger:
    def __init__(self, project_name, run_name, cfg):
        config = OmegaConf.to_container(cfg)
        wandb.init(project=project_name, config=config, name=run_name)
        
        # specifially for asynchronous logging events like images generated from diffusion models, or evals on a large number of samples.
        
    def log(self, data):
        wandb.log(data)

class TextLogger:
    def __init__(self, cfg=None, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.log(cfg)
    def log(self, data):
        self.logger.info(data)
