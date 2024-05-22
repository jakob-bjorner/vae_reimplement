from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import torch
import pprint
import wandb
import os

@hydra.main(version_base=None, config_path="configs", config_name='config')
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    encoder_model_name = cfg.encoder_model._target_.split('.')[-1]
    decoder_model_name = cfg.decoder_model._target_.split('.')[-1]
    cfg.run_name = f"pDim={cfg.prior_dim}_lr={cfg.trainer.lr}_en={encoder_model_name}_de={decoder_model_name}_ppEps={cfg.datasetloaders.epsilon_noise}_bKl={cfg.vae.beta_weight_on_kl}_priorEps={cfg.vae.prior_epsilon}_ep={cfg.trainer.epochs}_bs={cfg.datasetloaders.batch_size}_nw={cfg.datasetloaders.num_workers}"
    # use any of the following to get the nodename:
    # potential_names = [
    #     "SLURMD_NODENAME",
    #     "SLURM_STEP_NODELIST",
    #     "SLURM_TOPOLOGY_ADDR",
    #     "SLURM_NODELIST",
    #     "SLURM_JOB_NODELIST"
    # ]
    # for potential_name in potential_names:
    #     if potential_name in os.environ:
    #         cfg.device_name = os.environ[potential_name]
    #         break
    cfg.device_name = os.environ["SLURMD_NODENAME"]
    cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    wandb_cfg = OmegaConf.to_container(cfg)
    with wandb.init(**cfg.logger, config=wandb_cfg) as run:
        wandb.define_metric("custom_step")
        wandb.define_metric("evaluate_*", step_metric='custom_step') # allows for asynchronous logging of eval events.
        pprint.pprint(wandb_cfg)
        trainer = instantiate(cfg.trainer, logger=wandb)
        trainer.train()
if __name__ == "__main__":
    my_app()