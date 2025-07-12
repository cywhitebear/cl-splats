import hydra
from loguru import logger
import omegaconf
import wandb

import trainer

def setup_wandb(cfg: omegaconf.DictConfig):
    wandb.init(
        project=cfg.get("wandb_project", "cl-splats"),
        name=cfg.get("wandb_run_name", None),
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.get("wandb_mode", "online"),
    )

@hydra.main(version_base=None, config_path="configs", config_name="cl-splats")
def main(cfg: omegaconf.DictConfig):
    setup_wandb(cfg)
    
    # Load data
    
    # Setup components

    # Initialize trainer
    clsplats_trainer = trainer.CLSplatsTrainer(cfg)

    for time in range(cfg.train.start_time, cfg.train.num_times):
        logger.info(f"Optimizing observations at time {time}.")

        # should the user control this? or should i just do it?
        clsplats_trainer.prepare_timestep(time)
        clsplats_trainer.train()

        if cfg.history.log_history:
            ## should this belong here to this class? probably not
            clsplats_trainer.log_history()

if __name__ == "__main__":
    main()
