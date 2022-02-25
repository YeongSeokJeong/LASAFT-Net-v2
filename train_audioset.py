import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.distributed import rank_zero_info

from lasaft.scripts import trainer_audioset as trainer
from lasaft.utils.functions import print_config

dotenv.load_dotenv(override=True)


def main(cfg: DictConfig):
    # Load config
    rank_zero_info(OmegaConf.to_yaml(cfg))

    # Pretty print config using Rich library
    if cfg.get("print_config"):
        print_config(cfg, resolve=True)

    trainer.train(cfg)


@hydra.main(config_path="conf", config_name="train_audioset")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == '__main__':
    hydra_entry()
