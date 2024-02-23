import os

import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from trainer.DiffBEVTrainer import DiffBEVTrainer
from trainer.DiffDiTBEVTrainer import DiffDiTBEVTrainer
from utils import logging_utils


def run_task(config):
    logging = logging_utils.get_std_logging(config.logging.logfile, logging_utils.logging_levels[config.logging.level])

    if config.distributed_run:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        addn_configs = {'world_size': world_size, 'rank': rank, 'local_rank': local_rank} 
    else:
        addn_configs = {'world_size': 1, 'rank': 0, 'local_rank': 0}
    
    for k, v in addn_configs.items():
        OmegaConf.update(config, k, v, merge=False)

    logging.info(f'World_size: {addn_configs["world_size"]}, GPU: {addn_configs["local_rank"]}, Rank: {addn_configs["rank"]} Initialized.')

    if config.trainer == 'diffbev':
        trainer_module = DiffBEVTrainer
    elif config.trainer == 'diffditbev':
        trainer_module = DiffDiTBEVTrainer
    else:
        raise Exception(f'Unrecognized trainer module: {config.trainer}')

    trainer = trainer_module(config)
    trainer.resume_model()
    start_epoch = trainer.start_epoch

    for epoch in tqdm(range(start_epoch + 1, trainer.total_epochs + 1)):
        trainer.train_epoch(epoch, printer=logging.info)
        trainer.save_checkpoint(epoch)


@hydra.main(version_base=None, config_path="configs", config_name="test_one_epoch")
def main(cfg):
    run_task(cfg)

if __name__ == '__main__':
    main()
