#!/usr/bin/env python3

import boto3
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from MNISTModel import *


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    mnist_test = MNIST(cfg.mnist.path, train=False, transform=transform)
    test_dataloader = DataLoader(mnist_test, batch_size=cfg.training.batch_size)
    
    session = boto3.session.Session(profile_name=cfg.s3.profile)
    s3_client = session.client(
        service_name='s3',
        endpoint_url=cfg.s3.endpoint,
    )
    model = LitMNIST(data_dir=cfg.mnist.path, hidden_size=cfg.fcn.hidden_size, learning_rate=cfg.training.learning_rate)

    custom_checkpoint_io = CustomCheckpointIO(cfg.s3.bucket, s3_client)

    trainer = Trainer(
        plugins=[custom_checkpoint_io],
        callbacks=ModelCheckpoint(save_last=False),
        max_epochs=cfg.training.epochs,
    )
    if cfg.training.use_ckpt:
        trainer.test(model, test_dataloader, ckpt_path=cfg.training.ckpt_path)
    else:
        trainer.test(model, test_dataloader)


if __name__ == '__main__':
    main()
