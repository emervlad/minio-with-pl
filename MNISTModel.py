import io
import os

import pytorch_lightning as L
import torch
from pytorch_lightning.plugins import TorchCheckpointIO
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from torchvision.datasets import MNIST

PATH_DATASETS = "."


class LitMNIST(L.LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-5):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        
        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
    

class CustomCheckpointIO(TorchCheckpointIO):
    def __init__(self, bucket, s3_client):
        super().__init__()
        self.cwdl = len(os.getcwd())
        self.bucket = bucket
        self.client = s3_client

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)

        self.client.put_object(Body=buffer.getvalue(),
                        Bucket=self.bucket,
                        Key=path[self.cwdl:])

    def load_checkpoint(self, path, storage_options=None):
        object = self.client.get_object(Bucket=self.bucket, Key=path)

        loaded_buffer = io.BytesIO(object['Body'].read())
        
        return torch.load(loaded_buffer)

    def remove_checkpoint(self, path):
        #self.client.delete_object(Bucket=self.bucket, Key=path[self.cwdl:])
        pass
