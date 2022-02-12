import os
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import shuffle
from torch import nn
import numpy as np

# from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset

# from torchvision import transforms

import pytorch_lightning as pl
from torchmetrics import Accuracy

from data_inspection import create_merged_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(pl.LightningModule):
    def __init__(self, input_size=54, hidden_units=(32, 16), output_size=207):
        super().__init__()

        # new PL attributes:
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

        # Model similar to previous section:
        all_layers = [nn.Flatten()]
        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.ReLU())
            input_size = hidden_unit

        all_layers.append(nn.Linear(hidden_units[-1], output_size))
        all_layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outs):
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True)
        self.valid_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


if __name__ == "__main__":
    x, y = create_merged_dataset()
    x, y = shuffle(x, y)

    y += 103

    x_test, y_test = x[700:], y[700:]
    x, y = x[:700], y[:700]

    # TODO: so that there are no negative classes - add preprocessing and mapping

    tensor_x = torch.Tensor(x)  # transform to torch tensor
    tensor_y = torch.Tensor(y).type(torch.LongTensor)

    dataset_train = TensorDataset(tensor_x, tensor_y)  # create your datset
    dataloader_train = DataLoader(
        dataset_train, batch_size=32
    )  # create your dataloader

    tensor_x_test = torch.Tensor(x_test)  # transform to torch tensor
    tensor_y_test = torch.Tensor(y_test).type(torch.LongTensor)

    dataset_test = TensorDataset(tensor_x_test, tensor_y_test)  # create your datset
    dataloader_test = DataLoader(dataset_test, batch_size=32)  # create your dataloader

    pl.seed_everything(42)
    mlp = MLP()
    trainer = pl.Trainer(
        auto_scale_batch_size="power", gpus=0, deterministic=True, max_epochs=20
    )
    # trainer.fit(mlp, DataLoader(dataset))
    trainer.fit(mlp, dataloader_train)
    results = trainer.test(mlp, dataloader_test)
