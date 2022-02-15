import copy
import os
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from torch import nn
import numpy as np

# from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset

# from torchvision import transforms

import pytorch_lightning as pl
from torchmetrics import Accuracy

from data_inspection import create_merged_dataset, get_mapping_of_categories

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
    y_preprocessed = copy.deepcopy(y)
    x = preprocessing.StandardScaler().fit_transform(x)


    mapping = get_mapping_of_categories(y)
    mapping_orginal_to_new = dict((y, x) for x, y in mapping.items())

    y_original = np.array([mapping_orginal_to_new[elem] for elem in y])
    x_original = copy.deepcopy(x)

    #
    # x_test, y_test = x[700:], y[700:]
    # x, y = x[:700], y[:700]
    #
    # # TODO: so that there are no negative classes - add preprocessing and mapping
    #
    # tensor_x = torch.Tensor(x)  # transform to torch tensor
    # tensor_y = torch.Tensor(y).type(torch.LongTensor)
    #
    # dataset_train = TensorDataset(tensor_x, tensor_y)  # create your datset
    # dataloader_train = DataLoader(
    #     dataset_train, batch_size=32
    # )  # create your dataloader
    #
    # tensor_x_test = torch.Tensor(x_test)  # transform to torch tensor
    # tensor_y_test = torch.Tensor(y_test).type(torch.LongTensor)
    #
    # dataset_test = TensorDataset(tensor_x_test, tensor_y_test)  # create your datset
    # dataloader_test = DataLoader(dataset_test, batch_size=32)  # create your dataloader
    #
    # pl.seed_everything(42)
    # mlp = MLP()
    # trainer = pl.Trainer(
    #     auto_scale_batch_size="power", gpus=0, deterministic=True, max_epochs=20
    # )
    # # trainer.fit(mlp, DataLoader(dataset))
    # trainer.fit(mlp, dataloader_train)
    # results = trainer.test(mlp, dataloader_test)

    kfold = KFold()

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(x_original)):
        # train_loader = create_dataloader(train_df.iloc[train_idx])
        # valid_loader = create_dataloader(train_df.iloc[valid_idx])
        #

        # # Folder hack
        # tb_logger = TensorBoardLogger(save_dir=OUTPUT_PATH, name=f'{args.model_name}', version=f'fold_{fold + 1}')
        # os.makedirs(OUTPUT_PATH / f'{args.model_name}, exist_ok=True)
        # checkpoint_callback = ModelCheckpoint(filepath=tb_logger.log_dir + "/{epoch:02d}-{val_metric:.4f}",
        #                                       monitor='val_metric', mode='max')
        #
        # model = YourPLModule(args)
        # trainer = pl.Trainer(logger=tb_logger, early_stop_callback=early_stop_callback,
        #                      checkpoint_callback=checkpoint_callback)
        # trainer.fit(model, train_dataloader=train_loader, val_dataloaders=valid_loader)


        # TODO: add logger of training curves to tensor board
        x_test, y_test = x_original[valid_idx], y_original[valid_idx]
        x, y = x_original[train_idx], y_original[train_idx]

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
        dataloader_test = DataLoader(
            dataset_test, batch_size=32
        )  # create your dataloader

        pl.seed_everything(42)
        mlp = MLP()
        trainer = pl.Trainer(
            auto_scale_batch_size="power", gpus=0, deterministic=True, max_epochs=20
        )
        # trainer.fit(mlp, DataLoader(dataset))
        trainer.fit(mlp, train_dataloader=dataloader_train,val_dataloaders=dataloader_test)
        results = trainer.test(mlp, dataloader_test)
