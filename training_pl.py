import copy
import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn import preprocessing
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from torch import nn
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import pytorch_lightning as pl
from torchmetrics import Accuracy, F1

from data_inspection import create_merged_dataset, get_mapping_of_categories

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pl.seed_everything(42)


class MLP(pl.LightningModule):
    def __init__(self, input_size=54, hidden_units=(32, 16), output_size=100):
        super().__init__()
        self.l1_strength = 0.0
        self.l2_strength = 0.0

        # new PL attributes:
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

        self.train_f1 = F1(average="weighted", num_classes=output_size)
        self.valid_f1 = F1(average="weighted", num_classes=output_size)
        self.test_f1 = F1(average="weighted", num_classes=output_size)

        # Model similar to previous section:
        all_layers = [nn.Flatten()]
        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
            all_layers.append(nn.BatchNorm1d(num_features=hidden_unit))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(0.3))
            input_size = hidden_unit

        all_layers.append(nn.Linear(hidden_units[-1], output_size))
        all_layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.model(x)
        return x

    def predict(self, x):
        probs = self.forward(x)
        return probs.argmax(axis=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(self(x), y)
        preds = torch.argmax(logits, dim=1)

        self.train_acc.update(preds, y)
        self.train_f1.update(preds, y)

        # maybe regularization
        if self.l1_strength > 0:
            l1_reg = sum(param.abs().sum() for param in self.parameters())
            loss += self.l1_strength * l1_reg

        # L2 regularizer
        if self.l2_strength > 0:
            l2_reg = sum(param.pow(2).sum() for param in self.parameters())
            loss += self.l2_strength * l2_reg

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc", self.train_acc.compute())
        self.log("train_f1", self.train_f1.compute())

        self.train_acc.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.valid_f1.update(preds, y)
        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outs):
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True)
        self.log("valid_f1", self.valid_f1.compute(), prog_bar=True)
        self.valid_acc.reset()
        self.valid_f1.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(self(x), y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        # self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        # self.log("test_f1", self.test_f1.compute(), prog_bar=True)
        return loss

    def test_epoch_end(self, outs):
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        self.log("test_f1", self.test_f1.compute(), prog_bar=True)
        self.test_acc.reset()
        self.test_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return optimizer


def create_sampler(y_original, y_training):
    class_sample_count = np.array(
        [len(np.where(y_original == t)[0]) for t in np.unique(y_original)]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in y_training])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def create_dataloader_from_xy(x, y, shuffle=False, sampler=None) -> DataLoader:
    tensor_x = torch.Tensor(x)  # transform to torch tensor
    tensor_y = torch.Tensor(y).type(torch.LongTensor)

    dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    dataloader = DataLoader(
        dataset, batch_size=32, sampler=sampler, shuffle=shuffle
    )  # create your dataloader
    return dataloader


OUTPUT_PATH = os.path.join("artifacts", "outputs")
RESULTS_PATH = os.path.join("artifacts", "results")
MODEL_NAME_PREFIX = "mlp_reg"

HIDDEN_UNITS_TO_CHECK = [
    [8],
    [16],
    [32],
    [64],
    [128],
    [16, 8],
    [32, 16],
    [32, 8],
    [128, 128],
    [256, 128],
    [8, 8, 8],
    [32, 16, 8],
    [16, 16, 16],
    [128],
    [1024, 256],
]

if __name__ == "__main__":
    # Dataset creation
    x, y = create_merged_dataset()
    x, y = shuffle(x, y, random_state=42)
    y_preprocessed = copy.deepcopy(y)
    x = preprocessing.StandardScaler().fit_transform(x)

    mapping = get_mapping_of_categories(y)
    mapping_orginal_to_new = dict((y, x) for x, y in mapping.items())

    y_original = np.array([mapping_orginal_to_new[elem] for elem in y])
    x_original = copy.deepcopy(x)

    num_classes = max(mapping.keys()) + 1

    # for i in range(len(HIDDEN_UNITS_TO_CHECK)):
    for i in range(-1, 0, 1):
        # K fold crossval
        kfold = KFold(shuffle=True, random_state=42)

        # Config
        hidden_units = HIDDEN_UNITS_TO_CHECK[i]
        model_config = "_".join([str(elem) for elem in hidden_units])
        model_name = MODEL_NAME_PREFIX + f"_{model_config}"

        results = {}
        results_train = {}
        for fold, (train_idx, valid_idx) in enumerate(kfold.split(x_original)):
            # Folder hack
            tb_logger = TensorBoardLogger(
                save_dir=OUTPUT_PATH, name=f"{model_name}", version=f"fold_{fold + 1}"
            )
            os.makedirs(os.path.join(OUTPUT_PATH, model_name), exist_ok=True)
            checkpoint_callback = ModelCheckpoint(
                dirpath=tb_logger.log_dir,
                filename="{epoch:02d}-{valid_acc:.4f}",
                monitor="valid_acc",
                mode="max",
            )

            x_test, y_test = x_original[valid_idx], y_original[valid_idx]
            x, y = x_original[train_idx], y_original[train_idx]

            sampler = create_sampler(y_original, y)
            # dataloader_train = create_dataloader_from_xy(x, y,shuffle=True, sampler=sampler)
            dataloader_train = create_dataloader_from_xy(
                x, y, shuffle=True, sampler=None
            )

            dataloader_test = create_dataloader_from_xy(x_test, y_test)

            mlp = MLP(hidden_units=hidden_units, output_size=num_classes)
            trainer = pl.Trainer(
                auto_scale_batch_size="power",
                gpus=0,
                deterministic=True,
                max_epochs=200,
                logger=tb_logger,
                callbacks=[checkpoint_callback],
            )
            trainer.fit(
                mlp, train_dataloader=dataloader_train, val_dataloaders=dataloader_test
            )

            result = trainer.test(mlp, dataloader_test)
            print("result fold:", result)

            y_pred = mlp.predict(torch.Tensor(x_test).cpu())
            f1 = f1_score(y_test, y_pred, average="weighted")
            acc = accuracy_score(y_test, y_pred)
            results[fold] = {"test_acc": acc, "test_f1": f1}


            y_pred = mlp.predict(torch.Tensor(x).cpu())
            f1 = f1_score(y, y_pred, average="weighted")
            acc = accuracy_score(y, y_pred)
            results_train[fold] = {"train_acc": acc, "train_f1": f1}

        results_file_path = os.path.join(RESULTS_PATH, f"{model_name}.pkl")
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
        with open(results_file_path, "wb") as results_file:
            pickle.dump(results, results_file)

        # train
        results_file_path = os.path.join(RESULTS_PATH, f"{model_name}_train.pkl")
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
        with open(results_file_path, "wb") as results_file:
            pickle.dump(results_train, results_file)