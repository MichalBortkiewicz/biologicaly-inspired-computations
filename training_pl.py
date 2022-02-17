import copy
import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from torch import nn
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from torchmetrics import Accuracy

from data_inspection import create_merged_dataset, get_mapping_of_categories

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pl.seed_everything(42)


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


def create_dataloader_from_xy(x, y) -> DataLoader:
    tensor_x = torch.Tensor(x)  # transform to torch tensor
    tensor_y = torch.Tensor(y).type(torch.LongTensor)

    dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    dataloader = DataLoader(dataset, batch_size=32)  # create your dataloader
    return dataloader


OUTPUT_PATH = os.path.join("artifacts", "outputs")
RESULTS_PATH = os.path.join("artifacts", "results")
MODEL_NAME_PREFIX = "mlp"

HIDDEN_UNITS_TO_CHECK = [
    [8],
    [16],
    [32],
    [16, 8],
    [32, 16],
    [32, 8],
    [32, 16, 8],
    [16, 16, 16],
    [8, 8, 8],
    [64],
    [128],
    [128,128],
]

if __name__ == "__main__":
    # Dataset creation
    x, y = create_merged_dataset()
    x, y = shuffle(x, y)
    y_preprocessed = copy.deepcopy(y)
    x = preprocessing.StandardScaler().fit_transform(x)

    mapping = get_mapping_of_categories(y)
    mapping_orginal_to_new = dict((y, x) for x, y in mapping.items())

    y_original = np.array([mapping_orginal_to_new[elem] for elem in y])
    x_original = copy.deepcopy(x)

    # for i in range(len(HIDDEN_UNITS_TO_CHECK)):
    for i in range(-3,0, 1):
        # K fold crossval
        kfold = KFold(shuffle=True, random_state=42)

        # Config
        hidden_units = HIDDEN_UNITS_TO_CHECK[i]
        model_config = "_".join([str(elem) for elem in hidden_units])
        model_name = MODEL_NAME_PREFIX + f"_{model_config}"

        results = {}
        results[model_name] = {}
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

            dataloader_train = create_dataloader_from_xy(x, y)

            dataloader_test = create_dataloader_from_xy(x_test, y_test)

            mlp = MLP(hidden_units=hidden_units)
            trainer = pl.Trainer(
                auto_scale_batch_size="power",
                gpus=0,
                deterministic=True,
                max_epochs=100,
                logger=tb_logger,
                callbacks=[checkpoint_callback],
            )
            trainer.fit(
                mlp, train_dataloader=dataloader_train, val_dataloaders=dataloader_test
            )

            result = trainer.test(mlp, dataloader_test)
            print("results:", result)
            results[model_name][fold] = result

        results_file_path = os.path.join(RESULTS_PATH, f"{model_name}.pkl")
        os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
        results_file = open(results_file_path, "wb")
        pickle.dump(results, results_file)
