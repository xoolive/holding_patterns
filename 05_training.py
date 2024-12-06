# python 05_training.py --scn train_egll_test_3_trk_vr --arch 1 --gpu 1 --max_epochs 10 --vertical_rate

import argparse
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, Precision, Recall, F1Score


class TBLogger(pl.loggers.TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop("epoch", None)
        return super().log_metrics(metrics, step)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Classifier(pl.LightningModule):
    def __init__(self, arch=1, vertical_rate=False, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        factor = vertical_rate + 1
        models = {
            1: nn.Sequential(
                nn.Linear(30 * factor, 8 * factor),
                nn.ReLU(),
                nn.Linear(8 * factor, 4 * factor),
                nn.ReLU(),
                nn.Linear(4 * factor, 1),
                nn.Sigmoid(),
            ),
            2: nn.Sequential(
                nn.Linear(30 * factor, 8 * factor),
                Reshape(-1, 1 * factor, 8),
                nn.Conv1d(1 * factor, 10 * factor, 3),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Flatten(),
                nn.Linear(30 * factor, 8 * factor),
                nn.ReLU(),
                nn.Linear(8 * factor, 4 * factor),
                nn.ReLU(),
                nn.Linear(4 * factor, 1),
                nn.Sigmoid(),
            ),
        }
        self.classifier = models[arch]
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.BCELoss()
        self.acc_train, self.acc_val = Accuracy(), Accuracy()
        self.prec_train, self.prec_val = Precision(), Precision()
        self.recall_train, self.recall_val = Recall(), Recall()
        self.f1_train, self.f1_val = F1Score(), F1Score()

    def forward(self, x):
        return self.classifier(x)

    def _log_metrics(self, phase, y_hat, y, acc, prec, recall, f1):
        y = y.type(torch.int32)
        acc(y_hat, y)
        prec(y_hat, y)
        recall(y_hat, y)
        f1(y_hat, y)
        self.log(
            f"perf/acc/{phase}",
            acc,
            on_step=False,
            on_epoch=True,
        )
        self.log(f"perf/prec/{phase}", prec, on_step=False, on_epoch=True)
        self.log(
            f"perf/recall/{phase}",
            recall,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"perf/f1/{phase}",
            f1,
            on_step=False,
            on_epoch=True,
            prog_bar=phase == "val",
        )

    def training_step(self, batch, batch_idx):
        (x, y) = batch
        y_hat = self.classifier(x)
        y = y.view(-1, 1)
        loss = self.criterion(y_hat, y.view(-1, 1))
        self.log("loss/train", loss, on_step=False, on_epoch=True)
        self._log_metrics(
            "train",
            y_hat,
            y,
            self.acc_train,
            self.prec_train,
            self.recall_train,
            self.f1_train,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        (x, y) = batch
        y_hat = self.classifier(x)
        y = y.view(-1, 1)
        loss = self.criterion(y_hat, y)
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_metrics(
            "val",
            y_hat,
            y,
            self.acc_val,
            self.prec_val,
            self.recall_val,
            self.f1_val,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Classifier")
        parser.add_argument("--arch", type=int, default=1)
        return parent_parser


def get_model(scn, arch, vers):
    return Classifier.load_from_checkpoint(
        get_checkpoint(Path(f"{scn}/checkpoints/{arch}/{vers}"))
    )


def get_pred(scn, arch, vers, npy_file):
    model = get_model(scn, arch, vers)
    model.freeze()
    pred = (
        model(torch.from_numpy(np.load(npy_file)))
        .reshape(-1)
        .detach()
        .numpy()
        .round()
        .astype(bool)
    )
    return pred


def get_checkpoint(checkpoint_dir):
    if checkpoint_dir.is_dir():
        checkpoint_file = sorted(
            checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime, reverse=True
        )
        return str(checkpoint_file[0]) if checkpoint_file else None
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scn", type=str, default=".")
    parser.add_argument("--gpu", type=int, default=0)
    parser = Classifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args.gpus = [args.gpu]
    vertical_rate = args.scn.split("_")[-1] == "vr"

    model = Classifier(args.arch, vertical_rate)

    ds_train = TensorDataset(
        torch.from_numpy(np.load(f"{args.scn}/x_train.npy")),
        torch.from_numpy(np.load(f"{args.scn}/y_train.npy")),
    )
    ds_val = TensorDataset(
        torch.from_numpy(np.load(f"{args.scn}/x_val.npy")),
        torch.from_numpy(np.load(f"{args.scn}/y_val.npy")),
    )
    # ds_test = TensorDataset(
    #     torch.from_numpy(np.load(f"{args.scn}/x_test.npy")),
    #     torch.from_numpy(np.load(f"{args.scn}/y_test.npy")),
    # )

    dl_train = DataLoader(
        dataset=ds_train,
        shuffle=True,
        num_workers=10,
        batch_size=1000,  # len(ds_train)
    )
    dl_val = DataLoader(
        dataset=ds_val,
        shuffle=False,
        num_workers=10,
        batch_size=1000,  # len(ds_val)
    )

    for attempt in range(1):
        checkpoint_dir = Path(
            f"cache/{args.scn}/checkpoints/{args.arch}/{attempt}"
        )
        checkpoint_file = get_checkpoint(checkpoint_dir)
        monitor = "loss/val"  # "perf/f1/val"
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir, monitor=monitor
        )
        earlystopping_callback = EarlyStopping(monitor=monitor, patience=20)

        trainer = pl.Trainer.from_argparse_args(
            args,
            # ckpt_path=checkpoint,
            resume_from_checkpoint=checkpoint_file,
            logger=TBLogger(
                f"{args.scn}/lightning_logs",
                name=f"{args.arch}",
                default_hp_metric=False,
            ),
            callbacks=[
                checkpoint_callback,
                earlystopping_callback,
            ],
            # check_val_every_n_epoch=10
            # limit_val_batches=0.2
        )
        trainer.fit(model, dl_train, dl_val)
        print("Best model: ", checkpoint_callback.best_model_path)
