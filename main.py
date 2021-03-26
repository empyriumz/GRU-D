import numpy as np
import argparse
import os
import random

RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import torch

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

from utils import utils
from utils.preprocessing import MortalityDataLoader
from utils import metrics
from GRUD import GRUD

from torch.optim.lr_scheduler import ReduceLROnPlateau


def parse_arguments(parser):
    parser.add_argument(
        "--data_path",
        type=str,
        default="/host/StageNet/mortality_data",
        help="The path to the MIMIC-III data directory",
    )
    parser.add_argument("--file_name", type=str, help="File name to save model")
    parser.add_argument(
        "--partial_data", "-p", type=int, default=0, help="Use part of training data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.0015, help="Learing rate")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="Value controlling the coarse grain level",
    )
    parser.add_argument(
        "--input_dim", type=int, default=1, help="Dimension of visit record data"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=384, help="Dimension of hidden units in RNN"
    )
    parser.add_argument(
        "--dropout_dense",
        type=float,
        default=0.3,
        help="Dropout rate in the fully connected layer",
    )
    parser.add_argument(
        "--dropout_gru",
        type=float,
        default=0.3,
        help="Dropout rate inside the GRU",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import time

    start_time = time.time()
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    model_para = vars(args)
    """ Prepare training data"""
    print("Preparing training data ... ")
    train_data_loader = MortalityDataLoader(
        dataset_dir=os.path.join(args.data_path, "train"),
        listfile=os.path.join(args.data_path, "train-mortality.csv"),
        partial_data=model_para["partial_data"],
    )
    pos_weight = torch.sqrt(
        torch.tensor(train_data_loader.pos_weight, dtype=torch.float32)
    )

    val_data_loader = MortalityDataLoader(
        dataset_dir=os.path.join(args.data_path, "train"),
        listfile=os.path.join(args.data_path, "val-mortality.csv"),
        partial_data=model_para["partial_data"],
    )

    train_data_gen = utils.BatchDataGenerator(
        train_data_loader,
        model_para["batch_size"],
        shuffle=True,
    )
    val_data_gen = utils.BatchDataGenerator(
        val_data_loader,
        model_para["batch_size"],
        shuffle=False,
    )

    """Model structure"""
    print("Constructing model ... ")
    device = torch.device("cuda:0" if torch.cuda.is_available() == True else "cpu")
    print("available device: {}".format(device))

    model = GRUD(
        model_para["input_dim"],
        model_para["hidden_dim"],
        model_para["dropout_dense"],
        model_para["dropout_gru"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=args.weight_decay, lr=args.lr
    )
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.1)
    """Train phase"""
    print("Start training ... ")

    train_loss = []
    val_loss = []
    max_auroc = 0

    file_name = "./saved_weights/" + args.file_name
    for epoch in range(args.epochs):
        batch_loss = []
        model.train()
        for _ in range(train_data_gen.steps):
            batch_data = next(train_data_gen)
            batch_x = batch_data["data"][0]
            batch_x_last = batch_data["data"][1]
            batch_y = batch_data["data"][2]

            batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)
            batch_x_last = torch.tensor(batch_x_last, dtype=torch.float32).to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float32).to(device)
            batch_interval = torch.tensor(
                batch_data["interval"], dtype=torch.float32
            ).to(device)
            batch_mask = torch.tensor(batch_data["mask"], dtype=torch.float32).to(
                device
            )

            # cut long sequence
            if batch_x.size()[1] > 400:
                batch_x = batch_x[:, :400, :]
                batch_x_last = batch_x_last[:, :400, :]
                batch_y = batch_y[:, :400, :]
                batch_interval = batch_interval[:, :400, :]
                batch_mask = batch_mask[:, :400, :]

            optimizer.zero_grad()
            output = model(batch_x, batch_x_last, batch_interval, batch_mask, device)
            output = output.mean(axis=1)
            loss = pos_weight * batch_y * torch.log(output + 1e-7) + (
                1 - batch_y
            ) * torch.log(1 - output + 1e-7)
            loss = torch.neg(torch.sum(loss)) / batch_x.size()[0]
            batch_loss.append(loss.cpu().detach().numpy())

            loss.backward()
            optimizer.step()

        epoch_loss = np.mean(np.array(batch_loss))
        print("Epoch: {}, Training loss = {:.6f}".format(epoch, epoch_loss))
        train_loss.append(epoch_loss)

        print("\n==>Predicting on validation")
        with torch.no_grad():
            model.eval()
            cur_val_loss = []
            val_true = []
            val_pred = []
            for _ in range(val_data_gen.steps):
                val_data = next(val_data_gen)
                val_x = val_data["data"][0]
                val_x_last = val_data["data"][1]
                val_y = val_data["data"][2]

                val_x = torch.tensor(val_x, dtype=torch.float32).to(device)
                val_y = torch.tensor(val_y, dtype=torch.float32).to(device)
                val_interval = torch.tensor(
                    val_data["interval"], dtype=torch.float32
                ).to(device)
                val_mask = torch.tensor(val_data["mask"], dtype=torch.float32).to(
                    device
                )

                if val_x.size()[1] > 400:
                    val_x = val_x[:, :400, :]
                    val_y = val_y[:, :400, :]
                    val_interval = val_interval[:, :400, :]
                    val_mask = val_mask[:, :400, :]

                val_output = model(val_x, val_x_last, val_interval, val_mask, device)
                val_output = val_output.mean(axis=1)
                val_loss = pos_weight * val_y * torch.log(val_output + 1e-7) + (
                    1 - val_y
                ) * torch.log(1 - val_output + 1e-7)
                val_loss = torch.neg(torch.sum(val_loss)) / val_x.size()[0]
                cur_val_loss.append(val_loss.cpu().detach().numpy())

                for t, p in zip(
                    val_y.cpu().numpy().flatten(),
                    val_output.cpu().detach().numpy().flatten(),
                ):
                    val_true.append(t)
                    val_pred.append(p)
            cur_val_loss = np.mean(np.array(cur_val_loss))
            scheduler.step(cur_val_loss)
            print("Validation loss = {:.6f}".format(cur_val_loss))
            val_loss.append(cur_val_loss)
            print("\n")
            val_pred = np.array(val_pred)
            val_pred = np.stack([1 - val_pred, val_pred], axis=1)
            ret = metrics.print_metrics_binary(val_true, val_pred)
            cur_auroc = ret["auroc"]
            if cur_auroc > max_auroc:
                max_auroc = cur_auroc
                state = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "params": model_para,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                torch.save(state, file_name)
                print("\n------------ Save the best model ------------\n")
    end_time = time.time()
    print("total used time = {}".format(end_time - start_time))
