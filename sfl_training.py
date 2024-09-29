import argparse
import os

import numpy as np
import torch

import wandb
from datasets import Shard, get_dataset
from networks import ClientSideResNet18, ServerSideResNet18
from utils.misc import get_optimizer, load_config, random_seed
from utils.sfl_related import dataset_iid, federated_averaging


def get_clients(config, device):
    local_client_nets = []
    local_client_optimizers = []
    for _ in range(config["training"]["num_clients"]):
        local_client_net = ClientSideResNet18().to(device)
        local_client_nets.append(local_client_net)
        local_client_optimizer = get_optimizer(config, local_client_net)
        local_client_optimizers.append(local_client_optimizer)
    return local_client_nets, local_client_optimizers


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(config["experiment"]["work_dir"]):
        os.makedirs(config["experiment"]["work_dir"])

    # Load the dataset
    train_ds, test_ds = get_dataset(config["data"]["metadata_path"])

    dict_users_train = dataset_iid(train_ds, config["training"]["num_clients"])
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=config["data"]["batch_size"], shuffle=False
    )

    # Load the model
    server_net = ServerSideResNet18(num_classes=7).to(device)
    global_client_net = ClientSideResNet18().to(device)
    global_client_weight = global_client_net.state_dict()

    local_client_nets, local_client_optimizers = get_clients(config, device)

    criterion = torch.nn.CrossEntropyLoss()
    server_optimizer = get_optimizer(config, server_net)

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        m = max(int(config["training"]["num_clients"]), 1)
        idxs_users = np.random.choice(
            range(config["training"]["num_clients"]), m, replace=False
        )
        local_client_weights = []

        for idx in idxs_users:
            print(idx)
            train_dl = torch.utils.data.DataLoader(
                Shard(train_ds, dict_users_train[idx]),
                batch_size=config["data"]["batch_size"],
                shuffle=True,
            )

            local_client_nets[idx].load_state_dict(global_client_weight)
            local_client_nets[idx].train()
            server_net.train()
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_dl):
                labels = labels.type(torch.LongTensor)
                images, labels = images.to(device), labels.to(device)

                local_client_optimizers[idx].zero_grad()
                server_optimizer.zero_grad()

                # local client-side net forward-propagation
                fx = local_client_nets[idx](images)
                fx_client = fx.clone().detach().requires_grad_(True)

                # server-side net forward-propagation
                outputs = server_net(fx_client)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # server-side net backward-propagation
                loss.backward()
                dfx_client = fx_client.grad.clone().detach()
                server_optimizer.step()

                # local client-side net backward-propagation
                fx.backward(dfx_client)
                local_client_optimizers[idx].step()

                print(
                    f"Epoch [{epoch}/{config['training']['epochs']}], Step [{i}/{len(train_dl)}, Client {idx}], Loss: {loss.item()}"
                )
                if config["experiment"]["wandb_logging"]:
                    wandb.log({f"train/step/client_{idx}/train_loss": loss.item()})

            avg_train_loss = running_loss / len(train_dl)
            print(
                f"Epoch [{epoch}/{config['training']['epochs']}], Train Loss: {avg_train_loss}"
            )
            if config["experiment"]["wandb_logging"]:
                wandb.log({"train/epoch/client_{idx}/train_loss": avg_train_loss})

            local_client_weights.append(local_client_nets[idx].state_dict())

        print("Averaging Point")
        global_client_weight = federated_averaging(local_client_weights)
        global_client_net.load_state_dict(global_client_weight)

        # Evaluation loop
        global_client_net.eval()
        server_net.eval()
        running_val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_dl:
                labels = labels.type(torch.LongTensor)
                images, labels = images.to(device), labels.to(device)

                # local client-side net forward-propagation
                fx = global_client_net(images)
                fx_client = fx.clone().detach().requires_grad_(True)

                # server-side net forward-propagation
                outputs = server_net(fx_client)

                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            avg_val_loss = running_val_loss / len(test_dl)
            print(
                f"Epoch [{epoch}/{config['training']['epochs']}], Accuracy: {accuracy}, Test Loss: {avg_val_loss}"
            )
            if config["experiment"]["wandb_logging"]:
                wandb.log(
                    {
                        "test/epoch/accuracy": accuracy,
                        "test/epoch/val_loss": avg_val_loss,
                    }
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MultiSplitFed: Plain Training")
    parser.add_argument(
        "--config", "-c", required=True, help="Path to the JSON configuration file"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    random_seed(config["experiment"]["random_seed"])

    if config["experiment"]["wandb_logging"]:
        wandb.init(
            project="multisplitfed-experiments", name=config["experiment"]["wandb_run"]
        )
    main(config)
