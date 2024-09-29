import argparse
import os

import numpy as np
import torch

import wandb
from datasets import Shard, get_dataset
from networks import ClientSideLinear, ClientSideResNet18, SmallerServerSideResNet18
from utils.misc import get_optimizer, load_config, random_seed
from utils.sfl_related import dataset_iid, federated_averaging


def get_clients(config, device):
    local_client_nets = []
    local_client_net_optimizers = []
    local_client_linears = []
    local_client_linear_optimizers = []
    for _ in range(config["training"]["num_clients"]):
        local_client_net = ClientSideResNet18().to(device)
        local_client_linear = ClientSideLinear(num_classes=7).to(device)
        local_client_nets.append(local_client_net)
        local_client_linears.append(local_client_linear)

        local_client_net_optimizer = get_optimizer(config, local_client_net)
        local_client_net_optimizers.append(local_client_net_optimizer)
        local_client_linear_optimizer = get_optimizer(config, local_client_linear)
        local_client_linear_optimizers.append(local_client_linear_optimizer)
    return (
        local_client_nets,
        local_client_linears,
        local_client_net_optimizers,
        local_client_linear_optimizers,
    )


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
    server_net = SmallerServerSideResNet18().to(device)
    global_client_net = ClientSideResNet18().to(device)
    global_client_weight = global_client_net.state_dict()

    (
        local_client_nets,
        local_client_linears,
        local_client_net_optimizers,
        local_client_linear_optimizers,
    ) = get_clients(config, device)

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
            train_dl = torch.utils.data.DataLoader(
                Shard(train_ds, dict_users_train[idx]),
                batch_size=config["data"]["batch_size"],
                shuffle=True,
            )

            local_client_nets[idx].load_state_dict(global_client_weight)
            local_client_nets[idx].train()
            local_client_linears[idx].train()
            server_net.train()
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_dl):
                labels = labels.type(torch.LongTensor)
                images, labels = images.to(device), labels.to(device)

                local_client_net_optimizers[idx].zero_grad()
                local_client_linear_optimizers[idx].zero_grad()
                server_optimizer.zero_grad()

                # local client-side net forward-propagation
                fx_client = local_client_nets[idx](images)
                fx_client_net = fx_client.clone().detach().requires_grad_(True)

                # server-side net forward-propagation
                fx_server = server_net(fx_client_net)
                fx_server_net = fx_server.clone().detach().requires_grad_(True)

                # local client-side linear forward-propagation
                outputs = local_client_linears[idx](fx_server_net)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # local client-side linear backward-propagation
                loss.backward()
                dfx_server = fx_server_net.grad.clone().detach()
                local_client_linear_optimizers[idx].step()

                # server-side net backward-propagation
                fx_server.backward(dfx_server)
                dfx_client = fx_client_net.grad.clone().detach()
                server_optimizer.step()

                # local client-side net backward-propagation
                fx_client.backward(dfx_client)
                local_client_net_optimizers[idx].step()

                print(
                    f"Epoch [{epoch}/{config['training']['epochs']}], Step [{i}/{len(train_dl)}, Client {idx}], Loss: {loss.item()}"
                )
                if config["experiment"]["wandb_logging"]:
                    wandb.log({f"train/step/client_{idx}/loss": loss.item()})

            avg_train_loss = running_loss / len(train_dl)
            print(
                f"Epoch [{epoch}/{config['training']['epochs']}], Train Loss: {avg_train_loss}"
            )
            if config["experiment"]["wandb_logging"]:
                wandb.log({f"train/epoch/client_{idx}/loss": avg_train_loss})

            local_client_weights.append(local_client_nets[idx].state_dict())

        global_client_weight = federated_averaging(local_client_weights)
        global_client_net.load_state_dict(global_client_weight)

        # Evaluation loop
        global_client_net.eval()
        server_net.eval()
        running_test_losses = [0.0] * m
        corrects, totals = [0] * m, [0] * m
        with torch.no_grad():
            for images, labels in test_dl:
                labels = labels.type(torch.LongTensor)
                images, labels = images.to(device), labels.to(device)

                # local client-side net forward-propagation
                client_net_hidden_states = global_client_net(images)

                # server-side net forward-propagation
                server_net_hidden_states = server_net(client_net_hidden_states)

                # local client-side linear forward-propagation
                for idx in range(m):
                    outputs = local_client_linears[idx](server_net_hidden_states)
                    loss = criterion(outputs, labels)
                    running_test_losses[idx] += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    totals[idx] += labels.size(0)
                    corrects[idx] += (predicted == labels).sum().item()

            for idx in range(m):
                accuracy = corrects[idx] / totals[idx]
                avg_test_loss = running_test_losses[idx] / len(test_dl)
                print(
                    f"Epoch [{epoch}/{config['training']['epochs']}], Client {idx}, Accuracy: {accuracy}, Test Loss: {avg_test_loss}"
                )
                if config["experiment"]["wandb_logging"]:
                    wandb.log(
                        {
                            f"test/epoch/client_{idx}/accuracy": accuracy,
                            f"test/epoch/client_{idx}/loss": avg_test_loss,
                        }
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MultiSplitFed: MultiSplitFed Learning"
    )
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
