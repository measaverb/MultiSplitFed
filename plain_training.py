import argparse
import os

import torch

import wandb
from datasets import get_dataset
from networks import ResNet18
from utils.misc import load_config, random_seed


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(config["experiment"]["work_dir"]):
        os.makedirs(config["experiment"]["work_dir"])

    # Load the dataset
    train_ds, test_ds = get_dataset(config["data"]["metadata_path"])

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=config["data"]["batch_size"], shuffle=True
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=config["data"]["batch_size"], shuffle=False
    )

    # Load the model
    net = ResNet18(num_classes=7).to(device)  # 7 classes for HAM10000 dataset
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["networks"]["lr"])
    # Training loop
    step = 0
    for epoch in range(config["training"]["epochs"]):
        net.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_dl):
            labels = labels.type(torch.LongTensor)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if i % 1 == 0:
                print(
                    f"Epoch [{epoch}/{config['training']['epochs']}], Step [{i}/{len(train_dl)}], Loss: {loss.item()}"
                )
                if config["experiment"]["wandb_logging"]:
                    wandb.log({"train/step/train_loss": loss.item()}, step=step)

        avg_train_loss = running_loss / len(train_dl)
        print(
            f"Epoch [{epoch}/{config['training']['epochs']}], Train Loss: {avg_train_loss}"
        )
        if config["experiment"]["wandb_logging"]:
            wandb.log({"train/epoch/train_loss": avg_train_loss})

        # Evaluation loop
        net.eval()
        running_val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_dl:
                labels = labels.type(torch.LongTensor)
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            avg_val_loss = running_val_loss / len(test_dl)
            print(
                f"Epoch [{epoch}/{config['training']['epochs']}], Accuracy: {accuracy}, Validation Loss: {avg_val_loss}"
            )
            if config["experiment"]["wandb_logging"]:
                wandb.log(
                    {"val/epoch/accuracy": accuracy, "val/epoch/val_loss": avg_val_loss}
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
