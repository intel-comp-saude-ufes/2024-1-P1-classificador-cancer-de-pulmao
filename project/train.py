from typing import Tuple

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from project.datasets import LC25000
from project.models import CustomVGG19, CustomResNet


def eval_loss(
    model: nn.Module,
    device: str,
    loss_func: nn.Module,
    data_loader: DataLoader,
    softmax: bool = False,
) -> Tuple[float, float]:
    with torch.no_grad():
        loss, cnt = 0, 0
        true_preds, count = 0.0, 0

        y_true = []
        y_pred = []
        y_pred_class_0 = []
        y_pred_class_1 = []
        y_pred_class_2 = []

        for images, labels, _ in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images, softmax=softmax)
            loss += loss_func(outputs, labels).item()
            true_preds += (outputs.argmax(dim=-1) == labels).sum().item()
            count += labels.shape[0]
            cnt += 1

            outputs_batch = outputs.cpu().numpy()
            predicted_batch = np.argmax(outputs_batch, axis=1)
            labels_batch = labels.cpu().numpy()
            for i, j, k in zip(predicted_batch, labels_batch, outputs_batch):
                y_pred.append(i)
                y_true.append(j)
                y_pred_class_0.append(k[0])
                y_pred_class_1.append(k[1])
                y_pred_class_2.append(k[2])

    acc = true_preds / count
    loss = loss / cnt
    return (
        acc,
        loss,
        y_true,
        y_pred,
        y_pred_class_0,
        y_pred_class_1,
        y_pred_class_2,
    )


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_func: nn.Module,
    device: str,
    name: str,
    epochs: int = 50,
):
    model.to(device)
    writer = SummaryWriter()
    best_loss = np.inf

    losses_val = []
    losses_train = []

    accs_val = []
    accs_train = []

    for epoch in range(epochs):

        loss_epoch, cnt = 0, 0
        true_preds, count = 0.0, 0

        msg = f"- Epoch [{epoch+1}/{epochs}] "
        print(msg)

        for batch_images, batch_labels, _ in tqdm(train_dataloader):

            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_images)
            loss = loss_func(outputs, batch_labels)
            loss_epoch += loss.item()
            true_preds += (outputs.argmax(dim=-1) == batch_labels).sum().item()
            count += batch_labels.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1

        acc_train_epoch = true_preds / count
        loss_epoch = loss_epoch / cnt
        acc_val_epoch, loss_val_epoch, _, _, _, _, _ = eval_loss(
            model=model,
            device=device,
            data_loader=val_dataloader,
            loss_func=loss_func,
        )
        losses_train.append(loss_epoch)
        losses_val.append(loss_val_epoch)

        accs_train.append(acc_train_epoch)
        accs_val.append(acc_val_epoch)

        writer.add_scalar(f"LossTrain/{name}", loss_epoch, epoch)
        writer.add_scalar(f"LossVal/{name}", loss_val_epoch, epoch)
        writer.add_scalar(f"AccuracyTrain/{name}", acc_train_epoch, epoch)
        writer.add_scalar(f"AccuracyVal/{name}", acc_val_epoch, epoch)

        temp = {"Train": loss_epoch, "Val": loss_val_epoch}
        writer.add_scalars(f"Loss/{name}", temp, epoch)

        temp = {"Train": acc_train_epoch, "Val": acc_val_epoch}
        writer.add_scalars(f"Accuracy/{name}", temp, epoch)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_func,
            "loss_val": loss_epoch,
        }
        torch.save(checkpoint, f"./data/{name}/last_checkpoint.pth")
        if loss_val_epoch < best_loss:
            best_loss = loss_val_epoch
            torch.save(checkpoint, f"./data/{name}/best_checkpoint.pth")

        msg += f"- Loss: {loss_epoch:.4f} | Accuracy: {acc_train_epoch:.4f} "
        msg += f"| Loss Val: {loss_val_epoch:.4f} | Accuracy Val: {acc_val_epoch:.4f} "
        print(msg)

    checkpoint = torch.load(
        f"./data/{name}/best_checkpoint.pth",
        map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    (
        acc_test_epoch,
        loss_test_epoch,
        y_true,
        y_pred,
        y_pred_class_0,
        y_pred_class_1,
        y_pred_class_2,
    ) = eval_loss(
        model=model,
        device=device,
        data_loader=test_dataloader,
        loss_func=loss_func,
        softmax=True,
    )
    msg = f"- Final | Accuracy Test: {acc_test_epoch:.4f} | Loss Test: {loss_test_epoch:.4f} "
    print(msg)

    data_train = {
        "losses_train": losses_train,
        "losses_val": losses_val,
        "acc_train": accs_train,
        "acc_val": accs_val,
    }
    data_train = pd.DataFrame(data_train)
    data_train.to_csv(f"./data/{name}/train_metrics.csv")

    data_test = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_class_0": y_pred_class_0,
        "y_pred_class_1": y_pred_class_1,
        "y_pred_class_2": y_pred_class_2,
    }
    data_test = pd.DataFrame(data_test)
    data_test.to_csv(f"./data/{name}/test_inference.csv", index=False)


def main() -> None:
    train_set = pd.read_csv("./data/train.csv")
    test_set = pd.read_csv("./data/test.csv")
    val_set = pd.read_csv("./data/val.csv")

    tfs = {
        "train": transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    train_dataset = LC25000(
        images_paths=train_set["features"],
        labels=train_set["labels"],
        transform=tfs["train"],
    )
    val_dataset = LC25000(
        images_paths=val_set["features"],
        labels=val_set["labels"],
        transform=tfs["val"],
    )
    test_dataset = LC25000(
        images_paths=test_set["features"],
        labels=test_set["labels"],
        transform=tfs["val"],
    )
    batch_size = 30
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CustomResNet(num_class=3)
    model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_func=loss_func,
        device=device,
        name="resnet50",
        epochs=20,
    )

    model = CustomVGG19(num_class=3)
    model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_func=loss_func,
        device=device,
        name="vgg19",
        epochs=20,
    )


if __name__ == "__main__":
    main()
