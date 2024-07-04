import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import resnet50


from project.models import ConvNet, CustomResNet
from project.datasets import LC25000
from project.utils import get_paths_and_labels


def eval_loss(model, device, loss_func, data_loader):
    with torch.no_grad():
        loss, cnt = 0, 0
        true_preds, count = 0.0, 0
        for images, labels, img_id in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss += loss_func(outputs, labels).item()
            true_preds += (outputs.argmax(dim=-1) == labels).sum().item()
            count += labels.shape[0]
            cnt += 1
    acc = true_preds / count
    loss = loss / cnt
    return acc, loss


def main() -> None:
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    val = pd.read_csv("./data/val.csv")

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
        images_paths=train["features"],
        labels=train["labels"],
        transform=tfs["train"],
    )
    val_dataset = LC25000(
        images_paths=val["features"],
        labels=val["labels"],
        transform=tfs["val"],
    )
    test_dataset = LC25000(
        images_paths=test["features"],
        labels=test["labels"],
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
    writer = SummaryWriter()

    num_epochs = 30
    best_loss = np.inf

    for epoch in range(num_epochs):

        loss_epoch, cnt = 0, 0
        true_preds, count = 0.0, 0
        for k, (batch_images, batch_labels, id_img) in enumerate(train_dataloader):

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
        acc_val_epoch, loss_val_epoch = eval_loss(
            model=model,
            device=device,
            data_loader=val_dataloader,
            loss_func=loss_func,
        )

        writer.add_scalar("LossTrain", loss_epoch, epoch)
        writer.add_scalar("LossVal", loss_val_epoch, epoch)
        writer.add_scalar("AccuracyTrain", acc_train_epoch, epoch)
        writer.add_scalar("AccuracyVal", acc_val_epoch, epoch)

        temp = {"Train": loss_epoch, "Val": loss_val_epoch}
        writer.add_scalars("Loss", temp, epoch)

        temp = {"Train": acc_train_epoch, "Val": acc_val_epoch}
        writer.add_scalars("Accuracy", temp, epoch)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_func,
            "loss_val": loss_epoch,
        }
        torch.save(checkpoint, "last_checkpoint.pth")
        if loss_val_epoch < best_loss:
            best_loss = loss_val_epoch
            torch.save(checkpoint, "best_checkpoint.pth")
        print(
            f"- Epoch [{epoch+1}/{num_epochs}] | Loss: {loss_epoch:.4f} | Accuracy: {acc_train_epoch:.4f} | Loss Val: {loss_val_epoch:.4f} | Accuracy Val: {acc_val_epoch:.4f}"
        )

    acc_test_epoch, loss_test_epoch = eval_loss(
        model=model,
        device=device,
        data_loader=test_dataloader,
        loss_func=loss_func,
    )
    print(
        f"- Final | Accuracy Test: {acc_test_epoch:.4f} | Loss Test: {loss_test_epoch:.4f} "
    )


if __name__ == "__main__":
    main()
