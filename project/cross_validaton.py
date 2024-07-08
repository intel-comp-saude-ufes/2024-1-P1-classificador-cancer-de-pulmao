import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader

from project.datasets import LC25000
from project.models import CustomResNet, CustomVGG19
from project.train import train

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
)

def cross_validate(
    X: pd.DataFrame, 
    y: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
    epochs: int = 30,
    batch_size: int = 30,
) -> None:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            stratify=y_train,
            test_size=0.25,
            random_state=random_state,
        )

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
            images_paths=X_train.values,
            labels=y_train.values,
            transform=tfs["train"],
        )
        val_dataset = LC25000(
            images_paths=X_val.values,
            labels=y_val.values,
            transform=tfs["val"],
        )
        test_dataset = LC25000(
            images_paths=X_test.values,
            labels=y_test.values,
            transform=tfs["val"],
        )
    
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
            name=f"resnet50_fold_{fold}",
            epochs=epochs,
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
            name=f"vgg19_fold_{fold}",
            epochs=epochs,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    data = pd.read_csv("./data/all.csv")
    cross_validate(
        X=data["features"],
        y=data["labels"],
        n_folds=5,
        random_state=42,
        epochs=30,
        batch_size=30,
    )
