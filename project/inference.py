import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from project.datasets import LC25000
from project.models import CustomResNet


def main() -> None:
    checkpoint = "./data/resnet/best_checkpoint.pth"
    test = pd.read_csv("./data/resnet/predictions.csv")
    tfs = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_dataset = LC25000(
        images_paths=test["features"],
        labels=test["labels"],
        transform=tfs,
    )
    batch_size = 10
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CustomResNet(num_class=3)
    model = model.to(device)

    checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true = []
    y_pred = []
    y_pred_class_0 = []
    y_pred_class_1 = []
    y_pred_class_2 = []

    with torch.no_grad():
        for image, label, image_id in test_dataloader:
            image = image.to(device)

            outputs = model(image, softmax=True)
            outputs = outputs.cpu().numpy()
            predicted = np.argmax(outputs, axis=1)

            labels = label.cpu().numpy()
            for i, j, k in zip(predicted, labels, outputs):
                y_pred.append(i)
                y_true.append(j)
                y_pred_class_0.append(k[0])
                y_pred_class_1.append(k[1])
                y_pred_class_2.append(k[2])
    data = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_class_0":  y_pred_class_0,
        "y_pred_class_1":  y_pred_class_1,
        "y_pred_class_2":  y_pred_class_2,
    }
    df = pd.DataFrame(data)
    df.to_csv("./data/resnet/predictions.csv", index=False)

if __name__ == "__main__":
    main()
