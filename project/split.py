import pandas as pd
from sklearn.model_selection import train_test_split

from project.utils import get_paths_and_labels


def main() -> None:
    data = get_paths_and_labels(
        path="./data/lung_colon_image_set/lung_image_sets/",
        lab_names=[
            "lung_n",
            "lung_aca",
            "lung_scc",
        ],
    )
    df = pd.DataFrame(data)
    X = df["features"]
    y = df["labels"]

    train = pd.concat([X, y], axis=1)
    train.to_csv("./data/all.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        stratify=y_train,
        test_size=0.25,
        random_state=42,
    )
    train = pd.concat([X_train, y_train], axis=1)
    train.to_csv("./data/train.csv", index=False)

    test = pd.concat([X_test, y_test], axis=1)
    test.to_csv("./data/test.csv", index=False)

    val = pd.concat([X_val, y_val], axis=1)
    val.to_csv("./data/val.csv", index=False)


if __name__ == "__main__":
    main()
