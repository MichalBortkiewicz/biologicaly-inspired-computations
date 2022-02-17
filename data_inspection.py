import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle, class_weight
import seaborn as sns


ROOT_DIR = "."
DATA_DIR = os.path.join(ROOT_DIR, "dane_projekt")


def get_mapping_of_categories(y: np.ndarray):
    uniques = np.unique(y)
    return {idx: elem for idx, elem in enumerate(uniques)}


def load_file_and_split_into_x_and_y(file_name: str):
    data = np.loadtxt(os.path.join(DATA_DIR, file_name))
    return data[:, :-1], data[:, -1]


def create_merged_dataset(folder_name="dane_projekt"):
    file_names = os.listdir(folder_name)
    x_list = []
    y_list = []
    for file_name in file_names:
        x, y = load_file_and_split_into_x_and_y(file_name)
        x_list.append(x)
        y_list.append(y)
    return np.concatenate(x_list), np.concatenate(y_list)


if __name__ == "__main__":
    # x, y = create_merged_dataset()
    # x, y = shuffle(x, y)
    #
    # mapping = get_mapping_of_categories(y)
    # mapping_orginal_to_new = dict((y, x) for x, y in mapping.items())
    #
    # y_mapped = np.array([mapping_orginal_to_new[elem ] for elem in y])
    #
    # x = preprocessing.StandardScaler().fit_transform(x)

    x, y = create_merged_dataset()
    x, y = shuffle(x, y, random_state=42)
    y_preprocessed = copy.deepcopy(y)
    x = preprocessing.StandardScaler().fit_transform(x)

    mapping = get_mapping_of_categories(y)
    mapping_orginal_to_new = dict((y, x) for x, y in mapping.items())

    y_original = np.array([mapping_orginal_to_new[elem] for elem in y])
    x_original = copy.deepcopy(x)

    # Cross validation
    scoring = ["f1_weighted", "accuracy"]
    clf = make_pipeline(LogisticRegression(random_state=42, max_iter=500))
    scores = cross_validate(
        clf, x_original, y_original, cv=5, scoring=scoring, return_train_score=True
    )

    # corss check
    class_weights = class_weight.compute_class_weight(
        "balanced",
        classes=np.unique(y_original[: int(4 / 5 * x.shape[0])]),
        y=y_original[: int(4 / 5 * x.shape[0])],
    )

    class_weights = {
        cls: weight
        for cls, weight in zip(
            np.unique(y_original[: int(4 / 5 * x.shape[0])]), class_weights
        )
    }

    clf = LogisticRegression(random_state=42, max_iter=500, class_weight=class_weights)
    clf.fit(x[: int(4 / 5 * x.shape[0])], y_original[: int(4 / 5 * x.shape[0])])
    y_pred = clf.predict(x[int(4 / 5 * x.shape[0]) :])
    print(
        classification_report(
            y_original[int(4 / 5 * x.shape[0]) :],
            y_pred,
        )
    )

    # Feature importance
    forest = RandomForestClassifier(random_state=0)
    forest.fit(x, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    feature_names = [f"feature {i}" for i in range(x.shape[1])]
    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots(figsize=(12, 8))
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig("figures/importances.png")
    plt.show()

    # Correlation

    df = pd.DataFrame(x)
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.savefig("figures/corr.png")

    # Class imbalance
    classes, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(8, 8))
    plt.pie(list(counts), labels=[str(int(elem)) for elem in classes])
    plt.savefig("figures/imbalance.png")
    plt.show()
