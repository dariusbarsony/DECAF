from typing import Any, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


# It will apply a perturbation at each node provided in perturb.
def gen_data_nonlinear(
        G: Any,
        base_mean: float = 0,
        base_var: float = 0.3,
        mean: float = 0,
        var: float = 1,
        SIZE: int = 10000,
        err_type: str = "normal",
        perturb: list = [],
        sigmoid: bool = True,
        expon: float = 1.1,
) -> pd.DataFrame:
    list_edges = G.edges()
    list_vertex = G.nodes()

    order = []
    for ts in nx.algorithms.dag.topological_sort(G):
        order.append(ts)

    g = []
    for v in list_vertex:
        if v in perturb:
            g.append(np.random.normal(mean, var, SIZE))
            print("perturbing ", v, "with mean var = ", mean, var)
        else:
            if err_type == "gumbel":
                g.append(np.random.gumbel(base_mean, base_var, SIZE))
            else:
                g.append(np.random.normal(base_mean, base_var, SIZE))

    for o in order:
        for edge in list_edges:
            if o == edge[1]:  # if there is an edge into this node
                if sigmoid:
                    g[edge[1]] += 1 / 1 + np.exp(-g[edge[0]])
                else:
                    g[edge[1]] += g[edge[0]] ** 2
    g = np.swapaxes(g, 0, 1)

    return pd.DataFrame(g, columns=list(map(str, list_vertex)))


def load_adult() -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    path_test =  "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    
    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "label",
    ]
    df_train = pd.read_csv(path, names=names, index_col=False)
    df_test = pd.read_csv(path_test, names=names, index_col=False)[1:]

    df = pd.concat([df_train, df_test])
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)

    for col in df:
        if df[col].dtype == "object":
            df = df[df[col] != "?"]

    df["label"].replace({'<=50K.': '<=50K', '>50K.': '>50K'}, inplace=True)

    replace = [
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ],
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool",
        ],
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ],
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ],
        [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried",
        ],
        ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
        ["Female", "Male"],
        [
            "United-States",
            "Cambodia",
            "England",
            "Puerto-Rico",
            "Canada",
            "Germany",
            "Outlying-US(Guam-USVI-etc)",
            "India",
            "Japan",
            "Greece",
            "South",
            "China",
            "Cuba",
            "Iran",
            "Honduras",
            "Philippines",
            "Italy",
            "Poland",
            "Jamaica",
            "Vietnam",
            "Mexico",
            "Portugal",
            "Ireland",
            "France",
            "Dominican-Republic",
            "Laos",
            "Ecuador",
            "Taiwan",
            "Haiti",
            "Columbia",
            "Hungary",
            "Guatemala",
            "Nicaragua",
            "Scotland",
            "Thailand",
            "Yugoslavia",
            "El-Salvador",
            "Trinadad&Tobago",
            "Peru",
            "Hong",
            "Holand-Netherlands",
        ],
        [">50K", "<=50K"],
    ]

    for row in replace:
        df = df.replace(row, range(len(row)))

    df = pd.DataFrame(MinMaxScaler().fit_transform(df),
                      index=df.index, columns=df.columns)

    return df

# Helper functions from reproducibility study into DECAF (https://github.com/ShuaiWang97/UvA_FACT2022)

columns_adult = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "label",
]

columns_credit = [
    "male",
    "age",
    "debt",
    "married",
    "bankcustomer",
    "educationlevel",
    "ethnicity",
    "yearsemployed",
    "priordefault",
    "employed",
    "creditscore",
    "driverslicense",
    "citizen",
    "zip",
    "income",
    "approved",
]

def get_metrics(df_train, df_test):

    X_train, y_train = df_train.drop(columns=['label']), df_train['label']
    X_test, y_test = df_test.drop(columns=['label']), df_test['label']

    clf = MLPClassifier(verbose=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred)

    dp = DP(clf, X_test)
    ftu = FTU(clf, X_test)

    return {'precision': precision, 'recall': recall, 'auroc': auroc,'dp': dp, 'ftu': ftu}

def DP(mlp, X_test, dataset="adult"):
    """Calculate fairness metric DP"""

    columns = columns_adult if dataset == "adult" else columns_credit
    X_test_df = pd.DataFrame(X_test, columns=columns[:-1])
    if 'ethnicity' in X_test_df:
        X_test_0 = X_test_df[X_test_df["ethnicity"] < 0.5]
        X_test_1 = X_test_df[X_test_df["ethnicity"] > 0.5]
    else:
        X_test_0 = X_test_df[X_test_df["sex"] < 0.5]
        X_test_1 = X_test_df[X_test_df["sex"] > 0.5]
    dp = abs(np.mean(mlp.predict(X_test_0)) - np.mean(mlp.predict(X_test_1)))

    return dp


def FTU(mlp, X_test, dataset="adult"):
    """Calculate fairness metric FTU"""

    columns = columns_adult if dataset == "adult" else columns_credit
    X_test_df = pd.DataFrame(X_test, columns=columns[:-1])
    if 'ethnicity' in X_test_df:
        X_test_0 = X_test_df.assign(ethnicity=0)
        X_test_1 = X_test_df.assign(ethnicity=1)
    else:
        X_test_0 = X_test_df.assign(sex=0)
        X_test_1 = X_test_df.assign(sex=1)

    ftu = abs(np.mean(mlp.predict(X_test_0)) - np.mean(mlp.predict(X_test_1)))

    return ftu

    
