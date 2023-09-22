from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from DECAF.DECAF import DECAF
from DECAF.data import DataModule
from .utils import gen_data_nonlinear, load_adult, get_metrics


def generate_baseline(size: int = 100) -> Tuple[torch.Tensor, DataModule, list, dict]:
    # causal structure is in dag_seed
    dag_seed = [
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 5],
        [2, 0],
        [3, 0],
        [3, 6],
        [3, 7],
        [6, 9],
        [0, 8],
        [0, 9],
    ]
    # edge removal dictionary
    bias_dict = {6: [3]}  # This removes the edge into 6 from 3.

    # DATA SETUP according to dag_seed
    G = nx.DiGraph(dag_seed)
    data = gen_data_nonlinear(G, SIZE=size)
    dm = DataModule(data.values)

    return torch.Tensor(np.asarray(data)), dm, dag_seed, bias_dict


def test_sanity_params() -> None:
    _, dummy_dm, seed, _ = generate_baseline()

    model = DECAF(
        dummy_dm.dims[0],
        dag_seed=seed,
    )

    assert model.generator is not None
    assert model.discriminator is not None
    assert model.x_dim == dummy_dm.dims[0]
    assert model.z_dim == dummy_dm.dims[0]


def test_sanity_train() -> None:
    _, dummy_dm, seed, _ = generate_baseline()

    model = DECAF(
        dummy_dm.dims[0],
        dag_seed=seed,
    )
    trainer = pl.Trainer(max_epochs=2, logger=False)

    trainer.fit(model, dummy_dm)


def test_sanity_generate() -> None:
    raw_data, dummy_dm, seed, bias_dict = generate_baseline(size=10)

    model = DECAF(
        dummy_dm.dims[0],
        dag_seed=seed,
    )
    trainer = pl.Trainer(max_epochs=2, logger=False)

    trainer.fit(model, dummy_dm)

    synth_data = (
        model.gen_synthetic(
            raw_data, gen_order=model.get_gen_order(), biased_edges=bias_dict
        )
            .detach()
            .numpy()
    )
    assert synth_data.shape[0] == 10


@pytest.mark.parametrize("X,y, df", [load_adult()])
@pytest.mark.slow
def test_run_experiments(d_train: pd.DataFrame, d_test: pd.DataFrame, mode=''):
    """Normalize X"""

    dm = DataModule(d_train.values)
    dm_test = DataModule(d_test.values)

    # causal structure is in dag_seed
    dag_seed = [[8, 6], 
                [8, 14], 
                [8, 12], 
                [8, 3], 
                [8, 5], 
                [0, 6], 
                [0, 12], 
                [0, 14], 
                [0, 1],
                [0, 5], 
                [0, 3], 
                [0, 7],
                [9, 6],
                [9, 5],
                [9, 14], 
                [9, 1], 
                [9, 3],
                [9, 7], 
                [13, 5],
                [13, 12],
                [13, 3],
                [13, 1],
                [13, 14],
                [13, 7],
                [5, 6],
                [5, 12], 
                [5, 14], 
                [5, 1],
                [5, 7], 
                [5, 3], 
                [3, 6], 
                [3, 12],
                [3, 14],
                [3, 1],
                [3, 7], 
                [6, 14],
                [12, 14],
                [1, 14],
                [7, 14]]

    # no debiasing
    if mode =='':
        bias_dict = {}
    if mode == 'ftu':
        # edge removal dictionary
        bias_dict = {14: [9]}  # ftu
    if mode == 'cf':
        bias_dict = {14: [9],
                     14: [5]}
    if mode == 'dp':
        bias_dict = {14: [9],
                     14: [5],
                     14: [7],
                     14: [6],
                     14: [12],
                     14: [3],
                     14: [1]}

    model = DECAF(
        dm.dims[0],
        dag_seed=dag_seed,
        lr=0.5e-3,
        batch_size=64,
        d_updates=10,
        alpha=2,
        rho=2,
        l1_W=1e-2,
        use_mask=True,
        grad_dag_loss=False,
        lambda_privacy=0,
        lambda_gp=10,
        weight_decay=1e-2,
        l1_g=0,
        p_gen=-1,
    )

    trainer = pl.Trainer(max_epochs=1, logger=False)

    trainer.fit(model, dm)

    X_synth = (
        model.gen_synthetic(
            dm.dataset.x,
            gen_order=model.get_gen_order(), biased_edges=bias_dict
        )
            .detach()
            .numpy()
    )

    X_synth[:, -1] = X_synth[:, -1].astype(np.int8)

    synth_dataset = pd.DataFrame(X_synth,
                                 index=d_train.index,
                                 columns=d_train.columns)

    # binarise columns
    synth_dataset['sex'] = np.round(synth_dataset['sex'])
    synth_dataset['label'] = np.round(synth_dataset['label'])

    return get_metrics(synth_dataset, d_test)