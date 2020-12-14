"""
===========================
OpenDP Car Dataset Mwem (Multiclass)
===========================
"""

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Car Dataset Multiclass Classification
# Here we can see MWEM shine on a Multiclass classification problem when the data is purely categorical. The synthetic data performs quite comparably to the original dataset, and way outperforms random guessing.
#
# We can also note that the striking overlap between the synthetic data and the real data in the TSNE plot.

# %%
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import RidgeClassifier

import utils

from opendp.smartnoise.synthesizers.mwem import MWEMSynthesizer

from load_data import load_data

# %%
datasets = load_data(['car'])

# %%
synth = MWEMSynthesizer(400, 3.00, 40, 20, split_factor=7, max_bin_count=400)
synth.fit(datasets['car']['data'])

# %%
sample_size = len(datasets['car']['data'])
synthetic = synth.sample(int(sample_size))

# %%
utils.test_real_vs_synthetic_data(datasets['car']['data'], synthetic, RidgeClassifier, tsne=True, box=True, describe=True)

# %%
