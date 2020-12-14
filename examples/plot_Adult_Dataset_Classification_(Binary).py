"""
===========================
OpenDP Adult Dataset Classification (Binary)
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
# ## The Adult Dataset (Binary Classification with Synthetic Data)
# The adult dataset is a classic imbalanced classification task dataset. The final column specifies whether the person makes <= 50k a year, or more.
#
# Here we see MWEM stretched - we are forced to carefully select our feature dependence (via the "splits" feature) to acheive a reasonable data synthesis.
#
# We show that by either specifying a max_bin_count, or by dropping continuous columns, we can greatly speed up performance.

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
datasets = load_data(['adult'])

adult = datasets['adult']['data']
adult_cat_ord = datasets['adult']['data'].copy()

cat_ord_columns = ['workclass',
                       'marital-status', 
                       'occupation', 
                       'relationship', 
                       'race',
                       'gender',
                       'native-country',
                       'income',
                       'education',
                       'age',
                       'education-num',
                       'hours-per-week',
                  'earning-class']

for c in adult_cat_ord.columns.values:
    if not c in cat_ord_columns:
        adult_cat_ord = adult_cat_ord.drop([c], axis=1)

# %%
synth = MWEMSynthesizer(500, 0.1, 30, 15, splits=[[0,1,2],[3,4,5],[6,7,8],[9,10],[11,12],[13,14]], max_bin_count=400)
synth.fit(datasets['adult']['data'])

synth_cat_ord = MWEMSynthesizer(500, 0.1, 30, 15, split_factor=3)
synth_cat_ord.fit(adult_cat_ord)

# %%
sample_size = len(adult)
synthetic = synth.sample(int(sample_size))
synthetic_cat_ord = synth_cat_ord.sample(int(sample_size))

# %%
utils.test_real_vs_synthetic_data(adult, synthetic, RidgeClassifier, tsne=True, box=True, describe=True)

# %%
utils.test_real_vs_synthetic_data(adult_cat_ord, synthetic_cat_ord, ComplementNB, tsne=True, box=True, describe=True)

# %%
