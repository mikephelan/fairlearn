"""
===========================
OpenDP Visualizing Mwem
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
# ## Fake data to visualize MWEM's histograms
# MWEM works by first creating a uniformly distributed histogram out of real data. It then iteratively updates this histogram with noisy samples from the real data. In other words, using the multiplicative weights mechanism, MWEM updates the histograms "weights" via the DP exponential mechanism (for querying the original data).
#
# Here, we create a heatmap from the histograms. We visualize the histogram made from the real data, and the differentially private histogram. Brighter values correspond to more higher probability bins in each histogram.

# %%
import os
import pandas as pd
import numpy as np
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from opendp.smartnoise.synthesizers.mwem import MWEMSynthesizer


# %%
def plot_histo(title,histo):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    plt.imshow(histo)
    ax.set_aspect('equal')
    cax = fig.add_axes([0.1, 1.0, 1., 0.1])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.set_frame_on(False)
    plt.colorbar(orientation='horizontal')
    plt.show()



# %%
# Make ourselves some fake data, with a "hot-spot" in the distribution
# in the bottom right corner
df = pd.DataFrame({'fake_column_1': [random.randint(0,100) for i in range(3000)] + [random.randint(80,100) for i in range(1000)],
                   'fake_column_2': [random.randint(0,100) for i in range(3000)] + [random.randint(80,100) for i in range(1000)],})

synth = MWEMSynthesizer(400, 10.0, 30, 20,[[0,1]])
synth.fit(df)

plot_histo('"Real" Data', synth.synthetic_histograms[0][1])
plot_histo('"Fake" Data', synth.synthetic_histograms[0][0])

# %% [markdown]
# ## Effect of Bin Count
# Here we can visualize the effect of specifying a max_bin_count. In the original data, we have 100 bins. If we halve that, we see that we still do a pretty good job at capturing the overall distribution.

# %%
synth = MWEMSynthesizer(400, 10.0, 30, 20,[[0,1]], max_bin_count=50)
synth.fit(df)

plot_histo('"Real" Data', synth.synthetic_histograms[0][1])
plot_histo('"Fake" Data', synth.synthetic_histograms[0][0])
