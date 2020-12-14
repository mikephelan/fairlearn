"""
===========================
OpenDP Accuracy Pitfalls
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
# # Accuracy: Pitfalls and Edge Cases
#
# This notebook describes SmartNoise's accuracy calculations, and ways in which an analyst might be tripped up by them.
#
# ### Overview 
#
# #### Accuracy vs. Confidence Intervals
#
# Each privatizing mechanism (e.g. Laplace, Gaussian) in SmartNoise has an associated accuracy that is a function of the requested privacy usage, sensitivity of the function whose values is being privatized, etc. Imagine you have data $D$ and you want, for some function $\phi$ to return $\phi(D)$ in a differentially private way -- we will call this value $\phi_{dp}(D)$. An $\alpha$-level accuracy guarantee $a$ promises that, over infinite runs of the privatizing mechanism on the data in question, 
# $$ \phi(D) \in [\phi_{dp}(D) - a, \phi_{dp}(D) + a] $$
# with probability $1 - \alpha$.
#
# This looks very much like the traditional confidence interval, but it is important to note a major difference. In a canonical confidence interval, the uncertainty being represented is due to sampling error -- that is, how often will it be the case that $\phi(P)$ (the value of $\phi$ on the underlying population) is within some range of the realized $\phi(D)$. 
#
# In SmartNoise (and differentially private data analysis generally), there is an extra layer of uncertainty due to the noise added to $\phi(D)$ to produce $\phi_{dp}(D)$. SmartNoise's accuracy metrics deal only with the uncertainty of $\phi_{dp}(D)$ relative to $\phi(D)$ and not the uncertainty of $\phi(D)$ relative to $\phi(P)$.
#
# #### What is $D$?
#
# SmartNoise allows for analysis of data with an unknown number of rows by resizing the data to ensure consistency with an estimated size (see the [unknown dataset size notebook](https://github.com/opendifferentialprivacy/smartnoise-samples/blob/master/analysis/unknown_dataset_size.ipynb) for more details). Accuracy guarantees are always relative to the resized data $\tilde{D}$, not the data of unknown size.
#
# #### Synopsis
#
# Let's say an analyst releases $\phi_{dp}(D)$ and gets an accuracy guarantee of $a$ at accuracy-level $\alpha$. $D$ as a dataset of unknown size drawn from population $P$ and will be resized to $\tilde{D}$. This suggests that over infinite runs of this procedure,
#
# - $\phi_{dp}(D) \in [\phi(\tilde{D}) - a, \phi(\tilde{D}) + a]$ with probability $1 - \alpha$
# - It is likely that $\phi_{dp}(D) \in [\phi(D) - a, \phi(D) + a]$ with probability $\approx 1 - \alpha$, though we cannot make any guarantee. For many cases (e.g. resizing the data based on $n$ obtained from a differentially private count and reasonable bounds on the data elements), this is likely to be approximately true. In the next section, we will explore some examples of cases where this statement holds to varying extents.
#
# - We cannot directly make statements about the relationship uncertainty of $\phi_{dp}(D)$ relative to $\phi(P)$.
#
# ### Accuracy Guarantees In Practice
#
# We now move to some empirical evaluations of how well our accuracy guarantees translate from $\phi(\tilde{D})$ to $\phi(D)$. We first consider the case where we actually know the size of the underlying data and are able to set plausible lower/upper bounds on `age`.
#

# %% pycharm={"is_executing": true}
# load libraries
import os
import sys
import numpy as np
import pandas as pd
import opendp.smartnoise.core as sn

# establish data information
data_path = os.path.join('.', 'data-analysis', 'PUMS_california_demographics_1000', 'data.csv')
var_names = ["age", "sex", "educ", "race", "income", "married", "pid"]
D = pd.read_csv(data_path)['age']
D_mean_age = np.mean(D)

# establish extra information for this simulation
age_lower_bound = 0.
age_upper_bound = 100.
D_tilde = np.clip(D, age_lower_bound, age_upper_bound)
D_tilde_mean_age = np.mean(D_tilde)
data_size = 1000

n_sims = 1_000
releases = []
with sn.Analysis(dynamic = True) as analysis:
    data = sn.Dataset(path = data_path, column_names = var_names)
    D = sn.to_float(data['age'])
    # preprocess data (resize is a no-op because we have the correct data size)
    D_tilde = sn.resize(sn.clamp(data = D, lower=0., upper=100.), number_rows = data_size)
    
    for index in range(n_sims):
       # get DP mean of age
        releases.append(sn.dp_mean(
            data = sn.impute(D_tilde),
            privacy_usage = {'epsilon': 1}))

accuracy = releases[0].get_accuracy(0.05)

analysis.release()
dp_values = [release.value for release in releases]
print('Accuracy interval (with accuracy value {0}) contains the true mean on D_tilde with probability {1}'.format(
    round(accuracy, 4), 
    np.mean([(D_tilde_mean_age >= val - accuracy) & (D_tilde_mean_age <= val + accuracy) for val in dp_values])))

print('Accuracy interval (with accuracy value {0}) contains the true mean on D with probability {1}'.format(
    round(accuracy, 4), 
    np.mean([(D_mean_age >= val - accuracy) & (D_mean_age <= val + accuracy) for val in dp_values])))

# %% [markdown]
# This performance is as expected. $D$ and $\tilde{D}$ are actually the exact same data (the maximum age in the raw data is 93, so our clamp to $[0, 100]$ does not change any values, and we know the correct $n$), so our theoretical guarantees on $\tilde{D}$ map exactly to gaurantees on $D$.
#
# We now move to a scenario that is still realistic, but where the performance does not translate quite as well. In this case, we imagine that the analyst believes the data to be of size 1050 and uses the default imputation within resize so that the extra 50 elements are drawn uniformly from $\{ 0, 1, ..., 100 \}$.
#
# Note that our diagnostic testing of $\tilde{D}$ in the code above is not trivial in this case. In the first example, we knew that clamp/resize did not change the underlying data, so we could predict exactly the data on which the DP mean would actually be calculated. This will not be true for the following examples, so we will simulate finding the true underlying mean by releasing an extra DP mean with very high epsilon.  

# %% pycharm={"is_executing": true}
# establish extra information for this simulation
age_lower_bound = 0.
age_upper_bound = 100.
data_size = 1050

n_sims = 1_000
true_D_tilde_mean_ages = []
dp_D_tilde_mean_ages = []

# all nodes are retained in the final analysis
with sn.Analysis(filter_level='all') as analysis:
    
    data = sn.Dataset(path = data_path, column_names = var_names)
    
    # this is a no-op, but provides static guarantees
    D = sn.clamp(sn.to_float(data['age']), age_lower_bound, age_upper_bound)
    
    for index in range(n_sims):
        D_tilde = sn.impute(sn.resize(D, number_rows=data_size))
        
        # get true mean of age on D_tilde
        true_D_tilde_mean_ages.append(sn.mean(data = D_tilde))

        # get DP mean of age
        dp_D_tilde_mean_ages.append(sn.dp_mean(
            data = D_tilde,
            privacy_usage = {'epsilon': 1}))

accuracy = dp_D_tilde_mean_ages[0].get_accuracy(0.05)

analysis.release()

true_values = [true.value for true in true_D_tilde_mean_ages]
dp_values = [dp.value for dp in dp_D_tilde_mean_ages]

print('Accuracy interval (with accuracy value {0}) contains the true mean on D_tilde with probability {1}'.format(
    round(accuracy, 4), 
    np.mean([(true_val >= dp_val - accuracy) & (true_val <= dp_val + accuracy) for true_val,dp_val in zip(true_values, dp_values)])))

print('Accuracy interval (with accuracy value {0}) contains the true mean on D with probability {1}'.format(
    round(accuracy, 4), 
    np.mean([(D_mean_age >= dp_val - accuracy) & (D_mean_age <= dp_val + accuracy) for dp_val in dp_values])))

# %% [markdown]
# The accuracy guarantee still holds on $\tilde{D}$ (as it should), but we now see much worse performance relative to the true underlying data $D$.
