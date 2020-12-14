"""
===========================
OpenDP Covariance
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
# # Differentially Private Covariance
#
# SmartNoise offers three different functionalities within its `covariance` function:
#
# 1. Covariance between two vectors
# 2. Covariance matrix of a matrix
# 3. Cross-covariance matrix of a pair of matrices, where element $(i,j)$ of the returned matrix is the covariance of column $i$ of the left matrix and column $j$ of the right matrix.

# %%
# load libraries
import os
import opendp.smartnoise.core as sn
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

# establish data information
data_path = os.path.join('.', 'data-analysis', 'PUMS_california_demographics_1000', 'data.csv')
var_names = ["age", "sex", "educ", "race", "income", "married"]

data = np.genfromtxt(data_path, delimiter=',', names=True)

# %% [markdown]
# ### Functionality
#
# Below we show the relationship between the three methods by calculating the same covariance in each. We use a much larger $\epsilon$ than would ever be used in practice to show that the methods are consistent with one another.   

# %%
with sn.Analysis() as analysis:
    wn_data = sn.Dataset(path = data_path, column_names = var_names)
    
    # get scalar covariance
    age_income_cov_scalar = sn.dp_covariance(left = sn.to_float(wn_data['age']),
                                        right = sn.to_float(wn_data['income']),
                                        privacy_usage = {'epsilon': 5000},
                                        left_lower = 0.,
                                        left_upper = 100.,
                                        left_rows = 1000,
                                        right_lower = 0.,
                                        right_upper = 500_000.,
                                        right_rows = 1000)
    
    # get full covariance matrix
    age_income_cov_matrix = sn.dp_covariance(data = sn.to_float(wn_data['age', 'income']),
                                           privacy_usage = {'epsilon': 5000},
                                           data_lower = [0., 0.],
                                           data_upper = [100., 500_000],
                                           data_rows = 1000)

    # get cross-covariance matrix
    cross_covar = sn.dp_covariance(left = sn.to_float(wn_data['age', 'income']),
                                   right = sn.to_float(wn_data['age', 'income']),
                                   privacy_usage = {'epsilon': 5000},
                                   left_lower = [0., 0.],
                                   left_upper = [100., 500_000.],
                                   left_rows = 1_000,
                                   right_lower = [0., 0.],
                                   right_upper = [100., 500_000.],
                                   right_rows = 1000)

# analysis.release()
print('scalar covariance:\n{0}\n'.format(age_income_cov_scalar.value))
print('covariance matrix:\n{0}\n'.format(age_income_cov_matrix.value))    
print('cross-covariance matrix:\n{0}'.format(cross_covar.value))

# %% [markdown]
# ### DP Covariance in Practice
#   
# We now move to an example with a much smaller $\epsilon$. 

# %%
with sn.Analysis() as analysis:
    wn_data = sn.Dataset(path = data_path, column_names = var_names)
    # get full covariance matrix
    cov = sn.dp_covariance(data = sn.to_float(wn_data['age', 'sex', 'educ', 'income', 'married']),
                                          privacy_usage = {'epsilon': 1.},
                                          data_lower = [0., 0., 1., 0., 0.],
                                          data_upper = [100., 1., 16., 500_000., 1.],
                                          data_rows = 1000)
analysis.release()

# store DP covariance and correlation matrix
dp_cov = cov.value
dp_corr = dp_cov / np.outer(np.sqrt(np.diag(dp_cov)), np.sqrt(np.diag(dp_cov)))

# get non-DP covariance/correlation matrices
age = list(data[:]['age'])
sex = list(data[:]['sex'])
educ = list(data[:]['educ'])
income = list(data[:]['income'])
married = list(data[:]['married'])
non_dp_cov = np.cov([age, sex, educ, income, married])
non_dp_corr = non_dp_cov / np.outer(np.sqrt(np.diag(non_dp_cov)), np.sqrt(np.diag(non_dp_cov)))

print('Non-DP Correlation Matrix:\n{0}\n\n'.format(pd.DataFrame(non_dp_corr)))
print('DP Correlation Matrix:\n{0}'.format(pd.DataFrame(dp_corr)))

# %%
fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize = (9, 11))

# generate a mask for the upper triangular matrix
mask = np.triu(np.ones_like(non_dp_corr, dtype = np.bool))

# generate color palette
cmap = sns.diverging_palette(220, 10, as_cmap = True)

# get correlation plots
ax_1.title.set_text('Non-DP Correlation Matrix')
sns.heatmap(non_dp_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                          square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax = ax_1)
ax_1.set_xticklabels(labels = ['age', 'sex', 'educ', 'income', 'married'], rotation = 45)
ax_1.set_yticklabels(labels = ['age', 'sex', 'educ', 'income', 'married'], rotation = 45)


ax_2.title.set_text('DP Correlation Matrix')
sns.heatmap(dp_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                          square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax = ax_2)
ax_2.set_xticklabels(labels = ['age', 'sex', 'educ', 'income', 'married'], rotation = 45)
ax_2.set_yticklabels(labels = ['age', 'sex', 'educ', 'income', 'married'], rotation = 45)



# %% [markdown]
# Notice that the differentially private correlation matrix contains values outside of the feasible range for correlations, $[-1, 1]$. This is not uncommon, especially for analyses with small $\epsilon$, and is not necessarily indicative of a problem. In this scenario, we will not use these correlations for anything other than visualization, so we will leave our result as is.
#
# Sometimes, you may get a result that does cause problems for downstream analysis. For example, say your differentially private covariance matrix is not positive semi-definite. There are a number of ways to deal with problems of this type.
#
# 1. Relax your original plans: For example, if you want to invert your DP covariance matrix and are unable to do so, you could instead take the pseudoinverse.
# 2. Manual Post-Processing: Choose some way to change the output such that it is consistent with what you need for later analyses. This changed output is still differentially private (we will use this idea again in the next section). For example, map all negative variances to small positive value.
# 3. More releases: You could perform the same release again (perhaps with a larger $\epsilon$) and combine your results in some way until you have a release that works for your purposes.  Note that additional $\epsilon$ from will be consumed everytime this happens.  

# %% [markdown]
# ### Post-Processing of DP Covariance Matrix: Regression Coefficient
#
# Differentially private outputs are "immune" to post-processing, meaning functions of differentially private releases are also differentially private (provided that the functions are independent of the underlying data in the dataset). This idea provides us with a relatively easy way to generate complex differentially private releases from simpler ones.
#
# Say we wanted to run a linear regression of the form $income = \alpha + \beta \cdot educ$ and want to find an differentially private estimate of the slope, $\hat{\beta}_{DP}$. We know that 
# $$ \beta = \frac{cov(income, educ)}{var(educ)}, $$ 
# and so 
# $$ \hat{\beta}_{DP} = \frac{\hat{cov}(income, educ)_{DP}}{ \hat{var}(educ)_{DP} }. $$
#
# We already have differentially private estimates of the necessary covariance and variance, so we can plug them in to find $\hat{\beta}_{DP}$.
#
#

# %%
'''income = alpha + beta * educ'''
# find DP estimate of beta
beta_hat_dp = dp_cov[2,3] / dp_cov[2,2]
beta_hat = non_dp_cov[2,3] / non_dp_cov[2,2]

print('income = alpha + beta * educ')
print('DP coefficient: {0}'.format(beta_hat_dp))
print('Non-DP Coefficient: {0}'.format(beta_hat))

# %% [markdown] pycharm={"name": "#%% md\n"}
# This result is implausible, as it would suggest that an extra year of education is associated with, on average, a decrease in annual income of nearly $11,000. It's not uncommon for this to be the case for DP releases constructed as post-processing from other releases, especially when they involve taking ratios. 
#
# If you find yourself in such as situation, it is often worth it to spend some extra privacy budget to estimate your quantity of interest using an algorithm optimized for that specific use case.
