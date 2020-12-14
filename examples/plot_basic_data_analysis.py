"""
===========================
OpenDP Basic Data Analysis
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
# # Basic PUMS Analysis with SmartNoise
#
# This notebook will be a brief tutorial on doing data analysis within the SmartNoise system.
#
# We will start out by setting up our environment -- loading the necessary libraries and establishing the very basic things we need to know before loading our data (the file path and variable names).

# %%
# load libraries
import os
import sys
import numpy as np
import opendp.smartnoise.core as sn

# establish data information
data_path = os.path.join('.', 'data-analysis', 'PUMS_california_demographics_1000', 'data.csv')
var_names = ["age", "sex", "educ", "race", "income", "married", "pid"]

# %% [markdown]
# ### Properties
#
# The core SmartNoise library is made up of two key pieces; the runtime and the validator. The runtime is made up of low-level algorithms and operations. The validator contains logic for combining runtime elements into more complex operations, as well as methods for determining whether or not a computation is differentially private. If an analysis plan is deemed to produce data that are not differentially private, the validator will not allow this analysis to run. Importantly, this is done independent of the underlying data.
#
# Whether or not a set of computations produces differentially private data relies on a set of properties of the data. These properties can be statically determined (without touching the actual data) and can be updated at each step of the analysis. One pair of common properties is `lower/upper`. For a differentially private mean, for example, the validator requires the input data to have defined lower and upper bounds. An analyst can ensure that `lower/upper` are set with the `clamp` component, which clamps data to a given range.  
#
# Let's say that we have access to the PUMS codebook, and thus know some basic information about the possible values for the variables in the data. This is a convenient way to have reasonable baselines for properties like `lower/upper`.
#
# Another common property is `n`, an estimate of the sample size of the data in question. In general, this could be based on true knowledge of the data, an educated guess, or we could produce it via a differentially private process. We know, by construction of the data set, that this is a 1,000 person sample.
#
# Yet another property is `nullity`, whether or not the validator can guarantee that results are not null. We will see what it looks like for both the `lower/upper` and `nullity` properties to change within an analysis. 
#
# Let's start with `lower/upper`.

# %%
with sn.Analysis() as analysis:
    # load data
    data = sn.Dataset(path = data_path, column_names = var_names)
    
    # establish data 
    age_dt = sn.to_float(data['age'])

    # clamp data to set lower/upper properties
    clamped_age_dt = sn.clamp(age_dt, lower = 0., upper = 100.)

    # expand lower/upper by a factor of 2
    clamped_age_dt_2 = sn.multiply(clamped_age_dt, 2.)


analysis.release()
print('original properties:\n{0}\n\n'.format(age_dt.properties))
print('properties after clamping:\n{0}\n\n'.format(clamped_age_dt.properties))
print('properties after multiplication:\n{0}\n\n'.format(clamped_age_dt_2.properties))

# %% [markdown]
# We can now move onto `nullity`.

# %%
with sn.Analysis() as analysis:
    # load data
    data = sn.Dataset(path = data_path, column_names = var_names)
    
    # establish data 
    age_dt = sn.to_float(data['age'])

    # ensure data are non-null 
    non_null_age_dt = sn.impute(age_dt, distribution = 'Uniform', lower = 0., upper = 100.)

    # create null data
    potentially_null_age_dt = sn.multiply(non_null_age_dt, float('nan'))

analysis.release()
print('original properties:\n{0}\n\n'.format(age_dt.properties))
print('properties after imputation:\n{0}\n\n'.format(non_null_age_dt.properties))
print('properties after multiplication by nan:\n{0}\n\n'.format(potentially_null_age_dt.properties))


# %% [markdown]
# Note that the `nullity` property disappears after imputation (`nullity` disappearing is equivalent to `nullity: false`) and reappears after multiplication by `nan`. 

# %% [markdown]
# ### Analysis
#
# Now we can proceed to performing a basic analysis. Let's start by considering a differentially private mean of `age`. We will start with a few failed attempts in order to build an intuition for the requisite steps.

# %%
# set sample size
n = 1_000

# set ranges/feasible values
age_range = (0., 100.)
sex_vals = [0, 1]
educ_vals = [i for i in range(1, 17)]
race_vals = [i for i in range(1, 7)]
income_range = (0., 500_000.)
married_vals = [0, 1]

# %%
# attempt 1 - fails because of nullity
with sn.Analysis() as analysis:
    # load data
    data = sn.Dataset(path = data_path, column_names = var_names)
    
    ''' get mean age '''
    # establish data 
    age_dt = sn.to_float(data['age'])
    
    # calculate differentially private mean of age
    age_mean = sn.dp_mean(data = age_dt, privacy_usage={'epsilon': .65})

analysis.release()

# %% [markdown]
# Notice that `dp_mean` requires the data to have the property `nullity = False`.
# We can get around this by using `impute`. We will impute from a `Gaussian(mean = 45, sd = 10)` distribution, truncated such that no values fall outside of our age range we already established.

# %%
# attempt 2 - fails because of undefined lower/upper
with sn.Analysis() as analysis:
    # load data
    data = sn.Dataset(path = data_path, column_names = var_names)
    
    ''' get mean age '''
    # establish data 
    age_dt = sn.to_float(data['age'])
    
    # impute missing values
    age_dt = sn.impute(data = age_dt, distribution = 'Gaussian',
                                      lower = age_range[0], upper = age_range[1],
                                      shift = 45., scale = 10.)
    
    # calculate differentially private mean of age
    age_mean = sn.dp_mean(data = age_dt, privacy_usage={'epsilon': .65})
     
analysis.release()

# %% [markdown]
# Now we see that `dp_mean` needs to know the `lower` value (in fact, it also needs to know `upper`). We provide that with `clamp`. We paramaterize `clamp` with the lower and upper values of age we established at the beginning.

# %%
# attempt 3 - fails because of undefined n
with sn.Analysis() as analysis:
    # load data
    data = sn.Dataset(path = data_path, column_names = var_names)
    
    ''' get mean age '''
    # establish data 
    age_dt = sn.to_float(data['age'])
    
    # clamp data to range and impute missing values
    age_dt = sn.clamp(data = age_dt, lower = age_range[0], upper = age_range[1])
    age_dt = sn.impute(data = age_dt, distribution = 'Gaussian',
                                      lower = age_range[0], upper = age_range[1],
                                      shift = 45., scale = 10.)
    
    # calculate differentially private mean of age
    age_mean = sn.dp_mean(data = age_dt, privacy_usage={'epsilon': .65})

    
analysis.release()

# %% [markdown]
# SmartNoise requires `n` to be specified before a mean release can be considered valid.
# We know the true `n` in this case, but this will not always be true. We call `resize` to ensure that the data are consistent with the `n` we provide.

# %%
# attempt 4 - succeeds!
with sn.Analysis() as analysis:
    # load data
    data = sn.Dataset(path = data_path, column_names = var_names)
    
    ''' get mean age '''
    # establish data 
    age_dt = sn.to_float(data['age'])
    
    # clamp data to range and impute missing values
    age_dt = sn.clamp(data = age_dt, lower = age_range[0], upper = age_range[1])
    age_dt = sn.impute(data = age_dt, distribution = 'Gaussian',
                                      lower = age_range[0], upper = age_range[1],
                                      shift = 45., scale = 10.)
    
    # ensure data are consistent with proposed n
    age_dt = sn.resize(data = age_dt, number_rows = n, distribution = 'Gaussian',
                       lower = age_range[0], upper = age_range[1],
                       shift = 45., scale = 10.)
    
    # calculate differentially private mean of age
    age_mean = sn.dp_mean(data = age_dt, privacy_usage={'epsilon': .65})
        
    ''' get variance of age '''
    # calculate differentially private variance of age
    age_var = sn.dp_variance(data = age_dt, privacy_usage={'epsilon': .35})
    
analysis.release()

# print differentially private estimates of mean and variance of age
print(age_mean.value)
print(age_var.value)

# %% [markdown]
# Notice that we asked for an extra `dp_variance` at the end without having to use `clamp`, `impute`, or `resize`. Because these functions are updating the properties of `age_dt` as they are called, `dp_variance` has everything it needs from `age_dt` when we call it.
#
# Now that we have a sense for building up a statistic step-by-step, we can run through a much quicker version. We simply provide `data_lower, data_upper, data_rows` and the `clamp, impute, resize` steps are all performed implicitly. You'll notice that we don't even provide a `distribution` argument, even though it is needed for `impute`. For some arguments, we have (what we believe to be) reasonable defaults that are used if not provided explicitly.  For numerics, the default is to use a uniform distribution between the clamping min and max, but other distributions can be specified.

# %%
with sn.Analysis() as analysis:
    # load data
    data = sn.Dataset(path = data_path, column_names = var_names)
    
    # cast to float
    age_dt = sn.to_float(data['age'])
    
    # get mean of age
    age_mean = sn.dp_mean(data = age_dt,
                          privacy_usage = {'epsilon': .65},
                          data_lower = 0.,
                          data_upper = 100.,
                          data_rows = 1000
                         )
    # get variance of age
    age_var = sn.dp_variance(data = age_dt,
                             privacy_usage = {'epsilon': .35},
                             data_lower = 0.,
                             data_upper = 100.,
                             data_rows = 1000
                            )
analysis.release()

print("DP mean of age: {0}".format(age_mean.value))
print("DP variance of age: {0}".format(age_var.value))
print("Privacy usage: {0}".format(analysis.privacy_usage))

# %% [markdown]
# We see that the two DP releases within our analysis compose in a simple way, the individual epsilons we set add together for a total privacy usage of 1.   

# %% [markdown]
# One thing we have glossed over up until this point is the distinction between setting up and executing an analysis plan. An analysis plan is specified within the encapsulation of `sn.Analysis()` but until `analysis.release()` is run, the plan will not be validated and the data will not be touched.

# %%
with sn.Analysis() as analysis:
    # load data
    data = sn.Dataset(path = data_path, column_names = var_names)

    # get mean of age
    age_mean = sn.dp_mean(data = sn.to_float(data['age']),
                          privacy_usage = {'epsilon': .65},
                          data_lower = 0.,
                          data_upper = 100.,
                          data_rows = 1000
                         )
print("Pre-Release\n")
print("DP mean of age: {0}".format(age_mean.value))
print("Privacy usage: {0}\n\n".format(analysis.privacy_usage))

analysis.release()

print("Post-Release\n")
print("DP mean of age: {0}".format(age_mean.value))
print("Privacy usage: {0}\n\n".format(analysis.privacy_usage))


# %% [markdown]
# As a result, a user will not know whether or not the validator will allow a proposed analysis until running `analysis.release()`.

# %%
''' incomplete analysis plan, but no release => no failure '''
with sn.Analysis() as analysis:
    # load data
    data = sn.Dataset(path = data_path, column_names = var_names)

    age_mean_fail = sn.dp_mean(data = sn.to_float(data['age']),
                          privacy_usage = {'epsilon': .65})

# %%
''' fails upon release '''
analysis.release()
