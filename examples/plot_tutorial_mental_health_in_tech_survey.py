"""
===========================
OpenDP Tutorial Mental Health In Tech Survey
===========================
"""

# -*- coding: utf-8 -*-
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
# ## Privacy-Preserving Statistical Release
# ### Analysis of OSMI data 
# #### Mental Health in Tech Survey

# %% [markdown]
# The purpose of this demo is to showcase the utility of the OpenDP SmartNoise library. The notebook will focus on statistical releases in the *trusted curator* setting.
#
# Throughout his notebook, we make statistical queries to data with and without privacy-preserving mechanisms. As we compare query results side-by-side, we show that conclusions about the data are similar in both settings: without privacy-preserving mechanism, and with differential privacy mechanism. More precisely, the goals of this tutorial are:
#
# - serve an audience with basic understanding of differential privacy;
# - focus on reproducibility by using open data;
# - showcase the easy usability of the SmartNoise package;
# - focus on utility.
#
#
# **Disclaimer:** The present notebook is not intended to serve as a study of mental illness in the tech industry, or make any conclusions about the scenario of mental illness in tech. We use the data set as a illustrative example of the SmartNoise tool in survey and human subject studies.

# %% [markdown]
# ## 1 Data set
#
# Mental health in tech survey data set is an open data set licensed under CC BY-SA 4.0. 
#
# The data consists in 27 questions, aswered by 1,259 volunteers. For details on the data set, we refer the reader to the [Mental Health in Tech Survey](https://www.kaggle.com/osmi/mental-health-in-tech-survey)

# %% [markdown]
# The data used in the analysis was preprocessed. The original age, gender and country variables were mapped into categories for our analysis. We refer the reader to the [preprocessing notebook](data/Data%20set%20processing%20-%20Mental%20Health%20in%20Tech%20Survey.ipynb) for details on variable mappings.

# %% [markdown]
# The analysis will be focused on the following variables:
# - **age**: age of the participant. Categorical variable with 5 categories: 21-30yo (0), 31-40yo (1), 41-50yo (2), 51-60yo (3), 60yo+ (4) 
# - **gender**: gender declared by the participant, and in the survey the participants could input any string. We categorized answers as follows: Male/Man (1), Female/Woman(2), all other inputs (0).
# - **country**: participant's country of residence. We categorized answers as follows: United States (1), United Kingdom (2), Canada (3), other countries (0).
# - **remote_work**: binary value that indicates 
#     if participant work remotely more than 50% of the time
# - **family_history**: binary value that indicates if the 
#     participant has a family history of mental illness 
# - **treatment**: Binary value that 
#     indicates if the participants has seeked 
#     treatment for mental illness

# %% [markdown]
# ## 2 Data Analysis and Exploration

# %%
##Import packages
import os

import opendp.smartnoise.core as sn

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np
import pandas as pd
import scipy.stats as ss

import seaborn as sns
sns.set(style = "whitegrid")

# %matplotlib inline

# %%
path = './data/survey_V2.csv'

# %%
survey = pd.read_csv(path)
print('Data length: '+ str(len(survey)))
survey.head()

# %% [markdown]
# ### 2.1 Characteristics of survey participants

# %% [markdown]
# In the following analysis we explore the distribution of participants according to the following variables:
# - age;
# - gender;
# - country;
# - remote_work;
# - family_history; 
# - treatment.
#
# We make simple histogram queries for each of the variables.

# %%
# For multicategorical variables, we define the categories as lists
age_cat =['0','1','2','3','4']

country_cat = ['0','1','2','3']

gender_cat = ['0', '1', '2']

var_names =list(survey.columns)
with sn.Analysis() as analysis:
    data = sn.Dataset(path = path, column_names = var_names)
#Releases with Geometric Mechanism        
    age_histogram = sn.dp_histogram(
            data['age'],
            categories = age_cat,
            null_value = "4",
            privacy_usage = {'epsilon': 0.1}
        )
    gender_histogram = sn.dp_histogram(
            data['gender'],
            categories = gender_cat,
            null_value = '0',
            privacy_usage = {'epsilon': 0.1}
        )
    country_histogram = sn.dp_histogram(
            data['country'],
            categories = country_cat,
            null_value = '0',
            privacy_usage = {'epsilon': 0.1}
        )
    remotework_histogram = sn.dp_histogram(
            sn.cast(data['remote_work'], 'bool', true_label="1"),
            upper = 1213,
            privacy_usage = {'epsilon': 0.1}
        )
    
    family_histogram = sn.dp_histogram(
            sn.cast(data['family_history'], 'bool', true_label="1"),
            upper = 1213,
            privacy_usage = {'epsilon': 0.1}
        )
    
    treatment_histogram = sn.dp_histogram(
            sn.cast(data['treatment'], 'bool', true_label="1"),
            upper = 1213,
            privacy_usage = {'epsilon': 0.1}
        )
    
    

analysis.release()

print("Age histogram Geometric DP release:" + str(np.absolute(age_histogram.value)))
print("Country histogram Geometric DP release:" + str(np.absolute(country_histogram.value)))
print("Gender histogram Geometric DP release:" + str(np.absolute(gender_histogram.value)))
print("Remote Work histogram Geometric DP release:" + str(np.absolute(remotework_histogram.value)))
print("family history histogram Geometric DP release:" + str(np.absolute(family_histogram.value)))
print("treatment histogram Geometric DP release:" + str(np.absolute(treatment_histogram.value)))


# %% [markdown]
# Due to sequential composition, given $q_i$ queries, where each provide $\epsilon_i$-*differential privacy*, the sequence of queries provides $(\sum \epsilon_i)$-*differential privacy*. 

# %%
def pie_comparison(label, true_counts, dp_counts, title, subtitle1, subtitle2):
    # Make square figures and axes
    plt.figure(1, figsize=(20,25))
    the_grid = GridSpec(2, 2)
    plt.rcParams['text.color'] = '#000000'
    plt.rcParams['axes.labelcolor']= '#000000'
    plt.rcParams['xtick.color'] = '#000000'
    plt.rcParams['ytick.color'] = '#000000'
    plt.rcParams['font.size']=14

    cmap = plt.get_cmap('GnBu')
    colors = [cmap(i) for i in np.linspace(0, 1, 8)]


    plt.subplot(the_grid[0, 0], aspect=1, title= subtitle1)

    source_pie = plt.pie(true_counts, labels=label, autopct='%1.1f%%', shadow=True, colors=colors)


    plt.subplot(the_grid[0, 1], aspect=1, title= subtitle2)

    flavor_pie = plt.pie(dp_counts,labels=label, autopct='%.0f%%', shadow=True, colors=colors)

    plt.suptitle(title, fontsize=24)


    plt.show()

# %% [markdown]
# ### Age Distribution

# %%
age_true = [478, 554, 149, 26, 6]
age_geo = age_histogram.value
epsilon_ = 0.1
age_labels = ['21-30 years', '30-39 years', '40-49 years',
         '50-59 years', '60+ years']
title = 'Comparison of Results: Geometric Mechanism - epsilon '+str(epsilon_)
subtitle1 = 'True Counts: Age of Participants'
subtitle2 = 'Privacy-preserving: Age of Participants'
pie_comparison(age_labels, age_true, age_geo, title, subtitle1, subtitle2)

# %% [markdown]
# ## Country distribution 

# %%
country_true = [ 240, 732, 175, 66]
country_geo = country_histogram.value
epsilon_ = 0.1
country_labels = ['Other','US', 'UK','CA' ]
title = 'Comparison of Results: Geometric Mechanism - epsilon '+str(epsilon_)
subtitle1 = 'True Counts: Country of Participants'
subtitle2 = 'Privacy-preserving: Country of Participants'
pie_comparison(country_labels, country_true, country_geo, title, subtitle1, subtitle2)

# %% [markdown]
# ## Gender Distribution

# %%
gender_true = [16, 955, 242]
gender_geo = gender_histogram.value
epsilon_ = 0.1
gender_labels = ['Non-binary', 'Male', 'Female']
title = 'Comparison of Results: Geometric Mechanism - epsilon '+str(epsilon_)
subtitle1 = 'True Counts: Gender '
subtitle2 = 'Privacy-preserving: Gender'
pie_comparison(gender_labels, gender_true, gender_geo, title, subtitle1, subtitle2)

# %% [markdown]
# ### Remote work distribution

# %%
remote_true = [360, 853]
remote_geo = remotework_histogram.value
epsilon_ = 0.1
remote_labels = ['Does not work remotely', 'Works remotely' ]
title = 'Comparison of Results: Geometric Mechanism - epsilon '+str(epsilon_)
subtitle1 = 'True Counts: Remote work '
subtitle2 = 'Privacy-preserving: Remote work'
pie_comparison(remote_labels, remote_true, remote_geo, title, subtitle1, subtitle2)

# %% [markdown]
# ### Distribution of participants with family history of mental illness

# %%
family_true = [480, 733]
family_geo = family_histogram.value
epsilon_ = 0.1
family_labels = ['Family history of MI','No family history']

title = 'Comparison of Results: Geometric Mechanism - epsilon '+str(epsilon_)
subtitle1 = 'True Counts: Family history of MI'
subtitle2 = 'Privacy-preserving: Family history of MI'
pie_comparison(family_labels, family_true, family_geo, title, subtitle1, subtitle2)

# %% [markdown]
# ### Distribution of participants diagnosed with dental illness

# %%
treatment_true = [615, 598]
treatment_geo = treatment_histogram.value
epsilon_ = 0.1
treatment_labels = ['Not Diagnosed','Diagnosed']

title = 'Comparison of Results: Geometric Mechanism - epsilon '+str(epsilon_)
subtitle1 = 'True Counts: Participants diagnosed with MI'
subtitle2 = 'Privacy-preserving: Participants diagnosed with MI'
pie_comparison(treatment_labels, treatment_true, treatment_geo, title, subtitle1, subtitle2)

# %% [markdown]
# ### 2.2 Variable Interactions

# %% [markdown]
# For the queries in the following analysis, we will make histogram queries of disjoint subsets of the data.
#
# Due to parallel composition, when queries are applied to disjoint subsets of the data, the privacy guarantee depends only on the maximum $\epsilon_i$, not the sum.

# %%
epsilon_ = 0.4
with sn.Analysis() as analysis:
    data = sn.Dataset(path = path, column_names = var_names)
    
    ## percentage of patients 
    ##that received treatment in each Age class
    
    filter_age4 =  sn.filter(data['treatment'], mask = data['age'] == '4')
    age4_histogram = sn.dp_histogram(
            sn.cast(filter_age4, 'bool', true_label="0"),
            upper = 6,
            privacy_usage = {'epsilon': epsilon_}
        )
    filter_age3 =  sn.filter(data['treatment'], mask = data['age'] == '3')
    age3_histogram = sn.dp_histogram(
            sn.cast(filter_age3, 'bool', true_label="0"),
            upper = 26,
            privacy_usage = {'epsilon': epsilon_}
        )
    filter_age2 =  sn.filter(data['treatment'], mask = data['age'] == '2')
    age2_histogram = sn.dp_histogram(
            sn.cast(filter_age2, 'bool', true_label="0"),
            upper = 149,
            privacy_usage = {'epsilon': epsilon_}
        )
    filter_age1 =  sn.filter(data['treatment'], mask = data['age'] == '1')
    age1_histogram = sn.dp_histogram(
            sn.cast(filter_age1, 'bool', true_label="0"),
            upper = 554,
            privacy_usage = {'epsilon': epsilon_}
        )
    filter_age0 =  sn.filter(data['treatment'], mask = data['age'] == '0')
    age0_histogram = sn.dp_histogram(
            sn.cast(filter_age0, 'bool', true_label="0"),
            upper = 478,
            privacy_usage = {'epsilon': epsilon_}
        )
    
    
    ## percentage of patients 
    ##that received treatment in each Gender class 
    filter_gender2 =  sn.filter(data['treatment'], mask = data['gender'] == '2')
    gender2_histogram = sn.dp_histogram(
            sn.cast(filter_gender2, 'bool', true_label="0"),
            upper = 242,
            privacy_usage = {'epsilon': epsilon_}
        )
    filter_gender1 =  sn.filter(data['treatment'], mask = data['gender'] == '1')
    gender1_histogram = sn.dp_histogram(
            sn.cast(filter_gender1, 'bool', true_label="0"),
            upper = 955,
            privacy_usage = {'epsilon': epsilon_}
        )
    filter_gender0 =  sn.filter(data['treatment'], mask = data['gender'] == '0')
    gender0_histogram = sn.dp_histogram(
            sn.cast(filter_gender0, 'bool', true_label="0"),
            upper = 16,
            privacy_usage = {'epsilon': epsilon_}
        )
    
    
    
    ## percentage of patients 
    ##that received treatment in each country class 
    filter_country3 =  sn.filter(data['treatment'], mask = data['country'] == '3')
    country3_histogram = sn.dp_histogram(
            sn.cast(filter_country3, 'bool', true_label="0"),
            upper = 66,
            privacy_usage = {'epsilon': epsilon_}
        )
    filter_country2 =  sn.filter(data['treatment'], mask = data['country'] == '2')
    country2_histogram = sn.dp_histogram(
            sn.cast(filter_country2, 'bool', true_label="0"),
            upper = 175,
            privacy_usage = {'epsilon': epsilon_}
        )
    filter_country1 =  sn.filter(data['treatment'], mask = data['country'] == '1')
    country1_histogram = sn.dp_histogram(
            sn.cast(filter_country1, 'bool', true_label="0"),
            upper = 732,
            privacy_usage = {'epsilon': epsilon_}
        )
    filter_country0 =  sn.filter(data['treatment'], mask = data['country'] == '0')
    country0_histogram = sn.dp_histogram(
            sn.cast(filter_country0, 'bool', true_label="0"),
            upper = 240,
            privacy_usage = {'epsilon': epsilon_}
        )
    
    
    ## percentage of patients 
    ##that received treatment in each class of remote work
    filter_remote =  sn.filter(data['treatment'], mask = data['remote_work'] == '0')
    remote0_histogram = sn.dp_histogram(
            sn.cast(filter_remote, 'bool', true_label="0"),
            upper = 853,
            privacy_usage = {'epsilon': epsilon_}
        )
    
    filter_remote2 =  sn.filter(data['treatment'], mask = data['remote_work'] == '1')
    remote1_histogram = sn.dp_histogram(
            sn.cast(filter_remote2, 'bool', true_label="0"),
            upper = 360,
            privacy_usage = {'epsilon': epsilon_}
        )
    

    ## percentage of patients 
    ##that received treatment in each class of family history
    filtered =  sn.filter(data['treatment'], mask = data['family_history'] == '0')
    family0_histogram = sn.dp_histogram(
            sn.cast(filtered, 'bool', true_label="0"),
            upper = 733,
            privacy_usage = {'epsilon': epsilon_}
        )
    
    filtered2 =  sn.filter(data['treatment'], mask = data['family_history'] == '1')
    family1_histogram = sn.dp_histogram(
            sn.cast(filtered2, 'bool', true_label="0"),
            upper = 480,
            privacy_usage = {'epsilon': epsilon_}
        )
    
  

analysis.release()


# %% [markdown]
# Disposing all query results as a dataframe for our analysis. This dataframe will represent a contingency table that will display the frequency distribution of the variables.
#
# Contigency tables are heavily used in survey statistics, business intelligence, engineering and scientific research.
#
# The contingency table will allow us to explore the interrelations between two variables and also compute the Cramer's V coefficient, which is a statistic used to measure the strengh of association between two vaiables.

# %%
dpage0 = np.absolute(age0_histogram.value)
dpage1 = np.absolute(age1_histogram.value)
dpage2 = np.absolute(age2_histogram.value)
dpage3 = np.absolute(age3_histogram.value)
dpage4 = np.absolute(age4_histogram.value)

age0 = survey[survey.age == 0].treatment.value_counts(sort=False)
age1 = survey[survey.age == 1].treatment.value_counts(sort=False)
age2 = survey[survey.age == 2].treatment.value_counts(sort=False)
age3 = survey[survey.age == 3].treatment.value_counts(sort=False)
age4 = survey[survey.age == 4].treatment.value_counts(sort=False)

d1 = {'Group': ['21-30', '31-40', '41-50', '50-60', '60+','21-30', '31-40', '41-50', '50-60', '60+'],
     'Process':['true value', 'true value','true value','true value','true value','privacy-preserving','privacy-preserving','privacy-preserving','privacy-preserving','privacy-preserving'],
     'Participants on treatment MI':[ age0[1], age1[1], age2[1], age3[1], age4[1],dpage0[1], dpage1[1], dpage2[1], dpage3[1], dpage4[1]],
     'Participants not on treatment MI':[ age0[0], age1[0], age2[0], age3[0], age4[0],dpage0[0], dpage1[0], dpage2[0], dpage3[0], dpage4[0]],
     'Variable': 'Age'} 

age_df = pd.DataFrame(data=d1)

gender0 = survey[survey.gender == 0].treatment.value_counts(sort=False)
gender1 = survey[survey.gender == 1].treatment.value_counts(sort=False)
gender2 = survey[survey.gender == 2].treatment.value_counts(sort=False)

dpgender0 = np.absolute(gender0_histogram.value)
dpgender1 = np.absolute(gender1_histogram.value)
dpgender2 = np.absolute(gender2_histogram.value)

d5 = {'Group': ['Non-binary','Male', 'Female', 'Non-binary','Male', 'Female'],
     'Process':['true value', 'true value','true value','privacy-preserving','privacy-preserving','privacy-preserving'],
     'Participants on treatment MI':[ gender0[1], gender1[1], gender2[1],dpgender0[1], dpgender1[1], dpgender2[1]],
     'Participants not on treatment MI':[ gender0[0], gender1[0], gender2[0],dpgender0[0], dpgender1[0], dpgender2[0]],
     'Variable':'Gender'} 

gender_df = pd.DataFrame(data=d5)


country0 = survey[survey.country == 0].treatment.value_counts(sort=False)
country1 = survey[survey.country == 1].treatment.value_counts(sort=False)
country2 = survey[survey.country == 2].treatment.value_counts(sort=False)
country3 = survey[survey.country == 3].treatment.value_counts(sort=False)

dpcountry0 = np.absolute(country0_histogram.value)
dpcountry1 = np.absolute(country1_histogram.value)
dpcountry2 = np.absolute(country2_histogram.value)
dpcountry3 = np.absolute(country3_histogram.value)

d2 = {'Group': ['Other','US', 'UK','CA' ,'Other','US', 'UK','CA' ],
     'Process':['true value', 'true value','true value','true value','privacy-preserving','privacy-preserving','privacy-preserving','privacy-preserving'],
     'Participants on treatment MI':[ country0[1], country1[1], country2[1], country3[1],dpcountry0[1], dpcountry1[1], dpcountry2[1], dpcountry3[1]],
     'Participants not on treatment MI':[ country0[0], country1[0], country2[0], country3[0],dpcountry0[0], dpcountry1[0], dpcountry2[0], dpcountry3[0]],
     'Variable':'Country'} 

country_df = pd.DataFrame(data=d2)

# %%
dpremote0 = np.absolute(remote0_histogram.value)
dpremote1 = np.absolute(remote1_histogram.value)
dpfamily0 = np.absolute(family0_histogram.value)
dpfamily1 = np.absolute(family1_histogram.value)

remote0 = survey[survey.remote_work == 0].treatment.value_counts(sort=False)
remote1 = survey[survey.remote_work == 1].treatment.value_counts(sort=False)
family0 = survey[survey.family_history == 0].treatment.value_counts(sort=False)
family1 = survey[survey.family_history == 1].treatment.value_counts(sort=False)

d3 = {'Group': ['Does not work remotely', 'Works remotely', 'Does not work remotely', 'Works remotely'],
     'Process':['true value', 'true value','privacy-preserving','privacy-preserving'],
     'Participants on treatment MI':[ remote0[1], remote1[1],dpremote0[1], dpremote1[1]],
     'Participants not on treatment MI':[ remote0[0], remote1[0],dpremote0[0], dpremote1[0]],
     'Variable': 'Remote work'} 

remote_df = pd.DataFrame(data=d3)

d4 = {'Group': ['No family history','Family history of MI','No family history','Family history of MI'],
     'Process':['true value', 'true value','privacy-preserving','privacy-preserving'],
     'Participants on treatment MI':[ family0[1], family1[1], dpfamily0[1], dpfamily1[1]],
     'Participants not on treatment MI':[ family0[0], family1[0], dpfamily0[0], dpfamily1[0]],
     'Variable': 'Family history'} 

family_df = pd.DataFrame(data=d4)

# %%
dfs = [age_df, gender_df, country_df, remote_df, family_df]
DF = pd.concat(dfs)
DF['percentage'] = DF['Participants on treatment MI']*100/(DF['Participants not on treatment MI']+DF['Participants on treatment MI'])

# %% [markdown]
# ## Observation 1
#
# Participants with family history of mental illness are twice as likely to be diagnosed with mental illness, when compared with participants with no family history.

# %%
DF[DF.Variable == 'Family history'].sort_values(['Group'])

# %%
sns.catplot(x="Group", y="percentage", col="Process", data=DF[DF.Variable == 'Family history'], kind="bar")

# %% [markdown]
# ### Observation 2
#
# participants working remotetly have the same probably of being diagnosed with mental illness as participants that do not work remotely.

# %%
DF[DF.Variable == 'Remote work'].sort_values(['Group'])

# %%
sns.catplot(x="Group", y="percentage", col="Process", data=DF[DF.Variable == 'Remote work'], kind="bar")

# %% [markdown]
# # Privacy by design

# %% [markdown]
# An adversarial researcher might try to drill down the data in order to get information about specific survey participants.
#
# Differential privacy mechanisms are designed to address such queries with suficient noise to mask the participation of any individual.
#
# In scenarios without a trusted curator, the SmartNoise library provides accuracy intervals to the researcher. An $\alpha$-level accuracy guarantee a promise that (with probability $1-\alpha$)
#
# $$
# M(D) \in [M_{DP}(D)-a, M_{DP}(D)+a]
# $$
#
# where $M(D)$ is the query response of function $M$ on database $D$ without differential privacy, and $M_{DP}(D)$ is the response with differential privacy.
#
# Accuracy pitfalls and edge cases are discussed in [this notebook](https://github.com/opendifferentialprivacy/smartnoise-samples/tree/master/analysis/accuracy_pitfalls.ipynb)

# %% [markdown]
# ## Observation 3
#
# The US has the biggest percent of professionals in the tech industry diagnosed with mental illness (around 54%). In other countries having the lowest percentages, fewer than 40% of tech workers have been treated for mental illness.

# %%
DF[DF.Variable == 'Country'].sort_values(['Group'])

# %%
sns.catplot(x="Group", y="percentage", col="Process", data=DF[DF.Variable == 'Country'], kind="bar")

# %% [markdown]
# ## Observation 4
#
# As we explore the percentage of participants in each age group that sought MI treatment, we observe that very similar conclusions can be drawn from true values and from privacy preserving values. The exception in for age group of participants 60+ years old. 
#
# As expected, in very small data partitions the distortions are greater.
#

# %%
DF[DF.Variable == 'Age'].sort_values(['Group'])

# %%
sns.catplot(x="Group", y="percentage", col="Process", data=DF[DF.Variable == 'Age'], kind="bar")


# %% [markdown]
# # Comparing associations between variables
# ### Different variables and variable 'treatment for MI'

# %% [markdown]
# ## Intercorrelation of two discrete variables
#
# Cramér's V, sometimes referred to as Cramér's phi (denoted as φc), is a measure of association between two nominal variables, giving a value between 0 and +1 (inclusive). It is based on Pearson's chi-squared statistic.
#
# φc is the intercorrelation of two discrete variables. and may be used with variables having two or more levels. φc is a symmetrical measure, it does not matter which variable we place in the columns and which in the rows. Also, the order of rows/columns doesn't matter, so φc may be used with nominal data types or higher (notably ordered or numerical).
#
# Source: Wikipedia

# %%
def cramers_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = sum(confusion_matrix.sum())
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


# %%
Coefs_var = ['Family history', 'Gender', 'Remote work', 'Age', 'Country'] 

# %%
coefs = []
dpcoefs = []
for variable in Coefs_var:
    coefs.append(cramers_stat(DF[(DF.Variable == variable)&(DF.Process == 'true value')][['Participants on treatment MI', 'Participants not on treatment MI']]))
    dpcoefs.append(cramers_stat(DF[(DF.Variable == variable)&(DF.Process == 'privacy-preserving')][['Participants on treatment MI', 'Participants not on treatment MI']]))


# %%
c = {'True Coef': coefs,
        'Privacy Coef':dpcoefs}
cramer_coef = pd.DataFrame(data=c, index = Coefs_var)

# %%
cramer_coef
