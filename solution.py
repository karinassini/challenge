# Databricks notebook source
# MAGIC %md
# MAGIC # Import libs

# COMMAND ----------

!pip install --upgrade pip && pip install --no-cache-dir -r requirements/requirements.txt

# COMMAND ----------

import sys
import warnings
import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
%matplotlib inline

from scipy import stats
import association_metrics as am
import pingouin as pg

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
import lightgbm as lgbm
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce

from core.functions import continuous_to_binary, lgbm_class_hyperparameter_tuning_pipeline

import optuna
import shap

warnings.filterwarnings("ignore")

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

path = "data/dataset_SCL.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC # Challenges
# MAGIC 
# MAGIC - Below are the analyzes and answers to questions one through five.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 and 2

# COMMAND ----------

df = pd.read_csv(path)

# COMMAND ----------

dict_columns = {
    "Fecha-I": "scheduled_date_and_time",
    "Vlo-I": "scheduled_flight_number",
    "Ori-I": "scheduled_origin_city_code",
    "Des-I": "scheduled_destination_city_code",
    "Emp-I": "scheduled_airline_code",
    "Fecha-O": "operation_date_and_time",
    "Vlo-O": "operation_number",
    "Ori-O": "operation_origin_city_code",
    "Des-O": "operation_destination_city_code",
    "Emp-O": "operated_airline_code",
    "DIA": "day_operation",
    "MES": "month_operation",
    "AÑO": "year_operation",
    "DIANOM": "day_of_the_week_operation",
    "TIPOVUELO": "type_of_flight",
    "OPERA": "airline_that_operates",
    "SIGLAORI": "origin_city_name",
    "SIGLADES": "destination_city_name",
}

# COMMAND ----------

df = df.rename(columns=dict_columns).reset_index(drop=True)

# COMMAND ----------

df.head()

# COMMAND ----------

for column in df.columns:
    print(f"Column {column} has {df[column].nunique()} unique values.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Null data

# COMMAND ----------

# Just one value missing
df.info()

# COMMAND ----------

df.query("operation_number.isna()", engine="python")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Understanding correlation between flight operation data for filling null data in operation_number
# MAGIC 
# MAGIC -Note: It is important to review and validate these methods for filling in the missing values with experts from the business community.

# COMMAND ----------

operational_columns = [
    "operation_number",
    "operation_destination_city_code",
    "operated_airline_code",
    "month_operation",
    "day_operation",
    "day_of_the_week_operation",
    "destination_city_name",
    "airline_that_operates",
]

# COMMAND ----------

df_aux = df[operational_columns].dropna()

# COMMAND ----------

# Initialize a CramersV object: understanding categorical x categorial correlation 
df_aux = df_aux.apply(lambda x: x.astype("category") if x.dtype == "O" else x)

cramers_v = am.CramersV(df_aux)
cfit = cramers_v.fit().round(2)

# COMMAND ----------

# Plotting heatmap results
fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.imshow(cfit.values, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(ticks=range(len(cfit.columns)))
ax.set_xticklabels(labels=cfit.columns)
ax.set_label(cfit.columns)
ax.set_yticks(ticks=range(len(cfit.columns)))
ax.set_yticklabels(cfit.columns)
ax.tick_params(axis="x", labelsize=10, labelrotation=90)
ax.tick_params(axis="y", labelsize=12, labelrotation=0)
fig.colorbar(cax).ax.tick_params(labelsize=12)
for (x, y), t in np.ndenumerate(cfit):
    ax.annotate("{:.2f}".format(t), xy=(x, y), va="center", ha="center").set(
        color="black", size=12
    )

# COMMAND ----------

# Destination_city_name represents the same information as operation_destination_city_code
compare_values = np.all(
    pd.factorize(df["destination_city_name"])[0]
    == pd.factorize(df["operation_destination_city_code"])[0]
)
compare_values

# COMMAND ----------

df = df.drop("destination_city_name", axis=1)

# COMMAND ----------

f = lambda x: x.median() if np.issubdtype(x.dtype, np.number) else x.mode().iloc[0]

# Filling in only the nulls from the above data by grouping the columns correlated with operation_number.
df = df.fillna(
    df.groupby(
        [
            "operation_destination_city_code",
            "airline_that_operates",
            "operated_airline_code",
        ]
    ).transform(f)
)

# COMMAND ----------

df.iloc[6068]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Visualization and Outliers Understanding

# COMMAND ----------

# MAGIC %md
# MAGIC #### Understanding the months and days of the week when the scheduled target city was reached

# COMMAND ----------

df.query("scheduled_destination_city_code == operation_destination_city_code").shape[
    0
] / df.shape[0]

# COMMAND ----------

# Closes to uniform data
sns.catplot(
    y="month_operation",
    x="scheduled_destination_city_code",
    data=df.query("scheduled_destination_city_code == operation_destination_city_code"),
    order=df.query(
        "scheduled_destination_city_code == operation_destination_city_code"
    )["scheduled_destination_city_code"]
    .value_counts()
    .iloc[:5]
    .index,
    height=5,
    aspect=3,
    legend=True,
    jitter="0.25",
    hue="type_of_flight",
    kind="violin"
)

# COMMAND ----------

# Closes to uniform data
sns.catplot(
    y="day_operation",
    x="scheduled_destination_city_code",
    data=df.query("scheduled_destination_city_code == operation_destination_city_code"),
    order=df.query("scheduled_destination_city_code == operation_destination_city_code")["scheduled_destination_city_code"].value_counts().iloc[:5].index,
    height=5,
    aspect=3,
    legend=True,
    jitter = '0.25',
    hue="type_of_flight",
    kind="violin"
)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Distribution: most frequent destinations
# MAGIC 
# MAGIC - It seems that there is a bimodal distribution (for the most operated city destinations ) and the actual achieved destination operation to the programmed
# MAGIC 
# MAGIC - Peaks in January and December

# COMMAND ----------

x = df.query(
    "scheduled_destination_city_code == operation_destination_city_code and scheduled_destination_city_code=='SPJC'"
)["month_operation"]
y = df.query(
    "scheduled_destination_city_code == operation_destination_city_code and scheduled_destination_city_code=='SCFA'"
)["month_operation"]
z = df.query(
    "scheduled_destination_city_code == operation_destination_city_code and scheduled_destination_city_code=='SCTE'"
)["month_operation"]
w = df.query(
    "scheduled_destination_city_code == operation_destination_city_code and scheduled_destination_city_code=='SCIE'"
)["month_operation"]

# COMMAND ----------

plt.hist(x, alpha=0.5, label="SPJC")
plt.hist(y, alpha=0.5, label="SCFA")
plt.hist(z, alpha=0.5, label="SCTE")
plt.hist(w, alpha=0.5, label="SCIE")
plt.legend(loc="upper right")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pareto: most operated days of the week

# COMMAND ----------

pareto_df = pd.DataFrame(
    df["day_of_the_week_operation"].value_counts().sort_values(ascending=False)
)

pareto_df["cum_perc"] = (
    pareto_df["day_of_the_week_operation"].cumsum()
    / pareto_df["day_of_the_week_operation"].sum()
    * 100
)

# COMMAND ----------

fig, ax = plt.subplots()
ax.bar(pareto_df.index, pareto_df["day_of_the_week_operation"], color="pink")

ax2 = ax.twinx()
ax2.plot(pareto_df.index, pareto_df["cum_perc"], color="purple", marker="D", ms=4)
ax2.yaxis.set_major_formatter(PercentFormatter())

ax.tick_params(axis="y", colors="purple")
ax.tick_params(axis="x", colors="purple", rotation=45)
ax2.tick_params(axis="y", colors="purple")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Understanding the months of operation when the scheduled destination city was not reached
# MAGIC 
# MAGIC Operation different from scheduled city

# COMMAND ----------

df.query("scheduled_destination_city_code != operation_destination_city_code").shape[
    0
] / df.shape[0]

# COMMAND ----------

# SCIE close to normal
# More problems with international flights
sns.catplot(
    y="month_operation",
    x="scheduled_destination_city_code",
    data=df.query("scheduled_destination_city_code != operation_destination_city_code"),
    order=df.query(
        "scheduled_destination_city_code != operation_destination_city_code"
    )["scheduled_destination_city_code"]
    .value_counts()
    .iloc[:5]
    .index,
    height=5,
    aspect=3,
    legend=True,
    jitter="0.25",
    hue="type_of_flight",
    kind="violin",
).ax.set_ylim(0, 13)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pareto: days of the week that occur most change of destination

# COMMAND ----------

pareto_df = pd.DataFrame(
    df.query("scheduled_destination_city_code != operation_destination_city_code")[
        "day_of_the_week_operation"
    ]
    .value_counts()
    .sort_values(ascending=False)
)

pareto_df["cum_perc"] = (
    pareto_df["day_of_the_week_operation"].cumsum()
    / pareto_df["day_of_the_week_operation"].sum()
    * 100
)

# COMMAND ----------

# display Pareto chart
fig, ax = plt.subplots()
ax.bar(pareto_df.index, pareto_df["day_of_the_week_operation"], color="pink")

ax2 = ax.twinx()
ax2.plot(pareto_df.index, pareto_df["cum_perc"], color="purple", marker="D", ms=4)
ax2.yaxis.set_major_formatter(PercentFormatter())

ax.tick_params(axis="y", colors="pink")
ax2.tick_params(axis="y", colors="purple")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pareto: months that occur most change of destination
# MAGIC 
# MAGIC - Not all destination changes occur in months with more flights (see distribution: most frequent destinations). For example, there are few such events in January
# MAGIC - It seems that it is concentrated in the months of high season (Dec, Jul and March)

# COMMAND ----------

pareto_df = pd.DataFrame(
    df.query("scheduled_destination_city_code != operation_destination_city_code")[
        "month_operation"
    ]
    .map(
        {
            1: "Jan",
            2: "Fev",
            3: "Mar",
            4: "Abr",
            5: "Mai",
            6: "Jun",
            7: "Jul",
            8: "Ago",
            9: "Set",
            10: "Out",
            11: "Nov",
            12: "Dez",
        }
    )
    .value_counts()
    .sort_values(ascending=False)
)

pareto_df["cum_perc"] = (
    pareto_df["month_operation"].cumsum() / pareto_df["month_operation"].sum() * 100
)

# COMMAND ----------

# display Pareto chart
fig, ax = plt.subplots()
ax.bar(pareto_df.index, pareto_df["month_operation"], color="pink")

ax2 = ax.twinx()
ax2.plot(pareto_df.index, pareto_df["cum_perc"], color="purple", marker="D", ms=4)
ax2.yaxis.set_major_formatter(PercentFormatter())

ax.tick_params(axis="y", colors="pink")
ax2.tick_params(axis="y", colors="purple")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Understanding days of the week when the scheduled airline company was and was not reached

# COMMAND ----------

df.query("scheduled_airline_code != operated_airline_code").shape[
    0
] / df.shape[0]

# COMMAND ----------

sns.catplot(
    y="day_operation",
    x="scheduled_airline_code",
    data=df.query("scheduled_airline_code == operated_airline_code"),
    order=df.query("scheduled_airline_code == operated_airline_code")[
        "scheduled_airline_code"
    ]
    .value_counts()
    .iloc[:5]
    .index,
    height=5,
    aspect=3,
    legend=True,
    jitter="0.25",
    hue="type_of_flight",
  kind="violin"
)

# COMMAND ----------

# Companies that changed programming the most
sns.catplot(
    y="day_operation",
    x="scheduled_airline_code",
    data=df.query("scheduled_airline_code != operated_airline_code"),
    order=df.query("scheduled_airline_code != operated_airline_code")[
        "scheduled_airline_code"
    ]
    .value_counts()
    .iloc[:5]
    .index,
    height=5,
    aspect=3,
    legend=True,
    jitter="0.25",
    hue="type_of_flight",
  kind="violin"
)

# COMMAND ----------

sns.catplot(
    y="day_operation",
    x="operated_airline_code",
    data=df.query("scheduled_airline_code != operated_airline_code"),
    order=df.query("scheduled_airline_code != operated_airline_code")[
        "operated_airline_code"
    ]
    .value_counts()
    .iloc[:5]
    .index,
    height=5,
    aspect=3,
    legend=True,
    jitter="0.25",
    hue="type_of_flight",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Patterns understanding

# COMMAND ----------

df["operation_date_and_time"] = pd.to_datetime(df["operation_date_and_time"])
df["scheduled_date_and_time"] = pd.to_datetime(df["scheduled_date_and_time"])

# COMMAND ----------

# Everything seems right in these date columns
sum(df["operation_date_and_time"].dt.day.eq(df["day_operation"])), sum(
    df["operation_date_and_time"].dt.month.eq(df["month_operation"])
), sum(df["operation_date_and_time"].dt.year.eq(df["year_operation"]))

# COMMAND ----------

compare_values = np.all(
    pd.factorize(df["scheduled_airline_code"])[0]
    == pd.factorize(df["operated_airline_code"])[0]
)
compare_values

# COMMAND ----------

compare_values = np.all(
    pd.factorize(df["scheduled_destination_city_code"])[0]
    == pd.factorize(df["operation_destination_city_code"])[0]
)
compare_values

# COMMAND ----------

compare_values = np.all(
    pd.factorize(df["scheduled_flight_number"])[0]
    == pd.factorize(df["operation_number"])[0]
)
compare_values

# COMMAND ----------

df = df.drop(
    [
        "scheduled_origin_city_code",
        "operation_origin_city_code",
        "origin_city_name",
    ],
    axis=1,
)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Comments:
# MAGIC 
# MAGIC - Comparing scheduled x operated
# MAGIC 
# MAGIC - Not all flights planned for a certain city were destined for it
# MAGIC 
# MAGIC - scheduled_destination_city_code and operation_destination_city_code different can mean outliers or just routing change
# MAGIC 
# MAGIC - scheduled_origin_city_code, operation_origin_city_code, name_city_of_origin are the same, becauses consists in origin city name: SCL
# MAGIC 
# MAGIC - Need to deeply explore the data in future work

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate the following additional columns
# MAGIC 
# MAGIC - high_season : 1 if Date-I is between Dec-15 and Mar-3, or Jul-15 and Jul-31, or Sep-11 and Sep-30, 0 otherwise.
# MAGIC 
# MAGIC - min_diff : difference in minutes between Date-O and Date-I .
# MAGIC 
# MAGIC - delay_15 : 1 if min_diff > 15, 0 if not.
# MAGIC 
# MAGIC - period_day : morning (between 5:00 and 11:59), afternoon (between 12:00 and 18:59) and night (between 19:00 and 4:59), basedonDate-I .

# COMMAND ----------

# MAGIC %md
# MAGIC #### high_season

# COMMAND ----------

mask = (
    (df["scheduled_date_and_time"] >= "2017-12-15 00:00:00")
    | (df["scheduled_date_and_time"] < "2017-03-04 00:00:00")
    | (
        (df["scheduled_date_and_time"] >= "2017-07-15 00:00:00")
        & (df["scheduled_date_and_time"] < "2017-08-01 00:00:00")
    )
    | (
        (df["scheduled_date_and_time"] >= "2017-09-11 00:00:00")
        & (df["scheduled_date_and_time"] < "2017-10-01 00:00:00")
    )
)

# COMMAND ----------

df["high_season"] = np.where(mask, 1, 0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### min_diff

# COMMAND ----------

df["min_diff"] = (
    pd.to_datetime(df["operation_date_and_time"])
    - pd.to_datetime(df["scheduled_date_and_time"])
).dt.total_seconds() / 60

# COMMAND ----------

# MAGIC %md
# MAGIC #### delay_15

# COMMAND ----------

df["delay_15"] = np.where(df["min_diff"] > 15, 1, 0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### period_day

# COMMAND ----------

morning = (df["scheduled_date_and_time"].dt.time >= datetime.time(5, 0)) & (
    df["scheduled_date_and_time"].dt.time <= datetime.time(11, 59)
)

afternoon = (df["scheduled_date_and_time"].dt.time >= datetime.time(12, 0)) & (
    df["scheduled_date_and_time"].dt.time <= datetime.time(18, 59)
)

night = (df["scheduled_date_and_time"].dt.time >= datetime.time(19, 0)) | (
    df["scheduled_date_and_time"].dt.time >= datetime.time(0, 0)
) & (df["scheduled_date_and_time"].dt.time <= datetime.time(4, 59))

conditions = [morning, afternoon, night]

# COMMAND ----------

df["period_day"] = np.select(
    conditions, ["morning", "afternoon", "night"], default=None
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### count_flights_same_datetime
# MAGIC 
# MAGIC Counts number of flights scheduled for the same day and time by the same airline

# COMMAND ----------

aux = (
    df
    .groupby(["scheduled_date_and_time", "scheduled_airline_code"])
    .count()
    .reset_index()[
        ["scheduled_date_and_time", "scheduled_airline_code", "scheduled_flight_number"]
    ]
    .rename(columns={"scheduled_flight_number": "count_flights_same_datetime"})
)

# COMMAND ----------

df_merged = df.merge(
    aux,
    on=["scheduled_date_and_time", "scheduled_airline_code"],
    how="outer",
    validate="m:1",
)

# COMMAND ----------

assert df_merged.shape[0] == df.shape[0]

# COMMAND ----------

df["count_flights_same_datetime"] = (
    df_merged["count_flights_same_datetime"].map({1: 0, 2: 1, 3: 2, 4:3}).fillna(0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### scheduled_destination_city_airline

# COMMAND ----------

df["scheduled_destination_city_airline"] = (
    df["scheduled_destination_city_code"] + "_" + df["scheduled_airline_code"]
)

# COMMAND ----------

df["scheduled_destination_city_airline"] .value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. What is the behavior of the delay rate across destination, airline, month of the year, day of the week, season, type of flight? What variables would you expect to have the most influence in predicting delays?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Numerical features

# COMMAND ----------

# MAGIC %md
# MAGIC - Null Hypothesis (H0): True correlation is equal to zero.
# MAGIC 
# MAGIC - Alternative Hypothesis (H1): True correlation is not equal to zero.

# COMMAND ----------

# Perform a hypothesis testing: non-parametric method - Spearman rank-correlation.
subj = ['delay_15']
personality = list(set(df.columns) - set("delay_15"))
pg.pairwise_corr(df, columns=[subj, personality], method="spearman").round(3).sort_values(by="p-unc")

# COMMAND ----------

# MAGIC %md
# MAGIC - The results show that the probability **first 5** rows pair-wise correlation values are lower than the conventional 5% (P<0.05), **thus here the alternative hypothesis is true**.
# MAGIC - The features high_season, month_operation, count_flights_same_datetime are correlated with the target (delay_15). We can expect it to perform well in models.

# COMMAND ----------

fig, axs = plt.subplots(2, 2, figsize=(10,5))
sns.scatterplot(data=df, x="high_season", y="delay_15", ax=axs[0, 0], hue="type_of_flight")
axs[0, 0].set_title("high_season")
sns.scatterplot(data=df, x="month_operation", y="delay_15", ax=axs[0, 1],  hue="type_of_flight")
axs[0, 1].set_title("month_operation")
sns.scatterplot(data=df, x="count_flights_same_datetime", y="delay_15", ax=axs[1, 0],  hue="type_of_flight")
sns.scatterplot(data=df, x="day_operation", y="delay_15", ax=axs[1, 1],  hue="type_of_flight")

for ax in axs.flat:
    ax.set(ylabel="delay_15")

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

# COMMAND ----------

# 18% of flights with delays of more than 15 minutes were operated by another airline
df.query("scheduled_airline_code != operated_airline_code")["delay_15"].value_counts(normalize=True)

# COMMAND ----------

# 28% of flights with delays of more than 15 minutes were rescheduled to another city
df.query("scheduled_destination_city_code != operation_destination_city_code")["delay_15"].value_counts(normalize=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Destination
# MAGIC 
# MAGIC - Of the major destinations, international and national flights appear to have two modes. Planned destinations were maintained.

# COMMAND ----------

sns.catplot(
    x="scheduled_destination_city_code",
    y="delay_15",
    data=df.query("scheduled_destination_city_code == operation_destination_city_code"),
    order=df["scheduled_destination_city_code"].value_counts().iloc[:10].index,
    height=5,
    aspect=3,
    legend=True,
    hue="type_of_flight",
    kind="violin",
)

# COMMAND ----------

# More delays on international flights than domestic flights on some airlines, and vice versa
sns.catplot(
    x="scheduled_destination_city_code",
    y="delay_15",
    data=df.query("scheduled_destination_city_code != operation_destination_city_code"),
    order=df["scheduled_destination_city_code"].value_counts().iloc[:10].index,
    height=5,
    aspect=3,
    legend=True,
    hue="type_of_flight",
    kind="violin",
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Airline

# COMMAND ----------

sns.catplot(
    x="scheduled_airline_code",
    y="delay_15",
    data=df.query("scheduled_airline_code == operated_airline_code"),
    order=df["scheduled_airline_code"].value_counts().iloc[:10].index,
    height=5,
    aspect=3,
    legend=True,
    hue="type_of_flight",
    kind="violin",
)

# COMMAND ----------

sns.catplot(
    x="scheduled_airline_code",
    y="delay_15",
    data=df.query("scheduled_airline_code != operated_airline_code"),
    order=df["scheduled_airline_code"].value_counts().iloc[:10].index,
    height=5,
    aspect=3,
    legend=True,
    hue="type_of_flight",
    kind="violin",
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Day of the week

# COMMAND ----------

sns.catplot(
    x="day_of_the_week_operation",
    y="delay_15",
    data=df,
    height=5,
    aspect=3,
    legend=True,
    hue="type_of_flight",
    kind="violin",
)

# COMMAND ----------

sns.catplot(
    x="period_day",
    y="delay_15",
    data=df,
    height=5,
    aspect=3,
    legend=True,
    hue="type_of_flight",
    kind="violin",
)

# COMMAND ----------

sns.scatterplot(data=df.query("period_day=='morning' and delay_15==1"), x="period_day", y="delay_15",  hue="type_of_flight")

# COMMAND ----------

# MAGIC %md
# MAGIC #### All categorical features x delay_15
# MAGIC 
# MAGIC - ANOVA test: cannot be used because of your assumptions about the data
# MAGIC - I chose the Spearman rank correlation coefficient because it is a non-parametric measure of rank correlation

# COMMAND ----------

df.columns

# COMMAND ----------

features = [
    "airline_that_operates",
    "day_of_the_week_operation",
    "period_day",
    "scheduled_destination_city_airline"
]

# COMMAND ----------

# Pode ser que a feature combinada com outras traga bons resultados
for f in features:
    print(f" Feature: {f}, Result: {stats.spearmanr(df['delay_15'], df[f])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train one or several models (using the algorithm(s) of your choice) to estimate the likelihood of a flight delay. Feel free to generate additional variables and/or supplement with external variables:

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scenario 1
# MAGIC 
# MAGIC Operational data cannot be used for training because it can lead to leakege data, since we do not have this information at the time of prediction.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Treat categorical features
# MAGIC 
# MAGIC - One-hot encoding can be computationally expensive and can lead to a high-dimensional feature space, especially for variables with many categories. So, we'll do it only with the categories below.
# MAGIC 
# MAGIC - I used CatBoostEncounter for the other columns. I then fitted the encoder on the training data and transformed the training and test data using the trained encoder.

# COMMAND ----------

model_inputs = [
    
        "scheduled_destination_city_code",
        "scheduled_airline_code",
        "type_of_flight",
        "high_season",
        "delay_15",
        "period_day",
        "count_flights_same_datetime",
    
]

# COMMAND ----------

df = df.reset_index(drop=True)

# COMMAND ----------

encoded_df = pd.get_dummies(
    df[model_inputs],
    columns=["type_of_flight", "period_day"],
    prefix=["type_of_flight", "period_day"],
    drop_first=[True, False],
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train test split

# COMMAND ----------

X, y = train_test_split(
    encoded_df,
    stratify=encoded_df["delay_15"],
    test_size=0.20,
    random_state=42,
)

# COMMAND ----------

assert set(list(X.index)).intersection(set(y.index)) == set()

# COMMAND ----------

assert X["delay_15"].value_counts(normalize=True)[0] < 0.82

# COMMAND ----------

assert y["delay_15"].value_counts(normalize=True)[0] < 0.82

# COMMAND ----------

train_x = X.drop("delay_15", axis=1)
train_y = X["delay_15"]

test_x = y.drop("delay_15", axis=1)
test_y = y["delay_15"]


# COMMAND ----------

# MAGIC %md
# MAGIC #### Enconding with categorical encoder

# COMMAND ----------

# create a categorical encoder and fit it to the training data
encoder = ce.CatBoostEncoder(cols=['scheduled_destination_city_code', 'scheduled_airline_code'], random_state=42)
X_train_encoded = encoder.fit_transform(train_x, train_y)

# COMMAND ----------

base_parameters_lgbm_class = {
    "objective": "binary",
    "boosting_type": "dart",
    "metric": "binary_logloss",
    "device_type": "cpu",
    "num_threads": 4,
    "enable_bundle": True,
    "verbose": -1,
    "random_seed": 0,
    "is_unbalance": True,
    "extra_trees": True,
}

threshold = 0.5

# COMMAND ----------

# transform the testing data using the trained encoder
X_test_encoded = encoder.transform(test_x)

# COMMAND ----------

# MAGIC %md
# MAGIC #### First train

# COMMAND ----------

train_set = lgbm.Dataset(
    X_train_encoded,
    label=train_y
)

# COMMAND ----------

model = lgbm.train(base_parameters_lgbm_class, train_set)

# COMMAND ----------

y_pred_train = model.predict(
    X_train_encoded,
    num_iteration=model.best_iteration,
)

y_pred_test = model.predict(
    X_test_encoded,
    num_iteration=model.best_iteration,
)

# COMMAND ----------

# Convert to binary
y_pred_train = continuous_to_binary(y_pred_train, threshold)
y_pred_test = continuous_to_binary(y_pred_test, threshold)

current_score_test_recall = recall_score(
   y_pred_test, test_y, zero_division=0
)
current_score_train_recall = recall_score(
    y_pred_train, train_y, zero_division=0
)


current_score_test_precision = precision_score(
   y_pred_test, test_y, zero_division=0
)
current_score_train_precision = precision_score(
    y_pred_train, train_y, zero_division=0
)


current_score_test_f1_score = f1_score(
   y_pred_test, test_y, zero_division=0
)
current_score_train_f1_score = f1_score(
    y_pred_train, train_y, zero_division=0
)

# COMMAND ----------

current_score_test_recall, current_score_train_recall, current_score_test_precision, current_score_train_precision, current_score_test_f1_score, current_score_train_f1_score

# COMMAND ----------

print(classification_report(train_y, y_pred_train))

# COMMAND ----------

# Confusion Matrix - Train set
cm = confusion_matrix(train_y, y_pred_train)
fig = plt.gcf()
fig.tight_layout()
sns.heatmap(cm, annot=True, fmt="d")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Fining tunning

# COMMAND ----------

base_params = {
    "objective": "binary",
    "boosting_type": "dart",
    "metric": "binary_logloss",
    "device_type": "cpu",
    "num_threads": 4,
    "enable_bundle": True,
    "verbose": -1,
    "random_seed": 0,
    "is_unbalance": True,
    "extra_trees": True,
}

parameters_dict = {
    "learning_rate": {"min": 0.01, "max": 0.3, "type": "float"},
    "num_leaves": {"min": 500, "max": 3500, "step": 10, "type": "int"},
    "max_depth": {"min": 3, "max": 20, "step": 1, "type": "int"},
    "min_data_in_leaf": {"min": 10, "max": 300, "step": 5, "type": "int"},
    "feature_fraction": {"min": 0.5, "max": 1, "type": "float"},
    "bagging_fraction": {"min": 0.1, "max": 1, "type": "float"},
    "num_iterations": {"min": 100, "max": 300, "step": 10, "type": "int"},
    "lambda_l1": {"min": 1, "max": 10, "step": 1, "type": "int"},
    "lambda_l2": {"min": 1, "max": 10, "step": 1, "type": "int"},
}


WEIGHT_FOR_METRIC_OPTUNA = 1.5
OPTIMIZE_THRESHOLD_OPTUNA = True
N_TRIALS = 50

# COMMAND ----------

study = lgbm_class_hyperparameter_tuning_pipeline(
    X_train_encoded,
    train_y,
    base_params,
    parameters_dict,
    WEIGHT_FOR_METRIC_OPTUNA,
    OPTIMIZE_THRESHOLD_OPTUNA,
    N_TRIALS,
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Training with cross validation and applying fine tunning

# COMMAND ----------

parameters = study.best_params
parameters.update(base_parameters_lgbm_class)
threshold = study.best_params['threshold']

# COMMAND ----------


if "threshold" in parameters.keys():
    del parameters["threshold"]
    threshold = study.best_params['threshold']
else:
    threshold=0.5

scores_rs_train = []
scores_rs_test = []

scores_pr_train = []
scores_pr_test = []

# define stratified k-fold cross-validation
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


#  the data just to shuffle it
# Cross Validation
for train_index, test_index in kf.split(X_train_encoded, train_y):
    
    X_train, X_test = X_train_encoded.iloc[train_index], X_train_encoded.iloc[test_index]
    y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]
    
    
    train_set = lgbm.Dataset(
        X_train,
        label=y_train,
    )

    dval = lgbm.Dataset(
        X_test,
        y_test,
        reference=train_set,
    )

    model = lgbm.train(parameters, train_set, valid_sets=dval)

    y_pred_train = model.predict(
        X_train,
        num_iteration=model.best_iteration,
    )

    y_pred_test = model.predict(
        X_test,
        num_iteration=model.best_iteration,
    )

    # Convert to binary
    y_pred_train = continuous_to_binary(y_pred_train, threshold)
    y_pred_test = continuous_to_binary(y_pred_test, threshold)

    current_score_train_rs = recall_score(
        y_train, y_pred_train, zero_division=0
    )
    current_score_test_rs = recall_score(
        y_test, y_pred_test, zero_division=0
    )

    scores_rs_train += [current_score_train_rs]
    scores_rs_test += [current_score_test_rs]
    
    current_score_train_pr = precision_score(
        y_train, y_pred_train, zero_division=0
    )
    current_score_test_pr = precision_score(
        y_test, y_pred_test, zero_division=0
    )

    scores_pr_train += [current_score_train_pr]
    scores_pr_test += [current_score_test_pr]


scores_rs_train_mean = np.mean(scores_rs_train)
scores_rs_test_mean = np.mean(scores_rs_test)
scores_rs_train_std = np.std(scores_rs_train)
scores_rs_test_std = np.std(scores_rs_test)

scores_pr_train_mean = np.mean(scores_pr_train)
scores_pr_test_mean = np.mean(scores_pr_test)
scores_pr_train_std = np.std(scores_pr_train)
scores_pr_test_std = np.std(scores_pr_test)

# COMMAND ----------

scores_rs_train_mean, scores_pr_train_mean, scores_rs_test_mean, scores_pr_test_mean

# COMMAND ----------

scores_rs_train_std, scores_pr_train_std, scores_rs_test_std, scores_pr_test_std, 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Training final model

# COMMAND ----------

# Train model
train_set = lgbm.Dataset(X_train_encoded, label=train_y)

dval = lgbm.Dataset(
    X_test_encoded, test_y, reference=train_set
)

# Save train/validation plot
evals_results = {}  # to record eval results for plotting

model = lgbm.train(
    params=parameters,
    train_set=train_set,
    valid_sets=[train_set, dval],
    valid_names=["Train", "Valid"],
    callbacks=[lgbm.record_evaluation(evals_results)],
)

model_train_plot = lgbm.plot_metric(evals_results)

# COMMAND ----------

y_pred_train = model.predict(
    X_train_encoded,
    num_iteration=model.best_iteration,
)

y_pred_test = model.predict(
    X_test_encoded,
    num_iteration=model.best_iteration,
)
    
# Convert to binary
y_pred_train = continuous_to_binary(y_pred_train, threshold)
y_pred_test = continuous_to_binary(y_pred_test, threshold)

current_score_train_rs = recall_score(
train_y, y_pred_train, zero_division=0
)
current_score_test_rs = recall_score(
test_y, y_pred_test, zero_division=0
)

scores_rs_train += [current_score_train_rs]
scores_rs_test += [current_score_test_rs]

current_score_train_pr = precision_score(
train_y, y_pred_train, zero_division=0
)
current_score_test_pr = precision_score(
test_y, y_pred_test, zero_division=0
)


# COMMAND ----------

current_score_test_recall, current_score_train_recall, current_score_test_precision, current_score_train_precision, current_score_test_f1_score, current_score_train_f1_score

# COMMAND ----------

# Confusion Matrix - Train set
cm = confusion_matrix(train_y, y_pred_train)
fig = plt.gcf()
fig.tight_layout()
sns.heatmap(cm, annot=True, fmt="d")

# COMMAND ----------

# Confusion Matrix - Test set
cm = confusion_matrix(test_y, y_pred_test)
fig = plt.gcf()
fig.tight_layout()
sns.heatmap(cm, annot=True, fmt="d")

# COMMAND ----------

print(classification_report(test_y, y_pred_test))

# COMMAND ----------

print(classification_report(train_y, y_pred_train))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Test with catboost

# COMMAND ----------

X, y = train_test_split(
    df[model_inputs],
    stratify=df["delay_15"],
    test_size=0.20,
    random_state=42,
)

# COMMAND ----------

params = {
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'border_count': 64,
    'thread_count': 4,
    'random_seed': 42,
    'eval_metric': 'Recall',
    'verbose': 50,
    'class_weights': [1, 5] # set higher weight for minority class
}

# COMMAND ----------

assert set(list(X.index)).intersection(set(y.index)) == set()

# COMMAND ----------

assert X["delay_15"].value_counts(normalize=True)[0] < 0.82

# COMMAND ----------

assert y["delay_15"].value_counts(normalize=True)[0] < 0.82

# COMMAND ----------

train_x = X.drop("delay_15", axis=1)
train_y = X["delay_15"]

test_x = y.drop("delay_15", axis=1)
test_y = y["delay_15"]


# COMMAND ----------

categorical_features_indices = np.where(train_x.dtypes != np.float)[0]

train_pool = Pool(
    data=train_x,
    label=train_y,
    cat_features=categorical_features_indices
)

test_pool = Pool(
    data=test_x,
    label=test_y,
    cat_features=categorical_features_indices
)

# COMMAND ----------

cat_model = CatBoostClassifier(**params)
cat_model.fit(train_pool, eval_set=test_pool, use_best_model=True, plot=True)

# COMMAND ----------

y_pred_test = cat_model.predict(test_x)
y_pred_train = cat_model.predict(train_x)

# COMMAND ----------

current_score_train_rs = recall_score(
train_y, y_pred_train, zero_division=0
)
current_score_test_rs = recall_score(
test_y, y_pred_test, zero_division=0
)

current_score_train_pr = precision_score(
train_y, y_pred_train, zero_division=0
)
current_score_test_pr = precision_score(
test_y, y_pred_test, zero_division=0
)


# COMMAND ----------

current_score_train_rs, current_score_test_rs, current_score_train_pr, current_score_test_pr

# COMMAND ----------

print(classification_report(train_y, y_pred_train))

# COMMAND ----------

print(classification_report(test_y, y_pred_test))

# COMMAND ----------

# Confusion Matrix - Test set
cm = confusion_matrix(test_y, y_pred_test)
fig = plt.gcf()
fig.tight_layout()
sns.heatmap(cm, annot=True, fmt="d")

# COMMAND ----------

# Confusion Matrix - Train set

cm = confusion_matrix(train_y, y_pred_train)
fig = plt.gcf()
fig.tight_layout()
sns.heatmap(cm, annot=True, fmt="d")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Scenario 2
# MAGIC 
# MAGIC Operational data should be used for training because we will have this information at the time of training.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Treat categorical features
# MAGIC 
# MAGIC - One-hot encoding can be computationally expensive and can lead to a high-dimensional feature space, especially for variables with many categories. So, we'll do it only with the categories below.
# MAGIC 
# MAGIC - I used CatBoostEncounter for the other columns. I then fitted the encoder on the training data and transformed the training and test data using the trained encoder.

# COMMAND ----------

model_inputs = [
    "scheduled_destination_city_code",
    "scheduled_airline_code",
    "type_of_flight",
    "high_season",
    "delay_15",
    "period_day",
    "count_flights_same_datetime",
    "operation_destination_city_code",
    "operated_airline_code",
    "day_operation",
    "month_operation",
    "year_operation",
    "day_of_the_week_operation",
    "airline_that_operates",
]

# COMMAND ----------

df = df.reset_index(drop=True)

# COMMAND ----------

encoded_df = pd.get_dummies(
    df[model_inputs],
    columns=["type_of_flight", "period_day"],
    prefix=["type_of_flight", "period_day"],
    drop_first=[True, False],
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train test split

# COMMAND ----------

X, y = train_test_split(
    encoded_df,
    stratify=encoded_df["delay_15"],
    test_size=0.20,
    random_state=42,
)

# COMMAND ----------

assert set(list(X.index)).intersection(set(y.index)) == set()

# COMMAND ----------

assert X["delay_15"].value_counts(normalize=True)[0] < 0.82

# COMMAND ----------

assert y["delay_15"].value_counts(normalize=True)[0] < 0.82

# COMMAND ----------

train_x = X.drop("delay_15", axis=1)
train_y = X["delay_15"]

test_x = y.drop("delay_15", axis=1)
test_y = y["delay_15"]


# COMMAND ----------

# MAGIC %md
# MAGIC #### Enconding with categorical encoder

# COMMAND ----------

# create a categorical encoder and fit it to the training data
encoder = ce.CatBoostEncoder(cols=['scheduled_destination_city_code', 'scheduled_airline_code', 'operation_destination_city_code', 'operated_airline_code', 'day_of_the_week_operation', 'airline_that_operates'], random_state=42)
X_train_encoded = encoder.fit_transform(train_x, train_y)

# COMMAND ----------

base_parameters_lgbm_class = {
    "objective": "binary",
    "boosting_type": "dart",
    "metric": "binary_logloss",
    "device_type": "cpu",
    "num_threads": 4,
    "enable_bundle": True,
    "verbose": -1,
    "random_seed": 0,
    "is_unbalance": True,
    "extra_trees": True,
}

threshold = 0.5

# COMMAND ----------

# transform the testing data using the trained encoder
X_test_encoded = encoder.transform(test_x)

# COMMAND ----------

# MAGIC %md
# MAGIC #### First train

# COMMAND ----------

train_set = lgbm.Dataset(
    X_train_encoded,
    label=train_y
)

# COMMAND ----------

model = lgbm.train(base_parameters_lgbm_class, train_set)

# COMMAND ----------

y_pred_train = model.predict(
    X_train_encoded,
    num_iteration=model.best_iteration,
)

y_pred_test = model.predict(
    X_test_encoded,
    num_iteration=model.best_iteration,
)

# COMMAND ----------

# Convert to binary
y_pred_train = continuous_to_binary(y_pred_train, threshold)
y_pred_test = continuous_to_binary(y_pred_test, threshold)

current_score_test_recall = recall_score(
   y_pred_test, test_y, zero_division=0
)
current_score_train_recall = recall_score(
    y_pred_train, train_y, zero_division=0
)


current_score_test_precision = precision_score(
   y_pred_test, test_y, zero_division=0
)
current_score_train_precision = precision_score(
    y_pred_train, train_y, zero_division=0
)


current_score_test_f1_score = f1_score(
   y_pred_test, test_y, zero_division=0
)
current_score_train_f1_score = f1_score(
    y_pred_train, train_y, zero_division=0
)

# COMMAND ----------

current_score_test_recall, current_score_train_recall, current_score_test_precision, current_score_train_precision, current_score_test_f1_score, current_score_train_f1_score

# COMMAND ----------

print(classification_report(train_y, y_pred_train))

# COMMAND ----------

# Confusion Matrix - Train set
cm = confusion_matrix(train_y, y_pred_train)
fig = plt.gcf()
fig.tight_layout()
sns.heatmap(cm, annot=True, fmt="d")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Fining tunning

# COMMAND ----------

base_params = {
    "objective": "binary",
    "boosting_type": "dart",
    "metric": "binary_logloss",
    "device_type": "cpu",
    "num_threads": 4,
    "enable_bundle": True,
    "verbose": -1,
    "random_seed": 0,
    "is_unbalance": True,
    "extra_trees": True,
}

parameters_dict = {
    "learning_rate": {"min": 0.01, "max": 0.3, "type": "float"},
    "num_leaves": {"min": 500, "max": 3500, "step": 10, "type": "int"},
    "max_depth": {"min": 3, "max": 20, "step": 1, "type": "int"},
    "min_data_in_leaf": {"min": 10, "max": 300, "step": 5, "type": "int"},
    "feature_fraction": {"min": 0.5, "max": 1, "type": "float"},
    "bagging_fraction": {"min": 0.1, "max": 1, "type": "float"},
    "num_iterations": {"min": 100, "max": 300, "step": 10, "type": "int"},
    "lambda_l1": {"min": 1, "max": 10, "step": 1, "type": "int"},
    "lambda_l2": {"min": 1, "max": 10, "step": 1, "type": "int"},
}


WEIGHT_FOR_METRIC_OPTUNA = 1.5
OPTIMIZE_THRESHOLD_OPTUNA = True
N_TRIALS = 50

# COMMAND ----------

study = lgbm_class_hyperparameter_tuning_pipeline(
    X_train_encoded,
    train_y,
    base_params,
    parameters_dict,
    WEIGHT_FOR_METRIC_OPTUNA,
    OPTIMIZE_THRESHOLD_OPTUNA,
    N_TRIALS,
)

# COMMAND ----------

study.best_params

# COMMAND ----------

# MAGIC %md
# MAGIC #### Training with cross validation and applying fine tunning

# COMMAND ----------

parameters = study.best_params
parameters.update(base_parameters_lgbm_class)
threshold = study.best_params['threshold']

# COMMAND ----------


if "threshold" in parameters.keys():
    del parameters["threshold"]
    threshold = study.best_params['threshold']
else:
    threshold=0.5

scores_rs_train = []
scores_rs_test = []

scores_pr_train = []
scores_pr_test = []

# define stratified k-fold cross-validation
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


#  the data just to shuffle it
# Cross Validation
for train_index, test_index in kf.split(X_train_encoded, train_y):
    
    X_train, X_test = X_train_encoded.iloc[train_index], X_train_encoded.iloc[test_index]
    y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]
    
    
    train_set = lgbm.Dataset(
        X_train,
        label=y_train,
    )

    dval = lgbm.Dataset(
        X_test,
        y_test,
        reference=train_set,
    )

    model = lgbm.train(parameters, train_set, valid_sets=dval)

    y_pred_train = model.predict(
        X_train,
        num_iteration=model.best_iteration,
    )

    y_pred_test = model.predict(
        X_test,
        num_iteration=model.best_iteration,
    )

    # Convert to binary
    y_pred_train = continuous_to_binary(y_pred_train, threshold)
    y_pred_test = continuous_to_binary(y_pred_test, threshold)

    current_score_train_rs = recall_score(
        y_train, y_pred_train, zero_division=0
    )
    current_score_test_rs = recall_score(
        y_test, y_pred_test, zero_division=0
    )

    scores_rs_train += [current_score_train_rs]
    scores_rs_test += [current_score_test_rs]
    
    current_score_train_pr = precision_score(
        y_train, y_pred_train, zero_division=0
    )
    current_score_test_pr = precision_score(
        y_test, y_pred_test, zero_division=0
    )

    scores_pr_train += [current_score_train_pr]
    scores_pr_test += [current_score_test_pr]


scores_rs_train_mean = np.mean(scores_rs_train)
scores_rs_test_mean = np.mean(scores_rs_test)
scores_rs_train_std = np.std(scores_rs_train)
scores_rs_test_std = np.std(scores_rs_test)

scores_pr_train_mean = np.mean(scores_pr_train)
scores_pr_test_mean = np.mean(scores_pr_test)
scores_pr_train_std = np.std(scores_pr_train)
scores_pr_test_std = np.std(scores_pr_test)

# COMMAND ----------

scores_rs_train_mean, scores_pr_train_mean, scores_rs_test_mean, scores_pr_test_mean

# COMMAND ----------

scores_rs_train_std, scores_pr_train_std, scores_rs_test_std, scores_pr_test_std, 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Training final model

# COMMAND ----------

# Train model
train_set = lgbm.Dataset(X_train_encoded, label=train_y)

dval = lgbm.Dataset(
    X_test_encoded, test_y, reference=train_set
)

# Save train/validation plot
evals_results = {}  # to record eval results for plotting

model = lgbm.train(
    params=parameters,
    train_set=train_set,
    valid_sets=[train_set, dval],
    valid_names=["Train", "Valid"],
    callbacks=[lgbm.record_evaluation(evals_results)],
)

model_train_plot = lgbm.plot_metric(evals_results)

# COMMAND ----------

y_pred_train = model.predict(
    X_train_encoded,
    num_iteration=model.best_iteration,
)

y_pred_test = model.predict(
    X_test_encoded,
    num_iteration=model.best_iteration,
)
    
# Convert to binary
y_pred_train = continuous_to_binary(y_pred_train, threshold)
y_pred_test = continuous_to_binary(y_pred_test, threshold)

current_score_train_rs = recall_score(
train_y, y_pred_train, zero_division=0
)
current_score_test_rs = recall_score(
test_y, y_pred_test, zero_division=0
)

scores_rs_train += [current_score_train_rs]
scores_rs_test += [current_score_test_rs]

current_score_train_pr = precision_score(
train_y, y_pred_train, zero_division=0
)
current_score_test_pr = precision_score(
test_y, y_pred_test, zero_division=0
)


# COMMAND ----------

current_score_test_recall, current_score_train_recall, current_score_test_precision, current_score_train_precision, current_score_test_f1_score, current_score_train_f1_score

# COMMAND ----------

print(classification_report(train_y, y_pred_train))

# COMMAND ----------

print(classification_report(test_y, y_pred_test))

# COMMAND ----------

# Confusion Matrix - Train set
cm = confusion_matrix(train_y, y_pred_train)
fig = plt.gcf()
fig.tight_layout()
sns.heatmap(cm, annot=True, fmt="d")

# COMMAND ----------

# Confusion Matrix - Test set
cm = confusion_matrix(test_y, y_pred_test)
fig = plt.gcf()
fig.tight_layout()
sns.heatmap(cm, annot=True, fmt="d")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluate model performance in the predictive task across each model that you trained. Define and justify what metrics you used to assess model performance. Pick the best trained model and evaluate the following: What variables were the most influential in the prediction task? How could you improve the Performance?

# COMMAND ----------

# MAGIC %md
# MAGIC 1. **Define and justify what metrics you used to assess model performance:**
# MAGIC 
# MAGIC - In general, recall rate is a useful metric for applications where omitting a positive sample is more costly than misclassifying a negative sample as positive.
# MAGIC 
# MAGIC - Recall would be an important metric to optimize because it is more important to correctly identify all delayed flights (true positives), even if it means that some non-delayed flights are flagged as delayed (false positives) (**Very important to confirm with the specialists**)
# MAGIC 
# MAGIC - It is important to consider the context of the problem and the potential consequences of false negatives and false positives when choosing a metric to optimize for.
# MAGIC 
# MAGIC - A confusion matrix was also used, as four categories (TP, FP, TN, FN) allow evaluating the performance of the model in terms of its ability to correctly classify positive and negative samples.
# MAGIC 
# MAGIC 
# MAGIC 2. **How could you improve the Performance?**
# MAGIC 
# MAGIC 
# MAGIC - Feature engineering: Feature engineering involves selecting and transforming the input features that the model uses to make predictions. Example: We could collect data on weather conditions predicted at the time of flight planning and use it to develop statistics and new features.
# MAGIC 
# MAGIC - Hyperparameter tuning:  By tuning these hyperparameters, you can often improve the performance of the model. We could improve the objective function of parameter tuning to better understand the context of the problem. For example, if it is more expensive to have false negatives or false positives.
# MAGIC 
# MAGIC - Use more complex models: More complex models, such as deep neural networks, can sometimes achieve higher accuracy than simpler models. However, complex models can also be more difficult to train and require more data. **CatBoost model showed potential** in its performance and can be explored in the future
# MAGIC 
# MAGIC - Regularization: Regularization techniques can help prevent overfitting by penalizing models that are too complex.
# MAGIC 
# MAGIC - Preprocessing: Preprocessing techniques such as scaling, normalization, and imputation can sometimes improve model performance by making the data more suitable for the model to learn from.
# MAGIC 
# MAGIC - Test more categorical enconders: Choosing the correct categorical encoder can significantly improve model performance because it can help transform categorical features into a format that machine learning algorithms can better understand and make use of. By selecting the right categorical encoder, you can improve the quality of the input data provided to the model, which can in turn improve its accuracy and predictive power.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Most influential in the prediction task - Scenario 1

# COMMAND ----------

explainer = shap.TreeExplainer(model, X_train_encoded)
shap_values = explainer.shap_values(X_test_encoded)

# COMMAND ----------

shap.summary_plot(shap_values, X_test_encoded)

# COMMAND ----------

xmin = np.quantile(shap_values[:, feature].data, 0.05)
xmax = np.quantile(shap_values[:, feature].data, 0.95)

shap.plots.scatter(shap_values[:, feature], xmin=xmin, xmax=xmax)

# COMMAND ----------

explainer = shap.TreeExplainer(cat_model)
shap_values = explainer.shap_values(test_pool)

# COMMAND ----------

shap.summary_plot(shap_values, test_x, plot_type="bar")

# COMMAND ----------


