import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
pd.options.display.html.table_schema = True
import seaborn as sb

import pickle
import cdsw

## Create Spark Session
spark = SparkSession.builder \
      .appName("Telco Customer Churn") \
      .getOrCreate()
    
schemaData = StructType([StructField("state", StringType(), True),StructField("account_length", DoubleType(), True),StructField("area_code", StringType(), True),StructField("phone_number", StringType(), True),StructField("intl_plan", StringType(), True),StructField("voice_mail_plan", StringType(), True),StructField("number_vmail_messages", DoubleType(), True),     StructField("total_day_minutes", DoubleType(), True),     StructField("total_day_calls", DoubleType(), True),     StructField("total_day_charge", DoubleType(), True),     StructField("total_eve_minutes", DoubleType(), True),     StructField("total_eve_calls", DoubleType(), True),     StructField("total_eve_charge", DoubleType(), True),     StructField("total_night_minutes", DoubleType(), True),     StructField("total_night_calls", DoubleType(), True),     StructField("total_night_charge", DoubleType(), True),     StructField("total_intl_minutes", DoubleType(), True),     StructField("total_intl_calls", DoubleType(), True),     StructField("total_intl_charge", DoubleType(), True),     StructField("number_customer_service_calls", DoubleType(), True),     StructField("churned", StringType(), True)])
churn_data = spark.read.schema(schemaData).csv('data/churn.all')

## Data Exploration
sample_data = churn_data.sample(False, 0.5, 83).toPandas()
sample_data.head(23).transpose()

# Looking at joint distributions of data can also tell us a lot, 
# particularly about redundant features. [Seaborn's PairPlot](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.pairplot.html#seaborn.pairplot) 
# let's us look at joint distributions for many variables at once.

example_numeric_data = sample_data[["total_intl_minutes", "total_intl_calls",
                                       "total_intl_charge", "churned"]]
sb.pairplot(example_numeric_data, hue="churned")


## Churn Prediction
#
# Test with various number of trees. 
parser = argparse.ArgumentParser(description='Housing price predictor.')
parser.add_argument('--trees', type=int, default=10,
                   help='number of trees')
args = parser.parse_args()
cdsw.track_metric("numTrees",args.trees)

reduced_churn_data= churn_data.select("account_length", "number_vmail_messages", "total_day_calls",
                     "total_day_charge", "total_eve_calls", "total_eve_charge",
                     "total_night_calls", "total_night_charge", "total_intl_calls", 
                    "total_intl_charge","number_customer_service_calls")

label_indexer = StringIndexer(inputCol = 'churned', outputCol = 'label')
plan_indexer = StringIndexer(inputCol = 'intl_plan', outputCol = 'intl_plan_indexed')
pipeline = Pipeline(stages=[plan_indexer, label_indexer])
indexed_data = pipeline.fit(churn_data).transform(churn_data)

(train_data, test_data) = indexed_data.randomSplit([0.7, 0.3])

pdTrain = train_data.toPandas()
pdTest = test_data.toPandas()
features = ["intl_plan_indexed","account_length", "number_vmail_messages", "total_day_calls",
                     "total_day_charge", "total_eve_calls", "total_eve_charge",
                     "total_night_calls", "total_night_charge", "total_intl_calls", 
                    "total_intl_charge","number_customer_service_calls"]
randF=RandomForestClassifier(n_jobs=10,
                             n_estimators=args.trees)
randF.fit(pdTrain[features], pdTrain['label'])
predictions=randF.predict(pdTest[features])

## Feature Importance
list(zip(pdTrain[features], randF.feature_importances_))

## AUROC
y_true = pdTest['label']
y_scores = predictions
auroc = roc_auc_score(y_true, y_scores)
ap = average_precision_score (y_true, y_scores)
print(auroc, ap)

cdsw.track_metric("auroc", auroc)
cdsw.track_metric("ap", ap)

## Serialize Model
pickle.dump(randF, open("models/sklearn_rf.pkl","wb"))

cdsw.track_file("models/sklearn_rf.pkl")


## Stop Spark

spark.stop()
