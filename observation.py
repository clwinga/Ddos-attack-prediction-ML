#!/usr/bin/env python
# coding: utf-8

# In[3]:

# time spark-submit --master=local[1] observation.py datatset/unbalanced output/predictions.csv

import sys
from pyspark.sql import SparkSession, functions, types
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter
from pyspark.sql import Row
from pyspark.sql.functions import col, unix_timestamp, round
from pyspark.sql.functions import lit
import itertools
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier



spark = SparkSession.builder.appName('first Spark app').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+


schema = types.StructType([
    types.StructField('Unnamed: 0', types.StringType()),
    types.StructField('Flow ID', types.StringType()),
    types.StructField('Src IP', types.StringType()),
    types.StructField('Src Port', types.StringType()),
    types.StructField('Dst IP', types.StringType()),
    types.StructField('Dst Port', types.IntegerType()),
    types.StructField('Protocol', types.StringType()),
    types.StructField('Timestamp', types.StringType()),
    types.StructField('Flow Duration', types.IntegerType()),
    types.StructField('Tot Fwd Pkts', types.StringType()),
    types.StructField('Tot Bwd Pkts', types.StringType()),
    types.StructField('TotLen Fwd Pkts', types.StringType()),
    types.StructField('TotLen Bwd Pkts', types.StringType()),
    types.StructField('Fwd Pkt Len Max', types.StringType()),
    types.StructField('Fwd Pkt Len Min', types.StringType()),
    types.StructField('Fwd Pkt Len Mean', types.StringType()),
    types.StructField('Fwd Pkt Len Std', types.StringType()),
    types.StructField('Bwd Pkt Len Max', types.StringType()),
    types.StructField('Bwd Pkt Len Min', types.StringType()),
    types.StructField('Bwd Pkt Len Mean', types.StringType()),
    types.StructField('Bwd Pkt Len Std', types.StringType()),
    types.StructField('Flow Byts/s', types.StringType()),
    types.StructField('Flow Pkts/s', types.StringType()),
    types.StructField('Flow IAT Mean', types.StringType()),
    types.StructField('Flow IAT Std', types.StringType()),
    types.StructField('Flow IAT Max', types.StringType()),
    types.StructField('Flow IAT Min', types.StringType()),
    types.StructField('Fwd IAT Tot', types.StringType()),
    types.StructField('Fwd IAT Mean', types.StringType()),
    types.StructField('Fwd IAT Std', types.StringType()),
    types.StructField('Fwd IAT Max', types.StringType()),
    types.StructField('Fwd IAT Min', types.StringType()),
    types.StructField('Bwd IAT Tot', types.StringType()),
    types.StructField('Bwd IAT Mean', types.StringType()),
    types.StructField('Bwd IAT Std', types.StringType()),
    types.StructField('Bwd IAT Max', types.StringType()),
    types.StructField('Bwd IAT Min', types.StringType()),
    types.StructField('Fwd PSH Flags', types.StringType()),
    types.StructField('Bwd PSH Flags', types.StringType()),
    types.StructField('Fwd URG Flags', types.StringType()),
    types.StructField('Bwd URG Flags', types.StringType()),
    types.StructField('Fwd Header Len', types.StringType()),
    types.StructField('Bwd Header Len', types.StringType()),
    types.StructField('Fwd Pkts/s', types.StringType()),
    types.StructField('Bwd Pkts/s', types.StringType()),
    types.StructField('Pkt Len Min', types.StringType()),
    types.StructField('Pkt Len Max', types.StringType()),
    types.StructField('Pkt Len Mean', types.StringType()),
    types.StructField('Pkt Len Std', types.StringType()),
    types.StructField('Pkt Len Var', types.StringType()),
    types.StructField('FIN Flag Cnt', types.StringType()),
    types.StructField('SYN Flag Cnt', types.StringType()),
    types.StructField('RST Flag Cnt', types.StringType()),
    types.StructField('PSH Flag Cnt', types.StringType()),
    types.StructField('ACK Flag Cnt', types.StringType()),
    types.StructField('URG Flag Cnt', types.StringType()),
    types.StructField('CWE Flag Count', types.StringType()),
    types.StructField('ECE Flag Cnt', types.StringType()),
    types.StructField('Down/Up Ratio', types.StringType()),
    types.StructField('Pkt Size Avg', types.FloatType()),
    types.StructField('Fwd Seg Size Avg', types.StringType()),
    types.StructField('Bwd Seg Size Avg', types.StringType()),
    types.StructField('Fwd Byts/b Avg', types.StringType()),
    types.StructField('Fwd Pkts/b Avg', types.StringType()),
    types.StructField('Fwd Blk Rate Avg', types.StringType()),
    types.StructField('Bwd Byts/b Avg', types.StringType()),
    types.StructField('Bwd Pkts/b Avg', types.StringType()),
    types.StructField('Bwd Blk Rate Avg', types.StringType()),
    types.StructField('Subflow Fwd Pkts', types.StringType()),
    types.StructField('Subflow Fwd Byts', types.StringType()),
    types.StructField('Subflow Bwd Pkts', types.StringType()),
    types.StructField('Subflow Bwd Byts', types.StringType()),
    types.StructField('Init Fwd Win Byts', types.StringType()),
    types.StructField('Init Bwd Win Byts', types.StringType()),
    types.StructField('Fwd Act Data Pkts', types.StringType()),
    types.StructField('Fwd Seg Size Min', types.StringType()),
    types.StructField('Active Mean', types.StringType()),
    types.StructField('Active Std', types.StringType()),
    types.StructField('Active Max', types.StringType()),
    types.StructField('Active Min', types.StringType()),
    types.StructField('Idle Mean', types.StringType()),
    types.StructField('Idle Std', types.StringType()),
    types.StructField('Idle Max', types.StringType()),
    types.StructField('Idle Min', types.StringType()),
    types.StructField('Label', types.StringType())
])

output_schema = types.StructType([
    types.StructField('Real', types.StringType()),
    types.StructField('prediction', types.StringType())
])
# In[ ]:
def convert_string_to_datetime(dt_string):
    return datetime.strptime(dt_string, "%d/%m/%Y %I:%M:%S")

def get_data(data):
    return data.select(
            data['Flow ID'],
            data['Src IP'],
            data['Dst IP'],
            data['Dst Port'],
            data['Timestamp'],
            data['Flow Duration'],
            data['Pkt Size Avg'],
            data['Label'],
            data['Tot Fwd Pkts'],
            data['Tot Bwd Pkts'],
            data['Idle Mean'],
            data['Flow Byts/s'],
            data['Fwd Header Len'],
            data['Bwd Header Len']
        # TODO: also the y values
            #(data['id'] % 10).alias('bin'),
        )

def merge(list1, list2):
      
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

def organize_by_ping(data):
    grouped = data.groupBy(data['Flow ID'])
    groups = grouped.agg(
        functions.sum(data['Flow Duration']).alias('Flow Duration Sum'), #2
        (functions.sum(data['Flow Duration'])/functions.count('*')).alias('Flow Duration Avg'),#3
        (functions.sum(data['Tot Fwd Pkts'])).alias('Tot Fwd Pkts sum'), #4
        (functions.sum(data['Tot Bwd Pkts'])).alias('Tot Bwd Pkts sum'),#3
        (functions.sum(data['Fwd Header Len'])).alias('Fwd Header Len sum'), #4
        (functions.sum(data['Bwd Header Len'])).alias('Bwd Header Len sum'), #5
        (functions.sum(data['Tot Fwd Pkts'])/functions.count('*')).alias('Tot Fwd Pkts avg'),#6
        (functions.sum(data['Tot Bwd Pkts'])/functions.count('*')).alias('Tot Bwd Pkts avg'),#7
        (functions.sum(data['Idle Mean'])).alias('Idle avg'), #8
        (functions.sum(data['Flow Byts/s'])).alias('Flow Byts/s'), #/functions.count('*') #9
        functions.count('*').alias('ping')) #10
    
    return groups
    # We know groups has <=10 rows, so it can safely be moved into two partitions.
    #groups = groups.sort(groups['bin']).coalesce(2)
# In[ ]:
def t_test(data_1,data_2):
    return stats.ttest_ind(data_1, data_2)

def machine_leanring(combined_organized,x_1_col,x_2_col,y_1_col):
    X_1=combined_organized.rdd.map(lambda x: x[x_1_col]).collect() # Pings
    X_2=combined_organized.rdd.map(lambda x: x[x_2_col]).collect() # Total time
    Y_1 = combined_organized.rdd.map(lambda x: x[y_1_col]).collect()
    x_ml= np.stack([X_1,X_2], axis=1)   
    X_train, X_test, y_train, y_test = train_test_split(x_ml, Y_1)
    model = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=10))
    model.fit(X_train, y_train)
    return model
    
def main(in_directory, out_directory):
    # Read the data from the JSON files
    raw = spark.read.csv(in_directory, schema=schema)
    raw_filtered = get_data(raw)#.show(); #return
#     ddos = raw_filtered.filter(raw.Label=="ddos")
#     benign = raw_filtered.filter(raw.Label!="ddos")
    combined_organized = organize_by_ping(raw_filtered)
    
    combined_organized = combined_organized.join(
        raw_filtered,
        [(combined_organized['Flow ID'] == raw_filtered['Flow ID'])],
             "left"
            ).select("Tot Fwd Pkts","Tot Bwd Pkts","Label").cache()
    
    x_1_col = 0 
    x_2_col = 1
    y_1_col = 2
    X_1=combined_organized.rdd.map(lambda x: x[x_1_col]).collect() # Pings
    X_2=combined_organized.rdd.map(lambda x: x[x_2_col]).collect() # Total time
    Y_1 = combined_organized.rdd.map(lambda x: x[y_1_col]).collect()
    x_ml= np.stack([X_1,X_2], axis=1)   
    X_train, X_test, y_train, y_test = train_test_split(x_ml, Y_1)
    model = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=10))
    model.fit(X_train, y_train)
    N = 4000
    y_test_trim = (y_test[0:N])
    predictions = (model.predict(X_test[:N, :]))
    touple_temp = (merge(y_test_trim,predictions))
    print("Score :", model.score(X_test[:N, :], y_test_trim))
    rdd = spark.sparkContext.parallelize(touple_temp)
    sparkDF=spark.createDataFrame(rdd,output_schema)
    sparkDF.write.csv(out_directory, compression=None, mode='overwrite')
    print("+++++++++++++++++++++ done +++++++++++++++++++++")

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    #file = sys.argv[3]
    main(in_directory, out_directory)

