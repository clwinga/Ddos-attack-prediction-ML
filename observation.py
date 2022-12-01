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
from pyspark.sql.functions import col, unix_timestamp, round, lit
import itertools
from scipy import stats
from scipy.stats import mannwhitneyu

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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


def convert_string_to_datetime(dt_string):
    return datetime.strptime(dt_string, "%d/%m/%Y %I:%M:%S")

def get_data(data):
    return data.select(
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
            data['Flow Byts/s']
        # TODO: also the y values
            #(data['id'] % 10).alias('bin'),
        )


def organize_by_ping(data):
    grouped = data.groupBy(data['Src IP'],data['Dst IP'], data['Label'])
    groups = grouped.agg(
        functions.sum(data['Flow Duration']).alias('Flow Duration Sum'),
        (functions.sum(data['Flow Duration'])/functions.count('*')).alias('Flow Duration Avg'),
        (functions.sum(data['Tot Fwd Pkts'])).alias('Tot Fwd Pkts sum'),
        (functions.sum(data['Tot Bwd Pkts'])).alias('Tot Bwd Pkts sum'),
        (functions.sum(data['Tot Fwd Pkts'])*functions.count('*')).alias('Tot Fwd Pkts avg'),
        (functions.sum(data['Tot Bwd Pkts'])*functions.count('*')).alias('Tot Bwd Pkts avg'),
        (functions.sum(data['Idle Mean'])).alias('Idle avg'),
        (functions.sum(data['Flow Byts/s'])/functions.count('*')).alias('Flow Byts/s'),
        (functions.sum(data['Flow Byts/s'])).alias('Flow Byts/s avg'),
        functions.count('*').alias('ping'))
    return groups


def get_feature_scores(df1, df2, X_first, X_second, Y):

    x1 = df1.select(X_first).rdd.flatMap(lambda x: x).collect() + df2.select(X_first).rdd.flatMap(lambda x: x).collect()
    x2 = df1.select(X_second).rdd.flatMap(lambda x: x).collect() + df1.select(X_second).rdd.flatMap(lambda x: x).collect()
    y = df1.select(Y).rdd.flatMap(lambda x: x).collect() + df2.select(Y).rdd.flatMap(lambda x: x).collect()
    x = np.stack([x1, x2], axis=1)
    
    X_train, X_valid, y_train, y_valid = train_test_split(x, y)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    print(model.score(X_valid, y_valid))


def main(in_directory, out_directory):
    # Read the data from the JSON files
    raw = spark.read.csv(in_directory, schema=schema)
    raw_filtered = get_data(raw)#.show(); #return
    ddos = raw_filtered.filter(raw.Label=="ddos")
    benign = raw_filtered.filter(raw.Label!="ddos")
    ddos_organized = organize_by_ping(ddos).cache()
    benign_organized = organize_by_ping(benign).cache()

    raw_orgranized = organize_by_ping(raw_filtered)

    size_ddos = ddos_organized.count()
    limited_bening = benign_organized.limit(size_ddos)

    get_feature_scores(limited_bening, ddos_organized, 'Tot Fwd Pkts avg', 'Tot Bwd Pkts avg', 'Label')
    
if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
