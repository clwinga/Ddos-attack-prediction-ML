import sys
from pyspark.sql import SparkSession, functions, types
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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
    types.StructField('Tot Fwd Pkts', types.IntegerType()),
    types.StructField('Tot Bwd Pkts', types.IntegerType()),
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
    types.StructField('Flow Byts/s', types.DoubleType()),
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
    types.StructField('Pkt Size Avg', types.DoubleType()),
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
    types.StructField('Idle Mean', types.DoubleType()),
    types.StructField('Idle Std', types.StringType()),
    types.StructField('Idle Max', types.StringType()),
    types.StructField('Idle Min', types.StringType()),
    types.StructField('Label', types.StringType())
])

def convert_string_to_datetime(dt_string):
    return datetime.strptime(dt_string, "%d/%m/%Y %I:%M:%S")

def create_plot(df_balanced, feature1, feature2):
    ddos = df_balanced[df_balanced['Label'] == "ddos"]
    benign = df_balanced[df_balanced['Label'] != "ddos"]
    ddos_x = ddos.select(feature1).rdd.flatMap(lambda x: x).collect()
    ddos_y = ddos.select(feature2).rdd.flatMap(lambda x: x).collect()
    benign_x = benign.select(feature1).rdd.flatMap(lambda x: x).collect()
    benign_y = benign.select(feature2).rdd.flatMap(lambda x: x).collect()
    plt.scatter(ddos_x, ddos_y, c='r')
    plt.scatter(benign_x, benign_y, c='b')
    plt.show()
    
def get_data(data):
    timestamp_min_udf = functions.udf(convert_string_to_datetime,returnType=types.TimestampType())
    return data.select(
            data['Flow ID'],
            data['Src IP'],
            data['Dst IP'],
            data['Dst Port'],
            functions.date_trunc("minute",timestamp_min_udf(functions.regexp_replace(functions.regexp_replace(data['Timestamp'], " AM", ""), " PM", ""))).alias('Timestamp'),
            data['Flow Duration'],
            data['Pkt Size Avg'],
            data['Label'],
            data['Tot Fwd Pkts'],
            data['Tot Bwd Pkts'],
            data['Idle Mean'],
            data['Flow Byts/s'],
            (data['Fwd Pkt Len Mean'] / data['Bwd Pkt Len Mean']).alias("Fwd Bwd size ratio"),
            data['Fwd Header Len'],
            data['Bwd Header Len']
        # TODO: also the y values
            #(data['id'] % 10).alias('bin'),
        )

def merge(list1, list2):
      
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

def organize_by_ping(data):
    grouped = data.groupBy(data['Src IP'],data['Dst IP'],data['Timestamp'], data['Label'])
    groups = grouped.agg(
        functions.sum(data['Flow Duration']).alias('Flow Duration Sum'),
        (functions.sum(data['Flow Duration'])/functions.count('*')).alias('Flow Duration Avg'),
        (functions.sum(data['Tot Fwd Pkts'])).alias('Tot Fwd Pkts sum'),
        (functions.sum(data['Tot Bwd Pkts'])).alias('Tot Bwd Pkts sum'),
        (functions.sum(data['Idle Mean'])).alias('Idle avg'),
        (functions.sum(data['Flow Byts/s'])).alias('Sum Flow Byts/s'),
        (functions.sum(data['Flow Byts/s'])/functions.count('*')).alias('avg Flow Byts/s'),
        (functions.avg(data['Fwd Bwd size ratio'])).alias('avg Fwd Bwd size ratio'),
        functions.count('*').alias('ping')
        )
    return groups

def get_model():
    model = make_pipeline( 
        MinMaxScaler(), 
        KNeighborsClassifier(n_neighbors=5)
    )
    return model

def get_feature_scores(df):
    Y = df['Label']
    X = df.drop("Label", axis=1).values
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y)
    assert np.any(np.isnan(X_train)) == False
    assert np.all(np.isfinite(X_train)) == True
    model = get_model()
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    print(model.score(X_valid, y_valid))

    #cleans and balances data with one to one ratio
def balance_and_clean_df(df_bening, df_ddos,count):
    df_bening = df_bening.replace([np.inf, -np.inf], np.nan)
    df_ddos = df_ddos.replace([np.inf, -np.inf], np.nan)
    df_bening = df_bening.dropna()
    df_ddos = df_ddos.dropna()
#     we did the count and hard coded it to save time. feel free to change it
#     count_ddos = df_ddos.count()
#     count_benign = df_bening.count()
    count_ddos = 3496 
    count_benign = 2688115
    min_len = min(count_benign, count_ddos, count)
    balanced = df_bening.limit(min_len).union(df_ddos.limit(min_len))
    balanced = balanced.drop('Src IP','Dst IP', 'Timestamp')
    return balanced

def clean_df(combined):
    combined = combined.drop('Src IP','Dst IP', 'Timestamp')
    combined.replace([np.inf, -np.inf], np.nan)#, inplace=True)
    combined = combined.na.drop()
    return combined

def main(in_directory, count):
    # Read the data from the JSON files
    raw = spark.read.csv(in_directory, schema=schema)
    raw_filtered = get_data(raw)#.cache()
    ddos = raw_filtered.filter(raw_filtered.Label=="ddos")
    benign = raw_filtered.filter(raw_filtered.Label!="ddos")
    ddos_organized = organize_by_ping(ddos).cache()
    benign_organized = organize_by_ping(benign).cache()
    combined_organized = balance_and_clean_df(ddos_organized, benign_organized, count)
    create_plot(combined_organized, "Tot Fwd Pkts sum","Tot Bwd Pkts sum")
    combined_organized = combined_organized.select("Tot Fwd Pkts sum","Tot Bwd Pkts sum","Flow Duration Sum","ping","Label")#.cache()
    combined_organized.select("Label").distinct().show()
    x_1_col = 0
    x_2_col = 1
    x_3_col = 2
    x_4_col = 3
    y_1_col = 4
    X_1=combined_organized.rdd.map(lambda x: x[x_1_col]).collect() #
    X_2=combined_organized.rdd.map(lambda x: x[x_2_col]).collect() #
    X_3=combined_organized.rdd.map(lambda x: x[x_3_col]).collect() #
    X_4=combined_organized.rdd.map(lambda x: x[x_4_col]).collect() #
    Y_1 = combined_organized.rdd.map(lambda x: x[y_1_col]).collect() #
    x_ml= np.stack([X_1,X_2,X_3,X_4], axis=1)   
    X_train, X_test, y_train, y_test = train_test_split(x_ml, Y_1)
    model = make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=10))
    model.fit(X_train, y_train)
    y_test_trim = (y_test)
    print("Score :", model.score(X_test, y_test_trim))
    print("+++++++++++++++++++++ done +++++++++++++++++++++")
    

if __name__=='__main__':
    in_directory = sys.argv[1]
    count = int(sys.argv[2])
    main(in_directory, count)
