from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from functions import generateStock
import seaborn as sns

import os
import sys
os.environ['SPARK_HOME'] = "C:\spark-2.0.0-bin-hadoop2.7"

sys.path.append("C:\spark-2.0.0-bin-hadoop2.7\python")
sys.path.append("C:\spark-2.0.0-bin-hadoop2.7\python\lib\py4j-0.10.1-src.zip")

try:
  from pyspark import SparkContext
  from pyspark import SparkConf
  print("Successfully imported Spark Modules")

except ImportError as e:
  print("Can not import Spark Modules", e)
  sys.exit(1)

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml import Pipeline
from pyspark.ml.param import TypeConverters # Developer API
from pyspark.sql import SparkSession
from pyspark.ml.linalg import *
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import VectorAssembler

if __name__ == "__main__":

    spark = SparkSession\
        .builder\
        .appName("LinearRegressionWithElasticNet")\
        .config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")\
        .getOrCreate()

    # Sin el getOrCreate()
    #spark = SparkSession
    #sc = SparkContext()
    #sql = SQLContext(SparkContext)

    # Load training data as DataFrame
    train = spark.read.csv("Datos.csv")
    train.show()
    train.describe().show()

    # Renaming columns
    train = train.withColumnRenamed("_c0", "x").withColumnRenamed("_c1", "label")
    train.printSchema()
    # Columns DataType conversion
    train_1 = train.withColumnRenamed("x", "oldX").withColumnRenamed("label", "oldLabel")
    train_2 = train_1.withColumn("x", train_1["oldX"].cast("float")).drop("oldX")
    train_2 = train_2.withColumn("label", train_1["oldLabel"].cast("float")).drop("oldLabel")
    train_2.cache()
    train_2.show()
    train_2.printSchema()

    train_2.printSchema()
    print(train_2.dtypes)
    train_2.describe().show()

    # Converting "features" column in a Vector column
    train_2 = VectorAssembler(inputCols=["x"], outputCol="feature").transform(train_2)
    train_2.printSchema()

    # Plotting Dataset
    f, axarr = plt.subplots(2, sharex=True)
    # Converting "features" DenseVector column to NPy Array
    npFeatures = np.array([])
    for i in train_2.collect():
        npFeatures = np.append(npFeatures, i['feature'].toArray())
    # Converting "label" DenseVector column to NPy Array
    npLabels = np.array([])
    for i in train_2.collect():
        npLabels = np.append(npLabels, i['label'])
    axarr[0].plot(npFeatures, npLabels, label="Data", linewidth=2)

    # Pipeline: Polynomial expansion, Linear Regression and label vs. prediction charts for every degree
    for degree in [5, 6, 7]:
        px = PolynomialExpansion(degree=degree, inputCol="feature", outputCol="features")
        lr = LinearRegression(maxIter=5)
        pipeline = Pipeline(stages=[px, lr])
        model = pipeline.fit(train_2)
        # lr.write.overwrite().save("D:\\Users\\festevem\Desktop\Modelos\modelo1")    # No va de ninguna manera
        npPredictions = np.array([])
        for i in model.transform(train_2).collect():
            npPredictions = np.append(npPredictions, (i['prediction']))
        # Model plot
        axarr[0].plot(npFeatures, npPredictions, label="Degree %d" % degree)
        print("Degree " + str(degree) + " model coefficients: " + str(model.stages[1].coefficients))
        print("Degree " + str(degree) + " model intercept: " + str(model.stages[1].intercept))
        print("Degree " + str(degree) + " model Mean Squared Error: " + str(model.stages[1].summary.meanSquaredError))
        print("Degree " + str(degree) + " model Mean Absolute Error: " + str(model.stages[1].summary.meanAbsoluteError))

    # Intentos de guardar
    #pipeline.save("D:\\Users\\festevem\Desktop\Modelos\pipeline1.1")
    #model.stages[1].write.overwrite().save("D:\\Users\\festevem\Desktop\Modelos\pipeline1")
    # model.stages[1].save("D:\\Users\\festevem\Desktop\Modelos\pipeline1")
    #pipeline.write.overwrite().save("D:\\Users\\festevem\Desktop\Modelos")

    # Print the coefficients and intercept for the last linear regression
    print("//////////////////////////////////////////")
    model.transform(train_2).show()
    print(model.transform(train_2))

    print("End")

    # Plot adjustments
    axarr[0].legend(loc='upper left')
    axarr[0].set_title('Actual Sells vs. Hypotesis')
    axarr[0].axis([9, 22, 0, 25])
    axarr[0].set_ylabel('Sells')
    # axarr[0].grid()
    axarr[1].plot(npFeatures, generateStock(npFeatures, npPredictions, 200), label="Stock", color='red')
    axarr[1].set_title('Stock Evolution')
    axarr[1].axis([9, 22, 0, 205])
    axarr[1].set_ylabel('Stock')
    axarr[1].set_xlabel('Time (h)')
    # axarr[1].grid()

    plt.show()
    spark.stop()
