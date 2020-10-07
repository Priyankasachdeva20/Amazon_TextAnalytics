# Databricks notebook source
# Load in one of the tables
df1 = spark.sql("select * from default.video_games_5")
df2 = spark.sql("select * from default.home_and_kitchen_5_small")
df3 = spark.sql("select * from default.books_5_small")

#data = df1
#data = df1.union(df2)
data = df1.union(df2).union(df3)

data = data.sample(False, 0.01, seed=0)

data = data.cache()

print((data.count(), len(data.columns)))

# COMMAND ----------

# Check nulls
from pyspark.sql.functions import isnull, when, count, col
nacounts = data.select([count(when(isnull(c), c)).alias(c) for c in data.columns]).toPandas()
nacounts

# COMMAND ----------

data = data.dropna(thresh=2,subset=('summary','reviewerName'))

# COMMAND ----------

data = data.dropDuplicates(['reviewerID', 'asin'])

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql.types import FloatType

# Re-balancing (weighting) of records to be used in the logistic loss objective function
numPositives = data.filter(data["label"] == 1).count()
datasetSize = data.count()
balancingRatio = (datasetSize - numPositives) / datasetSize
print("numPositives   = {}".format(numPositives))
print("datasetSize    = {}".format(datasetSize))
print("balancingRatio = {}".format(balancingRatio))

def calculateWeights(d):
    if d == 1.0:
      return 1 * balancingRatio
    else:
      return 1 * (1.0 - balancingRatio)
    
udfcalculateWeights = f.udf(calculateWeights, FloatType())
    
data = data.withColumn("classWeightCol", udfcalculateWeights(data["label"]))
data.show(5)

# COMMAND ----------

display(data)

# COMMAND ----------

from sparknlp.base import*
from sparknlp.annotator import *
from nltk.corpus import *
from sparknlp.annotator import *

from sparknlp.base import DocumentAssembler, Finisher
from pyspark.ml.feature import StringIndexer, VectorIndexer
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Stemmer
from pyspark.ml.classification import MultilayerPerceptronClassifier, NaiveBayes, GBTClassifier, DecisionTreeClassifier

from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

# convert text column to spark nlp document
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")


# convert document to array of tokens
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")
 
# clean tokens 
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

#lemmatizer = LemmatizerModel.pretrained() \
 #    .setInputCols(['normalized']) \
  #   .setOutputCol('lemmatized')

# remove stopwords
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

# stems tokens to bring it to root form
stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")

# Convert custom document structure to array of tokens.
finisher = Finisher() \
    .setInputCols(["stem"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

# Generate Term Frequency
tf = CountVectorizer(inputCol="token_features", outputCol="rawFeatures", vocabSize=10000, minTF=1, minDF=50, maxDF=0.40)

# Generate Inverse Document Frequency weighting
idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures", minDocFreq=5)

# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["overall", "verified",  "idfFeatures"], outputCol="features")

# Index labels, adding metadata to the label column.
#labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel")

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
#featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4, handleInvalid="skip")

layers = [2337, 10, 5, 2]
# Machine Learning Algorithm
#ml_alg  = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.0)
#ml_alg  = RandomForestClassifier(numTrees=100).setLabelCol("label").setFeaturesCol("features")
#VERY POOR ON TEST DATA

ml_alg1=MultilayerPerceptronClassifier(maxIter=100, layers=layers, seed=1234).setLabelCol("label").setFeaturesCol("features")
ml_alg2=NaiveBayes(smoothing=1.0, modelType="multinomial", weightCol="classWeightCol")
#NB: 0.6401
ml_alg3=LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.0).setWeightCol("classWeightCol").setLabelCol("label").setFeaturesCol("features") 
#LR: 0.63
#ml_alg=GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10, seed=0)
#ml_alg = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",impurity="entropy", seed=1)


nlp_pipeline1 = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            normalizer,
            #lemmatizer,
            stopwords_cleaner, 
            stemmer, 
            finisher,
            tf,
            idf,
            assembler,
            #labelIndexer,
            #featureIndexer,
            ml_alg1])

nlp_pipeline2 = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            normalizer,
            #lemmatizer,
            stopwords_cleaner, 
            stemmer, 
            finisher,
            tf,
            idf,
            assembler,
            #labelIndexer,
            #featureIndexer,
            ml_alg2])

nlp_pipeline3 = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            normalizer,
            #lemmatizer,
            stopwords_cleaner, 
            stemmer, 
            finisher,
            tf,
            idf,
            assembler,
            #labelIndexer,
            #featureIndexer,
            ml_alg3])



# COMMAND ----------

(trainingData, testData) = data.randomSplit([0.8, 0.2], seed = 7)

# COMMAND ----------

pipeline_model_nn = nlp_pipeline1.fit(trainingData)

# COMMAND ----------

pipeline_model_nb = nlp_pipeline2.fit(trainingData)

# COMMAND ----------

pipeline_model_lr = nlp_pipeline3.fit(trainingData)

# COMMAND ----------

predictions_nn =  pipeline_model_nn.transform(testData)
display(predictions_nn)

# COMMAND ----------

predictions_nb =  pipeline_model_nb.transform(testData)
display(predictions_nb)

# COMMAND ----------

predictions_lr =  pipeline_model_lr.transform(testData)
display(predictions_lr)

# COMMAND ----------

print("NN  comparison of Label and Prediction")

predictions_nn.groupBy("label").count().show()
predictions_nn.groupBy("prediction").count().show()

# COMMAND ----------

print("NB  comparison of Label and Prediction")

predictions_nb.groupBy("label").count().show()
predictions_nb.groupBy("prediction").count().show()

# COMMAND ----------

print("LR  comparison of Label and Prediction")

predictions_lr.groupBy("label").count().show()
predictions_lr.groupBy("prediction").count().show()

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

acc_evaluator_nn = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
pre_evaluator_nn = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rec_evaluator_nn = MulticlassClassificationEvaluator(metricName="weightedRecall")
pr_evaluator_nn = BinaryClassificationEvaluator(metricName="areaUnderPR")
auc_evaluator_nn = BinaryClassificationEvaluator(metricName="areaUnderROC")

print("Test Accuracy       = %g" % (acc_evaluator_nn.evaluate(predictions_nn)))
print("Test Precision      = %g" % (pre_evaluator_nn.evaluate(predictions_nn)))
print("Test Recall         = %g" % (rec_evaluator_nn.evaluate(predictions_nn)))
print("Test areaUnderPR    = %g" % (pr_evaluator_nn.evaluate(predictions_nn)))
print("Test areaUnderROC   = %g" % (auc_evaluator_nn.evaluate(predictions_nn)))

# COMMAND ----------

acc_evaluator_nb = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
pre_evaluator_nb = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rec_evaluator_nb = MulticlassClassificationEvaluator(metricName="weightedRecall")
pr_evaluator_nb  = BinaryClassificationEvaluator(metricName="areaUnderPR")
auc_evaluator_nb = BinaryClassificationEvaluator(metricName="areaUnderROC")

print("Test Accuracy       = %g" % (acc_evaluator_nb.evaluate(predictions_nb)))
print("Test Precision      = %g" % (pre_evaluator_nb.evaluate(predictions_nb)))
print("Test Recall         = %g" % (rec_evaluator_nb.evaluate(predictions_nb)))
print("Test areaUnderPR    = %g" % (pr_evaluator_nb.evaluate(predictions_nb)))
print("Test areaUnderROC   = %g" % (auc_evaluator_nb.evaluate(predictions_nb)))

# COMMAND ----------

acc_evaluator_lr = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
pre_evaluator_lr = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rec_evaluator_lr = MulticlassClassificationEvaluator(metricName="weightedRecall")
pr_evaluator_lr  = BinaryClassificationEvaluator(metricName="areaUnderPR")
auc_evaluator_lr = BinaryClassificationEvaluator(metricName="areaUnderROC")

print("Test Accuracy       = %g" % (acc_evaluator_lr.evaluate(predictions_lr)))
print("Test Precision      = %g" % (pre_evaluator_lr.evaluate(predictions_lr)))
print("Test Recall         = %g" % (rec_evaluator_lr.evaluate(predictions_lr)))
print("Test areaUnderPR    = %g" % (pr_evaluator_lr.evaluate(predictions_lr)))
print("Test areaUnderROC   = %g" % (auc_evaluator_lr.evaluate(predictions_lr)))

# COMMAND ----------

# Reading Kaggle data from Database
Kaggle_test = spark.sql("select * from default.reviews_kaggle")

# COMMAND ----------

display(Kaggle_test)

# COMMAND ----------

predictions_nn2 =  pipeline_model_nn.transform(Kaggle_test)

# COMMAND ----------

predictions_nb2 =  pipeline_model_nb.transform(Kaggle_test)

# COMMAND ----------

predictions_lr2 =  pipeline_model_lr.transform(Kaggle_test)

# COMMAND ----------

pred1=predictions_nn2.select("reviewID","prediction").withColumnRenamed('prediction','prediction_nn')
display(pred1)

# COMMAND ----------

pred2=predictions_nb2.select("reviewID","prediction").withColumnRenamed('prediction','prediction_nb')
display(pred2)

# COMMAND ----------

pred3=predictions_lr2.select("reviewID","prediction").withColumnRenamed('prediction','prediction_lr')
display(pred3)

# COMMAND ----------

df=pred1.join(pred2, "reviewID", "inner").join(pred3, "reviewID", "inner")

# COMMAND ----------

df.head(5)

# COMMAND ----------

df=df.withColumn("VotingPred", df['prediction_nb']+df['prediction_nn']+df['prediction_lr'])

# COMMAND ----------

sums = [F.sum(x).alias(str(x)) for x in df.columns]
d = df.select(sums).collect()[0].asDict()

# COMMAND ----------

from pyspark.sql import functions as F
df = df.withColumn("FinalPrediction", F.when((df.VotingPred>=1),1).otherwise(0))

# COMMAND ----------

display(df.select('FinalPrediction'))

# COMMAND ----------


