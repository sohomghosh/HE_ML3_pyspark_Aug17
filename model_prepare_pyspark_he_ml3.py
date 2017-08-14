#./pyspark --master yarn --driver-memory 3g --executor-memory 6g --executor-cores 2 --num-executors 3

from pyspark.sql.types import StructType,StructField,LongType,StringType,TimestampType,ArrayType,IntegerType,FloatType
from pyspark.sql.functions import lit,udf,col,max,min,hour,minute,isnan, when, count
from pyspark.ml.feature import OneHotEncoder,VectorAssembler,StringIndexer
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.sql.functions import monotonically_increasing_id
import gc
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors


#|       ID|           datetime| siteid|offerid|category|merchant|countrycode|      browserid|  devid|click|
schema_train=StructType([StructField('ID', StringType(), False), StructField('datetime', TimestampType(), True), StructField('siteid', LongType(), True), StructField('offerid', LongType(), True),StructField('category',LongType(),True),StructField('merchant',LongType(),True),StructField('countrycode',StringType(),True),StructField('browserid',StringType(),True),StructField('devid',StringType(),True),StructField('click',IntegerType(),False)])
schema_test=StructType([StructField('ID', StringType(), False), StructField('datetime', TimestampType(), True), StructField('siteid', LongType(), True), StructField('offerid', LongType(), True),StructField('category',LongType(),True),StructField('merchant',LongType(),True),StructField('countrycode',StringType(),True),StructField('browserid',StringType(),True),StructField('devid',StringType(),True)])


train=spark.read.csv("/index/sohom_exeriment/HE_ML3/train.csv",header=True,schema=schema_train)
test=spark.read.csv("/index/sohom_exeriment/HE_ML3/test.csv",header=True,schema=schema_test)
gc.collect()

test=test.withColumn('click', lit(None).cast(StringType()))

#train.count()
#12137810

#test.count()
#3706907

train_test=train.union(test)

#train_test.count()
#15844717


#train.groupby(train.offerid).count().orderBy('count',ascending=False).show(100)
#train.groupby(train.offerid).count().count()
#847510

#test.groupby(test.offerid).count().orderBy('count',ascending=False).show(100)
#test.groupby(test.offerid).count().count()
#556519

#train.groupby(train.siteid).count().count()
#219174                                                                          

#test.groupby(test.siteid).count().count()
#84893

#train.groupby(train.category).count().count()
#271 

#test.groupby(test.category).count().count()
#267

#train.groupby(train.merchant).count().count()
#697

#test.groupby(test.merchant).count().count()
#650

#train.groupby(train.countrycode).count().count()
#6                                                                               

#test.groupby(test.countrycode).count().count()
#6

#train.groupby(train.countrycode).count().show()
'''
+-----------+-------+                                                           
|countrycode|  count|
+-----------+-------+
|          f|1190584|
|          e|1194449|
|          d| 717281|
|          c| 804218|
|          b|5285881|
|          a|2945397|
+-----------+-------+
'''

#test.groupby(test.countrycode).count().show()
'''
+-----------+-------+                                                           
|countrycode|  count|
+-----------+-------+
|          f| 356103|
|          e| 357068|
|          d| 252236|
|          c| 306111|
|          b|1561115|
|          a| 874274|
+-----------+-------+
'''

#train.groupby(train.browserid).count().count()
#12                                                                              

#test.groupby(test.browserid).count().count()
#12                                                                              

#train.groupby(train.browserid).count().show()
'''
+-----------------+-------+                                                     
|        browserid|  count|
+-----------------+-------+
|Internet Explorer| 230818|
|             null| 608327|
| InternetExplorer| 743821|
|          Firefox|3347105|
|          Mozilla|1120649|
|  Mozilla Firefox|1008609|
|           Safari| 114957|
|    Google Chrome| 700571|
|               IE| 346042|
|           Chrome| 345432|
|             Edge|3456150|
|            Opera| 115329|
+-----------------+-------+
'''


#test.groupby(test.browserid).count().show()
'''
+-----------------+-------+                                                     
|        browserid|  count|
+-----------------+-------+
|Internet Explorer|  69845|
|             null| 221906|
| InternetExplorer| 275252|
|          Firefox| 980698|
|          Mozilla| 328501|
|  Mozilla Firefox| 295403|
|           Safari|  34987|
|    Google Chrome| 249938|
|               IE| 104036|
|           Chrome| 104442|
|             Edge|1007035|
|            Opera|  34864|
+-----------------+-------+
'''


#train.groupby(train.devid).count().count()
#4

#test.groupby(test.devid).count().count()
#4

#train.groupby(train.devid).count().show()
'''
+-------+-------+                                                               
|  devid|  count|
+-------+-------+
|   null|1820299|
| Mobile|4035596|
| Tablet|3403479|
|Desktop|2878436|
+-------+-------+
'''

#test.groupby(test.devid).count().show()
'''
+-------+-------+                                                               
|  devid|  count|
+-------+-------+
|   null| 704619|
| Mobile|1180094|
| Tablet| 958157|
|Desktop| 864037|
+-------+-------+
'''

#train.groupby(train.click).count().show()
'''
+-----+--------+                                                                
|click|   count|
+-----+--------+
|    1|  437214|
|    0|11700596|
+-----+--------+
>>> 437214/(437214+11700596)
0.036020830775897794
###########ONLY 3% clicked###########
'''

#train.select(max("datetime")).show(truncate=False)
'''
+---------------------+                                                         
|max(datetime)        |
+---------------------+
|2017-01-20 23:59:54.0|
+---------------------+
'''

#train.select(min("datetime")).show(truncate=False)
'''
+---------------------+                                                         
|min(datetime)        |
+---------------------+
|2017-01-10 02:00:17.0|
+---------------------+
'''

#test.select(max("datetime")).show(truncate=False)
'''
+---------------------+                                                         
|max(datetime)        |
+---------------------+
|2017-01-23 23:52:51.0|
+---------------------+
'''

#test.select(min("datetime")).show(truncate=False)
'''
+---------------------+                                                         
|min(datetime)        |
+---------------------+
|2017-01-21 00:00:02.0|
+---------------------+
'''
#Check if any id common between train & test

####Features
#siteid|offerid|category|merchant :: Keep as it is; As too many values

'''
df = spark.createDataFrame([
    (0, "a"),
    (1, "b"),
    (2, "c"),
    (3, "a"),
    (4, "a"),
    (5, "c")
], ["id", "category"])

stringIndexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)

encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
encoded = encoder.transform(indexed)
encoded.show()

pyspark.ml.feature.IndexToString to reverse the numeric indices back to the original categorical values (which are often strings) at any time.

encoder = OneHotEncoder(inputCol="gender_numeric", outputCol="gender_vector")
encoded_df = encoder.transform(indexed_df)
encoded_df.drop("bar").show()

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 10 distinct values are treated as continuous.
featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=10).fit(data)
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)


# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only

'''



##Browsers merge
def browsers_merge(brid):
	if brid=='Internet Explorer' or brid=='InternetExplorer' or brid=='IE':
		return 1
	elif brid=='Firefox' or brid=='Mozilla' or brid =='Mozilla Firefox':
		return 2
	elif brid=='Google Chrome' or brid=='Chrome':
		return 3
	elif brid=='Safari':
		return 4
	elif brid=='Edge':
		return 5
	elif brid=='Opera':
		return 6
	else:
		return np.nan


udf_browsers_merge=udf(browsers_merge,IntegerType())
train_test=train_test.withColumn('browserid_merged',udf_browsers_merge(col('browserid')))

##** Extract features from datetime: hr,sec,hr*60+sec
train_test=train_test.withColumn('hour_of_day',hour("datetime")).withColumn('minute_of_hour',minute("datetime")).withColumn('hour_min',col('hour_of_day')*60+col('minute_of_hour'))

def countrycode_encode(country):
	if country=='a':
		return 1
	elif country=='b':
		return 2
	elif country=='c':
		return 3
	elif country=='d':
		return 4
	elif country=='e':
		return 5
	elif country=='f':
		return 6
	else:
		return np.nan


udf_countryencode=udf(countrycode_encode,IntegerType())
train_test=train_test.withColumn('encoded_country',udf_countryencode(col('countrycode')))

def devid_encode(devid):
	if devid=='Mobile':
		return 1
	elif devid=='Tablet':
		return 2
	elif devid=='Desktop':
		return 3
	else:
		return np.nan


udf_devid_encode=udf(devid_encode,IntegerType())
train_test=train_test.withColumn('devid_encode',udf_devid_encode(col('devid')))

#train_test.where(train_test.siteid.isNull()).count()
#1583291

#train_test.where(train_test.offerid.isNull()).count()
#0

#train_test.where(train_test.category.isNull()).count()
#0

#train_test.where(train_test.merchant.isNull()).count()
#0

#train_test.where(train_test.browserid_merged.isNull()).count()
#830233

#train_test.where(train_test.hour_of_day.isNull()).count()
#0

#train_test.where(train_test.minute_of_hour.isNull()).count()
#0

#train_test.where(train_test.hour_min.isNull()).count()
#0

#train_test.where(train_test.encoded_country.isNull()).count()
#0

#train_test.where(train_test.devid_encode.isNull()).count()
#2524918

#train_test.groupby('browserid_merged').count().orderBy('count').show(20)
'''
+----------------+-------+                                                      
|browserid_merged|  count|
+----------------+-------+
|               4| 149944|
|               6| 150193|
|            null| 830233|
|               3|1400383|
|               1|1769814|
|               5|4463185|
|               2|7080965|
+----------------+-------+
'''



train_test.groupby(train_test.siteid).count().orderBy('count',ascending=False).show(10)
#3696590 is the most ocurring site

#Missing value treatment with most ocurring
train_test=train_test.na.fill({'siteid':3696590,'browserid_merged':2, 'devid_encode':1})
#devid_encode:1 is Mobile


cols_now=["siteid","offerid","category","merchant","browserid_merged","hour_of_day","minute_of_hour","hour_min","encoded_country","devid_encode"]
assembler_features = VectorAssembler(inputCols=cols_now, outputCol='features')
labelIndexer = StringIndexer(inputCol='click', outputCol="label")
pipeline_train = Pipeline(stages=[assembler_features,labelIndexer])


train_test_use=train_test.select(["id","siteid","offerid","category","merchant","browserid_merged","hour_of_day","minute_of_hour","hour_min","encoded_country","devid_encode","click"])
train_test_use=train_test_use.withColumn("row_id", monotonically_increasing_id())

train_ids=train.select('id')
##########train_use=train_test_use.limit(train.count())
##########train_use=train_use.drop('row_id')
train_use=train_test_use.join(train_ids,train_test_use.id==train_ids.id,'inner').drop(train_ids.id).orderBy(col('row_id'))
train_use=train_use.drop('row_id')

train_use.count()
train_use.na.drop().count()
#These 2 counts should match

test_ids=test.select('id')
test_use=train_test_use.join(test_ids,train_test_use.id==test_ids.id,'inner').drop(test_ids.id).orderBy(col('row_id'))
test_use=test_use.drop('row_id').drop('click')

test_use.count()
test_use.na.drop().count()
#These 2 counts should match

#TAKES MORE TIME [AVOID THIS]
#test_row_ids=[i for i in range(train.count(),train_test_use.count())]
#test_use=train_test_use.filter(train_test_use.row_id.isin(test_row_ids)).orderBy(col('row_id'))
#train_use=train_test_use.join(train_ids,train_test_use.id==train_ids.id,'inner').drop(train_ids.id)

##OR ANOTHER METHOD
#test_use=train_test_use.subtrcat(train_use).orderBy(col('row_id')).drop(col('row_id'))

for i in cols_now:
	print(i,train_use.where(col(i).isNull()).count())

train_use.select([count(when(isnan(c), c)).alias(c) for c in train_use.columns]).show()
train_use.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in train_use.columns]).show()


###NOT RECOMMENDED: Dropping 42% values by this
##############################train_use=train_use.na.drop() #Droping na values: 
#train_use=train_test_use.limit(2482)

trainingData = pipeline_train.fit(train_use).transform(train_use)


#RandomForest Regressor
#You can feed your categorical features into a OneHotEncoder. When your new encoded feature is fed through the classifier, it will be treated as categorical FOR ML
rf=RandomForestRegressor(labelCol='label', featuresCol='features',numTrees=20,featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=4, maxBins=32,minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, subsamplingRate=1.0, seed=42)


#RandomForest Classifier
rf = RandomForestClassifier(labelCol='label', featuresCol='features',numTrees=20)
fit = rf.fit(trainingData)
print(fit.featureImportances)

####Predict
pipeline_test=Pipeline(stages=[assembler_features])

###NEED TO CHANGE IT
####NOT RECOMMENDED; DROPPING 31% VALUES
###test_use=test_use.na.drop()

testData=pipeline_test.fit(test_use).transform(test_use)
transformed = fit.transform(testData)

#transformed has 3 notable columns:  rawPrediction,probability and prediction
#predictions=transformed.select(transformed.prediction)

predictions=transformed.select(transformed.probability)
udf_second_element=udf(lambda x:float(x[1]),FloatType())#2nd element of probability select i.e the element at 1st position
ans=transformed.select([col('id'),udf_second_element('probability').alias('click')])

ans.count()
#3706907
#This should match with test.count()

ans.write.csv(path="/index/sohom_exeriment/HE_ML3/predictions_rf")


###Others
#** Smote ONLY 3% clicked



#############################TRYING ONE HOT ENCODED FEATURES#####################
train_test=train.union(test)

#train_test.columns
#['ID', 'datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid', 'click']

udf_browsers_merge=udf(browsers_merge,IntegerType())
train_test=train_test.withColumn('browserid_merged',udf_browsers_merge(col('browserid')))

train_test=train_test.na.fill({'siteid':3696590,'browserid_merged':2, 'devid':'Mobile'})

##** Extract features from datetime: hr,sec,hr*60+sec
train_test=train_test.withColumn('hour_of_day',hour("datetime")).withColumn('minute_of_hour',minute("datetime")).withColumn('hour_min',col('hour_of_day')*60+col('minute_of_hour'))

#train_test.columns
#['ID', 'datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid', 'click', 'browserid_merged', 'hour_of_day', 'minute_of_hour', 'hour_min']

categorical_cols = ['category', 'countrycode', 'devid','merchant','browserid_merged']

indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)) for c in categorical_cols]

encoders = [StringIndexer(inputCol=indexer.getOutputCol(), outputCol="{0}_encoded".format(indexer.getOutputCol())) for indexer in indexers]

onehotvector = [OneHotEncoder(inputCol=encoder.getOutputCol(), outputCol="{0}_vector".format(encoder.getOutputCol())) for encoder in encoders]
#onehotvector.transform(train_test)

pipeline = Pipeline(stages=indexers + encoders + onehotvector)
train_test_encoded_vector=pipeline.fit(train_test).transform(train_test)

#train_test_encoded_vector.columns
#['ID', 'datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid', 'click', 'hour_of_day', 'minute_of_hour', 'hour_min', 'browserid_merged', 'category_indexed', 'countrycode_indexed', 'devid_indexed', 'merchant_indexed', 'browserid_merged_indexed', 'category_indexed_encoded', 'countrycode_indexed_encoded', 'devid_indexed_encoded', 'merchant_indexed_encoded', 'browserid_merged_indexed_encoded', 'category_indexed_encoded_vector', 'countrycode_indexed_encoded_vector', 'devid_indexed_encoded_vector', 'merchant_indexed_encoded_vector', 'browserid_merged_indexed_encoded_vector']


#encoder = OneHotEncoder(inputCol="devid_encode", outputCol="devid_encodedVec")
#encoded = encoder.transform(train_test)
#encoded.show()

cols_now=["siteid","offerid","category_indexed_encoded_vector","merchant_indexed_encoded_vector","countrycode_indexed_encoded_vector","browserid_merged_indexed_encoded_vector","hour_of_day","minute_of_hour","hour_min","devid_indexed_encoded_vector"]





assembler_features = VectorAssembler(inputCols=cols_now, outputCol='features')
labelIndexer = StringIndexer(inputCol='click', outputCol="label")
pipeline_train = Pipeline(stages=[assembler_features,labelIndexer])


train_test_use=train_test_encoded_vector.select(["ID","siteid","offerid","category_indexed_encoded_vector","merchant_indexed_encoded_vector","countrycode_indexed_encoded_vector","browserid_merged_indexed_encoded_vector","hour_of_day","minute_of_hour","hour_min","devid_indexed_encoded_vector","click"])
train_test_use=train_test_use.withColumn("row_id", monotonically_increasing_id())

train_ids=train.select('id')
##########train_use=train_test_use.limit(train.count())
##########train_use=train_use.drop('row_id')
train_use=train_test_use.join(train_ids,train_test_use.ID==train_ids.id,'inner').drop(train_test_use.ID).orderBy(col('row_id'))
train_use=train_use.drop('row_id')

train_use.count()
train_use.na.drop().count()
#These 2 counts should match

test_ids=test.select('id')
test_use=train_test_use.join(test_ids,train_test_use.ID==test_ids.id,'inner').drop(train_test_use.ID).orderBy(col('row_id'))
test_use=test_use.drop('row_id').drop('click')

test_use.count()
test_use.na.drop().count()


trainingData = pipeline_train.fit(train_use).transform(train_use)

#trainingData.columns
#| siteid|offerid|category_indexed_encoded_vector|merchant_indexed_encoded_vector|countrycode_indexed_encoded_vector|browserid_merged_indexed_encoded_vector|hour_of_day|minute_of_hour|hour_min|devid_indexed_encoded_vector|click|       ID|            features|label|


####Predict
pipeline_test=Pipeline(stages=[assembler_features])

###NEED TO CHANGE IT
####NOT RECOMMENDED; DROPPING 31% VALUES
###test_use=test_use.na.drop()

testData=pipeline_test.fit(test_use).transform(test_use)


#############################################################################################################################
########################################RANDOM FOREST REGRESSOR#############################################################
#############################################################################################################################

#RandomForest Regressor
#You can feed your categorical features into a OneHotEncoder. When your new encoded feature is fed through the classifier, it will be treated as categorical FOR ML
rf=RandomForestRegressor(labelCol='label', featuresCol='features',numTrees=20,featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=4, maxBins=32,minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, subsamplingRate=1.0, seed=42)



fit = rf.fit(trainingData)
print(fit.featureImportances)

transformed = fit.transform(testData)

#transformed has 3 notable columns:  rawPrediction,probability and prediction
#predictions=transformed.select(transformed.prediction)

predictions=transformed.select(transformed.prediction)
#udf_second_element=udf(lambda x:float(x[1]),FloatType())#2nd element of probability select i.e the element at 1st position
ans=transformed.select([col('id'),col('prediction').alias('click')])


#############################################################################################################################
########################################RANDOM FOREST CLASSIFIER#############################################################
#############################################################################################################################

rf = RandomForestClassifier(labelCol='label', featuresCol='features',numTrees=20)
fit = rf.fit(trainingData)
print(fit.featureImportances)

####Predict
pipeline_test=Pipeline(stages=[assembler_features])

###NEED TO CHANGE IT
####NOT RECOMMENDED; DROPPING 31% VALUES
###test_use=test_use.na.drop()

testData=pipeline_test.fit(test_use).transform(test_use)
transformed = fit.transform(testData)

#transformed has 3 notable columns:  rawPrediction,probability and prediction
#predictions=transformed.select(transformed.prediction)

predictions=transformed.select(transformed.probability)
udf_second_element=udf(lambda x:float(x[1]),FloatType())#2nd element of probability select i.e the element at 1st position
ans=transformed.select([col('id'),udf_second_element('probability').alias('click')])


#############################################################################################################################
######################################################  GBM REGRESSOR #######################################################
#############################################################################################################################


#http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.GBTRegressor
gbt = GBTRegressor(featuresCol="features", labelCol="label", maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, subsamplingRate=1.0, checkpointInterval=10, lossType="squared", maxIter=20, stepSize=0.1, seed=None, impurity="variance")
fit=gbt.fit(trainingData)
transformed = fit.transform(testData)

predictions=transformed.select(transformed.prediction)
ans=transformed.select([col('id'),col('prediction').alias('click')])


#############################################################################################################################
######################################################  GBM CLASSIFIER ######################################################
#############################################################################################################################


#http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.GBTClassifier
gbt = GBTClassifier(featuresCol="features", labelCol="label",  maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, lossType="logistic", maxIter=20, stepSize=0.1, seed=None, subsamplingRate=1.0)
fit = gbt.fit(trainingData)
transformed = fit.transform(testData)
predictions=transformed.select(transformed.probability)
udf_second_element=udf(lambda x:float(x[1]),FloatType())#2nd element of probability select i.e the element at 1st position
ans=transformed.select([col('id'),udf_second_element('probability').alias('click')])


#############################################################################################################################
######################################################  MLP CLASSIFIER ######################################################
#############################################################################################################################

#http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.MultilayerPerceptronClassifier

#mlp=MultilayerPerceptronClassifier(featuresCol="features", labelCol="label", predictionCol="prediction", maxIter=100, tol=1e-6, seed=None, layers=None, blockSize=128, stepSize=0.03, solver="l-bfgs", initialWeights=None)

# specify layers for the neural network:
# input layer of size 989 (features), two intermediate of size 5 and 4
# and output of size 2 (classes)
# input layer = 989 get by checking testData.select(col('features')) ; See the first element of the bracket (
#+--------------------+                                                          
#|            features|
#+--------------------+
#|(989,[0,1,6,273,9...|

layers = [989, 5, 4, 2]

mlp=MultilayerPerceptronClassifier(featuresCol="features", labelCol="label",maxIter=100, layers=layers, blockSize=128, seed=1234)
fit = mlp.fit(trainingData)


transformed = fit.transform(testData)
predictions=transformed.select(transformed.prediction)
ans=transformed.select([col('id'),col('prediction')])


#save a model for later use
#model.save(path_to_model)  ###HERE model=mlp.fit(trainingData) so using fit
#model2 = MultilayerPerceptronClassificationModel.load(path_to_model) ###LOADING THE MODEL LATER FOR USE

#############################################################################################################################
######################################################  PARAM GRID ##########################################################
#############################################################################################################################

#PARAM GRID
#http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.ParamGridBuilder

rf=rf = RandomForestClassifier(labelCol='label', featuresCol='features')
grid = ParamGridBuilder().addGrid(rf.numTrees, [10, 20]).build()
evaluator = BinaryClassificationEvaluator()
cv = CrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator)
cvModel = cv.fit(trainingData)
cvModel.avgMetrics[0]
#0.9284708574892995

acry=evaluator.evaluate(cvModel.transform(trainingData))
#0.9345678816213042

#Not over here
acry=evaluator.evaluate(cvModel.transform(validationData))

#Best parameters from cross validayor
print(cvModel.bestModel)

# Make predictions on test test. cvModel uses the best model found in rf.
transformed=cvModel.transform(testData)
ans_prediction=transformed.select([col('id'),col('prediction')])

predictions=transformed.select(transformed.probability)
udf_second_element=udf(lambda x:float(x[1]),FloatType())#2nd element of probability select i.e the element at 1st position
ans=transformed.select([col('id'),udf_second_element('probability').alias('click')])


#############################################################################################################################
######################################################  STACKING ##########################################################
#############################################################################################################################

#Meta classifier: Train a logistic model: Taking ouputs of previous models as input; output should be the label; Remember: The predictions of the models should not be correlated

