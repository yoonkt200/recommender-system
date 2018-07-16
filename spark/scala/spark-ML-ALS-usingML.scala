import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{functions => F}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.types.{IntegerType, DoubleType, FloatType}
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.sql.Row

var df = spark.read.format("csv").option("header", "true").load(base_path + "{your_data_path}")
val Array(training, test) = df.randomSplit(Array(0.8, 0.2))

/* 
*
* Build the recommendation model using ALS on the training data
*
*/

val als = new ALS()
  .setMaxIter(20) // Iteration
  .setRegParam(0.05) // 정규화 파라미터
  .setImplicitPrefs(true) // true : cost function을 정의하는 방식을 Implitcit Data에 맞는 방식으로 변경. 
                          // --> sum(confidence * (preference - prediction)^2 + regularization parameter)
  .setRank(10) // latent 요소 개수 '=. 내재적 피처의 개수
  .setAlpha(1.0) // Implicit ALS에서 신뢰도 계산에 사용되는 상수
  .setUserCol("userid")
  .setItemCol("itemid")
  .setRatingCol("rating")
  .setNonnegative(true)
val model = als.fit(training)

// Evaluate the model by computing the RMSE on the test data
// Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
model.setColdStartStrategy("drop")
val predictions = model.transform(test)

val evaluator = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("rating")
  .setPredictionCol("prediction")
val rmse = evaluator.evaluate(predictions)
println(s"Root-mean-square error = $rmse")

// Generate top 10 recommendations for each user
var userRecs = model.recommendForAllUsers(10)
// Generate top 10 recommendations for each item
var itemRecs = model.recommendForAllItems(10)


/* 
*
* Data flatten and get personal recommendation list
*
*/


// data flatten
var userRecsFlatten = userRecs.withColumn("new", F.explode($"recommendations")).select("userid", "new.*")

userRecsFlatten.show(5)
+---------+--------+---------+
|   userid|  itemid|   rating|
+---------+--------+---------+
|      112|   81212|0.8959758|
|      112|   03627|0.8019955|
|      113|   20144|0.7881651|
|      113|   60643|0.7608514|
|      113|   71231|0.7271776|
+---------+--------+---------+