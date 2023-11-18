package com.allaboutscala.chapter.one.tutorial_04

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer,Normalizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{split,udf,concat_ws}
import org.apache.spark.ml.classification.{RandomForestClassifier}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.lower
import org.apache.spark.sql.functions.col
import scala.util.matching.Regex

object Link_Prediction_Part1 {


  def sameAuthor(auth1: String, auth2: String): Int = {
   // val t1 = System.nanoTime
    if (auth1 == null || auth2 == null)
      return 0
    val tmp1 = auth1.split(",")
    val tmp2 = auth2.split(",")
    for (i <- 0 until tmp1.length) {
      for (j <- 0 until tmp2.length) {
        if (tmp1(i).compareTo(tmp2(j)) == 0) {
          return 1
        }
      }
    }
   // val duration = (System.nanoTime - t1) / 1e9d
    return 0
  }

  def sameJournal(Journal1: String, Journal2: String): Int = {
    // val t2 = System.nanoTime
    if (Journal1 == null || Journal2 == null)
      return 0
    if (Journal1.compareTo(Journal2) == 0) {
      return 1
    }
   // val duration = (System.nanoTime - t2) / 1e9d
    return 0
  }

  def distance(vector1: Vector, vector2: Vector): Double = {
    val tmp1 = vector1.toArray
    val tmp2 = vector2.toArray
    return tmp1.zip(tmp2).map(x => x._1 * x._2).reduce((a, b) => a + b)
  }

  // in this def function we get a dataframe and we use Tf-Idf at a specific column of our data. we remove stopwords
  def Tf_Idf2(x: DataFrame, column: String, columnOut: String, numFeatures: Int): DataFrame = {
    // val t4 = System.nanoTime
    val tokenzr = new Tokenizer().setInputCol(column).setOutputCol("words")
    val wData = tokenzr.transform(x.na.fill(Map(column -> "")))

    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("words1")
    val w1Data = remover.transform(wData)
    val hashTF = new HashingTF()
      .setInputCol("words1").setOutputCol("rawFeatures").setNumFeatures(numFeatures)
    val featData = hashTF.transform(w1Data)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol(columnOut)
    val idfmodel = idf.fit(featData)
    val rescaledData = idfmodel.transform(featData)
    return rescaledData.drop("words").drop("words1").drop("rawFeatures").drop(column)
  }

  // in main function we set the spark context, we load the node information-training set and test set.
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new spark.SparkConf().setAppName("Link Classification Application").setMaster("local[*]")
    val sc = new spark.SparkContext(conf)
    val sparkSession = SparkSession
      .builder()
      .appName("Spark SQL ")
      .config("spark.some.config.option", "some-value")
      .enableHiveSupport()
      .getOrCreate()
    import sparkSession.implicits._

    // load node information
    val dfHeaderl = sparkSession.read.format("csv").option("header", "false").load("samples\\node_information.csv")
    var dfNodeInform = dfHeaderl.toDF("Id", "Year", "Title", "Authors", "Journal", "Abstract")

    val columnName1: String = "Authors"
    dfNodeInform = dfNodeInform.withColumn(columnName1, lower(col(columnName1)))

    val columnName2: String = "Title"
    dfNodeInform = dfNodeInform.withColumn(columnName2, lower(col(columnName2)))

    val columnName3: String = "Journal"
    dfNodeInform = dfNodeInform.withColumn(columnName3, lower(col(columnName3)))
    // load training set
    val df1_2 = sparkSession.read.format("csv").option("header", "false").load("samples\\training_set.txt").toDF("info")
    // load testing set
    val df1_3 = sparkSession.read.format("csv").option("header", "false").load("samples\\testing_set.txt").toDF("info")
    // val Array(dfNoStop, dfNoStop1)=Array(dfHive, dfHive1).map(RemoveStopwords)

    def ProcessDataf(NodeInfo: DataFrame, NodeCombination: DataFrame): DataFrame = {
     // val t6 = System.nanoTime
      // break info into 3 columns Id1,Id2 and label
      val dfComSplit = NodeCombination.withColumn("_tmp", split($"info", " ")).select(
        $"_tmp".getItem(0).as("Id1"),
        $"_tmp".getItem(1).as("Id2"),
        $"_tmp".getItem(2).as("label")
      ).drop("_tmp")
      // join the information for every node with id=Id1.
      // rename column names
      // cast the type of column label (string to Int).
      val test_feat = dfComSplit.join(NodeInfo, $"Id1" === $"Id").select($"Id1", $"Id2", $"label", $"fAbstract".as("fAbstract1"),
        $"Authors".as("Authors1"), $"fTitle".as("Title1"), $"Journal".as("Journal1"))
        .withColumn("label", dfComSplit.col("label").cast(IntegerType))
      // join the information for every node with id=Id2 rename column names
      val firstDf = test_feat.join(NodeInfo, $"Id2" === $"Id").select($"Id1", $"Id2", $"label", $"fAbstract1", $"Authors1", $"Title1", $"Journal1",
        $"fAbstract".as("fAbstract2"), $"Authors".as("Authors2"), $"fTitle".as("Title2"), $"Journal".as("Journal2"))
      val dist = udf(distance _)
      val samAuthor = udf(sameAuthor _)
      val samJournal = udf(sameJournal _)
      // produce final dataframe calculate euclidean distance between abstracts and title and check if there is the same author and journal between 2 articles
      val final_df = firstDf.withColumn("Dist", dist($"fAbstract1", $"fAbstract2"))
        .withColumn("DistTitle", dist($"Title1", $"Title2"))
        .withColumn("sameAuthors", samAuthor($"Authors1", $"Authors2"))
        .withColumn("Id", concat_ws(" ", $"Id1", $"Id2"))
        .withColumn("sameJournal", samJournal($"Journal1", $"Journal2"))
        .select("Id", "label", "Dist", "DistTitle", "sameAuthors", "sameJournal")
      // dataframe's last form is Id,label,cosine distance and same authors.
      return final_df
    }

    // process for abstact
    val dfNodeInfoProst = Tf_Idf2(dfNodeInform, "Abstract", "fAbstract", 1000).drop("Abstract")
    val dfNodeInfoP = Tf_Idf2(dfNodeInfoProst, "Title", "fTitle", 1000).drop("Title")

    // process the dataframe
    val ProcDf = ProcessDataf(dfNodeInfoP, df1_2)
    val Test_ProcDf = ProcessDataf(dfNodeInfoP, df1_3)
    val assembler1 = new VectorAssembler().
      setInputCols(Array("Dist", "DistTitle", "sameAuthors", "sameJournal")).
      setOutputCol("features")
    val assembledFinalDf = assembler1.transform(ProcDf).select("Id", "features", "label")
    val assembledTestDf = assembler1.transform(Test_ProcDf).select("Id", "features", "label")

    // split the training for train and test set.
    val Array(training, test) = assembledFinalDf.randomSplit(Array[Double](0.7, 0.3), 18)
    val Array(dt_training_all) = assembledFinalDf.randomSplit(Array[Double](xs = 1.0), 18)
    val Array(dt_test) = assembledTestDf.randomSplit(Array[Double](1.0), 18)

    // random forest model
    val randomf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(30)
    val rf_train = randomf.fit(dt_training_all)
    val rf_predict = rf_train.transform(dt_test)
    val randomf_evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")
    val randomf_accuracy = randomf_evaluator.evaluate(rf_predict)
    println(s"RF_F1 score = ${randomf_accuracy}")

 }
}
