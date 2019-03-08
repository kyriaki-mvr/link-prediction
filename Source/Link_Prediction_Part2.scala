package allaboutscala.shapter.one.tutorial_04

import org.apache.log4j.{Level, Logger}
import org.apache.spark
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{split, udf}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.storage.StorageLevel
import org.apache.spark.ml.feature.MinHashLSH
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

object Link_Prediction_Part2 {

  def YearDiff(Year1: Int, Year2: Int):Int={
    return math.abs(Year1 - Year2)
  }

  def getSrcId(Year1: Int, Year2: Int,Id1:String,Id2:String):String={
    if(Year1<Year2){
      return Id2
    }else{
      return Id1
    }
  }

  def getDistanceId(Year1: Int, Year2: Int,Id1:String,Id2:String):String={
    if(Year1<Year2){
      return Id1
    }else{
      return Id2
    }
  }

  def Setlabel(cluster1: Int, cluster2: Int,Year1: Int, Year2: Int):Int={
    if(cluster1==cluster2){
      return 1
    }else{
      return 0
    }
  }

  def sameAuthor(auth1: String,auth2: String):Int={
    if(auth1==null || auth2 ==null)
      return 0
    val temp1=auth1.split(",")
    val temp2=auth2.split(",")
    var sameAuthors=0
    for( i <- 0 until temp1.length){
      for(j <-0 until temp2.length){
        if(temp1(i).compareTo(temp2(j))==0){
          sameAuthors=sameAuthors+1
        }
      }
    }
    return sameAuthors
  }

  //In this def function we get a dataframe and we use Tf-Idf at a specific column of our data.we get tokens and then we rescale
  def Tf_Idf(x: DataFrame, column: String, columnOut: String, numFeatures: Int): DataFrame = {
    val tokenzr = new Tokenizer().setInputCol(column).setOutputCol("words")
    val wData = tokenzr.transform(x.na.fill(Map(column -> "")))
    val hashTF = new HashingTF()
      .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(numFeatures)
    val featData = hashTF.transform(wData)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol(columnOut)
    val idfmodel = idf.fit(featData)
    val rescaledData = idfmodel.transform(featData)
    return rescaledData.drop("words").drop("rawFeatures").drop(column)
  }
  //In this def function we get a dataframe and we use Tf-Idf at a specific column of our data.we get tokens we remove stopwords and then we rescale
  def Tf_Idf2(x: DataFrame, column: String, columnOut: String, numFeatures: Int): DataFrame = {
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

  //in main function we set the spark context, we load the node information-training and we find linked papers and F1 score
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new spark.SparkConf().setAppName("Link Classification Application").setMaster("local[*]")
    val sc = new spark.SparkContext(conf)
    val sparkSession = SparkSession
      .builder()
      .appName("Spark SQL basic example")
      .config("spark.some.config.option", "some-value")
      .enableHiveSupport()
      .getOrCreate()

    import sparkSession.implicits._
// read the data
    val dfHeaderl = sparkSession.read.format("csv").option("header", "false").load("C:\\Users\\dell\\Desktop\\node_information.csv")
    val dfNodeInform = dfHeaderl.toDF("Id", "Year1", "Title", "Authors", "Journal", "Abstract").withColumn("Year",$"Year1".cast(IntegerType))
    val dfNodeInfoProst=Tf_Idf(dfNodeInform,"Title","fTitle",1000).drop("Title")
    val dfNodeInfoP=Tf_Idf2(dfNodeInfoProst,"Abstract","fAbstract",10).drop("Abstract")
    val assem_Hash = new VectorAssembler().
      setInputCols(Array("fTitle"))
      .setOutputCol("hashVector")
    val hash_Df=assem_Hash.transform(dfNodeInfoP)
    val assembler1 = new VectorAssembler().
      setInputCols(Array("Year")).
      setOutputCol("featuresTmp")
    val assembledDf = assembler1.transform(hash_Df)
    //clustering, seed<10 in our case?
    val kmeans = new KMeans().setK(3).setSeed(1L).setFeaturesCol("featuresTmp")
    val cluster = kmeans.fit(assembledDf)

    // Make predictions
    val predict = cluster.transform(assembledDf)
    //predictions.show()

    /*
    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator().setFeaturesCol("featuresTmp")
    val silhouette = evaluator.evaluate(predict)
    println(s"Silhouette = $silhouette " )
*/
    val minh = new MinHashLSH()
      .setNumHashTables(45)
      .setInputCol("hashVector")
      .setOutputCol("hashes")
    val model = minh.fit(predict)

    // Feature Transformation
    val hashedDf=model.transform(predict)
    val sAuthor= udf(sameAuthor _)
    val difYear = udf(YearDiff _ )
    val setLabel = udf(Setlabel _)
    val getDst = udf(getDistanceId _)
    val getSrc = udf(getSrcId _)
    val c=model.approxSimilarityJoin(hashedDf, hashedDf, 0.7, "EuclideanDistance")
      .select($"datasetA.Id".alias("idA"),
        $"datasetB.Id".alias("idB"),
        $"datasetA.prediction".alias("cluster1"),
        $"datasetB.prediction".alias("cluster2"),
        $"EuclideanDistance",
        $"datasetA.Year".alias("YearA"),
        $"datasetB.Year".alias("YearB"))
      .filter($"idA"=!=$"idB")
      .withColumn("label",setLabel($"cluster1",$"cluster2",$"YearA",$"YearB"))
      .withColumn("IdSrc",getSrc($"YearA",$"YearB",$"IdA",$"IdB"))
      .withColumn("IdDst",getDst($"YearA",$"YearB",$"IdA",$"IdB"))
    c.persist(StorageLevel.MEMORY_AND_DISK)

    val Right1=c.filter($"label"===1).count()
    println("Evaluation start now ")
    val groundTruth = sparkSession.read.format("csv").option("header", "false").load("C:\\Users\\dell\\Desktop\\Cit-HepTh.txt").toDF("Id").withColumn("_tmp", split($"Id", "\t")).select(
      $"_tmp".getItem(0).as("Id1"),
      $"_tmp".getItem(1).as("Id2")).drop("_tmp")
    val alledges=groundTruth.count()
    println("all the edges from ground truth:",alledges)
    val eval=groundTruth.join(c,($"IdSrc"===$"Id1"&&$"IdDst"===$"Id2"))
    val Right2=eval.filter($"label"===1).count()
    println(Right2,alledges)
    val recall = (Right2).toDouble / alledges
    val precision = (Right2).toDouble/Right1
    println(2*(precision*recall/(precision+recall)))
  }
}