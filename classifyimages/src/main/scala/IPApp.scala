import java.nio.file.{Files, Paths}

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, RandomForestModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.bytedeco.javacpp.opencv_highgui._

import scala.collection.mutable
/**
  * Created by NAVEENA on 16-02-2017.
  */
object IPApp {

  val featureVectorCluster = new mutable.MutableList[String]

  val IMAGE_INPUT = List("bulldog", "germanshep", "labdrador","yorkshire")

  def extractDescriptors(sc: SparkContext, images: RDD[(String, String)]): Unit = {

    if (Files.exists(Paths.get(IPsettings.FEATURES_PATH))) {
      println(s"${IPsettings.FEATURES_PATH} exists, skipping feature extraction..")
      return
    }

    val data = images.map {
      case (name, contents) => {
        val desc = ImageUtils.descriptors(name.split("file:/")(1))
        val list = ImageUtils.matToString(desc)
        println("-- " + list.size)
        list
      }
    }.reduce((x, y) => x ::: y)

    val featuresSeq = sc.parallelize(data)

    featuresSeq.saveAsTextFile(IPsettings.FEATURES_PATH)
    println("Total size : " + data.size)
  }

  def kMeansCluster(sc: SparkContext): Unit = {
    if (Files.exists(Paths.get(IPsettings.KMEANS_PATH))) {
      println(s"${IPsettings.KMEANS_PATH} exists, skipping clusters formation..")
      return
    }
    val data = sc.textFile(IPsettings.FEATURES_PATH)
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble)))
    val numClusters = 400
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    clusters.save(sc, IPsettings.KMEANS_PATH)
    println(s"Saves Clusters to ${IPsettings.KMEANS_PATH}")
    sc.parallelize(clusters.clusterCenters.map(v => v.toArray.mkString(" "))).saveAsTextFile(IPsettings.KMEANS_CENTERS_PATH)
  }

  def createHistogram(sc: SparkContext, images: RDD[(String, String)]): Unit = {
    if (Files.exists(Paths.get(IPsettings.HISTOGRAM_PATH))) {
      println(s"${IPsettings.HISTOGRAM_PATH} exists, skipping histograms creation..")
      return
    }

    val sameModel = KMeansModel.load(sc, IPsettings.KMEANS_PATH)

    val kMeansCenters = sc.broadcast(sameModel.clusterCenters)

    val categories = sc.broadcast(IMAGE_INPUT)

    val data = images.map {
      case (name, contents) => {

        val vocabulary = ImageUtils.vectorsToMat(kMeansCenters.value)

        val desc = ImageUtils.bowDescriptors(name.split("file:/")(1), vocabulary)
        val list = ImageUtils.matToString(desc)
        println("-- " + list.size)

        val segments = name.split("/")
        val cat = segments(segments.length - 2)
        List(categories.value.indexOf(cat) + "," + list(0))
      }
    }.reduce((x, y) => x ::: y)

    val featuresSeq = sc.parallelize(data)

    featuresSeq.saveAsTextFile(IPsettings.HISTOGRAM_PATH)
    println("Total size : " + data.size)
  }

  def generateDecisionTreeModel(sc: SparkContext): Unit = {
    if (Files.exists(Paths.get(IPsettings.Decision_Tree_PATH))) {
      println(s"${IPsettings.Decision_Tree_PATH} exists, skipping Decision model formation..")
      return
    }

    val data = sc.textFile(IPsettings.HISTOGRAM_PATH)
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }


    val splits = parsedData.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = parsedData
    val test = splits(1)


    val numClasses = 4
    val categoricalFeaturesInfo = Map[Int, Int]()

    val maxBins = 100


    val maxDepths = 3 to(6, 1)
    val impurities = List("gini", "entropy")

    var bestModel: Option[DecisionTreeModel] = None
    var bestErr = 1.0
    val bestParams = new mutable.HashMap[Any, Any]()
    var bestimpurity = ""
    var bestmaxdepth = 0

    impurities.foreach(impurity => {
      maxDepths.foreach(maxDepth => {

        println(" impurity " + impurity + " maxDepth " + maxDepth)

        val model = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

        val predictionAndLabel = test.map { point =>
          val prediction = model.predict(point.features)
          (point.label, prediction)
        }

        val testErr = predictionAndLabel.filter(r => r._1 != r._2).count.toDouble / test.count()
        println("Test Error = " + testErr)
        ModelEvaluation.evaluateModel(predictionAndLabel)

        if (testErr < bestErr) {
          bestErr = testErr
          bestModel = Some(model)
          bestParams.put("impurity", impurity)
          bestParams.put("maxDepth", maxDepth)
          bestimpurity = impurity
          bestmaxdepth = maxDepth
        }
      })
    })

    println("Best Err " + bestErr)
    println("Best params " + bestParams.toArray.mkString(" "))


    val DTM = DecisionTree.trainClassifier(parsedData, numClasses, categoricalFeaturesInfo, bestimpurity, bestmaxdepth, maxBins)

    DTM.save(sc, IPsettings.Decision_Tree_PATH)
    println("Decision Model generated")
  }
  def classifyImage(sc: SparkContext, path: String): Double = {

    val model = KMeansModel.load(sc, IPsettings.KMEANS_PATH)
    val vocabulary = ImageUtils.vectorsToMat(model.clusterCenters)

    val desc = ImageUtils.bowDescriptors(path, vocabulary)

    val histogram = ImageUtils.matToVector(desc)

    println("--Histogram size : " + histogram.size)

    val nbModel = DecisionTreeModel.load(sc, IPsettings.Decision_Tree_PATH)

    val p = nbModel.predict(histogram)

    p
  }

  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir","C:\\Users\\NAVEENA\\Desktop\\classifyimages");
    val conf = new SparkConf()
      .setAppName(s"IPApp")
      .setMaster("local[*]")
      .set("spark.executor.memory", "6g")
      .set("spark.driver.memory", "6g")
    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")

    val sc=new SparkContext(sparkConf)

    val images = sc.wholeTextFiles(s"${IPsettings.INPUT_DIR}/*/*.jpg")
    extractDescriptors(sc, images)
    kMeansCluster(sc)


    createHistogram(sc, images)


    generateDecisionTreeModel(sc)



    val testImages = sc.wholeTextFiles(s"${IPsettings.TEST_INPUT_DIR}/*/*.jpg")
    val testImagesArray = testImages.collect()
    var predictionLabels = List[String]()
    testImagesArray.foreach(f => {
      println(f._1)
      val splitStr = f._1.split("file:/")
      val predictedClass: Double = classifyImage(sc, splitStr(1))
      val segments = f._1.split("/")
      val cat = segments(segments.length - 2)
      val GivenClass = IMAGE_INPUT.indexOf(cat)
      println(s"Predicting test image : " + cat + " as " + IMAGE_INPUT(predictedClass.toInt))
      predictionLabels = predictedClass + ";" + GivenClass :: predictionLabels
    })

    val pLArray = predictionLabels.toArray

    predictionLabels.foreach(f => {
      val ff = f.split(";")
      println(ff(0), ff(1))
    })
    val predictionLabelsRDD = sc.parallelize(pLArray)


    val pRDD = predictionLabelsRDD.map(f => {
      val ff = f.split(";")
      (ff(0).toDouble, ff(1).toDouble)
    })
    val accuracy = 1.0 * pRDD.filter(x => x._1 == x._2).count() / testImages.count

    println(accuracy)
    ModelEvaluation.evaluateModel(pRDD)


  }
}