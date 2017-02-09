import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
/**
  * Created by NAVEENA on 08-02-2017.
  */
object kmeans {

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\Users\\NAVEENA\\Desktop\\CS5542-Tutorial3-SourceCode\\naveenameans");
    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);
    val data = sc.textFile("data/kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()
    parsedData.foreach(f => println(f))
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)
    println("Clustering on training data: ")
    clusters.predict(parsedData).zip(parsedData).foreach(f => println(f._2, f._1))
    clusters.save(sc, "data/KMeansModel")
    val sameModel = KMeansModel.load(sc, "data/KMeansModel")
  }


}
