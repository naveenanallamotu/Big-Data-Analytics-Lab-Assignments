import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
/**
  * Created by NAVEENA on 08-02-2017.
  */
object naveenalinear {
  def main(args: Array[String]): Unit ={


    System.setProperty("hadoop.home.dir","C:\\Users\\NAVEENA\\Desktop\\CS5542-Tutorial3-SourceCode\\untitled");

    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")

    val sc=new SparkContext(sparkConf)


    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    val data = sc.textFile("data\\lpsa.data")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()

    parsedData.take(1).foreach(f=>println(f))


    val Array(training, test) = parsedData.randomSplit(Array(0.95, 0.05))

    val numIterations = 100
    val stepSize = 0.00000001
    val model = LinearRegressionWithSGD.train(training, numIterations, stepSize)

    val valuesAndPreds = training.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    println("training Mean Squared Error = " + MSE)


    val valuesAndPreds2 = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE2 = valuesAndPreds2.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    println("test Mean Squared Error = " + MSE2)

    model.save(sc, "data\\LinearRegressionWithSGDModel")
    val sameModel = LinearRegressionModel.load(sc, "data\\LinearRegressionWithSGDModel")
  }
}
