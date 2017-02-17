import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
/**
  * Created by NAVEENA on 16-02-2017.
  */
object ModelEvaluation {
  def evaluateModel(predictionAndLabels: RDD[(Double, Double)]) = {
      val metrics = new MulticlassMetrics(predictionAndLabels)
      val cfMatrix = metrics.confusionMatrix
      println(" |=================== Confusion matrix ==========================")
      println(cfMatrix)
      println(metrics.fMeasure)

    }
  }


