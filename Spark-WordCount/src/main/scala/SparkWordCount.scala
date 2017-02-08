

import org.apache.spark.{SparkContext, SparkConf}


object SparkWordCount {

  def main(args: Array[String]) {

    System.setProperty("hadoop.home.dir", "C:\\Users\\NAVEENA\\Desktop\\CS5542-Tutorial2-SparkSourceCode\\Spark WordCount");
    val sparkConf = new SparkConf().setAppName("WordCount").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    val input=sc.textFile("input")
    val wordcount=input.flatMap(line=>{line.split(" ").sliding(4)}).map( p => (p.mkString(" "),1))
    val output=wordcount.reduceByKey(_+_,1).sortBy(_._2,false)
    output.saveAsTextFile("output")

  }
}
