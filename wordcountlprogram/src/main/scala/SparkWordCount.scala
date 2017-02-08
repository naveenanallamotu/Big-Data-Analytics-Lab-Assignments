

import org.apache.spark.{SparkContext, SparkConf}
object SparkWordCount {

  def main(args: Array[String]) {

    System.setProperty("hadoop.home.dir","C:\\Users\\NAVEENA\\Desktop\\CS5542-Tutorial2-SparkSourceCode\\Spark WordCount");

    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")

    val sc=new SparkContext(sparkConf)

    val input=sc.textFile("input")

    val wordcount=input.flatMap(line=>{line.split(" ")}).map(word=>(word{0},1)).cache()

    val output=wordcount.reduceByKey(_+_)

    output.saveAsTextFile("output")

    val o=output.collect()

    var s:String="Words:Count \n"
    o.foreach{case(word,count)=>{

      s+=word+" : "+count+"\n"

    }}

  }

}
