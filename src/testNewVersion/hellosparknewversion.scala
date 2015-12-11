package testNewVersion

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructField, StringType, StructType}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by HM on 2015/12/8.
 */
object hellosparknewversion {

  /*
* transform 1048576,[96163,650202,693856],[8.242560621923047,12.899221076089338,7.330876572328241]
*into  96163:8.242560621923047 650202:12.899221076089338 693856:7.330876572328241
*
* */
  def tfidfToSVM(string: String): String = {
    val str = string.substring(8, string.length)
    val array = str.split("],")
    val a1: String = array(0).replaceAll("\\[|\\]", "")
    val a2: String = array(1).replaceAll("\\[|\\]", "")
    val arr1: Array[String] = a1.split(",")
    val arr2: Array[String] = a2.split(",")
    var svm: String = ""
    var i = 0
    for (i <- 0 to arr1.size - 1) {
      svm = svm + arr1(i) + ":" + arr2(i) + " "
    }
    return svm
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("testNewVersion").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val rdd = sc.textFile("hdfs://192.168.131.192:9000///user/humeng/MLWork/ML_files/flag-termList")


    val schemaString = "flag words"

    val schema =
      StructType(
        schemaString.split(" ").map(fieldName => StructField(fieldName, StringType, true)))


    val rowRDD = rdd.map(p => {
      val field = p.split("\t")
      if (field.size == 2) {
        Row(field(0), field(1))
      } else {
        Row(field(0), "")
      }
    })

    val sentenceData = sqlContext.createDataFrame(rowRDD, schema).toDF("label", "sentence")

//    sentenceData.registerTempTable("message")

//    val messages = sqlContext.sql("SELECT sentence FROM message WHERE label > 0 ")
//
//    messages.map(t => "Name: " + t(0)).collect().foreach(println)

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
    val featurizedData = hashingTF.transform(wordsData)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)



    val result = rescaledData.select("label", "features").take(10)
//    val result = featurizedData.select("label", "features").take(5)//.foreach(println _)

    result.foreach(f => {
      println(f)
      println(f.get(0))
      println(f.get(1))
    })


//    result.take(5).foreach(f => println(f.get(0)+" "+tfidfToSVM(f.get(1).toString.substring(1,f.get(1).toString.size-2))))
//    val svm = result.take(result.count().toInt).foreach(f => println(f.get(0)+" "+tfidfToSVM(f.get(1).toString.substring(1,f.get(1).toString.size-2))))

//    val svm2 = result.take(result.count().toInt).map(f => {
//      f.get(0)+" "+tfidfToSVM(f.get(1).toString.substring(1,f.get(1).toString.size-2))
//    })


//    sc.parallelize(svm2).saveAsTextFile("hdfs://192.168.131.192:9000//user/humeng/MLWork/ML_files/flag_termlist-TF_data")




//    svm2.foreach(println _)


    sc.stop()
  }

}
