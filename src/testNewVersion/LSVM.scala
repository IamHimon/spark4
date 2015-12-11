package testNewVersion

import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

import scala.beans.BeanInfo

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.{Row, SQLContext}

@BeanInfo
case class Document(id: Long, text: String)

object LSVM {

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("SimpleTextClassificationPipeline").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val schemaString = "label text"

    val schema =
      StructType(
        schemaString.split(" ").map { fieldName =>
          if (fieldName.equals("text")) {
            StructField(fieldName, StringType)
          }
          else {
            StructField(fieldName, DoubleType)
          }
        }
      )


    val training = sc.textFile("G:\\MSGDATA\\rs\\train.txt")
    val test = sc.textFile("G:\\MSGDATA\\rs\\test.txt")

    val trainRDD = training.map(p => Row(p.substring(0, 1).toDouble, p.substring(1, p.length)))

    val sentenceData = sqlContext.createDataFrame(trainRDD, schema).toDF("label", "text")

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))
    val model = pipeline.fit(sentenceData)

    val testSchema =
      StructType(
        schemaString.split(" ").map { fieldName =>
          if (fieldName.equals("text")) {
            StructField(fieldName, StringType)
          }
          else {
            StructField(fieldName, DoubleType)
          }
        }
      )

    val testRDD = test.map(p => Row(p.substring(0, 7).toDouble, p.substring(7, p.length)))
    val testData = sqlContext.createDataFrame(testRDD, testSchema).toDF("id", "text")


    sc.parallelize(model.transform(testData).select("id", "prediction").collect()).repartition(1)
      .saveAsTextFile("hdfs://192.168.131.192:9000/user/liuguangfu/data/IDPrediction-Data")



    sc.stop()
  }
}