import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.classification.{SVMWithSGD, SVMModel}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by HM on 2015/12/9.
 */
object pipeline {

  case class Document(id: Long, text: String)

  def main(args: Array[String]) {


    val conf = new SparkConf().setAppName("SimpleTextClassificationPipeline").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)


    //载入训练集
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
    val training = sc.textFile("D:\\MLWork\\train.txt")


    val trainRDD = training.map(p => Row(p.substring(0, 1).toDouble, p.substring(1, p.length)))

    val sentenceData = sqlContext.createDataFrame(trainRDD, schema).toDF("label", "text")

   /* val training2 = sc.textFile("D:\\MLWork\\HM_data\\ansj_flag_seg_result")
//    val trainRDD2 = training2.map(f => {
//      val field = f.split(",")
//      if(field.size == 2){
//        Row(field(0).toDouble,field(1))
//      }else{
//        Row(field(0).toDouble,"")
//      }
//    })

    val trainRDD2 = training2.map(p => Row(p.substring(1,2).toDouble,p.substring(3,p.size-1)))


    val sentenceData2 = sqlContext.createDataFrame(trainRDD2,schema).toDF("label","text")*/



    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)

//    val svm = new SVMWithSGD()

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

    val test = sc.textFile("D:\\MLWork\\test.txt")
    val testRDD = test.map(p => Row(p.substring(0,7).toDouble,p.substring(7,p.length)))
    val testData = sqlContext.createDataFrame(testRDD,testSchema).toDF("id","text")


    val IDPreResult = model.transform(testData).select("id","prediction")
    val result = sc.parallelize(IDPreResult.collect()).repartition(1)//.saveAsTextFile("hdfs://192.168.131.192:9000//user/humeng/MLWork/ML_files/result")

    val IDLabelCSV = result.map(f => {
      f.get(0).toString.substring(0,f.get(0).toString.size-2)+","+f.get(1).toString.substring(0,1)
    })

    IDLabelCSV.repartition(1).saveAsTextFile("hdfs://192.168.131.192:9000//user/humeng/MLWork/ML_files/IDLabelCSV1292")

   sc.stop()
  }

}
