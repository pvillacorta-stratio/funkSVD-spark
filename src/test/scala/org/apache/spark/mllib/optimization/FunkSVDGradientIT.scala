package org.apache.spark.mllib.optimization

import com.stratio.spaceai.SparkBase
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector => MLDense}
import org.apache.spark.mllib.linalg.{Vectors, Vector => V}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{functions => F}
import org.scalatest.FunSuite

class FunkSVDGradientIT extends FunSuite with SparkBase {

    test("Testing with nf_subsample.csv"){

        val schema = new StructType(Array(
            StructField("user", DoubleType, true),
            StructField("item", DoubleType, true),
            StructField("rating", DoubleType, true)
        ))

        val ratings = spark.read.option("header", "false")
                .schema(schema)
                .csv("src/test/resources/nf_subsample.csv")
                .withColumnRenamed("_c0", "user")
                .withColumnRenamed("_c1", "item")
                .withColumnRenamed("_c2", "rating")

        val vectorAssembler = new VectorAssembler().setInputCols(Array("user", "item")).setOutputCol("features")
        val myspark = spark
        import myspark.implicits._

        val data = vectorAssembler.transform(ratings).map(
            r => (r.getAs[Double]("rating"), Vectors.fromML(r.getAs[MLDense]("features")))).cache()
        data.count()

        ratings.select(F.min("user"), F.max("user")).show()

        val maxUserItemRow = ratings.select(F.max("user").alias("maxUser"), F.max("item").alias("maxItem")).first()
        val maxUser = maxUserItemRow.getAs[Double](maxUserItemRow.fieldIndex("maxUser")).toInt
        val maxItem = maxUserItemRow.getAs[Double](maxUserItemRow.fieldIndex("maxItem")).toInt
        val nUsers = maxUser+1
        val nItems = maxItem+1
        val nLatent = 10

        // Run training algorithm to build the model
        val numCorrections = 10
        val convergenceTol = 1e-4
        val maxNumIterations = 20
        val regParam = 0.1
        val funkSVDGradient = new FunkSVDGradient(nLatent, nUsers, nItems)
        val initialWeights = funkSVDGradient.generateZeroWeights()

        val (weights, loss) = LBFGS.runLBFGS(
            data.rdd,
            funkSVDGradient,
            new SquaredL2Updater(),
            numCorrections,
            convergenceTol,
            maxNumIterations,
            regParam,
            initialWeights)

        println(weights)
    }

}
