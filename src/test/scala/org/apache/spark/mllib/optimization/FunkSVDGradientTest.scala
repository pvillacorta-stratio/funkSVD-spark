package org.apache.spark.mllib.optimization

import com.stratio.spaceai.SparkBase
import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.FunSuite

class FunkSVDGradientTest extends FunSuite with SparkBase {

    test("Test FunkSVDGradient compute returning loss and gradient"){
        val nLatent = 10
        val nUsers = 3
        val nItems = 4
        val funkSVDGradient = new FunkSVDGradient(nLatent, nUsers, nItems)

        val label = 3.2
        val data = Array(1, 3, label) // user 1 rates item 3 with a rating of 3.2
        val weights = Array.fill(nLatent*nUsers + nLatent*nItems)(1.0)

        val (gradient, loss) = funkSVDGradient.compute(Vectors.dense(data), label, Vectors.dense(weights))
        println(gradient)
        println(loss)
    }

    test("Test FunkSVDGradient compute returning loss only"){
        val nLatent = 10
        val nUsers = 3
        val nItems = 4
        val funkSVDGradient = new FunkSVDGradient(nLatent, nUsers, nItems)

        val label = 3.2
        val data = Array(1, 3, label) // user 1 rates item 3 with a rating of 3.2
        val weights = Array.fill(nLatent*nUsers + nLatent*nItems)(1.0)
        val cumGradient = Array.fill(nLatent*nUsers + nLatent*nItems)(0.0)

        val loss = funkSVDGradient.compute(Vectors.dense(data), label, Vectors.dense(weights), Vectors.dense(cumGradient))
        println(loss)
    }

}
