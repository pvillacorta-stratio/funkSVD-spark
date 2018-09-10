package org.apache.spark.mllib.optimization

import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.BLAS.axpy
import org.apache.spark.mllib.linalg.Vectors

import scala.util.Random

/**
  * Class to compute the gradient of the Funk SVD optimization problem:
  * minimize $ _{(p_u, q_i)} \sum_{r_{ui} ∈ R} (r_{ui} − p_u · q_i)^2 $
  * where $p_u$ and $q_i$ are real vectors of dimension nLatent
  * @param nLatent Number of latent factors to use. Increasing this value enhances the accuracy of the decomposition
  * @param nUsers Total number of different users in our system. Needed to know how the weights Vector is
  *               arranged, since this vector actually contains all the elements of two matrices P (nUsers x nLatent)
  *               and Q (nItems x nLatent) unrolled and juxtaposed in a single linalg.Vector
  */
class FunkSVDGradient(nLatent: Int, nUsers: Int, nItems: Int) extends Gradient {

    /**
      * Computes the gradient and loss for a single rating (a single data point) assuming that the feature vector
      * has the structure [userId, itemId, rating] and both userId and itemId start at index 0
      * @param data Data point containing a single rating with the structure [userId, itemId, rating]
      * @param label Numerical rating (positive real number)
      * @param weights Array of coefficients with all the elements of every p_u and q_i vectors. First we have
      *                the elements of P (i.e. all the p_i vectors, which are nLatent * nUsers real values)
      *                and then the q_i
      * @return
      */
    override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector): (linalg.Vector, Double) = {

        val offsetU = nUsers * nLatent
        val (userId, itemId) = (data(0).toInt, data(1).toInt)

        val weightsArray:Array[Double] = weights.toArray

        val p_u:Array[Double] = weightsArray.slice(userId * nLatent, (userId+1) * nLatent)
        val q_i:Array[Double] = weightsArray.slice(offsetU + itemId * nLatent, offsetU + (itemId+1) * nLatent)

        val sqrt_loss = label - p_u.zip(q_i).map{case (p, q) => p*q}.sum

        val gradient_p_u = q_i.map(-2 * _ * sqrt_loss)
        val gradient_q_i = p_u.map(-2 * _ * sqrt_loss)

        val gradient = Array.fill(weightsArray.length)(0.0)

        for(i <- 0 to (nLatent-1)){
            gradient.update((userId * nLatent) + i, gradient_p_u(i))
            gradient.update(offsetU + itemId * nLatent + i, gradient_q_i(i))
        }

        (Vectors.dense(gradient), sqrt_loss*sqrt_loss / 2.0)
    }

    override def compute(data: linalg.Vector, label: Double, weights: linalg.Vector, cumGradient: linalg.Vector): Double = {
        val (gradient, loss) = compute(data, label, weights)
        axpy(1, gradient, cumGradient)
        loss
    }

    def generateZeroWeights(): linalg.Vector ={
        Vectors.zeros(nLatent * (nUsers + nItems))
    }

    def generateInitialWeights(seed: Int = 12345): linalg.Vector = {
        Vectors.dense(Array.fill(nLatent * (nUsers + nItems))(Random.nextDouble()))
    }
}
