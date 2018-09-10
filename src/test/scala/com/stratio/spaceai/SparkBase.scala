package com.stratio.spaceai

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, Suite}

trait SparkBase extends BeforeAndAfterAll {
  _: Suite =>

  val sparkConfig: SparkConf = new SparkConf()
  implicit var spark: SparkSession = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession.builder().config(sparkConfig).master("local[2]").getOrCreate()
  }

  override def afterAll(): Unit = {
    super.afterAll()
    spark.stop()
  }
}
