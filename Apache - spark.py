What is Apache Spark:
	is a in memory execution environment
	is a fast(in memory) data processing engine which allows to:
    - Data classification through Spark machine learning
    - Streaming data through source via Spark Streaming
    - Querying data in real time throgh Spark SQL

Spark is built on the top of Scala programming language, which runs on the Java Virtual Machine

from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("word count").setMaster("local[3]")
sc = SparkContext(conf=conf)							# this is the main entry point to the spark functionality
														# "word count" - the app name
														# "local[3]" - run locally using 3 cores of our cpu
														# local[2] - 2 cores; local[*] - all available cores; local - 1 core;
lines = sc.textFile("<file path + name>")				# the file loaded as a RDD (resilient dristibuted dataset)
rdd = sc.parallelize(range(1, 7))						# Create a RDD
result = lines.collect()								# Return a list that contains all of the elements in this RDD.
result = lines.first()									# Return the first element in this RDD.
result = lines.take(num)								# Take the first num elements of the RDD.
result = lines.count()									# Return the number of elements in this RDD.
result = lines.distinct().collect()						# Return a new RDD containing the distinct elements in this RDD.
result = lines.filter(lambda x: x % 2 == 0).collect()	# Return a new RDD containing only the elements that satisfy a predicate.
result = lines.flatMap(lambda x: range(1, x)).collect())# Return a new RDD by first applying a function to all elements of this RDD, and then flattening the results.
result= (rdd + rdd).collect()							# Return the union of this RDD and another one.
result= lines.sample(withreplacement=True, fraction=0.1, seed=None)
														# withreplacement - Elements can be sampled multiple times
														# Fraction - size of the sample as a fraction to the RDD's size
														# seed - seed for the random number generator
result= rdd1.cartesian(rdd2)							# all possible pairs: 1,a 1,b 2,a 2,b
content.saveAsTextFile("out/output_file.text")			# Save the rdd to another file
rdd.persists(StorageLevel.MEMORY_ONLY)					# Keeps the RDD in memory for repeated actions
$ spark - submit < file path + name >					# running the spark job
http://localhost:4040/stages							# visualize
$ spark-shell											# spark shell
