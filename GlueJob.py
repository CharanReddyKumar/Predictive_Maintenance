import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, when

# @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Parameters for Input and Output
input_database = "ai4i_database"
input_table_name = "machine_failure_source_bucket"
output_s3_path = "s3://machine-failure-transformed-bucket/TransformedData/"

datasource = glueContext.create_dynamic_frame.from_catalog(
    database=input_database,
    table_name=input_table_name
)

df = datasource.toDF()

# Debugging
print("Columns in DataFrame:", df.columns)

df = df.withColumnRenamed("process temperature [k]", "process_temperature") \
       .withColumnRenamed("air temperature [k]", "air_temperature") \
       .withColumnRenamed("rotational speed [rpm]", "rotational_speed") \
       .withColumnRenamed("torque [nm]", "torque")

df = df.withColumn(
    'Temp_Diff',
    col('process_temperature') - col('air_temperature')
).withColumn(
    'Power',
    col('torque') * (col('rotational_speed') * 2 * 3.14159 / 60)
).withColumn(
    'Failure',
    when(
        (col('twf') == 1) | (col('hdf') == 1) | (col('pwf') == 1) |
        (col('osf') == 1) | (col('rnf') == 1), 1
    ).otherwise(0)
)

cols_to_drop = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df = df.drop(*cols_to_drop)

# Debugging: Show transformed data
print("Sample data after dropping columns:")
df.show(5)

df.write.mode("overwrite").option("header", "true").csv(output_s3_path)

# Commit the job
job.commit()
