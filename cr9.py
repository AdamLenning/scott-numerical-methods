import mysql.connector
import pandas as pd

mydb = mysql.connector.connect(
  host="localhost",
  user="alenning",
  password="Paris8889~",
  database="sakila"
)

df = pd.read_sql("select * from actor", mydb);

print(df)

statsdb = mysql.connector.connect(
  host="localhost",
  user="alenning",
  password="Paris8889~",
  database="my_db"
)

df.to_sql('actor', statsdb)