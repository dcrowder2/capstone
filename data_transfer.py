import sqlite3
import numpy
import os

if __name__ == "__main__":
    database = sqlite3.connect("data.db3")
    cur = database.cursor()

    cur.execute("Select * from games")
    data = cur.fetchall()
    array = numpy.array(data)
    database.close()
