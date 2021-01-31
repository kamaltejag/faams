import mysql.connector

try : 
    mydb = mysql.connector.connect(host="localhost", user="homestead", password="secret", database="faams")
    mycursor = mydb.cursor()
except:
    print("Unable to establish database connection!")
    raise

# import pickledb

# db = pickledb.load('attendance.db', False)

# db.set('key2', 'testing')
# db.dump()

# db.dexists('key')