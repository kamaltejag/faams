import json
import requests
import datetime

date = datetime.datetime.now()

# attendance = {}

# # Add date and roll number to dictionary
# attendance['date'] = date("%d-%m-%Y")
# attendance['roll_number'] = '17ra1a0537'

# # Add attendance to database and retrieve name
# url = 'http://faams.web:8000/rest_api/api/attendance/create.php'
# headers = {'Accept' : 'application/json', 'Content-Type' : 'application/json'}
# r = requests.post(url, data=json.dump(attendance), headers=headers)
# data = r.json()
# name = data['name']

with open('students.json') as f:
    students = json.load(f)

for student in students:
    if "17ra1a0537" in student.values():
        name = student.get('name')

print(name)