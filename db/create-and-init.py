import pymongo
import json

client = pymongo.MongoClient("mongodb://localhost:27017")

db = client["lazy"]
buildings = db["buildings"]
buildings.drop()

with open('buildings.geojson') as f:
	data = json.load(f)

buildings.insert_many(data['features'])
