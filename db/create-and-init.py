import pymongo
import json

client = pymongo.MongoClient("mongodb://localhost:27017")

db = client["lazy"]
places = db["places"]
risk_points = db["risk_points"]
places.drop()
risk_points.drop()

with open('dangerous_object.geojson') as f:
	dan_obj_data = json.load(f)

risk_points.insert_many(dan_obj_data['features'])
point = dan_obj_data['features'][0]

with open('buildings.geojson') as f:
	places_data = json.load(f)
	places_data['features'][0]['restricted_by'] = { 'UID': point['properties']['UID'], 'RESTRICTION': 'red' }

places.insert_many(places_data['features'])

