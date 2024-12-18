from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pandas as pd
from transformers import pipeline

app = Flask(__name__)
api = Api(app)

# Load sample property data
property_data = pd.read_csv('property_data.csv')

# Load the NLP model
chatbot = pipeline('conversational', model="microsoft/DialoGPT-small")

class PropertyRecommendations(Resource):
    def get(self):
        user_preferences = request.args
        # For simplicity, let's use a basic filter; a real implementation would be more complex.
        filtered_properties = property_data[
            (property_data['price'] <= float(user_preferences.get('max_price', float('inf')))) &
            (property_data['bedrooms'] >= int(user_preferences.get('min_bedrooms', 0)))
        ]
        return jsonify(filtered_properties.to_dict(orient='records'))

class Chatbot(Resource):
    def get(self):
        user_message = request.args.get('message')
        response = chatbot(user_message)
        return jsonify({'response': response})

api.add_resource(PropertyRecommendations, '/recommendations')
api.add_resource(Chatbot, '/chatbot')

if __name__ == '__main__':
    app.run(debug=True)
