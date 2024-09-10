from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the model and the scaler
with open('kmeans_model.pkl', 'rb') as model_file:
    kmeans = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Convert form data to a numpy array
    features = np.array([[
        float(data['Age']), float(data['Income']), float(data['Kidhome']), float(data['Teenhome']),
        float(data['Recency']), float(data['Complain']), float(data['MntWines']), float(data['MntFruits']),
        float(data['MntMeatProducts']), float(data['MntFishProducts']), float(data['MntSweetProducts']),
        float(data['MntGoldProds']), float(data['NumDealsPurchases']), float(data['AcceptedCmp1']),
        float(data['AcceptedCmp2']), float(data['AcceptedCmp3']), float(data['AcceptedCmp4']),
        float(data['AcceptedCmp5']), float(data['Response']), float(data['NumWebPurchases']),
        float(data['NumCatalogPurchases']), float(data['NumStorePurchases']), float(data['NumWebVisitsMonth'])
    ]])

    # Scale the data
    scaled_data = scaler.transform(features)

    # Predict the cluster
    cluster = kmeans.predict(scaled_data)

    # Return the prediction as a JSON response
    return jsonify({'Cluster': int(cluster[0])})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)