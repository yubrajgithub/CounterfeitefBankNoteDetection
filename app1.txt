from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
import pickle
import pywt
import base64

# Initialize Flask application
app = Flask(__name__)

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Load the kmeans model
kmeans = joblib.load('kmeans_model.pkl')

# Load the cluster labels dataframe
with open('cluster_labels.pkl', 'rb') as f:
    cluster_labels_df = pickle.load(f)

# Load the PCA model
pca = joblib.load('pca_model.pkl')

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize image to 640x480
    resized_image = cv2.resize(image, (640, 480))
    
    # Convert image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)  # Adjust kernel size if needed
    
    return blurred

# Function to extract features from the preprocessed image
def extract_features(image):
    # Perform wavelet transform
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs

    # Extract statistical features from each subband
    features = {
        'variance': np.var(cA),
        'skewness': np.mean(cH),
        'kurtosis': np.mean(cV),
        'entropy': np.mean(cD)
    }
    
    return features

# Function to predict class label for input features
def predict_class(input_values):
    # Scale the input features using the loaded scaler
    input_scaled = scaler.transform([input_values])
    # Predict cluster for scaled input using the loaded kmeans model
    cluster = kmeans.predict(input_scaled)[0]
    # Retrieve majority class label assigned to the cluster
    majority_class = cluster_labels_df.loc[cluster_labels_df['cluster'] == cluster, 'predicted_class'].iloc[0]
    
    # Map numerical labels to text labels
    label_text = "fake" if majority_class == 0 else "real"
    
    return majority_class, label_text

# Route for the home page
@app.route('/')
def index():
    return render_template('upload.html')

# Route for uploading images and detecting bank notes
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Check if the file extension is not jpg
            if uploaded_file.filename.endswith('.png'):
                return render_template('upload.html', error="Please upload a  image.")
            
            # Read the uploaded image
            img_np = np.fromstring(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            8
            # Preprocess the image
            preprocessed_image = preprocess_image(img)
            
            # Extract features from the preprocessed image
            image_features = extract_features(preprocessed_image)
            
            # Predict class label for extracted features
            predicted_class, predicted_label = predict_class(list(image_features.values()))
            
            # Filter cluster labels dataframe for the predicted class
            cluster_labels_pred_class = cluster_labels_df[cluster_labels_df['predicted_class'] == predicted_class]
            
            # Calculate accuracy for this predicted class
            accuracy = cluster_labels_df.loc[cluster_labels_df['predicted_class'] == predicted_class, 'class'].mean()
            
            # Convert preprocessed image to base64 format
            _, encoded_image = cv2.imencode('.png', preprocessed_image)
            base64_image = base64.b64encode(encoded_image).decode('utf-8')

            # Render the results page with detected class and features
            return render_template('result.html', 
                                   features=image_features,
                                   predicted_class=predicted_class,
                                   predicted_label=predicted_label,
                                   accuracy=accuracy,
                                   base64_image=base64_image)
    return render_template('upload.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
