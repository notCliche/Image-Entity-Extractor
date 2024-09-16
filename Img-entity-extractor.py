# Install required libraries
!pip install torch torchvision transformers pillow tqdm scikit-learn

import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
from tqdm import tqdm
import random
import requests
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Global variables
unit_classifier = None
number_predictor = None
feature_extractor = None
ENTITY_UNIT_MAP = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

def load_models():
    global unit_classifier, feature_extractor, number_predictor
    unit_classifier = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    num_labels = sum(len(units) for units in ENTITY_UNIT_MAP.values())
    unit_classifier.classifier = torch.nn.Linear(unit_classifier.classifier.in_features, num_labels)
    unit_classifier.eval()
    
    number_predictor = RandomForestRegressor(n_estimators=100, random_state=42)

def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert('RGB')

def extract_features(image):
    return feature_extractor(images=image, return_tensors="pt")['pixel_values']

def predict_unit(features):
    with torch.no_grad():
        outputs = unit_classifier(features)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    all_units = [unit for units in ENTITY_UNIT_MAP.values() for unit in units]
    return all_units[predicted_class_idx]

def predict_number(features, group_id, entity_name):
    # Flatten the features and combine with other inputs
    flat_features = features.squeeze().flatten().numpy()
    input_vector = np.concatenate([flat_features, [int(group_id)]])
    
    return number_predictor.predict(input_vector.reshape(1, -1))[0]

def predictor(image_url, group_id, entity_name):
    try:
        image = download_image(image_url)
        features = extract_features(image)
        
        predicted_unit = predict_unit(features)
        predicted_number = predict_number(features, group_id, entity_name)
        
        return f"{predicted_number:.2f} {predicted_unit}"
    except Exception as e:
        print(f"Error processing {image_url}: {str(e)}")
        return ""

def truncate_dataset(input_file, num_samples):
    df = pd.read_csv(input_file)
    return df.sample(n=min(num_samples, len(df)), random_state=42)

def calculate_f1_score(true_values, predicted_values):
    true_positives = sum((t != "" and p != "" and t == p) for t, p in zip(true_values, predicted_values))
    false_positives = sum((t != "" and p != "" and t != p) or (t == "" and p != "") for t, p in zip(true_values, predicted_values))
    false_negatives = sum((t != "" and p == "") for t, p in zip(true_values, predicted_values))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1_score

if __name__ == "__main__":
    # For Google Colab, you might need to mount your Google Drive to access the files
    from google.colab import drive
    drive.mount('/content/drive')

    DATASET_FOLDER = '/content/drive/MyDrive/dataset'  # Adjust this path as needed
    train_filename = os.path.join(DATASET_FOLDER, 'train.csv')
    output_filename = os.path.join(DATASET_FOLDER, 'train_predictions.csv')

    # Load and truncate dataset
    num_samples = 1000  # Change this to the desired number of samples
    df = truncate_dataset(train_filename, num_samples)

    # Split the data for training the number predictor
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print("Loading models...")
    load_models()

    print("Training number predictor...")
    X_train = []
    y_train = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        try:
            image = download_image(row['image_link'])
            features = extract_features(image)
            flat_features = features.squeeze().flatten().numpy()
            X_train.append(np.concatenate([flat_features, [int(row['group_id'])]]))
            number, _ = row['entity_value'].split(' ', 1)
            y_train.append(float(number))
        except Exception as e:
            print(f"Error processing training sample: {e}")
    
    number_predictor.fit(np.array(X_train), np.array(y_train))

    print("Making predictions...")
    predictions = []
    true_values = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prediction = predictor(row['image_link'], row['group_id'], row['entity_name'])
        predictions.append(prediction)
        true_values.append(row['entity_value'])

    test_df['prediction'] = predictions

    # Save the output
    test_df[['index', 'prediction']].to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")

    # Calculate F1 score
    f1_score = calculate_f1_score(true_values, predictions)
    print(f"\nF1 Score: {f1_score:.4f}")

    print(f"\nPredictions have been saved as {output_filename}")
