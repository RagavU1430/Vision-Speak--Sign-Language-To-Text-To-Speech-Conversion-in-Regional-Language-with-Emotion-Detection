import os
import sys
import pickle
from utils import suppress_c_stderr

def main():
    model_path = "models/sign_cnn.h5"
    encoder_path = "models/label_encoder.pkl"
    
    print("Checking if files exist...")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
    if not os.path.exists(encoder_path):
        print(f"Error: Encoder not found at {encoder_path}")
        sys.exit(1)
        
    print("Loading model and numpy...")
    try:
        with suppress_c_stderr():
            import tensorflow as tf
            import numpy as np
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        model.summary()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
        
    print("Loading label encoder...")
    try:
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"Label encoder loaded. Classes: {list(label_encoder.classes_)}")
    except Exception as e:
        print(f"Failed to load label encoder: {e}")
        sys.exit(1)
        
    print("Performing dummy inference...")
    try:
        # Realistic random input rather than saturated ones
        dummy_input = np.random.rand(1, 64, 64, 3).astype(np.float32)
        preds = model(dummy_input, training=False).numpy()[0]
        class_idx = np.argmax(preds)
        confidence = preds[class_idx]
        label = label_encoder.inverse_transform([class_idx])[0]
        print(f"Inference test passed! Predicted class: {label} with confidence {confidence:.2f}")
    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)
        
    print("All checks passed successfully!")

if __name__ == "__main__":
    main()
