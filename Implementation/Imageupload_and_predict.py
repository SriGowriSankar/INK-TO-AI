from google.colab import files
from PIL import Image
import io
import os



model_exists = os.path.exists('./models/best_model.h5') and os.path.exists('./models/class_names.json')

if not model_exists:
    print("Model files not found. Running the quick demo to train a model...")
   
    print("Please run the previous code cell labeled '2dIDwsdrul1p' to train the model first.")
else:
    # Upload the image
    print("Please upload an image for prediction.")
    uploaded = files.upload()

    uploaded_image_path = None
    # Assuming only one file is uploaded for simplicity
    for filename, content in uploaded.items():
        print(f'Uploaded file: {filename}')
        # Save the uploaded file to disk
        with open(filename, 'wb') as f:
            f.write(content)
        uploaded_image_path = filename
        break

    if uploaded_image_path:
        try:
            recognizer = HandwritingRecognizer(
                model_path='./models/best_model.h5',
                class_names_path='./models/class_names.json'
            )
            print(f"\nPredicting for: {uploaded_image_path}")
            recognizer.predict(uploaded_image_path, top_k=1, show_image=True)
        except Exception as e:
            print("Error during prediction. Please ensure the model is correctly loaded.")
            print(f"Details: {e}")
    else:
        print("No image uploaded for prediction.")
