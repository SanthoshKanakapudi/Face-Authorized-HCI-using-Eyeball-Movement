import os
import numpy as np
from deepface import DeepFace
import joblib

# Path to your collected images
my_images_path = "faces_db/Me/"

embeddings = []

for img_name in os.listdir(my_images_path):
    img_path = os.path.join(my_images_path, img_name)
    try:
        embedding = DeepFace.represent(img_path, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
        embeddings.append(embedding)
    except:
        print(f"Skipping {img_path}")

embeddings = np.array(embeddings)

# Save embeddings to use later
joblib.dump(embeddings, "my_face_embeddings.pkl")
print(f"✅ Stored {len(embeddings)} embeddings for your face.")
