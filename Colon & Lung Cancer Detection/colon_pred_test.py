import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from model_parameters import selected_model
from sklearn.metrics import accuracy_score

model = selected_model()

class_labels = [
    'Colon Adenocarcinoma',
    'Colon Benign Tissue',
    'Lung Adenocarcinoma',
    'Lung Benign Tissue',
    'Lung Squamous Cell Carcinoma'
]

test_dir = 'local_test_images'

# Store ground truth and predictions
y_true = []
y_pred = []
images = []
titles = []

for class_name in os.listdir(test_dir):
    print(class_name)
    class_folder = os.path.join(test_dir, class_name)
    print(class_folder)
    
    if not os.path.isdir(class_folder):
        continue

    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        pred = model.predict(img_array)
        class_index = np.argmax(pred)
        confidence = np.max(pred)
        predicted_class = class_labels[class_index]

        y_true.append(class_name)
        y_pred.append(predicted_class)

        images.append(img)
        titles.append(f'True: {class_name}\nPred: {predicted_class}\nConf: {confidence:.2f}')

# ✅ After all predictions, plot in one figure
num_images = len(images)
cols = 5
rows = (num_images + cols - 1) // cols  # Ceiling division

fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))

for i, ax in enumerate(axes.flat):
    if i < num_images:
        ax.imshow(images[i])
        ax.set_title(titles[i], fontsize=10)
        ax.axis('off')
    else:
        ax.axis('off')

        # 🎯 Print and visualize
        # print(f"\n🖼️ Image: {img_name}")
        # print(f"✅ True label: {class_name}")
        # print(f"🎯 Predicted label: {predicted_class} (Confidence: {confidence:.2f})")

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # ax1.imshow(img)
        # ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2f}', color='blue', fontsize=14)
        # ax1.axis('off')

        # ax2.barh(class_labels, pred[0], color='skyblue')
        # ax2.set_xlim(0, 1)
        # ax2.set_xlabel('Probability')
        # ax2.set_title('Class Probabilities')
        # for index, value in enumerate(pred[0]):
        #     ax2.text(value + 0.01, index, f'{value:.2f}', va='center')

        # plt.tight_layout()
        # plt.show()

accuracy = accuracy_score(y_true, y_pred)
print(f"\n\n📈 Overall Test Accuracy: {accuracy*100:.2f}%")

plt.tight_layout()
plt.show()
