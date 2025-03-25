import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
cap = cv2.VideoCapture(0)
offset = 20
imgSize = 300
counter = 0
# Initialize variables for evaluation metrics
true_labels = []  # Ground truth labels
predicted_labels = []  # Predicted labels

# Load the model
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Initialize HandDetector
detector = HandDetector(maxHands=2)

# Define labels
labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

# Main loop for real-time sign language detection and evaluation
while True:
    # Read frame from webcam
    success, img = cap.read()
    imgOutput = img.copy()
    
    # Find hands in the frame
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Crop hand region
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        if not imgCrop.size == 0:
            # Resize and preprocess cropped image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
            imgWhite[:imgSize, :imgSize] = imgResize

            # Predict label
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            predicted_label = labels[index]
            
            # Update true labels with ground truth
            true_label = "Actual_Label"  # Replace this with the actual label of the frame
            true_labels.append(true_label)
            
            # Update predicted labels
            predicted_labels.append(predicted_label)

            # Visualization code
            cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x -offset+400, y - offset+60-50),(0,255,0),cv2.FILLED)  
            cv2.putText(imgOutput, predicted_label, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2) 
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y+h + offset), (0,255,0), 4)   

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
        else:
            print("imgCrop has an invalid size. Skipping...")

    cv2.imshow('Image', imgOutput)
    
    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:")
print(conf_matrix)

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
