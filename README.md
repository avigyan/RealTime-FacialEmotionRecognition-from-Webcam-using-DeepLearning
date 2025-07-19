AUTHOR: AVIGYAN SINHA

This project uses a modified VGGnet architecture of deep learning and applies it to the fer2013 dataset

The Fer2013:
The Kaggle Emotion and Facial Expression Recognition challenge training dataset consists of 28,709 images, each of which are 48×48 grayscale images. The faces have been
automatically aligned such that they are approximately the same size in each image. Given these images(real-time webcam video input), our goal is to categorize the emotion expressed on each face into six distinct classes:
angry, fear, happy, sad, surprise, and neutral.

This facial expression dataset is called the FER13 dataset and can be found at the official Kaggle
competition page and downloading the fer2013.tar.gz file:
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data 

The .tar.gz archive of the dataset is ≈ 92MB, so make sure you have a decent internet connection before downloading it. After downloading the dataset, you’ll find a file named fer2013.csv
with with three columns:
• emotion: The class label.
• pixels: A flattened list of 48×48 = 2304 grayscale pixels representing the face itself.
• usage: Whether the image is for Training, PrivateTest (validation), or PublicTest(testing).
Our goal is to now take this .csv file and convert it to HDF5 format so we can more easily
train a Convolutional Neural Network on top of it.


To run this program use the following command:
python3 realtime_emotion.py

Note: If you are using anything from this repository please consider citing - Avigyan Sinha, Aneesh R P, "Real Time Facial Emotion Recognition using Deep Learning", International Journal of Innovations and Implementations in Engineering(ISSN 2454-3489), 2019, vol 1
