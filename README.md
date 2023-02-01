# Object-detection-in-Images-and-Videos
Object Detection using Keras, YOLO V3, Docker and Streamlit

### Run the prediction using one of following,

#### 1. Perform detection using trained weights on image, set of images, video, or webcam
    python predict.py -i /path/to/image/or/video

It carries out detection on the image and write the image with detected bounding boxes to the default output folder.

#### 2. Run with Streamlit
    streamlit run app.py server.port=8501
then point to [localhost:8501](https://localhost:8501) for streamlit app
#### 3. Run with Docker
    docker build -t dannylee1020/object_detection_video .
    docker run -p 8501:8501 dannylee1020/object_detection_video:latest
then point to [localhost:8501](https://localhost:8501) for streamlit app
