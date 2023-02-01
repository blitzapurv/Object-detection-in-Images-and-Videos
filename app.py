import streamlit as st
import os
import time
from predict_bb import make_prediction


def run():
    DIR_PATH = '/app/' #docker
    #DIR_PATH = './'     #local testing
    st.title('Object Detection in Video')
    option = st.radio('', ['Choose a test image', 'Upload your own image',
                            'Choose a test video', 'Upload your own video (.mp4 only)'])
    #st.sidebar.title('Parameters')
    #confidence_slider = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, config.DEFALUT_CONFIDENCE, 0.05)
    #nms_slider = st.sidebar.slider('Non-Max Suppression Threshold', 0.0, 1.0, config.NMS_THRESHOLD, 0.05)

    if option == 'Choose a test image':
        test_images = os.listdir(DIR_PATH + 'sample_images/')
        test_image = st.selectbox('Please choose a test video', test_images)
    
    elif option == 'Choose a test video':
        test_videos = os.listdir(DIR_PATH + 'sample_videos/')
        test_video = st.selectbox('Please choose a test video', test_videos)
    
    elif option == 'Upload your own image':
        test_image = st.file_uploader('Upload an image', type = ['png', 'jpg', 'jpeg'])
        if test_image is not None:
            with open(os.path.join(DIR_PATH, "uploaded_image."+test_image.name.split(".")[-1]),"wb") as ff:
                ff.write(test_image.getbuffer())
        else:
            st.write('** Please upload a test image **')
    
    else:
        test_video = st.file_uploader('Upload a video', type = ['mp4'])
        if test_video is not None:
            with open(os.path.join(DIR_PATH, "uploaded_video."+test_video.name.split(".")[-1]),"wb") as ff:
                ff.write(test_video.getbuffer())
        else:
            st.write('** Please upload a test video **')


    if st.button ('Detect Objects'):
        
        time.sleep(2)

        if option == "Choose a test image":
            st.write(f"[INFO] Processing Image....")
            elap = make_prediction(DIR_PATH + 'sample_images/' + test_image)
            output_image = open(DIR_PATH + 'output/output_image.jpg', 'rb')
            image_bytes = output_image.read()
            
            st.write(f"[INFO] Time required to process the entire image: {round((elap)/60, 2)} minutes")
            final_image = st.image(image_bytes)
            st.download_button(label="Download Result", data=image_bytes, file_name="result.jpg")
        
        elif option == "Upload your own image":
            st.write(f"[INFO] Processing Image....")
            elap = make_prediction(os.path.join(DIR_PATH, "uploaded_image."+test_image.name.split(".")[-1]))
            output_image = open(DIR_PATH + 'output/output_image.jpg', 'rb')
            image_bytes = output_image.read()
            
            st.write(f"[INFO] Time required to process the entire image: {round((elap)/60, 2)} minutes")
            final_image = st.image(image_bytes)
            st.download_button(label="Download Result", data=image_bytes, file_name="result.jpg")

        elif option == "Choose a test video":
            st.write(f"[INFO] Processing Video....")
            elap = make_prediction(DIR_PATH + 'sample_videos/' + test_video)
            output_video = open(DIR_PATH + 'output/output_video.mp4', 'rb')
            # output_video = open(config.VIDEO_PATH + test_video, 'rb')
            video_bytes = output_video.read()

            st.write(f"[INFO] Time required to process the entire video: {round((elap)/60, 2)} minutes")
            final_video = st.video(video_bytes)
            st.download_button(label="Download Result", data=video_bytes, file_name="result.mp4")
        
        else:
            st.write(f"[INFO] Processing Video....")
            elap = make_prediction(os.path.join(DIR_PATH, "uploaded_video."+test_video.name.split(".")[-1]))
            output_video = open(DIR_PATH + 'output/output_video.mp4', 'rb')
            # output_video = open(config.VIDEO_PATH + test_video, 'rb')
            video_bytes = output_video.read()

            st.write(f"[INFO] Time required to process the entire video: {round((elap)/60, 2)} minutes")
            final_video = st.video(video_bytes)
            st.download_button(label="Download Result", data=video_bytes, file_name="result.mp4")




if __name__ == '__main__':

    run()

