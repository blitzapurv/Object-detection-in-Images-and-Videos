FROM python:3.10
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN python3 -m pip install --upgrade pip

# define working directory within docker image
WORKDIR /app

# copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy necessary folders to run the app
COPY predict.py /app/predict.py
COPY predict_bb.py /app/predict_bb.py
COPY app.py /app/app.py
COPY utils /app/utils
COPY weights /app/weights
COPY output /app/output
COPY sample_videos /app/sample_videos
COPY sample_images /app/sample_images


# for local build
EXPOSE 8501

# for running locally
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
#ENTRYPOINT ["streamlit", "run"]
#CMD ["app/app.py"]

