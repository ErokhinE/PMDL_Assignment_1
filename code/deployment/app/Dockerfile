# Use the official Python image from the Docker Hub
FROM python:3.9

COPY /code/deployment/app/requirements.txt /requirements.txt
COPY /code/deployment/app/app.py /app.py
RUN pip install --upgrade -r /requirements.txt

# Expose the port the app runs on
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
