# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory in the container
COPY /code/deployment/api/requirements.txt /requirements.txt
COPY /code/deployment/api/api.py /api.py
COPY /models/random_forest_model.pkl /random_forest_model.pkl
RUN pip install --upgrade -r /requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
