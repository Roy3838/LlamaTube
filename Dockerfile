# Use an official Python runtime as a parent image
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run train.py when the container launches
CMD ["python", "train.py"]
