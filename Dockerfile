# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . /app

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Run main.py when the container launches
CMD ["python", "main.py","-e", "production"]
ENTRYPOINT ["python", "main.py", "-e", "production"]

##Buildujemy tak:
    #docker build -t my_image_name .
##Odpalamy kontener tak:
    #docker run -p 7860:7860 my_image_name