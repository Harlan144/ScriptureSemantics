#Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app
RUN pip install --upgrade pip setuptools wheel

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
COPY requirements.txt /
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Change permissions of all files in the working directory to 777
RUN chmod -R 777 /app

CMD ["bash","main.sh"]
RUN chmod -R 777 /app
