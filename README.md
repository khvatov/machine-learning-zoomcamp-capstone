## Alex Khvatov Capstone project for ML ZoomCamp class (2024)

### Description:
Diabetic retinopathy is a complication of diabetes that affects the eyes. It occurs when high blood sugar levels cause damage to the blood vessels in the retina, the light-sensitive tissue at the back of the eye. Over time, these damaged blood vessels can leak fluid or bleed, leading to vision problems.

### Project Setup

The main file is [capstone.ipynb](capstone.ipynb). It contains EDA, feature importance analysis as well as several models' creation, tuning and comparison.

In order to run this Jupyter notebook one must have Python 3.11 installed and `pipenv`. Once inside this cloned repo directory, execute 
`pipenv install` 
this will install all the necessary dependecies to launch Jupyter and interact with this notebook. You may execute` pipenv shell` in order to swith context to the virtual environment created by pipenv. Execute `jupyter lab` in order to launch Jupyter.

### Cloud and local deployment

In order to build a Docker image you should be on a computer with _x86_ architecture (not on Mac or other arm-based processor-based machines) because the base layer has only been built for x86 machines. You need to `cd dist` from the main directory and then run `docker build -t khvatov/zoomcamp-capstone-ak:1.0.0 .`in order to build Docker image locally. Then you can create and run Docker container by executing `docker run -p 9696:9696 khvatov/zoomcamp-capstone-ak:1.0.0`
<br/>
`dist` directory is used to hold `Pipfile` modified to minimize the size of the created Docker image. 

The Dockerfile included used to build the Docker image uploaded to Docker hub.

You may just pull the docker image or recreate it yourself by running the following command `docker build -t khvatov/zoomcamp-capstone-ak:1.0.0 .` given you are in capstone project directory, have installed the required libraries via `pipenv install` and executed `python train.py` in order to retrain and save the model. I have included the model.bin just in case.

You may run this Docker image using `docker run -p 9696:9696 khvatov/zoomcamp-capstone-ak:1.0.0` - I pushed this image to the Dockerhub

The webservice is going to be available on the port 9696. Please use `web_client.ipynb` notebook to interact with the webservice.