# pull python base image
FROM python:3.10

# specify working directory
WORKDIR /survival_pred_api

ADD /survival_pred_api/requirements.txt .
ADD /survival_pred_api/xgboost-model.pkl .

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# copy application files
ADD /survival_pred_api/app/* ./app/

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]
