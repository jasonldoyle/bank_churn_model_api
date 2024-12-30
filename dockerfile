FROM python:3.6-slim
COPY ./model_api.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./best_rf_model.pkl /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "model_api.py"]