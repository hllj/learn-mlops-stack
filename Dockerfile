FROM amazon/aws-lambda-python
# FROM huggingface/transformers-pytorch-cpu:latest
# COPY ./ /app
# WORKDIR /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG MODEL_DIR=./models
RUN mkdir $MODEL_DIR

ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
	AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

RUN yum install git -y && yum -y install gcc-c++


COPY ./ ./
ENV PYTHONPATH "${PYTHONPATH}:./"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# install requirements
RUN pip install "dvc[s3]" --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir

# init dvc
RUN dvc init --no-scm -f
RUN dvc remote add -d model-store s3://mlops-dvc-1/models/

RUN cat .dvc/config

RUN dvc pull dvcfiles/best_model.dvc
RUN dvc pull dvcfiles/model_onnx.dvc

# EXPOSE 3499
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3499"]

RUN python lambda_handler.py
RUN chmod -R 0755 $MODEL_DIR
CMD [ "lambda_handler.lambda_handler"]
