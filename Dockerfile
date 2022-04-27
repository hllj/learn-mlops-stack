FROM huggingface/transformers-pytorch-gpu:3.3.1
COPY ./ /app
WORKDIR /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
	AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# install requirements
RUN pip install "dvc[s3]"
RUN pip install -r requirements.txt

# init dvc
RUN dvc init --no-scm
RUN dvc remote add -d model-store s3://mlops-dvc-1/models/

RUN cat .dvc/config

RUN dvc pull dvcfiles/best_model.dvc
RUN dvc pull dvcfiles/model_onnx.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 3499
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3499"]

