FROM huggingface/transformers-pytorch-gpu:3.3.1
COPY ./ /app
WORKDIR /app
# install requirements
RUN pip install "dvc[gdrive]"
RUN pip install -r requirements.txt

# init dvc
RUN dvc init --no-scm
RUN dvc remote add -d storage gdrive://1wz7gbmVnzqYxfsI2teMYXyYMgX_TFcwv
RUN dvc remote modify storage gdrive_use_service_account true
RUN dvc remote modify storage gdrive_service_account_json_file_path creds.json

RUN dvc pull dvcfiles/best_model.dvc
RUN dvc pull dvcfiles/model_onnx.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 3499
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3499"]

