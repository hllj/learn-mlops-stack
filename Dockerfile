FROM huggingface/transformers-pytorch-gpu:3.3.1
COPY ./ /app
WORKDIR /app
RUN pip install -r requirements.txt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 3499
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3499"]

