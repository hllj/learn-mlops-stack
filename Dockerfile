FROM huggingface/transformers-pytorch-gpu:latest
COPY ./ /app
WORKDIR /app
RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html 
RUN pip install -r requirements.txt
EXPOSE 3499
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3499"]

