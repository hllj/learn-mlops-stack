import torch
from model import ColaModel
from data import DataModule
from utils import timing

class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ColaModel.load_from_checkpoint(model_path)
        self.model.to("cuda:0")
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.labels = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]], device="cuda:0"),
            torch.tensor([processed["attention_mask"]], device="cuda:0"),
        )
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})
        return predictions

if __name__ == '__main__':
    sentence = "The boy is sitting on a bench"
    predictor = ColaPredictor("./models/best.ckpt")
    print(predictor.predict(sentence))
    sentences = [
        "The boy is sitting on a bench",
        "The cat is crying on a door",
        "I am a death lover",
        "Je suis vietnamien",
        "J'ai un chien",
        "The boy is sitting on a bench",
        "The cat is crying on a door",
        "I am a death lover",
        "Je suis vietnamien",
        "J'ai un chien",
    ]
    for sentence in sentences:
        predictor.predict(sentence)
