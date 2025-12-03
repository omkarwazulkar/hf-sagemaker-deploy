import os
import json
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# -------------------------
# SageMaker-required methods
# -------------------------

def model_fn(model_dir):
    """
    Loads the HuggingFace model from /opt/ml/model.
    This model.tar.gz is automatically downloaded from S3 by SageMaker.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    return {"tokenizer": tokenizer, "model": model}


def input_fn(request_body, request_content_type="application/json"):
    """Parse incoming request."""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        text = data["text"]
        return text
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_text, model_dict):
    """Make prediction and return label index."""
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]

    inputs = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)

    return int(preds.item())


def output_fn(prediction, accept="application/json"):
    """Format output JSON."""
    response = {"prediction": prediction}
    return json.dumps(response), accept


# -------------------------
# Load Model at Container Start
# -------------------------

model_dir = "/opt/ml/model"
model_dict = model_fn(model_dir)


# -------------------------
# HTTP Endpoints
# -------------------------

@app.route("/ping", methods=["GET"])
def ping():
    """Health check."""
    return jsonify({"status": "Healthy"}), 200


@app.route("/invocations", methods=["POST"])
def invoke():
    """Prediction endpoint."""
    data = request.data.decode("utf-8")
    content_type = request.content_type

    text = input_fn(data, content_type)
    prediction = predict_fn(text, model_dict)
    response, content_type = output_fn(prediction, content_type)

    return response, 200, {"Content-Type": content_type}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
