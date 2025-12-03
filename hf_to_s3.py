from transformers import AutoModelForCausalLM, AutoTokenizer
import boto3
import subprocess
import os

# ----------------------------
# Config
# ----------------------------
model_name = "omkarwazulkar/SentimentModel-V1.0"   # Hugging Face repo
s3_bucket = "648758970526-us-east-1-models"  # S3 bucket
tar_file_path = "model.tar.gz"
s3_key = os.path.basename(tar_file_path)

# ----------------------------
# Step 1: Download Hugging Face model
# ----------------------------
model_folder = "hf_model"
os.makedirs(model_folder, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(model_folder)
tokenizer.save_pretrained(model_folder)
print(f"Downloaded Hugging Face model '{model_name}' to '{model_folder}'")

# ----------------------------
# Step 2: Compress the model folder
# ----------------------------
if os.path.exists(tar_file_path):
    os.remove(tar_file_path)

subprocess.run(
    ["tar", "-czvf", tar_file_path, "-C", model_folder, "."],
    check=True
)
print(f"Compressed model folder to '{tar_file_path}'")

# ----------------------------
# Step 3: Upload to S3
# ----------------------------
s3_client = boto3.client("s3")
s3_client.upload_file(tar_file_path, s3_bucket, s3_key)
print(f"Uploaded model to s3://{s3_bucket}/{s3_key}")

# ----------------------------
# Step 4: Output S3 location
# ----------------------------
model_data = f"s3://{s3_bucket}/{s3_key}"
print(f"Model Data S3 Location: {model_data}")
