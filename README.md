# **IMDb Sentiment Classification with Hugging Face & SageMaker**

This repository demonstrates an end-to-end workflow for training, optimizing, deploying, and running inference on a sentiment classification model using Hugging Face Transformers, PyTorch, ONNX optimization, and Amazon SageMaker.

### The workflow includes:

1. Training and fine-tuning a pre-trained Hugging Face model on IMDb data.
2. Packaging and deploying the trained model as a Docker image in SageMaker for inference.
3. Cleaning up resources post-deployment.

## **Step 1: Train and Optimize the Model**

Notebook: IMDB_DataClassification_BERT_HF_Train_Push_Inference.ipynb

	•	Fetches a pre-trained Hugging Face model for sentiment classification.
	
	•	Fine-tunes the model on IMDb dataset for binary sentiment classification.
	
	•	Performs optimizations:
	
		•	Quantization to reduce model size and improve inference speed.
		•	ONNX export for further model optimization and deployment flexibility.
		
	•	Pushes the trained model to Hugging Face Hub for version control and reuse.

## **Step 2: Prepare and Deploy the Model in SageMaker**

Notebook: SM_HF_Model_Image_Deployment_Inference_Script.ipynb

This notebook demonstrates how to package the trained model for deployment in SageMaker, build a Docker image, deploy it, and perform inference. Multiple files are used in this process:

**Files and their purpose***:

	•	Dockerfile – Defines the Docker image for the inference environment. Uses a public Python image as the base and installs required dependencies.
	•	requirements.txt – Lists Python dependencies required for inference and model loading.
	•	inference.py – Contains the prediction function and inference logic for the deployed model.
	•	buildspec.yml – Defines the CodeBuild pipeline to build the Docker image, push it to ECR, and prepare the SageMaker model.
	
Workflow Steps:
	1.	Package the trained model into a tarball and upload it to Amazon S3.
	2.	Build the Docker image using the Dockerfile, dependencies, and inference script.
	3.	Push the Docker image to Amazon Elastic Container Registry (ECR).
	4.	Define and deploy the SageMaker model using the ECR image and S3 model artifact.
	5.	Create a SageMaker endpoint and run inference using the Predictor API.
	6.	Clean up by deleting the SageMaker endpoint, endpoint configuration, and model.

## **Step 3: Cleanup**
	•	Delete the SageMaker endpoint to stop ongoing usage and charges.
	•	Delete the endpoint configuration from SageMaker.
	•	Delete the deployed SageMaker model to complete the cleanup process.
