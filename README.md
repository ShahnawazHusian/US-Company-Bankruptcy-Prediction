# MLOps Project - US Bankruptcy Data Pipeline

Welcome to this MLOps project, designed to demonstrate a robust pipeline for managing us bankruptcy data. This project aims to impress recruiters and visitors by showcasing the various tools, techniques, services, and features that go into building and deploying a machine learning pipeline for real-world data management. Follow along to learn about project setup, data processing, model deployment, and CI/CD automation!

---

## 📁 Project Setup and Structure

### Step 1: Project Template
- Start by executing the `template.py` file to create the initial project template, which includes the required folder structure and placeholder files.

### Step 2: Package Management
- Write the setup for importing local packages in `setup.py` and `pyproject.toml` files.
- **Tip**: Learn more about these files from `project.txt`.

### Step 3: Virtual Environment and Dependencies
- Create a virtual environment and install required dependencies from `requirements.txt`:
  ```bash
  python -m venv myenv
  myenv/Scripts/activate
  pip install -r requirements.txt
  ```
- Verify the local packages by running:
  ```bash
  pip list
  ```

---

## 📊 MongoDB Setup and Data Management

### Step 4: MongoDB Atlas Configuration
1. Sign up for [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) and create a new project.
2. Set up a free M0 cluster, configure the username and password, and allow access from any IP address (`0.0.0.0/0`).
3. Retrieve the MongoDB connection string for Python and save it (replace `<password>` with your password).

### Step 5: Pushing Data to MongoDB
1. Create a folder named `notebook`, add the dataset, and create a notebook file `mongoDB_demo.ipynb`.
2. Use the notebook to push data to the MongoDB database.
3. Verify the data in MongoDB Atlas under Database > Browse Collections.

---

## 📝 Logging, Exception Handling, and EDA

### Step 6: Set Up Logging and Exception Handling
- Create logging and exception handling modules. Test them on a demo file `demo.py`.

### Step 7: Exploratory Data Analysis (EDA) and Feature Engineering
- Analyze and engineer features in the `EDA` and `Feature Engg` notebook for further processing in the pipeline.

---

## 📥 Data Ingestion

### Step 8: Data Ingestion Pipeline
- Define MongoDB connection functions in `configuration.mongo_db_connections.py`.
- Develop data ingestion components in the `data_access` and `components.data_ingestion.py` files to fetch and transform data.
- Update `entity/config_entity.py` and `entity/artifact_entity.py` with relevant ingestion configurations.
- Run `demo.py` after setting up MongoDB connection as an environment variable.

### Setting Environment Variables
- Set MongoDB URL:
  ```bash
  # For Bash
  export MONGODB_URL="mongodb+srv://<username>:<password>...."
  # For Powershell
  $env:MONGODB_URL = "mongodb+srv://<username>:<password>...."
  ```
- **Note**: On Windows, you can also set environment variables through the system settings.

---

## 🔍 Data Validation, Transformation & Model Training

### Step 9: Data Validation
- Define schema in `config.schema.yaml` and implement data validation functions in `utils.main_utils.py`.

### Step 10: Data Transformation
- Implement data transformation logic in `components.data_transformation.py` and create `estimator.py` in the `entity` folder.

### Step 11: Model Training
- Define and implement model training steps in `components.model_trainer.py` using code from `estimator.py`.

---

### 🌐 MongoDB Atlas Setup for Data Storage

## Step 12: MongoDB Atlas Setup

- Log in to the MongoDB Atlas Console.
-  Create a free cluster.
-  Add a database named US-Company-Bankruptcy.
- Create a user with username/password authentication.
- Copy your connection string (example --> `mongodb+srv://<username>:<password>@cluster0.abcd.mongodb.net/US-Company-Bankruptcy`)
- Set MongoDB connection as an environment variable.
```bash
# For Bash
export MONGODB_URL="your-mongodb-connection-string"
# For Powershell
$env:MONGODB_URL="your-mongodb-connection-string"
```

3. Configure MongoDB Atlas connection and add credentials in constants.__init__.py.

### Step 13: Model Evaluation and Pushing to MongoDB

Create a MongoDB Atlas database named `US-Company-Bankruptcy`.
Add a collection named MLmodel to store model metadata (model name, version, accuracy, timestamp).
Save trained models as model.pkl locally in the `artifact/ directory`.
Develop code to push/pull model metadata to/from MongoDB in src.mongo_storage and `entity/mongo_estimator.py`.

## 🚀 Model Evaluation, Model Pusher, and Prediction Pipeline

### Step 14: Model Evaluation & Model Pusher
- Implement model evaluation and deployment components.
- Create `Prediction Pipeline` and set up `app.py` for API integration.

### Step 15: Static and Template Directory
- Add `static` and `template` directories for web UI.

---

## 🔄 CI/CD Setup with Docker, GitHub Actions, and MongoDB

### Step 16: Docker and GitHub Actions

1. Create a `Dockerfile` and `.dockerignore`.
2. Set up GitHub Actions workflow to:
    - Build the Docker image.
    - Push the image to Docker Hub.
    - Create GitHub secrets for:
        - `DOCKERHUB_USERNAME`
        - `DOCKERHUB_TOKEN`
        - `MONGODB_URL`

### Step 17: Deployment with Docker

1. Install Docker Desktop (for local deployment) or use any VM/cloud server.
2. Pull the image from Docker Hub:
    ```bash
    # For Bash
    - "docker pull <your-dockerhub-username>/us-bankruptcy-data-prediction:latest"
    ```


### Step 18: Final Steps

1. Run your container locally (Docker Desktop) or on any VM/Cloud with Docker installed.
2. Expose the app on port 5000 (or any port you set in APP_PORT).
    ```bash
    # For Bash
    - "docker run -d -p 5000:5000 -e MONGODB_URL="your-mongo-url" <your-dockerhub-username>/us-bankruptcy-data-prediction:latest"
    ```
3. Access the deployed app by visiting (`http://localhost:5000`)

---

## 🛠️ Additional Resources
- **Crash Course on setup.py and pyproject.toml**: See `project.txt` for details.
- **GitHub Secrets**: Manage secrets for secure CI/CD pipelines.

---

## 🎯 Project Workflow Summary

1. **Data Ingestion** ➔ **Data Validation** ➔ **Data Transformation**
2. **Model Training** ➔ **Model Evaluation** ➔ **Model Deployment**
3. **CI/CD Automation** with GitHub Actions, Docker,MongoDB

---

## 💬 Connect
If you found this project helpful or have any questions, feel free to reach out!

---

This README provides a structured walkthrough of the MLOps project, showcasing the end-to-end pipeline, cloud integration, CI/CD setup, and robust data handling capabilities.