# from src.pipline.training_pipeline import TrainPipeline

# pipeline = TrainPipeline()
# pipeline.run_pipeline()

import os
from src.constants import MONGODB_URL_KEY

mongo_url = os.getenv(MONGODB_URL_KEY)
if not mongo_url:
    raise Exception(f"Environment variable '{MONGODB_URL_KEY}' is not set.")
else:
    print(mongo_url)