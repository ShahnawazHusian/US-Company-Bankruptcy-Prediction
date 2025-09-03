import pickle
import sys
import gridfs
from io import StringIO
import os,sys
from pymongo import MongoClient
from src.logger import logging
from typing import List, Dict, Union
from src.exception import MyException
from pandas import DataFrame,read_csv
import pandas as pd
from src.constants import DATABASE_NAME, MONGODB_URL_KEY, COLLECTION_NAME2,COLLECTION_NAME
import certifi
from bson.binary import Binary
from datetime import datetime
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure, ConfigurationError

ca = certifi.where()

# Load environment variables
# load_dotenv()

# Get environment variables with debug info
MONGODB_URL_KEY = os.getenv(MONGODB_URL_KEY) or os.getenv("MONGODB_URI", MONGODB_URL_KEY)
DATABASE_NAME = os.getenv("DATABASE_NAME", DATABASE_NAME)
COLLECTION_NAME2 = os.getenv("COLLECTION_NAME", COLLECTION_NAME2)

class MongoModelService:
    def __init__(self, mongo_uri: str = MONGODB_URL_KEY, db_name: str = DATABASE_NAME, model_collection: str = COLLECTION_NAME2):
        # Debug: Print configuration (hide password)
        self.debug_connection_info(mongo_uri, db_name, model_collection)
        
        try:
            # Create client with proper timeouts
            self.client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=5000,  # 5 seconds
                connectTimeoutMS=5000,          # 5 seconds
                socketTimeoutMS=5000,           # 5 seconds
                retryWrites=True
            )
            
            # Test the connection immediately
            self.client.admin.command('ping')
            
            self.db = self.client[db_name]
            self.collection = self.db["MLmodel"]
            
            print(f"✅ Successfully connected to MongoDB!")
            print(f"   Database: {db_name}")
            print(f"   Collection: {model_collection}")
            
        except ServerSelectionTimeoutError as e:
            print(f"❌ MongoDB Server Selection Timeout:")
            print(f"   This usually means MongoDB server is not running or URI is incorrect")
            print(f"   Error: {e}")
            raise Exception(f"MongoDB server not accessible. Check if MongoDB is running and URI is correct.")
            
        except ConnectionFailure as e:
            print(f"❌ MongoDB Connection Failure:")
            print(f"   Network connectivity issue or authentication failure")
            print(f"   Error: {e}")
            raise Exception(f"MongoDB connection failed. Check network connectivity and credentials.")
            
        except ConfigurationError as e:
            print(f"❌ MongoDB Configuration Error:")
            print(f"   Invalid connection string format")
            print(f"   Error: {e}")
            raise Exception(f"Invalid MongoDB URI format. Check your connection string.")
            
        except Exception as e:
            print(f"❌ Unexpected MongoDB connection error: {e}")
            raise Exception(f"Failed to connect to MongoDB: {e}")
    
    def debug_connection_info(self, mongo_uri: str, db_name: str, collection: str):
        """Print debug information about the connection"""
        print("🔍 MongoDB Connection Debug Info:")
        
        # Hide password in URI for debug output
        if '@' in mongo_uri and '://' in mongo_uri:
            protocol = mongo_uri.split('://')[0]
            rest = mongo_uri.split('://')[1]
            if '@' in rest:
                credentials_part = rest.split('@')[0]
                host_part = rest.split('@')[1]
                # Hide password
                if ':' in credentials_part:
                    username = credentials_part.split(':')[0]
                    debug_uri = f"{protocol}://{username}:***@{host_part}"
                else:
                    debug_uri = f"{protocol}://{credentials_part}@{host_part}"
            else:
                debug_uri = mongo_uri
        else:
            debug_uri = mongo_uri
            
        print(f"   URI: {debug_uri}")
        print(f"   Database: {db_name}")
        print(f"   Collection: {collection}")
        print()
    
    def test_connection(self):
        """Test MongoDB connection with detailed diagnostics"""
        try:
            # Test basic connectivity
            result = self.client.admin.command('ping')
            print("✅ MongoDB ping successful")
            
            # Test database access
            db_stats = self.db.command("dbstats")
            print(f"✅ Database '{self.db.name}' accessible")
            
            # Test collection access
            collection_count = self.collection.count_documents({})
            print(f"✅ Collection '{self.collection.name}' accessible (documents: {collection_count})")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def save_model(self, model, model_name: str):
        """
        Save a serialized model into MongoDB collection.
        """
        try:
            # Test connection before saving
            if not self.test_connection():
                raise Exception("MongoDB connection test failed before saving model")
            
            print(f"📦 Serializing model '{model_name}'...")
            model_obj = pickle.dumps(model)
            model_size_mb = len(model_obj) / (1024 * 1024)
            print(f"   Model size: {model_size_mb:.2f} MB")
            
            # Check if model is too large for regular collection (16MB limit)
            if len(model_obj) > 16 * 1024 * 1024:  # 16MB
                raise Exception(f"Model too large ({model_size_mb:.2f} MB). Use GridFS for models > 16MB")
            
            # Create document with metadata
            document = {
                "name": model_name,
                "model": Binary(model_obj),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "size_bytes": len(model_obj),
                "size_mb": round(model_size_mb, 2)
            }

            print(f"💾 Saving model to MongoDB...")
            # Update or insert the model
            result = self.collection.update_one(
                {"name": model_name},
                {"$set": document},
                upsert=True
            )
            
            if result.upserted_id:
                logging.info(f"Model '{model_name}' saved with new ID: {result.upserted_id}")
                print(f"✅ Model '{model_name}' saved successfully (new document)")
                return str(result.upserted_id)
            else:
                logging.info(f"Model '{model_name}' updated successfully")
                print(f"✅ Model '{model_name}' updated successfully")
                # Get the existing document ID
                existing_doc = self.collection.find_one({"name": model_name}, {"_id": 1})
                return str(existing_doc["_id"]) if existing_doc else None
                
        except ServerSelectionTimeoutError:
            error_msg = f"MongoDB connection timeout while saving model '{model_name}'. Check if MongoDB server is running."
            logging.error(error_msg)
            raise Exception(error_msg)
            
        except ConnectionFailure:
            error_msg = f"MongoDB connection failed while saving model '{model_name}'. Check network connectivity."
            logging.error(error_msg)
            raise Exception(error_msg)
            
        except Exception as e:
            error_msg = f"Failed to save model '{model_name}': {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)
        
    def load_model(self, model_name: str):
        """
        Load a serialized model from MongoDB collection.
        """
        try:
            if not self.test_connection():
                raise Exception("MongoDB connection test failed before loading model")
            
            print(f"📥 Loading model '{model_name}' from MongoDB...")
            doc = self.collection.find_one({"name": model_name})

            if not doc:
                raise Exception(f"Model '{model_name}' not found in MongoDB")

            if "model" not in doc:
                raise Exception(f"No 'model' field found in document for '{model_name}'")

            model_bin = doc["model"]

            # If it's pymongo.binary.Binary → convert to bytes
            if isinstance(model_bin, Binary):
                model_bin = bytes(model_bin)

            # Deserialize the ML model
            model = pickle.loads(model_bin)
            print(f"✅ Model '{model_name}' loaded successfully ({type(model)})")
            return model

        except Exception as e:
            error_msg = f"Failed to load model '{model_name}': {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)

    
        
    # def load_model(self,model_name):
    #     record = self.collection.find_one({"name": model_name})
    #     if not record:
    #         raise FileNotFoundError(f"Model '{model_name}' not found in MongoDB")

    #     model_binary = record["model"]

    #     # Ensure we have bytes
    #     if isinstance(model_binary, Binary):
    #         model_bytes = bytes(model_binary)
    #     else:
    #         model_bytes = model_binary

    #     # ✅ Unpickle to actual sklearn model
    #     model = pickle.loads(model_bytes)

    #     if not hasattr(model, "predict"):
    #         raise TypeError(f"Loaded object is {type(model)}, not a ML model")
        
    # def load_model(self, model_name: str):
    #     """
    #     Load a serialized model from the model collection.
    #     """
    #     try:
    #         record = self.collection.find_one({"model_name": model_name})
    #         if not record:
    #             raise FileNotFoundError(f"Model '{model_name}' not found in MongoDB collection")

    #         model = pickle.loads(record["model_blob"])
    #         logging.info(f"Model '{model_name}' loaded from MongoDB collection")
    #         return model
    #     except Exception as e:
    #         raise MyException(e, sys) from e
    
    def model_available(self, model_name: str) -> bool:
        """
        Checks if a specified model (file) is available in MongoDB GridFS.

        Args:
            model_name (str): Name of the model file in GridFS.

        Returns:
            bool: True if the model exists, False otherwise.
        """
        try:
            file_object = self.fs.find_one({"filename": model_name})
            return file_object is not None
        except Exception as e:
            raise MyException(e, sys) from e

    # def insert_data(self, record: dict):
    #     """
    #     Insert a data record into the data collection.
    #     """
    #     try:
    #         result = self.data_collection.insert_one(record)
    #         logging.info(f"Data inserted with ID {result.inserted_id}")
    #         return result.inserted_id
    #     except Exception as e:
    #         raise MyException(e, sys) from e

    # def fetch_data(self, query: dict = {}):
    #     """
    #     Fetch data records from the data collection.
    #     """
    #     try:
    #         records = list(self.data_collection.find(query))
    #         return records
    #     except Exception as e:
    #         raise MyException(e, sys) from e
