import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            numeric_transformer = StandardScaler()
            # min_max_scaler = MinMaxScaler()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")

            # Load schema configurations
            num_features = self._schema_config['num_features']
            # mm_columns = self._schema_config['mm_columns']
            logging.info("Cols loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),
                    # ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e

    def _map_target_column(self, col):
        """Map Company Status column to 0 for Alive and 1 for Failed."""
        logging.info("Mapping 'Company Status column to binary values")
        col = col.map({'alive': 0, 'failed': 1}) 
        return col
    
    def _feature_construction(self,df):
        """Creates a new feature years_since"""
        df["year"] = df["year"].astype(np.int64)
        current_year = 2025
        df['years_since'] = current_year - df['year']
        return df
    
    # def _create_dummy_columns(self, df):
    #     """Create dummy variables for categorical features."""
    #     logging.info("Creating dummy variables for categorical features")
    #     df = pd.get_dummies(df, drop_first=True)
    #     return df

    # def _rename_columns(self, df):
    #     """Rename specific columns and ensure integer types for dummy columns."""
    #     logging.info("Renaming specific columns and casting to int")
    #     df = df.rename(columns={"status_label" : "Company Status","X1" : "Current assets","X2": "Cost of goods sold",
    #                "X3" : "Depreciation and amortization","X4": "EBDITDA","X5" : "Inventory","X6" : "Net Income",
    #                "X7" : "Total Receivable","X8" : "Market Value","X9" : "Net Sales","X10" : "Total Assets",
    #                "X11" : "Total Long-term Debt","X12" : "EBIT" ,"X13" : "Gross Profit",
    #                "X14" : "Total Current Liabilitie","X15" : "Retained Earnings" , "X16" :"Total Revenue",
    #                "X17" : "Total Liabilities", "X18" : "Total Operating Expenses"},inplace= True)
    #     for col in df.columns:
    #         if col in df.columns:
    #             df[col] = df[col].astype('int')
    #     return df

    def _drop_column(self, df):
        """Drop the Unnecessary column if it exists."""
        logging.info("Dropping Unnecessary column")
        drop_col = self._schema_config['drop_columns']
        for col in drop_col:
            if col in df.columns:    
                df = df.drop(columns=[col],axis= 1)
        return df
        

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            # Apply custom transformations in specified sequence
            target_feature_train_df = self._map_target_column(target_feature_train_df)
            input_feature_train_df = self._drop_column(input_feature_train_df)
            input_feature_train_df = self._feature_construction(input_feature_train_df)
            # input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
            # input_feature_train_df = self._rename_columns(input_feature_train_df)

            target_feature_test_df = self._map_target_column(target_feature_test_df)
            input_feature_test_df = self._drop_column(input_feature_test_df)
            input_feature_test_df = self._feature_construction(input_feature_test_df)
            # input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
            # input_feature_test_df = self._rename_columns(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            logging.info("SMOTEENN applied to train-test df.")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e