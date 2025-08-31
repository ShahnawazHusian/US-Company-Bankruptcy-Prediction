from src.cloud_storage.mongo_storage import MongoModelService
from src.exception import MyException
from src.entity.estimator import MyModel
import sys
from pandas import DataFrame


class US_Company_Bankruptcy_Estimator:
    """
    This class is used to save and retrieve our model from mongo collection and to do prediction
    """

    def __init__(self,collection_name,model_path,):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.collection_name = collection_name
        self.mongo = MongoModelService()
        self.model_path = model_path
        self.loaded_model:MyModel=None


    def is_model_present(self,model_path):
        try:
            return self.mongo.model_available(model_name=model_path)
        except MyException as e:
            print(e)
            return False

    def load_model(self,)->MyModel:
        """
        Load the model from the model_path
        :return:
        """

        return self.mongo.load_model(self.model_path)

    def save_model(self,from_file)->None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            self.mongo.save_model(model = from_file,
                                model_name=self.model_path,
                                )
        except Exception as e:
            raise MyException(e, sys)


    def predict(self,dataframe:DataFrame):
        """
        :param dataframe:
        :return:
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise MyException(e, sys)