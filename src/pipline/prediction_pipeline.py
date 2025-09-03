import sys
from src.entity.config_entity import BankruptcyPredictorConfig
from src.entity.mongo_estimator import US_Company_Bankruptcy_Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class BankruptcyData:
    def __init__(self,
                 Current_assets ,
                 Cost_of_goods_sold ,
                 Depreciation_and_amortization ,
                 EBDITDA ,
                 Inventory ,
                 Net_Income ,
                 Total_Receivable ,
                 Market_Value ,
                 Net_Sales ,
                 Total_Assets ,
                 Total_Long_term_Debt ,
                 EBIT ,
                 Gross_Profit ,
                 Total_Current_Liabilitie ,
                 Retained_Earnings ,
                 Total_Revenue ,
                 Total_Liabilities ,
                 Total_Operating_Expenses,
                 year
                ):
        """
        Bankruptcy Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Current_assets = Current_assets
            self.Cost_of_goods_sold = Cost_of_goods_sold 
            self.Depreciation_and_amortization = Depreciation_and_amortization 
            self.EBDITDA = EBDITDA 
            self.Inventory = Inventory 
            self.Net_Income = Net_Income 
            self.Total_Receivable = Total_Receivable 
            self.Market_Value = Market_Value 
            self.Net_Sales = Net_Sales 
            self.Total_Assets = Total_Assets 
            self.Total_Long_term_Debt = Total_Long_term_Debt 
            self.EBIT = EBIT 
            self.Gross_Profit = Gross_Profit 
            self.Total_Current_Liabilitie = Total_Current_Liabilitie 
            self.Retained_Earnings = Retained_Earnings 
            self.Total_Revenue = Total_Revenue 
            self.Total_Liabilities = Total_Liabilities 
            self.Total_Operating_Expenses =  Total_Operating_Expenses
            self.year = year
            

        except Exception as e:
            raise MyException(e, sys) from e

    def get_bankruptcy_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from BankruptcyData class input
        """
        try:
            
            Bankruptcy_input_dict = self.get_bankruptcy_data_as_dict()
            return DataFrame(Bankruptcy_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e


    def get_bankruptcy_data_as_dict(self):
        """
        This function returns a dictionary from VehicleData class input
        """
        logging.info("Entered get_usvisa_data_as_dict method as VehicleData class")

        try:
            input_data = {
                "Current assets": [self.Current_assets],
                "Cost of goods sold": [self.Cost_of_goods_sold],
                "Depreciation and amortization": [self.Depreciation_and_amortization],
                "EBDITDA": [self.EBDITDA],
                "Inventory": [self.Inventory],
                "Net Income": [self.Net_Income],
                "Total Receivable": [self.Total_Receivable],
                "Market Value": [self.Market_Value],
                "Net Sales": [self.Net_Sales],
                "Total Assets": [self.Total_Assets],
                "Total Long term Debt": [self.Total_Long_term_Debt],
                "EBIT": [self.EBIT],
                "Gross Profit": [self.Gross_Profit],
                "Total Current Liabilitie": [self.Total_Current_Liabilitie],
                "Retained Earnings": [self.Retained_Earnings],
                "Total Revenue": [self.Total_Revenue],
                "Total Liabilities": [self.Total_Liabilities],
                "Total Operating Expenses": [self.Total_Operating_Expenses],
                "year": [self.year]
            }

            logging.info("Created Bankruptcy data dict")
            logging.info("Exited get_bankruptcy_data_as_dict method as Bankruptcy Data class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

class BankruptcyDataClassifier:
    def __init__(self,prediction_pipeline_config: BankruptcyPredictorConfig = BankruptcyPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of VehicleDataClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of VehicleDataClassifier class")
            model = US_Company_Bankruptcy_Estimator(
                collection_name=self.prediction_pipeline_config.model_collection_name,
                model_path=self.prediction_pipeline_config.mongo_model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)