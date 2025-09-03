from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import BankruptcyData, BankruptcyDataClassifier
from src.pipline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    DataForm class to handle and process incoming form data.
    This class defines the vehicle-related attributes expected from the form.
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.Current_assets: Optional[float] = None
        self.Cost_of_goods_sold: Optional[float] = None
        self.Depreciation_and_amortization: Optional[float] = None
        self.EBDITDA: Optional[float] = None
        self.Inventory: Optional[float] = None
        self.Net_Income: Optional[float] = None
        self.Total_Receivable: Optional[float] = None
        self.Market_Value: Optional[float] = None
        self.Net_Sales: Optional[float] = None
        self.Total_Assets: Optional[float] = None
        self.Total_Long_term_Debt: Optional[float] = None
        self.EBIT: Optional[float] = None
        self.Gross_Profit: Optional[float] = None
        self.Total_Current_Liabilitie: Optional[float] = None
        self.Retained_Earnings: Optional[float] = None
        self.Total_Revenue: Optional[float] = None
        self.Total_Liabilities: Optional[float] = None
        self.Total_Operating_Expenses: Optional[float] = None
        self.Year: Optional[int] = None


    async def get_financial_data(self):
        """
        Method to retrieve and assign form data to class attributes.
        This method is asynchronous to handle form data fetching without blocking.
        """
        form = await self.request.form()
        self.Current_assets = form.get("Current assets")
        self.Cost_of_goods_sold = form.get("Cost of goods sold")
        self.Depreciation_and_amortization = form.get("Depreciation and amortization")
        self.EBDITDA = form.get("EBDITDA")
        self.Inventory = form.get("Inventory")
        self.Net_Income = form.get("Net Income")
        self.Total_Receivable = form.get("Total Receivable")
        self.Market_Value = form.get("Market Value")
        self.Net_Sales = form.get("Net Sales")
        self.Total_Assets = form.get("Total Assets")
        self.Total_Long_term_Debt = form.get("Total Long term Debt")
        self.EBIT = form.get("EBIT")
        self.Gross_Profit = form.get("Gross Profit")
        self.Total_Current_Liabilitie = form.get("Total Current Liabilitie")
        self.Retained_Earnings = form.get("Retained Earnings")
        self.Total_Revenue = form.get("Total Revenue")
        self.Total_Liabilities = form.get("Total Liabilities")
        self.Total_Operating_Expenses = form.get("Total Operating Expenses")
        self.year = form.get("year")


# Route to render the main page with the form
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Renders the main HTML form page for vehicle data input.
    """
    return templates.TemplateResponse(
            "BankruptcyData.html",{"request": request, "context": "Rendering"})

# Route to trigger the model training process
@app.get("/train")
async def trainRouteClient():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

# Route to handle form submission and make predictions
@app.post("/predict")
async def predictRouteClient(request: Request):
    """
    Endpoint to receive form data, process it, and make a prediction.
    """
    try:
        form = DataForm(request)
        await form.get_financial_data()
        
        Bankruptcy_data = BankruptcyData(
                                Current_assets = form.Current_assets,
                                Cost_of_goods_sold = form.Cost_of_goods_sold,
                                Depreciation_and_amortization = form.Depreciation_and_amortization,
                                EBDITDA = form.EBDITDA,
                                Inventory = form.Inventory,
                                Net_Income = form.Net_Income,
                                Total_Receivable = form.Total_Receivable,
                                Market_Value = form.Market_Value,
                                Net_Sales = form.Net_Sales,
                                Total_Assets = form.Total_Assets,
                                Total_Long_term_Debt = form.Total_Long_term_Debt,
                                EBIT = form.EBIT,
                                Gross_Profit = form.Gross_Profit,
                                Total_Current_Liabilitie = form.Total_Current_Liabilitie,
                                Retained_Earnings = form.Retained_Earnings,
                                Total_Revenue = form.Total_Revenue,
                                Total_Liabilities = form.Total_Liabilities,
                                Total_Operating_Expenses = form.Total_Operating_Expenses,
                                year = form.year
                            )

        # Convert form data into a DataFrame for the model
        Bankruptcy_df = Bankruptcy_data.get_bankruptcy_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = BankruptcyDataClassifier()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=Bankruptcy_df)[0]

        # Interpret the prediction result (example: Risk-High or Risk-Low)
        status = "Risk-High" if value == 1 else "Risk-Low"

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "BankruptcyData.html",
            {"request": request, "context": status},
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)