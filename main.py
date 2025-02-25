import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBRegressor
from typing import Optional

# Load the pre-trained model
model = joblib.load('xgboost_model.pkl')

# Constants for Inventory Management
AVG_SALES = 50  # Example: Average sales per day
SAFETY_STOCK_PERCENTAGE = 0.2  # Keep 20% as safety stock
LEAD_TIME_DAYS = 5  # Average lead time in days

# Initialize FastAPI app
app = FastAPI(title="Sales Quantity & Inventory Management API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input data model
class SalesPredictionInput(BaseModel):
    date: str
    daily_sales_percentage: float
    market_share: int
    political: int
    marketing: int
    budget: float
    machineries: str
    region: str

    class Config:
        schema_extra = {
            "example": {
                "date": "2025-01-01",
                "daily_sales_percentage": 0.034463806,
                "market_share": 35,
                "political": 1,
                "marketing": 1,
                "budget": 5000.56,
                "machineries": "Backhoe Loader",
                "region": "Sherrichester"
            }
        }

# Output data model
class PredictionOutput(BaseModel):
    predicted_quantity: float
    safety_stock: float
    reorder_point: float
    inventory_suggestion: str

@app.post("/predict", response_model=PredictionOutput)
async def predict_sales(input_data: SalesPredictionInput):
    try:
        # Add this logging
        print("Received input:", input_data.dict())
        
        # Prepare new data
        new_data = pd.DataFrame({
            'Date': [input_data.date],
            'Daily_Sales _Percentage': [input_data.daily_sales_percentage],
            'Market_Share': [input_data.market_share],
            'Political': [input_data.political],
            'Marketing': [input_data.marketing],
            'Budget': [input_data.budget],
            'Infrastructure_Machineries': [input_data.machineries],
            'Region': [input_data.region]
        })

        # Process new data similar to training data
        new_data['Date'] = pd.to_datetime(new_data['Date'])
        new_data['year'] = new_data['Date'].dt.year
        new_data['month'] = new_data['Date'].dt.month
        new_data['day'] = new_data['Date'].dt.day
        new_data['dayofweek'] = new_data['Date'].dt.dayofweek

        # Ensure all columns match training data
        new_data_encoded = pd.get_dummies(new_data, columns=['Infrastructure_Machineries', 'Region'])
        for col in model.feature_names_in_:
            if col not in new_data_encoded.columns:
                new_data_encoded[col] = 0
        new_data_encoded = new_data_encoded[model.feature_names_in_]

        # Predict sales quantity
        predicted_quantity = float(model.predict(new_data_encoded)[0])

        # Calculate Safety Stock & Reorder Point
        safety_stock = SAFETY_STOCK_PERCENTAGE * predicted_quantity
        reorder_point = (predicted_quantity * LEAD_TIME_DAYS) + safety_stock

        # Inventory Management Suggestions
        if predicted_quantity > AVG_SALES:
            inventory_suggestion = "Increase stock levels to meet demand."
        elif predicted_quantity < AVG_SALES * 0.5:
            inventory_suggestion = "Reduce inventory to avoid overstocking."
        else:
            inventory_suggestion = "Maintain current inventory levels."

        result = PredictionOutput(
            predicted_quantity=predicted_quantity,
            safety_stock=safety_stock,
            reorder_point=reorder_point,
            inventory_suggestion=inventory_suggestion
        )
        
        # Add this logging
        print("Sending response:", result.dict())
        return result

    except Exception as e:
        print("Error:", str(e))  # Add this logging
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)