from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = None


# -------------------------------
# Upload Dataset
# -------------------------------

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):

    global df

    df = pd.read_csv(file.file)

    rows = df.shape[0]
    cols = df.shape[1]

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    missing_values = df.isnull().sum().to_dict()

    duplicate_rows = int(df.duplicated().sum())

    quality_score = 100 - (sum(missing_values.values()) / (rows*cols) * 100)

    summary = {
        "rows": rows,
        "columns": cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "data_quality_score": round(quality_score,2)
    }

    return summary


# -------------------------------
# Dataset Preview
# -------------------------------

@app.get("/preview")
def preview():
    return df.head(20).to_dict(orient="records")


# -------------------------------
# Missing Value Questions
# -------------------------------

@app.get("/missing_value_questions")
def missing_value_questions():

    missing = df.isnull().sum()

    questions = {}

    for col,val in missing.items():
        if val > 0:
            questions[col] = {
                "missing_count": int(val),
                "question": "How do you want to fill null values?",
                "options": ["fill_zero","fill_mean","fill_median"]
            }

    return questions


# -------------------------------
# Handle Null Values
# -------------------------------

@app.post("/handle_null_values")
def handle_null_values(column:str,method:str):

    if method == "fill_zero":
        df[column] = df[column].fillna(0)

    elif method == "fill_mean":
        df[column] = df[column].fillna(df[column].mean())

    elif method == "fill_median":
        df[column] = df[column].fillna(df[column].median())

    return {"message":f"{column} cleaned using {method}"}


# -------------------------------
# Text Formatting
# -------------------------------

@app.post("/text_format")
def text_format(column:str,format_type:str):

    if format_type == "lower":
        df[column] = df[column].str.lower()

    elif format_type == "upper":
        df[column] = df[column].str.upper()

    elif format_type == "title":
        df[column] = df[column].str.title()

    return {"message":f"{column} formatted using {format_type}"}


# -------------------------------
# Fix Inconsistent Values
# -------------------------------

@app.post("/fix_inconsistent")
def fix_inconsistent(column:str):

    df[column] = df[column].str.lower().str.strip()

    df[column] = df[column].replace({
        "m":"male",
        "male ":"male",
        "f":"female"
    })

    return {"message":f"Inconsistent values fixed in {column}"}


# -------------------------------
# Insights
# -------------------------------

@app.get("/insights")
def insights():

    insights = []

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    for col in numeric_cols:

        top = df[col].max()
        low = df[col].min()
        avg = df[col].mean()

        insights.append(
            f"{col}: Highest value is {top}, lowest value is {low}, average is {round(avg,2)}"
        )

    return {"insights":insights}


# -------------------------------
# Charts
# -------------------------------

@app.get("/charts")
def charts():

    chart_data = []

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    for col in numeric_cols:

        chart_data.append({
            "chart_type":"histogram",
            "column":col,
            "values":df[col].dropna().tolist()
        })

    return {"charts":chart_data}


# -------------------------------
# Business Suggestions
# -------------------------------

@app.get("/business_suggestions")
def business_suggestions():

    suggestions = []

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    for col in numeric_cols:

        avg = df[col].mean()
        max_val = df[col].max()

        if max_val > avg*2:
            suggestions.append(
                f"{col} has unusually high values. Check outliers."
            )

        suggestions.append(
            f"Analyze trends in {col} to improve business performance."
        )

    suggestions.append(
        "Handle missing values and duplicates to improve data quality."
    )

    suggestions.append(
        "Use visualization dashboards for better decision making."
    )

    return {"business_suggestions":suggestions}


# -------------------------------
# Data Story
# -------------------------------

@app.get("/story")
def story():

    rows = df.shape[0]
    cols = df.shape[1]

    return {
        "story":f"This dataset contains {rows} rows and {cols} columns. "
        "It can be used to identify trends, patterns and business insights."
    }


# -------------------------------
# Ask Questions About Data
# -------------------------------

@app.get("/ask")
def ask(question:str):

    q = question.lower()

    if "highest" in q:

        col = df.select_dtypes(include=['int64','float64']).columns[0]
        val = df[col].max()

        return {"answer":f"The highest value in {col} is {val}"}

    if "average" in q:

        col = df.select_dtypes(include=['int64','float64']).columns[0]
        val = df[col].mean()

        return {"answer":f"The average of {col} is {round(val,2)}"}

    if "rows" in q:

        return {"answer":f"The dataset has {df.shape[0]} rows"}

    return {"answer":"I couldn't understand the question yet"}