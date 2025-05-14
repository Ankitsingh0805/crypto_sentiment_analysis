# File: src/data_loader.py

import pandas as pd
import numpy as np
from datetime import datetime
from src.config import HISTORICAL_DATA_PATH, FEAR_GREED_INDEX_PATH

def load_historical_data():
    '''Load and preprocess the historical trading data'''
    df = pd.read_csv(HISTORICAL_DATA_PATH)
    
    # Convert timestamp to datetime
    df['Timestamp IST'] = pd.to_datetime(df['Timestamp IST'], format='%d-%m-%Y %H:%M')
    
    # Extract date for easier joining with sentiment data
    df['Date'] = df['Timestamp IST'].dt.date
    
    # Convert relevant columns to numeric
    numeric_cols = ['Execution Price', 'Size Tokens', 'Size USD', 'Closed PnL', 'Fee']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate additional metrics
    df['Profit_Factor'] = df['Closed PnL'].apply(lambda x: 1 if x >= 0 else 0)  # 1 for profit, 0 for loss
    
    return df

def load_fear_greed_index():
    '''Load and preprocess the fear and greed index data'''
    df = pd.read_csv(FEAR_GREED_INDEX_PATH)
    
    # Convert timestamp to datetime and extract date
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['Date'] = df['datetime'].dt.date
    
    # Ensure the date column matches the format from historical data
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    return df

def merge_datasets():
    '''Merge historical data with sentiment data'''
    historical_df = load_historical_data()
    sentiment_df = load_fear_greed_index()
    
    # Merge datasets on date
    merged_df = pd.merge(
        historical_df, 
        sentiment_df[['Date', 'value', 'classification']], 
        on='Date', 
        how='left'
    )
    
    # Handle any missing sentiment values with forward/backward fill
    merged_df['value'] = merged_df['value'].fillna(method='ffill').fillna(method='bfill')
    merged_df['classification'] = merged_df['classification'].fillna(method='ffill').fillna(method='bfill')
    
    return merged_df