# File: src/analysis.py

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from src.config import SENTIMENT_CATEGORIES

def calculate_trading_metrics(df):
    '''Calculate various trading metrics'''
    metrics = {}
    
    # Overall metrics
    metrics['total_trades'] = len(df)
    metrics['profitable_trades'] = len(df[df['Closed PnL'] > 0])
    metrics['losing_trades'] = len(df[df['Closed PnL'] <= 0])
    
    if metrics['total_trades'] > 0:
        metrics['win_rate'] = metrics['profitable_trades'] / metrics['total_trades']
    else:
        metrics['win_rate'] = 0
        
    metrics['total_profit'] = df['Closed PnL'].sum()
    metrics['avg_profit_per_trade'] = df['Closed PnL'].mean()
    metrics['max_profit'] = df['Closed PnL'].max()
    metrics['max_loss'] = df['Closed PnL'].min()
    
    # Calculate profit factor (sum of profits / sum of losses)
    profits = df[df['Closed PnL'] > 0]['Closed PnL'].sum()
    losses = abs(df[df['Closed PnL'] < 0]['Closed PnL'].sum())
    
    if losses > 0:
        metrics['profit_factor'] = profits / losses
    else:
        metrics['profit_factor'] = float('inf') if profits > 0 else 0
    
    return metrics

def analyze_sentiment_performance(df):
    '''Analyze performance metrics grouped by sentiment category'''
    sentiment_metrics = {}
    
    for sentiment in df['classification'].unique():
        sentiment_df = df[df['classification'] == sentiment]
        sentiment_metrics[sentiment] = calculate_trading_metrics(sentiment_df)
        
    return pd.DataFrame(sentiment_metrics).T

def calculate_sentiment_transition_matrix(df):
    '''Calculate sentiment transition probabilities'''
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp')
    
    # Create shifted column for next sentiment
    df_sorted['next_sentiment'] = df_sorted['classification'].shift(-1)
    
    # Count transitions
    transitions = df_sorted.groupby(['classification', 'next_sentiment']).size().unstack(fill_value=0)
    
    # Convert to probabilities
    for idx in transitions.index:
        row_sum = transitions.loc[idx].sum()
        if row_sum > 0:
            transitions.loc[idx] = transitions.loc[idx] / row_sum
    
    return transitions

def calculate_performance_by_account(df):
    '''Calculate performance metrics grouped by account'''
    account_metrics = {}
    
    for account in df['Account'].unique():
        account_df = df[df['Account'] == account]
        account_metrics[account] = calculate_trading_metrics(account_df)
        
    return pd.DataFrame(account_metrics).T

def analyze_side_by_sentiment(df):
    '''Analyze trading side (buy/sell) distribution by sentiment'''
    side_sentiment = pd.crosstab(df['classification'], df['Side'])
    side_sentiment_pct = side_sentiment.div(side_sentiment.sum(axis=1), axis=0)
    
    return side_sentiment, side_sentiment_pct

def calculate_sentiment_correlation(df):
    '''Calculate correlation between sentiment value and trading metrics'''
    # Group by date and calculate daily metrics
    daily_metrics = df.groupby('Date').agg({
        'Closed PnL': 'sum',
        'Size USD': 'sum',
        'value': 'mean',  # sentiment value
        'Side': lambda x: (x == 'BUY').mean()  # Buy ratio
    }).reset_index()
    
    # Calculate correlations
    numeric_metrics = daily_metrics.select_dtypes(include='number')
    correlations = numeric_metrics.corr()
    
    return correlations