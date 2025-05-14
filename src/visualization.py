# File: src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.config import OUTPUT_DIR, SENTIMENT_CATEGORIES
import os

def set_plotting_style():
    '''Set consistent plotting style'''
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def plot_sentiment_distribution(df):
    '''Plot the distribution of sentiment categories'''
    plt.figure(figsize=(10, 6))
    
    # Count by sentiment category
    sentiment_counts = df['classification'].value_counts().reindex(SENTIMENT_CATEGORIES)
    
    # Create bar plot
    ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    
    plt.title('Distribution of Market Sentiment', fontsize=16)
    plt.xlabel('Sentiment Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(sentiment_counts.values):
        ax.text(i, v + 5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_distribution.png'))

def plot_profit_by_sentiment(df):
    '''Plot profit metrics by sentiment category'''
    plt.figure(figsize=(12, 8))
    
    # Group by sentiment and calculate profit metrics
    sentiment_profit = df.groupby('classification').agg({
        'Closed PnL': ['mean', 'sum', 'count']
    })
    
    sentiment_profit.columns = ['Mean Profit', 'Total Profit', 'Trade Count']
    sentiment_profit = sentiment_profit.reindex(SENTIMENT_CATEGORIES)
    
    # Create plot
    ax = sentiment_profit['Mean Profit'].plot(kind='bar', color='skyblue')
    
    plt.title('Average Profit by Market Sentiment', fontsize=16)
    plt.xlabel('Sentiment Category', fontsize=14)
    plt.ylabel('Average Profit (USD)', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(sentiment_profit['Mean Profit']):
        ax.text(i, v + (0.1 if v >= 0 else -0.1), 
                f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'profit_by_sentiment.png'))

def plot_win_rate_by_sentiment(df):
    '''Plot win rate by sentiment category'''
    plt.figure(figsize=(12, 8))
    
    # Calculate win rate by sentiment
    win_rates = []
    
    for sentiment in SENTIMENT_CATEGORIES:
        sentiment_df = df[df['classification'] == sentiment]
        if len(sentiment_df) > 0:
            win_rate = len(sentiment_df[sentiment_df['Closed PnL'] > 0]) / len(sentiment_df)
            win_rates.append(win_rate)
        else:
            win_rates.append(0)
    
    # Create bar plot
    ax = sns.barplot(x=SENTIMENT_CATEGORIES, y=win_rates)
    
    plt.title('Win Rate by Market Sentiment', fontsize=16)
    plt.xlabel('Sentiment Category', fontsize=14)
    plt.ylabel('Win Rate', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add percentage labels
    for i, v in enumerate(win_rates):
        ax.text(i, v + 0.02, f"{v:.1%}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'win_rate_by_sentiment.png'))

def plot_sentiment_value_time_series(df):
    '''Plot sentiment value over time with trading results'''
    plt.figure(figsize=(16, 12))
    
    # Create daily summary
    daily_summary = df.groupby('Date').agg({
        'value': 'mean',
        'Closed PnL': 'sum',
        'Size USD': 'sum'
    }).reset_index()
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    
    # Plot sentiment value over time
    ax1.plot(daily_summary['Date'], daily_summary['value'], 'b-', linewidth=2)
    ax1.set_title('Sentiment Value Over Time', fontsize=16)
    ax1.set_ylabel('Sentiment Value', fontsize=14)
    ax1.grid(True)
    
    # Plot profit over time
    ax2.bar(daily_summary['Date'], daily_summary['Closed PnL'], color='g')
    ax2.set_title('Daily Trading Profit/Loss', fontsize=16)
    ax2.set_xlabel('Date', fontsize=14)
    ax2.set_ylabel('Profit/Loss (USD)', fontsize=14)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_time_series.png'))

def plot_trade_size_vs_sentiment(df):
    '''Plot trade size vs sentiment value'''
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    ax = sns.scatterplot(x='value', y='Size USD', data=df, hue='Closed PnL', 
                         palette='coolwarm', size='Size USD', sizes=(20, 200), alpha=0.7)
    
    plt.title('Trade Size vs Sentiment Value', fontsize=16)
    plt.xlabel('Sentiment Value', fontsize=14)
    plt.ylabel('Trade Size (USD)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'trade_size_vs_sentiment.png'))

def plot_heatmap_correlation(correlations):
    '''Plot correlation heatmap'''
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    mask = np.triu(np.ones_like(correlations, dtype=bool))
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap='coolwarm', mask=mask,
                vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
    
    plt.title('Correlation Between Metrics', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))

def plot_trading_volume_by_sentiment(df):
    '''Plot trading volume by sentiment category'''
    plt.figure(figsize=(12, 8))
    
    # Group by sentiment and calculate volume
    volume_by_sentiment = df.groupby('classification')['Size USD'].sum().reindex(SENTIMENT_CATEGORIES)
    
    # Create bar plot
    ax = sns.barplot(x=volume_by_sentiment.index, y=volume_by_sentiment.values)
    
    plt.title('Trading Volume by Market Sentiment', fontsize=16)
    plt.xlabel('Sentiment Category', fontsize=14)
    plt.ylabel('Volume (USD)', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, v in enumerate(volume_by_sentiment.values):
        ax.text(i, v + (v * 0.02), f"{v:.0f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'volume_by_sentiment.png'))