# File: main.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.config import OUTPUT_DIR
from src.data_loader import load_historical_data, load_fear_greed_index, merge_datasets
from src.analysis import (
    calculate_trading_metrics, 
    analyze_sentiment_performance, 
    calculate_sentiment_transition_matrix,
    calculate_performance_by_account,
    analyze_side_by_sentiment,
    calculate_sentiment_correlation
)
from src.visualization import (
    set_plotting_style,
    plot_sentiment_distribution,
    plot_profit_by_sentiment,
    plot_win_rate_by_sentiment,
    plot_sentiment_value_time_series,
    plot_trade_size_vs_sentiment,
    plot_heatmap_correlation,
    plot_trading_volume_by_sentiment
)

def main():
    print("Starting Crypto Trading Sentiment Analysis...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set plotting style
    set_plotting_style()
    
    # Load and merge datasets
    print("Loading and preprocessing data...")
    historical_data = load_historical_data()
    sentiment_data = load_fear_greed_index()
    merged_data = merge_datasets()
    
    # Print basic dataset information
    print(f"\nHistorical Trading Data: {len(historical_data)} records")
    print(f"Sentiment Data: {len(sentiment_data)} records")
    print(f"Merged Dataset: {len(merged_data)} records")
    
    # Calculate overall trading metrics
    print("\nCalculating overall trading metrics...")
    overall_metrics = calculate_trading_metrics(historical_data)
    
    # Print overall metrics
    print("\nOverall Trading Metrics:")
    print(f"Total Trades: {overall_metrics['total_trades']}")
    print(f"Win Rate: {overall_metrics['win_rate']:.2%}")
    print(f"Total Profit: ${overall_metrics['total_profit']:.2f}")
    print(f"Profit Factor: {overall_metrics['profit_factor']:.2f}")
    print(f"Average Profit per Trade: ${overall_metrics['avg_profit_per_trade']:.2f}")
    
    # Analyze performance by sentiment
    print("\nAnalyzing performance by sentiment...")
    sentiment_performance = analyze_sentiment_performance(merged_data)
    
    # Save sentiment performance to CSV
    sentiment_performance.to_csv(os.path.join(OUTPUT_DIR, 'sentiment_performance.csv'))
    print(f"Sentiment performance metrics saved to {os.path.join(OUTPUT_DIR, 'sentiment_performance.csv')}")
    
    # Calculate sentiment transition matrix
    print("\nCalculating sentiment transition probabilities...")
    transition_matrix = calculate_sentiment_transition_matrix(sentiment_data)
    
    # Save transition matrix to CSV
    transition_matrix.to_csv(os.path.join(OUTPUT_DIR, 'sentiment_transitions.csv'))
    print(f"Sentiment transition matrix saved to {os.path.join(OUTPUT_DIR, 'sentiment_transitions.csv')}")
    
    # Analyze performance by account
    print("\nAnalyzing performance by account...")
    account_performance = calculate_performance_by_account(historical_data)
    
    # Save account performance to CSV
    account_performance.to_csv(os.path.join(OUTPUT_DIR, 'account_performance.csv'))
    print(f"Account performance metrics saved to {os.path.join(OUTPUT_DIR, 'account_performance.csv')}")
    
    # Analyze trading side by sentiment
    print("\nAnalyzing trading side distribution by sentiment...")
    side_sentiment, side_sentiment_pct = analyze_side_by_sentiment(merged_data)
    
    # Save side by sentiment analysis to CSV
    side_sentiment.to_csv(os.path.join(OUTPUT_DIR, 'side_by_sentiment.csv'))
    side_sentiment_pct.to_csv(os.path.join(OUTPUT_DIR, 'side_by_sentiment_pct.csv'))
    print(f"Trading side by sentiment analysis saved to {os.path.join(OUTPUT_DIR, 'side_by_sentiment.csv')}")
    
    # Calculate correlation between sentiment and trading metrics
    print("\nCalculating correlation between sentiment and trading metrics...")
    correlations = calculate_sentiment_correlation(merged_data)
    
    # Save correlations to CSV
    correlations.to_csv(os.path.join(OUTPUT_DIR, 'sentiment_correlations.csv'))
    print(f"Sentiment correlations saved to {os.path.join(OUTPUT_DIR, 'sentiment_correlations.csv')}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot sentiment distribution
    plot_sentiment_distribution(sentiment_data)
    print(f"Sentiment distribution plot saved to {os.path.join(OUTPUT_DIR, 'sentiment_distribution.png')}")
    
    # Plot profit by sentiment
    plot_profit_by_sentiment(merged_data)
    print(f"Profit by sentiment plot saved to {os.path.join(OUTPUT_DIR, 'profit_by_sentiment.png')}")
    
    # Plot win rate by sentiment
    plot_win_rate_by_sentiment(merged_data)
    print(f"Win rate by sentiment plot saved to {os.path.join(OUTPUT_DIR, 'win_rate_by_sentiment.png')}")
    
    # Plot sentiment value time series
    plot_sentiment_value_time_series(merged_data)
    print(f"Sentiment time series plot saved to {os.path.join(OUTPUT_DIR, 'sentiment_time_series.png')}")
    
    # Plot trade size vs sentiment
    plot_trade_size_vs_sentiment(merged_data)
    print(f"Trade size vs sentiment plot saved to {os.path.join(OUTPUT_DIR, 'trade_size_vs_sentiment.png')}")
    
    # Plot correlation heatmap
    plot_heatmap_correlation(correlations)
    print(f"Correlation heatmap saved to {os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')}")
    
    # Plot trading volume by sentiment
    plot_trading_volume_by_sentiment(merged_data)
    print(f"Trading volume by sentiment plot saved to {os.path.join(OUTPUT_DIR, 'volume_by_sentiment.png')}")
    
    print("\nAnalysis complete! Results are saved in the 'output' directory.")
    
    return merged_data, sentiment_performance, correlations

if __name__ == "__main__":
    merged_data, sentiment_performance, correlations = main()