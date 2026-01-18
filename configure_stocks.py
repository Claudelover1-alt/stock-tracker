#!/usr/bin/env python3
"""
Interactive Stock Configuration Tool
Helps users set up which stocks to track and their target prices
"""

import json
import os

def get_stock_info():
    """Get stock ticker and target price from user"""
    while True:
        ticker = input("Enter stock ticker (e.g., AAPL, TSLA, RCAT): ").strip().upper()
        if ticker and ticker.replace('.', '').replace('-', '').isalnum():
            break
        print("Invalid ticker. Please use standard stock symbols.")
    
    while True:
        try:
            target = float(input(f"Enter target price for {ticker}: $"))
            if target > 0:
                break
            print("Price must be positive.")
        except ValueError:
            print("Please enter a valid number.")
    
    return {"ticker": ticker, "target_price": target, "enabled": True}

def main():
    print("="*60)
    print("STOCK TRACKER CONFIGURATION")
    print("="*60)
    print("\nThis tool helps you set up which stocks to track.")
    print("You can track up to 6 stocks simultaneously.\n")
    
    stocks = []
    
    while len(stocks) < 6:
        print(f"\n--- Stock #{len(stocks) + 1} ---")
        stock = get_stock_info()
        stocks.append(stock)
        
        if len(stocks) < 6:
            add_more = input("\nAdd another stock? (y/n): ").strip().lower()
            if add_more != 'y':
                break
        else:
            print("\nMaximum of 6 stocks reached.")
    
    # Create config
    config = {
        "stocks": stocks,
        "update_interval_seconds": 1,
        "max_stocks": 6
    }
    
    # Save to file
    with open('stocks_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("CONFIGURATION SAVED!")
    print("="*60)
    print(f"\nYou are tracking {len(stocks)} stock(s):")
    for stock in stocks:
        print(f"  • {stock['ticker']} → Target: ${stock['target_price']:.2f}")
    print("\nConfiguration saved to: stocks_config.json")
    print("\nYou can edit stocks_config.json directly anytime to:")
    print("  - Change target prices")
    print("  - Add/remove stocks (max 6)")
    print("  - Enable/disable stocks (set 'enabled' to true/false)")
    print("\nRun 'python server.py' to start tracking!")
    print("="*60)

if __name__ == "__main__":
    main()
