import os
import requests
import numpy as np
import pandas as pd
import time
import logging
from telegram import Bot
from flask import Flask, jsonify
import asyncio
import signal
import sys
import tracemalloc
import talib
from logging.handlers import RotatingFileHandler
import aiohttp
import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Enable memory tracking
tracemalloc.start()

# Configure logging with file rotation
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
handler = RotatingFileHandler('bot_trading.log', maxBytes=5*1024*1024, backupCount=3)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(handler)
logger = logging.getLogger(__name__)
logger.debug("Starting the application.")

# Environment variables
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
PORT = int(os.getenv("PORT", 8001))

if not DISCORD_WEBHOOK_URL:
    logger.error("The DISCORD_WEBHOOK_URL environment variable is missing. Please set it.")
    sys.exit(1)

# Initialize Flask
app = Flask(__name__)

# Configure Flask logging with file rotation
flask_handler = RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=3)
flask_handler.setLevel(logging.INFO)
flask_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
flask_handler.setFormatter(flask_formatter)
app.logger.addHandler(flask_handler)
app.logger.setLevel(logging.INFO)

# Constants
CURRENCY = "USD"
CRYPTO_LIST = ["BTC", "ETH"]
MAX_POSITION_PERCENTAGE = 0.1
CAPITAL = 100
PERFORMANCE_LOG = "trading_performance.csv"
SIGNAL_LOG = "signal_log.csv"

# Function to verify the content of the trading_performance.csv file
def verify_trading_performance():
    try:
        df = pd.read_csv(PERFORMANCE_LOG)
        print(df)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")

# Fetch historical data for cryptocurrencies
async def fetch_historical_data(crypto_symbol, currency="USD", interval="minute", limit=2000, max_retries=5, backoff_factor=2):
    logger.debug(f"Starting to fetch historical data for {crypto_symbol}.")
    base_url = "https://min-api.cryptocompare.com/data/v2/"

    # Determine the correct endpoint based on the interval
    endpoint = "histominute" if interval == "minute" else "histohour"
    url = f"{base_url}{endpoint}"
    params = {
        "fsym": crypto_symbol.upper(),
        "tsym": currency.upper(),
        "limit": limit,
        "api_key": "799a75ef2ad318c38dfebc92c12723e54e5a650c7eb20159a324db632e35a1b4"
    }

    attempt = 0
    while attempt < max_retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

            if data.get("Response") == "Success" and "Data" in data:
                prices = []
                for item in data["Data"].get("Data", []):
                    if all(key in item for key in ["time", "open", "high", "low", "close", "volumeto"]):
                        prices.append({
                            "time": item["time"],
                            "open": item["open"],
                            "high": item["high"],
                            "low": item["low"],
                            "close": item["close"],
                            "volume": item["volumeto"]
                        })

                opens = np.array([item["open"] for item in prices])
                highs = np.array([item["high"] for item in prices])
                lows = np.array([item["low"] for item in prices])
                closes = np.array([item["close"] for item in prices])
                volumes = np.array([item["volume"] for item in prices])

                logger.debug(f"Data fetched for {crypto_symbol}: {len(prices)} items.")
                logger.debug(f"Finished fetching historical data for {crypto_symbol}.")
                return prices, opens, highs, lows, closes, volumes

            else:
                logger.error(f"API error: {data.get('Message', 'Invalid data.')}")
                return [], [], [], [], [], []

        except aiohttp.ClientError as e:
            attempt += 1
            if attempt >= max_retries:
                logger.error(f"Failed after {max_retries} attempts: {e}")
                return [], [], [], [], [], []
            logger.warning(f"Attempt {attempt}/{max_retries} failed, retrying in {backoff_factor ** attempt} seconds.")
            await asyncio.sleep(backoff_factor ** attempt)

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return [], [], [], [], [], []

    logger.error(f"Definitive failure for {crypto_symbol}.")
    return [], [], [], [], [], []

# Function to calculate indicators with TA-Lib
def calculate_indicators(prices):
    logger.debug("Starting to calculate indicators.")
    if len(prices) < 20:
        raise ValueError("Not enough data to calculate indicators.")

    opens = np.array([price["open"] for price in prices])
    highs = np.array([price["high"] for price in prices])
    lows = np.array([price["low"] for price in prices])
    closes = np.array([price["close"] for price in prices])

    # Calculate indicators
    ema_50 = talib.EMA(closes, timeperiod=50)
    ema_200 = talib.EMA(closes, timeperiod=200)
    macd, macd_signal, _ = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
    rsi = talib.RSI(closes, timeperiod=14)
    upper_band, middle_band, lower_band = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)

    logger.debug("Finished calculating indicators.")

    return {
        "EMA_50": ema_50,
        "EMA_200": ema_200,
        "MACD": macd,
        "MACD_Signal": macd_signal,
        "RSI": rsi,
        "Upper_Band": upper_band,
        "Middle_Band": middle_band,
        "Lower_Band": lower_band,
        "Stochastic_K": slowk,
        "Stochastic_D": slowd
    }

# Adjusted function to calculate SL and TP based on ATR with increased multiplier
def calculate_sl_tp(entry_price, signal_type, atr, multiplier=2.0):  # Adjusted multiplier
    logger.debug("Starting to calculate Stop Loss and Take Profit levels.")
    if signal_type == "Buy":
        sl_price = entry_price - (multiplier * atr)
        tp_price = entry_price + (multiplier * atr)
    elif signal_type == "Sell":
        sl_price = entry_price + (multiplier * atr)
        tp_price = entry_price - (multiplier * atr)
    else:
        logger.error("Unknown signal type.")
        return None, None

    logger.debug(f"Stop Loss calculated at: {sl_price}, Take Profit calculated at: {tp_price} (Entry Price: {entry_price})")
    logger.debug("Finished calculating Stop Loss and Take Profit levels.")
    return sl_price, tp_price

# Function to analyze signals
def analyze_signals(prices):
    logger.debug("Starting to analyze signals.")
    indicators = calculate_indicators(prices)

    current_price = prices[-1]["close"]
    ema_50 = indicators["EMA_50"][-1]
    ema_200 = indicators["EMA_200"][-1]
    macd = indicators["MACD"][-1]
    macd_signal = indicators["MACD_Signal"][-1]
    rsi = indicators["RSI"][-1]
    upper_band = indicators["Upper_Band"][-1]
    lower_band = indicators["Lower_Band"][-1]
    slowk = indicators["Stochastic_K"][-1]
    slowd = indicators["Stochastic_D"][-1]

    buy_conditions = (
        current_price > ema_50 > ema_200,
        macd > macd_signal,
        rsi > 30,
        slowk < 20,
        current_price <= lower_band
    )

    sell_conditions = (
        current_price < ema_50 < ema_200,
        macd < macd_signal,
        rsi < 70,
        slowk > 80,
        current_price >= upper_band
    )

    if all(buy_conditions):
        decision = "Buy"
    elif all(sell_conditions):
        decision = "Sell"
    else:
        decision = "Do nothing"

    logger.debug(f"Action decision: {decision}")
    logger.debug("Finished analyzing signals.")
    return decision

# Function to send a message to Discord
async def send_discord_message(webhook_url, message):
    logger.debug(f"Starting to send a Discord message via webhook.")
    data = {
        "content": message
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=data, timeout=10) as response:
                response.raise_for_status()
                response_text = await response.text()
                logger.debug(f"Message sent successfully. Response: {response_text}")
    except aiohttp.ClientError as e:
        logger.error(f"Error sending message to Discord: {e}")
        response_text = await response.text()
        logger.error(f"Response content: {response_text}")
    except asyncio.TimeoutError:
        logger.error("Request timed out.")
    logger.debug("Finished sending Discord message.")

# Function to log memory usage
def log_memory_usage():
    logger.debug("Starting to log memory usage.")
    current, peak = tracemalloc.get_traced_memory()
    logger.info(f"Memory usage - Current: {current / 10**6} MB, Peak: {peak / 10**6} MB")
    tracemalloc.clear_traces()
    logger.debug("Finished logging memory usage.")

# Main trading bot function
async def trading_bot():
    logger.info("Starting trading task.")
    last_sent_signals = {}  # Dictionary to store the last sent signal for each cryptocurrency
    active_positions = {}   # Dictionary to store the state of active positions

    while True:
        logger.info("Starting a new trading iteration.")
        
        for crypto in CRYPTO_LIST:
            logger.debug(f"Fetching historical data for {crypto}.")
            prices, opens, highs, lows, closes, volumes = await fetch_historical_data(crypto, CURRENCY)
            
            if prices:
                logger.debug(f"Data fetched for {crypto}: {prices[-1]}")
                signal = analyze_signals(prices)
                
                if signal:
                    logger.debug(f"Signal analyzed for {crypto}: {signal}")

                    if last_sent_signals.get(crypto) == signal:
                        if active_positions.get(crypto) == signal:
                            logger.info(f"Signal already sent for {crypto} and position is still active. Ignored.")
                            continue
                        else:
                            logger.info(f"Signal already sent for {crypto}, but no active position. Proceeding with new signal.")

                    last_sent_signals[crypto] = signal
                    active_positions[crypto] = signal  # Mark the position as active

                    entry_price = prices[-1]["close"]
                    atr = talib.ATR(highs, lows, closes, timeperiod=7)[-1]
                    sl_price, tp_price = calculate_sl_tp(entry_price, signal, atr, multiplier=2.0)  # Use the new multiplier
                    
                    if sl_price is None or tp_price is None:
                        logger.error(f"Error calculating SL/TP levels for {crypto}")
                        continue
                    
                    message = (f"Trading signal for {crypto}/{CURRENCY}: {signal}\n"
                               f"Entry Price: {entry_price}\n"
                               f"Stop Loss: {sl_price}\n"
                               f"Take Profit: {tp_price}\n")
                    logger.debug(f"Sending Discord message for {crypto}: {message}")
                    await send_discord_message(DISCORD_WEBHOOK_URL, message)
                    logger.info(f"Discord message sent for {crypto}: {signal}")
                
                logger.info(f"Signal generated for {crypto}/{CURRENCY}: {signal}")
            else:
                logger.error(f"Cannot analyze data for {crypto}, data not available.")

        log_memory_usage()

        await asyncio.sleep(60)
        logger.debug("Finished waiting 1 minute.")
    
    logger.info("Finished trading task.")

# Function to send daily summary
async def send_daily_summary(webhook_url):
    logger.debug("Starting to send daily summary on Discord.")
    
    # Call the verification function
    verify_trading_performance()
    
    try:
        df = pd.read_csv(PERFORMANCE_LOG)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        today = datetime.datetime.utcnow().date()
        daily_trades = df[df['timestamp'].dt.date == today]
        
        if not daily_trades.empty:
            summary = daily_trades.to_string(index=False)
            message = f"Daily trading summary for {today}:\n\n{summary}"
        else:
            message = f"No trades made on {today}."

        await send_discord_message(webhook_url, message)
        logger.debug("Daily summary sent successfully.")
    except Exception as e:
        logger.error(f"Error sending daily summary: {e}")

    logger.debug("Finished sending daily summary on Discord.")

# Schedule the daily summary job
scheduler = AsyncIOScheduler()
scheduler.add_job(send_daily_summary, 'interval', days=1, args=[DISCORD_WEBHOOK_URL], next_run_time=datetime.datetime.now() + datetime.timedelta(seconds=10))
scheduler.start()

# Handle shutdown signals
async def handle_shutdown_signal(signum, frame):
    logger.info(f"Shutdown signal received: {signum}")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Clean shutdown of the bot.")
    sys.exit(0)

# Configure signal handlers
def configure_signal_handlers(loop):
    logger.debug("Configuring signal handlers.")
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda sig=sig: asyncio.create_task(handle_shutdown_signal(sig, None)))
    logger.debug("Finished configuring signal handlers.")

# Flask route
@app.route("/")
def home():
    logger.info("Request received on '/'")
    return jsonify({"status": "Trading bot operational."})

# Run Flask application
async def run_flask():
    logger.debug("Starting Flask application.")
    await asyncio.to_thread(app.run, host='0.0.0.0', port=PORT, threaded=True, use_reloader=False, debug=True)
    logger.debug("Finished starting Flask application.")

# Main function
async def main():
    logger.info("Starting main execution.")
    await asyncio.gather(
        trading_bot(),
        run_flask()
    )
    logger.info("Finished main execution.")

# Entry point
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        logger.info("Complete shutdown.")
