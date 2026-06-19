from django.shortcuts import render
from django.http import JsonResponse
import yfinance as yf
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import json

# Path for the pre-trained model and scaler
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'ml_model', 'model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'ml_model', 'scaler.pkl')

# Global variables for caching
_model_cache = None
_scaler_cache = None

from django.views.decorators.csrf import csrf_exempt

def get_model():
    global _model_cache
    if _model_cache is None:
        if os.path.exists(MODEL_PATH):
            _model_cache = load_model(MODEL_PATH)
    return _model_cache

def get_scaler():
    global _scaler_cache
    if _scaler_cache is None:
        if os.path.exists(SCALER_PATH):
            _scaler_cache = joblib.load(SCALER_PATH)
    return _scaler_cache

def home(request):
    return render(request, 'index.html')

@csrf_exempt
def predict_api(request):
    if request.method == 'POST':
        try:
            data_json = json.loads(request.body)
            stock = data_json.get('stock', 'AAPL').upper().strip()
            period = data_json.get('period', '1y') # 1mo, 3mo, 6mo, 1y, 5y
            forecast_days = int(data_json.get('forecast_days', 1))

            # Map UI periods to yfinance periods
            period_map = {
                '1 Month': '1mo',
                '3 Months': '3mo',
                '6 Months': '6mo',
                '1 Year': '1y',
                '5 Years': '5y'
            }
            yf_period = period_map.get(period, '1y')

            # 📥 Fetch data
            df = yf.download(stock, period=yf_period)

            if df.empty:
                return JsonResponse({'error': 'Invalid stock symbol or no data found.'}, status=400)

            # Use the most recent data for prediction
            close_prices = df['Close'].values.reshape(-1, 1)
            
            # Always create a fresh scaler for each stock's price range
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)

            # Load model from cache
            model = get_model()
            if not model:
                return JsonResponse({'error': 'Model file not found on server.'}, status=500)

            # Last 60 days for prediction
            if len(scaled_data) < 60:
                return JsonResponse({'error': 'Not enough data for prediction (need at least 60 days).'}, status=400)
            
            last_60_days = scaled_data[-60:]
            current_batch = last_60_days.reshape(1, 60, 1)
            
            predictions = []
            for _ in range(forecast_days):
                pred = model.predict(current_batch, verbose=0)
                predictions.append(pred[0])
                # Update batch: remove first, append pred
                current_batch = np.append(current_batch[:, 1:, :], [pred], axis=1)

            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            
            # Calculate RMSE on the last portion of fetched data as a proxy for accuracy
            test_data = scaled_data[-80:-20] # previous window
            test_actual = scaled_data[-20:] # actuals
            if len(test_data) == 60:
                test_pred = model.predict(test_data.reshape(1, 60, 1), verbose=0)
                test_pred_inv = scaler.inverse_transform(test_pred)
                test_actual_inv = scaler.inverse_transform(test_actual[0].reshape(-1, 1))
                rmse = np.sqrt(mean_squared_error(test_actual_inv, test_pred_inv))
            else:
                rmse = 0.0

            # Prepare historical data for chart
            history_dates = df.index.strftime('%Y-%m-%d').tolist()
            history_prices = df['Close'].values.flatten().tolist()
            
            current_price = history_prices[-1]
            predicted_price = float(predictions[-1][0])
            price_change = predicted_price - current_price
            percent_change = (price_change / current_price) * 100
            
            trend = "Stable"
            if percent_change > 0.5: trend = "Increasing"
            elif percent_change < -0.5: trend = "Decreasing"

            return JsonResponse({
                'stock': stock,
                'current_price': round(current_price, 2),
                'predicted_price': round(predicted_price, 2),
                'rmse': round(float(rmse), 2),
                'percent_change': round(percent_change, 2),
                'trend': trend,
                'history_dates': history_dates,
                'history_prices': history_prices,
                'forecast_prices': predictions.flatten().tolist(),
                'forecast_days': forecast_days
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)
