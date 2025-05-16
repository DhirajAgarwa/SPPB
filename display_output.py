import requests
import base64
from PIL import Image
from io import BytesIO

def display_prediction(ticker):
    url = f"http://localhost:5000/predict?ticker={ticker}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: {response.json().get('error', 'Unknown error')}")
        return
    data = response.json()
    print(f"Ticker: {data['ticker']}")
    print(f"Current Price: {data.get('current_price', 'N/A')}")
    print(f"Best Model: {data['best_model']}")
    print(f"RMSE: {data['best_rmse']}")
    print(f"Best Price Prediction: {data.get('best_price_pred', 'N/A')}")
    print(f"Best Confidence: {data.get('best_confidence', 'N/A')}")
    img_data = base64.b64decode(data['prediction_plot'])
    img = Image.open(BytesIO(img_data))
    save_path = f"{ticker}_prediction_plot.png"
    img.save(save_path)
    print(f"Prediction plot image saved as {save_path}")

if __name__ == "__main__":
    ticker = input("Enter stock ticker symbol: ")
    display_prediction(ticker)
