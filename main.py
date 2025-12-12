import os
os.environ['SSL_CERT_FILE'] = r"C:\certifi\cacert.pem"
os.environ['REQUESTS_CA_BUNDLE'] = r"C:\certifi\cacert.pem"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from indicators import add_indicators


import yfinance as yf

data = yf.download("AAPL", period="1mo")
data = add_indicators(data)
print(data.tail())

