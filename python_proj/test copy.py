import MetaTrader5 as mt5

# Connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed:", mt5.last_error())
else:
    print("MT5 initialized successfully")

# Get account info
account_info = mt5.account_info()
if account_info:
    print("Account info:")
    print(account_info._asdict())
else:
    print("Failed to get account info")

# Shutdown
mt5.shutdown()
