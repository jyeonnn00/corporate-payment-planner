import requests
import pandas as pd
import datetime

# CURRENT DAY DATE
current_date = datetime.date.today()
current_year = current_date.year
current_month = current_date.month

print(f"Today is: {current_date}")
print(f"Fetching data up to Year: {current_year}, Month: {current_month}...")

years_to_fetch = [current_year - 1, current_year]
all_data = []


# FETCHING DATA FROM BMN API
for year in years_to_fetch:
    # If its the current year, only loop up to the current month
    # If its past year, all 12 months will be taken
    if year == current_year:
        months = range(1, current_month + 1)
    else:
        months = range(1, 13)

    for month in months:
        url = f"https://api.bnm.gov.my/public/usd-interbank-intraday-rate/year/{year}/month/{month}"
        headers = {"Accept": "application/vnd.BNM.API.v1+json"}

        try:
            response = requests.get(url, headers=headers, timeout=5)
            
            # To check whether BNM have data to provide or no
            if response.status_code == 200:
                json_data = response.json()
                if 'data' in json_data and json_data['data']:
                    df_month = pd.json_normalize(json_data['data'])
                    
                    if 'date' in df_month.columns:
                        cols = ['date', 'highest_rate', 'lowest_rate']
                        existing_cols = [c for c in cols if c in df_month.columns]
                        all_data.append(df_month[existing_cols])
                        print(f"Got data for {year}-{month}")
                else:
                    pass
        except Exception as e:
            print(f" Error connection for {year}-{month}")


# SAVE AND CALCULATE AVERAGE
if all_data:
    df_final = pd.concat(all_data, ignore_index=True)
    
    # To make sure app.py didnt crash, average will be calculated here
    df_final['highest_rate'] = pd.to_numeric(df_final['highest_rate'])
    df_final['lowest_rate'] = pd.to_numeric(df_final['lowest_rate'])
    df_final['average_rate'] = (df_final['highest_rate'] + df_final['lowest_rate']) / 2
    
    # Save to CSV "historical_usd_myr"
    df_final.to_csv("historical_usd_myr.csv", index=False)
    print(f"\nSUCCESS!! Saved {len(df_final)} days of data.")
    print(f"Data goes from {df_final['date'].min()} to {df_final['date'].max()}")
else:
    print("No data found.")