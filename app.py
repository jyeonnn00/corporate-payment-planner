import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import datetime

# PAGE SETUP
st.set_page_config(page_title="USD/MYR Currency Payment Planner", layout="wide")


# LOAD DATA
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("historical_usd_myr.csv")
    except FileNotFoundError:
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    # calculate average if missing
    if 'average_rate' not in df.columns or df['average_rate'].isnull().any():
        df['average_rate'] = (df['highest_rate'] + df['lowest_rate']) / 2
    df = df.sort_values('date')
    df['day_id'] = np.arange(len(df))
    return df

df = load_data()

if df is None:
    st.error("CSV file not found. Please run get_data.py file first.")
    st.stop()


# MODEL CALCULATION
X = df[['day_id']]
y = df['average_rate']

# use two models to get a more accurate prediction
model_1 = LinearRegression()
model_2 = RandomForestRegressor(n_estimators=100, random_state=42)
final_model = VotingRegressor([('lr', model_1), ('rf', model_2)])
final_model.fit(X, y)

# use R-Squared to calculate confidence level of the prediction
r2_score = final_model.score(X, y)
confidence_pct = r2_score * 100
y_pred_internal = final_model.predict(X)
mae = np.mean(np.abs(y - y_pred_internal))

# predict next 30 days
last_day_id = df['day_id'].max()
last_date_in_data = df['date'].max()

future_days = np.array([[last_day_id + i] for i in range(1, 32)])
future_predictions = final_model.predict(future_days) 
future_dates = [last_date_in_data + datetime.timedelta(days=i) for i in range(1, 32)]

# key metrics
next_day_rate = future_predictions[0]
current_rate = df['average_rate'].iloc[-1]
next_day_date = future_dates[0].strftime("%d %b %Y")
difference = next_day_rate - current_rate


# SIDEBAR (Calculator + Data Reference)
st.sidebar.header("Tools & Data")

# Calculator
st.sidebar.subheader("Today Currency Convertor")
usd_input = st.sidebar.number_input("USD Amount", min_value=1.0, value=10.0, step=1.0)
myr_output = usd_input * current_rate
st.sidebar.write(f"= RM {myr_output:,.2f}")
st.sidebar.caption(f"Rate: {current_rate:.3f}")
st.sidebar.markdown("---")

# Raw Data Table
st.sidebar.subheader("Forecast Reference")
future_df = pd.DataFrame({
    "Date": [d.strftime("%d-%b") for d in future_dates], 
    "Rate": [f"{r:.3f}" for r in future_predictions]
})
st.sidebar.dataframe(future_df, height=250) 


# MAIN DASHBOARD 
st.title("Corporate Loan Repayment Dashboard")
st.markdown("### USD/MYR Forecast & Payment Scheduler")

col1, col2, col3 = st.columns(3)
col1.metric("Current Rate", f"{current_rate:.4f}", f"Updated: {last_date_in_data.date()}")
col2.metric(f"Forecast ({next_day_date})", f"{next_day_rate:.4f}", delta=f"{difference:.4f}")
col3.info("Model: (Linear + Random Forest)")

st.markdown("---")


# DECISION TOOL 
st.subheader("Payment Planner")
st.write("Analyze the best time to pay any USD transaction.")

col_main1, col_main2 = st.columns([1, 2])

with col_main1:
    st.markdown("#### Invoice Details")
    invoice_ref = st.text_input("Invoice Reference", "Invoice #0001")
    payment_amount = st.number_input("Payment Amount USD ($)", min_value=1000, value=123456, step=1000)
    
    # User selects time for payment
    payment_terms = st.selectbox("Payment Terms (Due Date)", 
                                 ["Net 7 (1 Week)", "Net 30 (1 Month)"])

with col_main2:
    st.markdown(f"#### Analysis for {invoice_ref}")
    
    # set days and confidence in the prediction
    if "Net 7" in payment_terms:
        days_ahead = 7
    elif "Net 30" in payment_terms:
        days_ahead = 30
        
    # get the prediction from model
    future_days_extended = np.array([[last_day_id + i] for i in range(1, 32)])
    extended_predictions = final_model.predict(future_days_extended)
    
    # calculate the specific target date
    target_date_obj = last_date_in_data + datetime.timedelta(days=days_ahead)
    target_date_str = target_date_obj.strftime("%d %b %Y") # e.g., "04 Feb 2026"
    target_rate = extended_predictions[days_ahead - 1]
    
    # calculate cost
    cost_now = payment_amount * current_rate
    cost_later = payment_amount * target_rate
    savings = cost_now - cost_later
    
    # cost diplay
    c1, c2 = st.columns(2)
    c1.metric("Cost if Paid TODAY", f"RM {cost_now:,.2f}")
    c2.metric(f"Estimated Cost on {target_date_str}", f"RM {cost_later:,.2f}", delta=f"{-savings:,.2f}" if savings < 0 else f"{savings:,.2f}", delta_color="inverse")
    
    st.divider()

    # display Results
    st.write(f"Model Accuracy (R² Score): {confidence_pct:.1f}%")
    st.caption(f"Error Margin (MAE): ± RM {mae:.4f}")

    if target_rate < current_rate:
        st.success(f"Suggestion: Wait {days_ahead} Days")
        st.write(f"Predicted Rate on {target_date_str}: {target_rate:.3f} (Lower than today)")
        st.metric("Estimated Savings", f"RM {savings:,.2f}", delta="Cheaper")
    else:
        st.warning("Suggestion: Pay Now!!")
        st.write(f"Predicted Rate on {target_date_str}: {target_rate:.3f} (Higher than today)")
        st.metric("Avoided Extra Cost", f"RM {abs(savings):,.2f}", delta="Avoided Loss")


# GRAPH 
st.markdown("---")
st.subheader("Reference Graph")
fig, ax = plt.subplots(figsize=(10, 4))

# last 6 months data only
recent_df = df.tail(180) 
ax.plot(recent_df['date'], recent_df['average_rate'], label='Actual Rate', color='blue', alpha=0.5)

# trend line
trend_line = final_model.predict(X)
ax.plot(df['date'].tail(180), trend_line[-180:], label='Trend Line', color='red', linestyle='--', linewidth=1.5)

# prediction scatter plots
ax.scatter(future_dates, future_predictions, color='green', label='Forecast', zorder=5)

ax.set_title(f"USD/MYR Movement Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Rate (MYR)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)