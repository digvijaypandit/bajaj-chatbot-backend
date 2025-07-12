import pandas as pd

# Step 1: Load and clean the data
df = pd.read_csv("data/stock_prices/bajajfinserv_prices.csv")
df.columns = [col.strip().lower() for col in df.columns]  # normalize headers

# Rename columns for consistency
df.rename(columns={"close price": "close"}, inplace=True)

# Step 2: Parse date and create month column
df["date"] = pd.to_datetime(df["date"], format="%d-%b-%y")  # exact format in your CSV
df["month"] = df["date"].dt.to_period("M")  # used for querying by month

# Get stats for a month (e.g. Jan-2024)
def get_stats_for_month(month_str):
    try:
        month_period = pd.Period(month_str, freq='M')
        monthly = df[df["month"] == month_period]

        if monthly.empty:
            return {"error": f"No data found for {month_str}"}

        return {
            "month": month_str,
            "average": round(monthly["close"].mean(), 2),
            "high": round(monthly["close"].max(), 2),
            "low": round(monthly["close"].min(), 2)
        }
    except Exception as e:
        return {"error": str(e)}

# Compare two months (e.g. Jan-2024 vs Apr-2024)
def compare_months(month1, month2):
    stats1 = get_stats_for_month(month1)
    stats2 = get_stats_for_month(month2)

    if "error" in stats1:
        return {"error": f"From Month Error: {stats1['error']}"}
    if "error" in stats2:
        return {"error": f"To Month Error: {stats2['error']}"}

    return {
        "period_1": stats1,
        "period_2": stats2,
        "change_in_avg": round(stats2["average"] - stats1["average"], 2)
    }
