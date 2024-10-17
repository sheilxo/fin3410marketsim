import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Define your updated portfolio
portfolio = {
    'AAPL': {'shares': 447},
    'ACN': {'shares': 72},
    'ANET': {'shares': 166},
    'BRK-B': {'shares': 215},  # Changed from BRK.B to BRK-B for yfinance compatibility
    'JPM': {'shares': 587},
    'LULU': {'shares': 375},
    'MSFT': {'shares': 234},
    'NVDA': {'shares': 1351},
    'VSPGX': {'shares': 131},  # Vanguard Fund
    'WMT': {'shares': 1246},
}

# List of symbols
symbols = list(portfolio.keys())

# Download historical price data for the last 2 months
start_date = '2024-08-17'  # Adjust the start date to 2 months before the current date
end_date = '2024-10-17'
data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']

# Handle missing data by forward filling
data = data.ffill().dropna()

# Multiply each stock's price by the number of shares to get the daily value
for symbol in symbols:
    data[symbol] = data[symbol] * portfolio[symbol]['shares']

# Sum across all stocks to get the total portfolio value each day
portfolio_value = data.sum(axis=1)

# Download benchmark data (S&P 500 ETF - SPY)
benchmark = yf.download('SPY', start=start_date, end=end_date)['Adj Close']
benchmark = benchmark.ffill().dropna()

# Calculate daily returns
portfolio_returns = portfolio_value.pct_change().dropna()
benchmark_returns = benchmark.pct_change().dropna()

# Ensure that both portfolio_returns and benchmark_returns have the same datetime format (without timezone)
portfolio_returns.index = portfolio_returns.index.tz_localize(None)
benchmark_returns.index = benchmark_returns.index.tz_localize(None)

# Align the index of benchmark returns with the portfolio returns
benchmark_returns = benchmark_returns.reindex(portfolio_returns.index, method='ffill')

# Calculate cumulative returns
portfolio_cum_returns = (1 + portfolio_returns).cumprod() - 1
benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1

# Plot cumulative returns for both the portfolio and the S&P 500
plt.figure(figsize=(12, 6))
plt.plot(portfolio_cum_returns.index, portfolio_cum_returns.values, label='Our Portfolio')
plt.plot(benchmark_cum_returns.index, benchmark_cum_returns.values, label='S&P 500', linestyle='--')  # S&P 500 line
plt.title('Cumulative Returns Over the Last 2 Months')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cumulative_returns.png')  # Save the plot as a PNG file
plt.close()  # Close the figure to free up memory

# Check if portfolio_cum_returns is empty to avoid the IndexError
if not portfolio_cum_returns.empty:
    # Calculate period performance metrics (not annualized)
    period_return = portfolio_cum_returns.iloc[-1]  # Use iloc to fix the warning
    period_volatility = portfolio_returns.std()

    # Risk-free rate over the period (assume 5% annualized, adjust for period)
    risk_free_rate_annual = 0.05
    days_in_period = len(portfolio_returns)
    risk_free_rate_period = risk_free_rate_annual * (days_in_period / 252)

    # Sharpe Ratio Calculation for the Portfolio
    portfolio_sharpe_ratio = (period_return - risk_free_rate_period) / (period_volatility * np.sqrt(days_in_period))

    # Print performance metrics
    print('Portfolio Return over Period: {:.2%}'.format(period_return))
    print('Portfolio Volatility over Period: {:.2%}'.format(period_volatility))
    print('Portfolio Sharpe Ratio over Period: {:.2f}'.format(portfolio_sharpe_ratio))
else:
    print("No data available for the selected period. Check if stock symbols are correct.")

# Monte Carlo Simulation
num_simulations = 1000
num_days = days_in_period  # Simulate for the same number of days as your period

# Log returns of the portfolio
log_returns = np.log(1 + portfolio_returns)

# Calculate drift and standard deviation
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5 * var)
stdev = log_returns.std()

# Starting value of the portfolio
start_value = portfolio_value.iloc[-1]

# Initialize an array to hold simulation results
simulation_results = np.zeros((num_days, num_simulations))

for sim in range(num_simulations):
    prices = [start_value]
    for day in range(1, num_days):
        shock = drift + stdev * np.random.normal()
        price = prices[-1] * np.exp(shock)
        prices.append(price)
    simulation_results[:, sim] = prices

# Calculate percentiles for confidence intervals
p10 = np.percentile(simulation_results, 10, axis=1)  # 10th percentile (pessimistic)
p50 = np.percentile(simulation_results, 50, axis=1)  # 50th percentile (median)
p90 = np.percentile(simulation_results, 90, axis=1)  # 90th percentile (optimistic)

# Plot the simulation results with confidence intervals
plt.figure(figsize=(12, 6))

# Plot all simulations in very light gray for better visibility of the median line
plt.plot(simulation_results, color="lightgray", alpha=0.05)  

# Plot the median outcome with a thicker blue line to make it stand out
plt.plot(p50, color="blue", label="Median Outcome (50th Percentile)", linewidth=2)  

# Confidence band between the 10th and 90th percentiles
plt.fill_between(range(num_days), p10, p90, color="lightblue", alpha=0.3, label="10th-90th Percentile")

# Titles and labels
plt.title('Monte Carlo Simulation of Portfolio Future Value (Next 2 Months)')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')

# Display the legend
plt.legend(loc="upper left")

# Adjust the layout and save the figure
plt.tight_layout()
plt.savefig('monte_carlo_simulation_with_confidence.png')  # Save the plot as a PNG file
plt.close()



# Histogram of ending portfolio values
ending_values = simulation_results[-1, :]
plt.figure(figsize=(10, 6))
plt.hist(ending_values, bins=50)
plt.title('Distribution of Ending Portfolio Values After 2 Months')
plt.xlabel('Portfolio Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('ending_values_histogram.png')  # Save the plot as a PNG file
plt.close()

# Calculate the probability of loss
probability_of_loss = np.mean(ending_values < start_value)
expected_ending_value = np.mean(ending_values)

print('Probability of Loss over Next 2 Months: {:.2%}'.format(probability_of_loss))
print('Expected Portfolio Value after 2 Months: ${:,.2f}'.format(expected_ending_value))
