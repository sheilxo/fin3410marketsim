import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

portfolio = {
    'AAPL': {'shares': 447},
    'ACN': {'shares': 72},
    'ANET': {'shares': 166},
    'BRK-B': {'shares': 215},  
    'JPM': {'shares': 587},
    'LULU': {'shares': 375},
    'MSFT': {'shares': 234},
    'NVDA': {'shares': 1351},
    'VSPGX': {'shares': 131},
    'WMT': {'shares': 1246},
}

symbols = list(portfolio.keys())
start_date = '2024-08-17'
end_date = '2024-10-17'
data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
data = data.ffill().dropna()

for symbol in symbols:
    data[symbol] = data[symbol] * portfolio[symbol]['shares']

portfolio_value = data.sum(axis=1)
benchmark = yf.download('SPY', start=start_date, end=end_date)['Adj Close']
benchmark = benchmark.ffill().dropna()

portfolio_returns = portfolio_value.pct_change().dropna()
benchmark_returns = benchmark.pct_change().dropna()

portfolio_returns.index = portfolio_returns.index.tz_localize(None)
benchmark_returns.index = benchmark_returns.index.tz_localize(None)
benchmark_returns = benchmark_returns.reindex(portfolio_returns.index, method='ffill')

portfolio_cum_returns = (1 + portfolio_returns).cumprod() - 1
benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1

plt.figure(figsize=(12, 6))
plt.plot(portfolio_cum_returns.index, portfolio_cum_returns.values, label='Our Portfolio')
plt.plot(benchmark_cum_returns.index, benchmark_cum_returns.values, label='S&P 500', linestyle='--')
plt.title('Cumulative Returns Over the Last 2 Months')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cumulative_returns.png')
plt.close()

if not portfolio_cum_returns.empty:
    period_return = portfolio_cum_returns.iloc[-1]
    
    days_in_period = len(portfolio_returns)
    
    period_volatility = portfolio_returns.std() * np.sqrt(days_in_period)

    risk_free_rate_annual = 0.05
    risk_free_rate_period = risk_free_rate_annual * (days_in_period / 252)

    portfolio_sharpe_ratio = (period_return - risk_free_rate_period) / period_volatility

    print('Portfolio Return over Period: {:.2%}'.format(period_return))
    print('Portfolio Volatility over Period: {:.2%}'.format(period_volatility))
    print('Portfolio Sharpe Ratio over Period: {:.2f}'.format(portfolio_sharpe_ratio))
else:
    print("No data available for the selected period. Check if stock symbols are correct.")



num_simulations = 1000
num_days = days_in_period
log_returns = np.log(1 + portfolio_returns)
u = log_returns.mean()
var = log_returns.var()
drift = u - (0.5 * var)
stdev = log_returns.std()
start_value = portfolio_value.iloc[-1]

simulation_results = np.zeros((num_days, num_simulations))

for sim in range(num_simulations):
    prices = [start_value]
    for day in range(1, num_days):
        shock = drift + stdev * np.random.normal()
        price = prices[-1] * np.exp(shock)
        prices.append(price)
    simulation_results[:, sim] = prices

p10 = np.percentile(simulation_results, 10, axis=1)  
p50 = np.percentile(simulation_results, 50, axis=1)  
p90 = np.percentile(simulation_results, 90, axis=1) 

plt.figure(figsize=(12, 6))
plt.plot(simulation_results, color="lightgray", alpha=0.05)  
plt.plot(p50, color="blue", label="Median Outcome (50th Percentile)", linewidth=2)  
plt.fill_between(range(num_days), p10, p90, color="lightblue", alpha=0.3, label="10th-90th Percentile")
plt.title('Monte Carlo Simulation of Portfolio Future Value (Next 2 Months)')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig('monte_carlo_simulation_with_confidence.png')  
plt.close()

ending_values = simulation_results[-1, :]
plt.figure(figsize=(10, 6))
plt.hist(ending_values, bins=50)
plt.title('Distribution of Ending Portfolio Values After 2 Months')
plt.xlabel('Portfolio Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('ending_values_histogram.png') 
plt.close()

probability_of_loss = np.mean(ending_values < start_value)
expected_ending_value = np.mean(ending_values)

print('Probability of Loss over Next 2 Months: {:.2%}'.format(probability_of_loss))
print('Expected Portfolio Value after 2 Months: ${:,.2f}'.format(expected_ending_value))
