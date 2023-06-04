import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import numpy as np
tickers = ['SLGG', 'AAPL', '^RUT', '^IXIC','^GSPC']
D = yf.download(tickers, start='2022-06-01', end='2023-06-01')
P = D['Adj Close']
Returns = P / P.shift(1) -1
R = P.pct_change()
R = R.dropna()

R2 = R.tail(250)
R2 = R2.rename(columns={'^GSPC': 'SnP'})
R2 = R2.rename(columns={'^RUT': 'Russell'})
R2 = R2.rename(columns={'^IXIC': 'Nasdaq'})
# Calculating Beta and use beta * percentage changes in SnP to represent percentage change in Stocks
results1 = sm.OLS.from_formula(formula='SLGG ~ SnP', data=R2).fit()
results2 = sm.OLS.from_formula(formula='AAPL ~ SnP', data=R2).fit()
B1 = results1.params['SnP']
B2 = results2.params['SnP']
sp= R2['SnP']
Dailyreturn1 = sp * B1 * 1000000 - sp * B2 * 500000
var1 = np.percentile(Dailyreturn1, 5)

# Mapping long $1mm to Russell 2000 (Ticker RTY <Index>) and short to Nasdaq (Ticker CCMP <Index>)
results3 = sm.OLS.from_formula(formula='SLGG ~ Russell', data=R2).fit()
results4 = sm.OLS.from_formula(formula='AAPL ~ Nasdaq', data=R2).fit()
B3 = results3.params['Russell']
B4 = results4.params['Nasdaq']
rul = R2['Russell']
nas = R2['Nasdaq']
Dailyreturn2 = rul * B3 * 1000000 - nas * B2 * 500000
var2 = np.percentile(Dailyreturn2, 5)

# Individual time serious
slg= R2['SLGG']
aap = R2['AAPL']
Dailyreturnslg = slg * 1000000
Dailyreturnaap = - aap * 500000
Dailyreturnidv = slg * 1000000 - aap * 500000
varslg = np.percentile(Dailyreturnslg, 5)
varaap = np.percentile(Dailyreturnaap, 5)
var3 = np.percentile(Dailyreturnidv, 5)
#print(varslg, varaap, var3)
data = {'SnP_Mapping': [var1], 'Hybrid_Mapping': [var2], 'Individual_Mapping': [var3], 'slgg': [varslg], 'aapl': [varaap]}
VaR = pd.DataFrame(data)
print(VaR)

import matplotlib.pyplot as plt

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(Dailyreturn1.index, Dailyreturn1, label='Mapping_SnP')
plt.plot(Dailyreturn2.index, Dailyreturn2, label='Mapping_Hybrid')
plt.plot(Dailyreturnidv.index, Dailyreturnidv, label='Dailyreturnidv')
plt.plot(Dailyreturnslg.index, Dailyreturnslg, label='Dailyreturnslg')
plt.plot(Dailyreturnaap.index, Dailyreturnaap, label='Dailyreturnaap')

# 添加图例
plt.legend()

# 添加标题和标签
plt.title('Daily Returns')
plt.xlabel('Date')
plt.ylabel('Return')

# 显示图表
plt.show()
plt.savefig('trend_plot.png')










