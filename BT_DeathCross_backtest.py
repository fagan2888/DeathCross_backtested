import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from backtest import Strategy, Portfolio


class MovingAverageCrossStrategy(Strategy):


    def __init__(self,name,bars,win_s,win_l):
        self.name = name
        self.bars = bars
        self.wins = win_s
        self.winl = win_l

    def generate_signals(self):
        sig = pd.DataFrame(index = self.bars.index)
        sig['signal'] = 0
        sig[self.wins] = pd.rolling_mean(self.bars,self.wins,min_periods=1)
        sig[self.winl] = pd.rolling_mean(self.bars,self.winl,min_periods=1)
        sig['signal'] = np.where(sig[self.wins]>sig[self.winl],1,-1)

        return sig


class MarketOnClosePortfolio(Portfolio):


    def __init__(self,name,no_of_hold,sig,bars,initial_capital):
        self.name = name
        self.no_of_hold = no_of_hold
        self.bars = bars
        self.sig = sig
        self.initial_capital = initial_capital
        self.positions = self.generate_positions()

    def generate_positions(self):
        positions = pd.DataFrame(index = self.sig.index)
        positions[self.name] = self.sig['signal']*self.no_of_hold

        return positions

    def backtest_portfolio(self):
        pos_diff  = self.positions[self.name].diff()
        initial_sig_val = self.positions.iloc[0,0]
        pos_diff.iloc[0] = initial_sig_val

        prt = pd.DataFrame(index=self.sig.index)
        prt['holding'] = self.positions[self.name].mul(self.bars[self.name],axis=0)
        prt['cash'] = self.initial_capital - (pos_diff.mul(self.bars[self.name], axis=0).cumsum())

        prt['total'] = prt['holding'] + prt['cash']
        prt['TR%'] = prt['total'].pct_change()
        return prt


if __name__ == '__main__':

    df = pd.read_csv('data_new.csv',index_col='Date',parse_dates=True).dropna()
    df.columns = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB', 'SPX']

    security_name = 'AAPL'  # you can also use any of these ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB', 'SPX']
    no_of_hold = 1000

    bars = df[[security_name]]

    #first step is to generate signals
    mac = MovingAverageCrossStrategy(security_name,bars,50,200)
    sig = mac.generate_signals()

    #generate portfolios
    portfolio = MarketOnClosePortfolio(security_name,no_of_hold,sig,bars,initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()

    Comb = pd.DataFrame(index=returns.index)
    Comb['DCMA'] = returns['TR%']
    Comb['LO'] = (bars[security_name]*no_of_hold+100000).pct_change()
    Comb.dropna(inplace=True)
    Comb['LO_cum'] = np.cumprod(1 + Comb['LO']) - 1
    Comb['DCMA_cum'] = np.cumprod(1 + Comb['DCMA']) - 1


    fsize = 9
    plt.subplot(3, 1, 1)
    plt.plot(Comb[['LO_cum','DCMA_cum']])
    plt.legend(['Long_only','Death Cross MA'])
    #Comb['DCMA_cum'].plot(secondary_y=True)
    #plt.legend(['Death_CrossMA'])
    plt.title('Returns', fontsize=fsize)

    plt.subplot(3, 1, 2)
    plt.plot(sig[[50, 200]])
    plt.legend(['50','200'])
    plt.title('Moving Averages', fontsize=fsize)

    plt.subplot(3, 1, 3)
    plt.plot(sig['signal'])
    plt.title('signals', fontsize=fsize)

    plt.show()

    print(Comb)