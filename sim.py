#!/usr/bin/env python

"""
sim.py: a stochastic simulator for a set of monthly financial derivatives like futures
		PCA is used in volatility analysis with seasonality included
"""

__author__ = "Hua Zhao"

from datain import file
from sup import basic
from src import vol
import pandas as pd
import numpy as np
from datetime import datetime
import copy
from warnings import warn
import random
from sup.decor import timmer

dt_dic = {'M': 1 / 12, 'D': 1 / 365.25}
dt_dic_2 = {'M': 30, 'A': 365.25, 'D': 1}


class ItoMC:
	"""
	Generalized Wiener process to simulate Monthly Fowward contracts
	params:
	- price: fowrad gas market price at specific date. dtype: pd.series, index of forward months, and name of that date
	- pca: MxN pca where M is number of forward months, and N is number of (loading) factors
	"""

	def __init__(self, price=pd.Series(), pca=None, seas=pd.Series(), freq='M', nsim=1000, addcost=False):
		if not price.name or not isinstance(price.name, datetime):
			raise ValueError("unknown name of price, price (future gas market price) series name has to be the market date")
		self._pca = pca
		self._tenors, self._factors = self._pca.shape
		self._var = (self._pca.values**2).sum(axis=1).reshape([self._tenors, 1])
		if price.empty:
			raise ValueError('Future Contract Market Price Series is empty')
		if self._tenors != len(price):
			raise ValueError(f"PCA or default has {self._tenors} tenors while market price at {price.name.date()} has {len(price)} tenors")
		if not (price.name < min(price.index) <= basic.add_month(price.name, 1)):
			raise ValueError(f'{min(price.index).date()} is start month of forward curve at {price.name.date()}')

		self._seas = seas
		self._price = price.sort_index()
		self._start = price.name
		self._nsim = nsim
		self._trans_cost = file.gas_trans_cost() if addcost else 0
		self._idx_cost = file.gas_idx_cost() if addcost else 0
		self._freq = freq
		self._fwd_gas, self._fwd_mths = self._preprocess(self._price)
		self._sim = pd.Series()
		if self._freq == 'M':
			self._ts = pd.date_range(start=basic.add_month(self._start, 1), end=basic.add_month(self._start, self._tenors), freq='MS', normalize=True)
			self._ts = list(self._ts)
			self._dt = (self._ts[0] - self._start).days / 360
		elif self._freq == 'D':
			self._ts = pd.date_range(start=self._start, end=basic.add_month(self._start, self._tenors), freq='D', normalize=True)[1:]
			self._ts = list(self._ts)
			self._dt = 1 / 365.25
		self._maturity, self._count, self._date = 0, 0, self._start
		self._beat_equal = False if self._freq == 'M' else True

	def next(self):
		"""
		simulated results are in property sim, dtype: pd.series, where index is time seires (dates), and sim_prices are np.arrays, where rows
		are forward dates, and columns are simulated returns on that date 
		"""
		if self._pca.empty:
			raise ValueError("PCA is empty in date-by-date simulation")
		if self._maturity >= self._tenors or self._date == self._ts[-1]:
			warn("Reached the end of simulation period or all future contracts in simulation process have matured")
			return

		self._date = self._ts[self._count]
		sf = self._seas.loc[self._date.month] if not self._seas.empty else 1
		self._maturity = basic.find_order(self._fwd_mths, self._date, beat_equal=self._beat_equal)
		drift = -1 / 2 * self._var[:self._tenors - self._maturity] * (sf**2) * self._dt
		walk = self._pca.values[:self._tenors - self._maturity, :].dot(np.random.normal(0, 1, [self._factors, self._nsim])) * sf * (self._dt**0.5)
		self._fwd_gas[self._maturity:, :] = self._fwd_gas[self._maturity:, :] * np.exp(walk + drift)

		self._sim.loc[self._date] = copy.deepcopy(self._fwd_gas)
		self._count += 1
		self._dt = dt_dic[self._freq]
		return

	@timmer(0, 0, 0)
	def done(self):
		"""Simulating Nat Gas forward curves until all expired"""
		if self._pca.empty:
			raise ValueError("PCA is empty in one-time simulation")
		if self._count > 0:
			warn("Simulation already started, can not do one-time simulation anymore")
			return

		for t in self._ts:
			self.next()
		return

	def _preprocess(self, series):
		lis = list(series.index)
		series += (self._idx_cost + self._trans_cost)
		arr = series.values.reshape([self._tenors, 1]).repeat(self._nsim, axis=1)
		return arr, lis

	@property
	def sim_prices(self):
		"""series, date indexed, each cell has tenors x nsim of simulations"""
		return self._sim.sort_index()

	@property
	def pick_sim_price(self):
		"""randomly pick one of the simulations"""
		draw = random.choice(list(range(self._nsim)))
		res = pd.DataFrame(index=sorted(self._sim.index), columns=list(range(self._tenors)))
		for t in res.index:
			_mat = basic.find_order(self._fwd_mths, t, beat_equal=self._beat_equal)
			res.loc[t, :len(res.columns) - _mat - 1] = self.sim_prices.loc[t][:, draw][_mat:]
		return res

	@property
	def static(self):
		df = pd.DataFrame(index=self._ts, columns=list(range(self._tenors)))
		# _arr = np.array(self.price).reshape(1, len(self.price)).repeat(len(df.index), axis=0)
		for t in df.index:
			_mat = basic.find_order(self._fwd_mths, t, beat_equal=self._beat_equal)
			df.loc[t, :len(df.columns) - _mat - 1] = self.price.values[_mat:]
		return df

	@property
	def means(self):
		df = pd.DataFrame(index=self._sim.index, columns=list(range(self._tenors)))
		for t in df.index:
			_mat = basic.find_order(self._fwd_mths, t, beat_equal=self._beat_equal)
			df.loc[t, :len(df.columns) - _mat - 1] = np.mean(self.sim_prices.loc[t][_mat:, :], axis=1)
		return df

	def percentile(self, pct):
		df = pd.DataFrame(index=self._sim.index, columns=list(range(self._tenors)))
		for t in df.index:
			_mat = basic.find_order(self._fwd_mths, t, beat_equal=self._beat_equal)
			df.loc[t, :len(df.columns) - _mat - 1] = np.percentile(self.sim_prices.loc[t][_mat:, :], pct, axis=1)
		return df

	@property
	def sim_rtn(self):
		_rtn = 1 + self.sim_prices.pct_change()
		_rtn.iloc[0] = self.sim_prices.iloc[0] / self.price.values.reshape(len(self.price), 1).repeat(self.nsim, axis=1)  # first simulate date
		for t in _rtn.index:
			_rtn.loc[t] = np.log(_rtn.loc[t])
		return _rtn

	@property
	def pick_sim_rtn(self):
		_rtn = self.sim_rtn
		draw = random.choice(list(range(self._nsim)))
		res = pd.DataFrame(index=sorted(self._sim.index), columns=list(range(self._tenors)))
		for t in res.index:
			_mat = basic.find_order(self._fwd_mths, t, beat_equal=self._beat_equal)
			res.loc[t, :len(res.columns) - _mat - 1] = _rtn.loc[t][:, draw][_mat:]
		return res

	@property
	def std(self):
		_rtn = self.sim_rtn
		df = pd.DataFrame(index=self._sim.index, columns=list(range(self._tenors)))
		for t in df.index:
			_mat = basic.find_order(self._fwd_mths, t, beat_equal=self._beat_equal)
			df.loc[t, :len(df.columns) - _mat - 1] = np.std(_rtn.loc[t][_mat:, :], axis=1)
		return df
	
	def vol(self, freq):
		return self.std * (np.sqrt(dt_dic_2[freq]) / np.sqrt(dt_dic_2[self._freq]))

	@property
	def maturity(self):
		return self._maturity

	@property
	def count(self):
		return self._count

	@property
	def current_date(self):
		return self._date
	
	@property
	def current_value(self):
		return self._fwd_gas

	@property
	def nsim(self):
		return self._nsim

	@property
	def date_range(self):
		return self._ts

	@property
	def price(self):
		return self._price

	@property
	def pca(self):
		return self._pca


class PCA:
	def __init__(self, rtn=pd.DataFrame(), ve_thres=0.95):
		if rtn.empty:
			raise ValueError("null rtn")
		self._tenors = rtn.shape[1]
		self._rtn = rtn
		self._pca = pd.DataFrame()
		self._std, self._std_pca = np.array([]), np.array([])
		self._ve = None
		self._ve_thres = ve_thres
		# self._rtn, self._seas = vol.desea(self._rtn)
		self._pca, self._ve = vol.pca(self._rtn, thres=self._ve_thres)
	
	@property
	def pca(self):
		return self._pca

	@property
	def factors(self):
		return self._pca.shape[1]
	
	@property
	def ve(self):
		return self._ve
	
	@property
	def tenors(self):
		return self._tenors

	@property
	def std(self):
		self._std = self._rtn.std(axis=0).values.reshape([self._rtn.shape[1], 1])
		return self._std
	
	@property
	def std_pca(self):
		self._std_pca = ((self._pca.values**2).sum(axis=1)**0.5).reshape([self._pca.shape[0], 1])
		return self._std_pca

	@property
	def rtn(self):
		return self._rtn


def test():
	from datetime import datetime
	date = datetime.strptime('2019-12-31', '%Y-%m-%d')
	pca = PCA(tenors=24, PriceIndex='Henry Hub', trailing_year=3)
	fwd = file.current_future(date, tenors=23, priceIndex='HH')
	date1 = datetime.strptime('2020-01-01', '%Y-%m-%d')
	fwd.loc[date1] = 2.196
	fwd = fwd.sort_index()
	sim = ItoMC(PCA=pca, price=fwd, freq='M')
	sim.done()
	print(sim.at95)
	sim.at95.to_csv('test.csv')


if __name__ == '__main__':
	test()
