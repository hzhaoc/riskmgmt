#!/usr/bin/env python

"""plot.py: A class to plot charts where certain quantiles like 5%, 95% are indicated, as a risk tool"""

__author__ = "Hua Zhao"

import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io


class RiskBox:
    """
    A class used to plot box-like chart for time series simulations, similar to a typical box-plot seen in finance,
    but whisker caps indicate quantiles such as 5%th and 95%th from simulation.

    ...
    Params
    ----------
    table: boolean
        whether or not to attach data table below the plot.
    title: str
        plot title.
    df: pandas.DataFrame
        simulation data. Shape(m, n) where m is simulation #, n is number of timestamps.
        Currently columns only accepts datetime.timestamp object.

    Methods
    -------
    _get_img(self, df):
        convert simulation data to the plot
        return (bytes, Image.PIL)
    PIL:
        return Image.PIL of the plot for display
    bytes:
        return image data
    data:
        return simulation data
    """

    def __init__(self, *, table=True, title='', df=pd.DataFrame()):
        self._table = table
        self._title = title
        self._df = df.copy(deep=True).astype(float)  # simulated results, column by date, row by simulations
        self._r2c = 6 / self._df.shape[1]  # row to column shape ratio
        if self._df.empty:
            raise ValueError("No Dataframe found to plot, check kind or data source")
        self._img = self._get_img(self._df)

    @property
    def PIL(self):
        return self._img[1]

    @property
    def bytes(self):
        return self._img[0]

    @property
    def data(self):
        return self._df

    @property
    def data_path(self):
        return os.path.join(self._data_path, self._data_file)

    @property
    def img_path(self):
        return os.path.join(self._img_path, self._img_file)

    @property
    def img_file(self):
        return self._img_file

    @property
    def data_file(self):
        return self._data_file

    def _get_img(self, df):
        fig = plt.figure(figsize=(5 / self._r2c, 8)) if self._table else plt.figure(figsize=(5.5 / self._r2c, 7))
        # plot Box
        df.columns = [x.strftime("%y-%b") for x in df.columns]
        (ax, dic) = df.boxplot(
            rot=0,
            grid=False,
            whis=[5, 95],  # set whisker cap to 5% and 95%
            showmeans=True,
            return_type='both',
            patch_artist=True  # set True to use face color attribute
        )  # dic: dict with all box chart attributes, each dict value is a list
        for box in dic['boxes']:
            box.set(color='black', linewidth=1.5)
            box.set(facecolor='lightgrey')
        plt.setp(dic['medians'], color='green', linewidth=2.0)
        plt.setp(dic['whiskers'], linewidth=1.5, color='black')
        plt.setp(dic['caps'], linewidth=3.0, color='red')
        plt.setp(dic['means'], marker='*', markerfacecolor='yellow', markersize=6, markeredgecolor='yellow')
        plt.setp(dic['fliers'], marker='+', markersize=4.0, markeredgewidth=0.2, linestyle='none', markeredgecolor='black')
        # plot table
        col_labels = list(df.columns.values)
        row_labels = ['Min', '5%', '50%', 'Mean', '95%', 'Max']
        table_vals = pd.DataFrame(index=row_labels, columns=col_labels)
        table_text = table_vals.copy(deep=True)
        nsim = len(df)
        for col in list(table_vals.columns.values):
            series = sorted(df.loc[:, col])
            for idx in list(table_vals.index.values):
                if idx == 'Min':
                    table_vals.loc[idx, col] = series[0]
                if idx == '5%':
                    table_vals.loc[idx, col] = series[int(0.05 * nsim)]
                if idx == '50%':
                    table_vals.loc[idx, col] = series[int(0.5 * nsim)]
                if idx == 'Mean':
                    table_vals.loc[idx, col] = sum(series) / nsim
                if idx == '95%':
                    table_vals.loc[idx, col] = series[int(0.95 * nsim)]
                if idx == 'Max':
                    table_vals.loc[idx, col] = series[-1]
                table_text.loc[idx, col] = '${:,.2f}'.format(table_vals.loc[idx, col])

        if self._table:
            the_table = plt.table(cellText=table_text.values.tolist(),
                                  rowLabels=row_labels,
                                  colLabels=col_labels,
                                  loc='bottom',
                                  bbox=[0, -0.7, 1, 0.6])
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(12)
            plt.subplots_adjust(left=0.08, right=0.95, bottom=0.4, top=0.95)
        # plot points for min, 5%, 50%, mean, 95%, max, y-axis line, emphasize-circle
        ax.plot([''] + col_labels, [None] + list(table_vals.loc['Min', :]), '+', markersize=4, markeredgewidth=1.5,
                color='black', label='min')
        ax.plot([''] + col_labels, [None] + list(table_vals.loc['5%', :]), '_', markersize=6, markeredgewidth=1.5,
                color='red', label='5%')
        ax.plot([''] + col_labels, [None] + list(table_vals.loc['50%', :]), '_', markersize=6, markeredgewidth=1.5,
                color='green', label='50%')
        ax.plot([''] + col_labels, [None] + list(table_vals.loc['Mean', :]), '*', markersize=6, markeredgewidth=1.5,
                color='yellow', label='mean')
        ax.plot([''] + col_labels, [None] + list(table_vals.loc['95%', :]), '_', markersize=6, markeredgewidth=1.5,
                color='red', label='95%')
        ax.plot([''] + col_labels, [None] + list(table_vals.loc['Max', :]), '+', markersize=4, markeredgewidth=1.5,
                color='black', label='max')
        ax.axhline(y=0, xmin=0, xmax=1, linestyle=':', linewidth=2, color='b')
        # annotate
        series = pd.Series(list(table_vals.loc['Min', :]), index=df.columns.values)
        self._annotate_local_min(ax, series, n=2)

        ax.yaxis.grid()
        ax.set_ylabel('$/Millions', fontsize=11)
        ax.legend(loc='best', fontsize=8, markerscale=1.5)
        ax.set_xticklabels(col_labels, rotation=0, fontsize=9)
        y_ticks = np.arange(int(min(table_vals.loc['Min', :])) - 6, int(max(table_vals.loc['Max', :])) + 4, 1.5)
        # y_ticks = np.arange(min(table_vals.loc['Min', :]), max(table_vals.loc['Min', :]), 1.50, dtype=float)
        ax.set_yticks(ticks=y_ticks)
        ax.set_yticklabels([f'$ {x:.2f} M' for x in y_ticks], rotation=0, fontsize=10)
        # plt.rc('ytick', labelsize=18) # set ticklabel font size globally
        # plt.rc('xtick', labelsize=23)
        # plt.rcParams['xtick.labelsize'] = 8 # same as above
        # plt.rcParams['ytick.labelsize'] = 8
        plt.title(self._title + ' Distribution', loc='center', fontsize=15)
        img_buf = io.BytesIO()
        fig.savefig(img_buf, dpi=500, format='png')
        # fig.savefig(os.path.join(self._img_path, self._img_file), dpi=500)
        # plt.show()
        plt.close(fig)
        img = Image.open(img_buf)
        return img_buf, img

    @staticmethod
    def _annotate_local_min(ax, series, n):
        mins = local_min(series, n)
        textprops = dict(boxstyle="round", pad=0.6, fc="0.8")
        arrowprops = dict(arrowstyle='<-', color="0.1", shrinkB=8)

        pre_loc = 0
        is_dir = 1
        for key, val in mins.items():
            cur_loc = series.index.get_loc(key)
            if abs(cur_loc - pre_loc) <= 5 and is_dir == 1:
                x = 250
            if abs(cur_loc - pre_loc) > 5 and is_dir == 1:
                x = -875
                is_dir = -1
            if is_dir == -1:
                x = -875
            x = x + 125 if cur_loc >= len(series) - 2 else x
            y = -300 if len(series) - 2 > cur_loc > 1 else -180

            ax.annotate("{}: ${}M".format(key, round(val, 2)),
                        xy=(key, val), xytext=(x, y),
                        xycoords='data', textcoords='offset pixels',
                        bbox=textprops, arrowprops=arrowprops)
            pre_loc = cur_loc
        return


def local_min(series, number=2):
    """
    find local min points in a series
    :param series: pandas series
    :param number: number of local mins
    :return: index, value in dict
    """
    mins = dict()
    for i in range(len(series)):
        if i == 0 and series.iloc[i] <= series.iloc[i + 1]:
            mins[series.index[i]] = series.iloc[i]
        if i >= 1 and (i <= len(series) - 2) and series.iloc[i] <= series.iloc[i + 1] and series.iloc[i] <= series.iloc[i - 1]:
            mins[series.index[i]] = series.iloc[i]
        if i == len(series) - 1 and series.iloc[i] <= series.iloc[i - 1]:
            mins[series.index[i]] = series.iloc[i]
    thres = sorted(mins.values())[number - 1]
    mins = {key: val for key, val in mins.items() if val <= thres}
    return mins
