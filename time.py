import pandas as pd
import copy
from dateutil import parser
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import OrderedDict


def last_weekday(date):
    dt = date
    if isinstance(dt, str):
        dt = parser.parse(dt)
    while True:
        dt -= timedelta(days=1)
        if dt.weekday() <= 4 and dt.date() not in holiday:
            break
    return dt


def add_weekday(date, add=0):
    dt = date
    if isinstance(dt, str):
        dt = parser.parse(dt)
    if not isinstance(add, int):
        raise ValueError("add must be an integer")
    while True:
        if add == 0:
            break
        if add < 0:
            dt -= timedelta(days=1)
            if dt.weekday() <= 4 and dt.date() not in holiday:
                add += 1
        if add > 0:
            dt += timedelta(days=1)
            if dt.weekday() <= 4 and dt.date() not in holiday:
                add -= 1
    return dt


def add_month(date, add=0):
    dt = date
    if isinstance(dt, str):
        dt = parser.parse(dt)
    dt += relativedelta(months=add)
    dt = datetime(year=dt.year, month=dt.month, day=1)
    return dt