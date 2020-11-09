#!/usr/bin/env python

"""decor.py: some wrappers"""

__author__ = "Hua Zhao"

import time


def timmer(flagtime=True, flagname=True, flagdoc=True):
    def wrapper(f):
        def inner(*args, **kwargs):
            # print(f"executing function {f.__name__}\n")

            if flagname:
                print(f"function in execution: {f.__name__}\n")

            if flagdoc:
                print(f"         document: {f.__doc__}\n")

            if flagtime:
                t0 = time.time()
                ret = f(*args, **kwargs)
                t1 = time.time()
                print(f"         execution time: {round(t1-t0, 2)} s\n")
            else:
                ret = f(*args, **kwargs)

            return ret
        return inner
    return wrapper
