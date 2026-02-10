# Common imports for all files.
import numpy as np 
import pandas as pd
from tabulate import tabulate 
import statistics 
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from matplotlib import rc
rc('text',usetex=True)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})
from matplotlib.ticker import StrMethodFormatter
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
import faulthandler; faulthandler.enable()

from stargazer.stargazer import Stargazer