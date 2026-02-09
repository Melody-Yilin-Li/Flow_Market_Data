# Common imports for all files.
import numpy as np 
import pandas as pd
import itertools 
import statistics
import matplotlib.pyplot as plt 
from matplotlib.ticker import StrMethodFormatter
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
import faulthandler; faulthandler.enable()