import numpy as np
import pandas as pd
import sqlalchemy
from arch import arch_model
from sys import argv
import datetime

from pandas.tseries.offsets import DateOffset

#importar pacotes
import pandas as pd
import numpy as np
from scipy.stats import *
from matplotlib import pyplot as plt
from sqlalchemy import create_engine, text
import pyodbc
import xlsxwriter
from openpyxl import load_workbook
import openpyxl
import win32com.client
import time
from arch import arch_model
import dateutil.parser
from pandas.io import parsers
import argparse
# from datetime import datetime, timedelta
from tkinter import *
import warnings
import statsmodels.api as sm
import os
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import DateOffset

import plotly.express as px

input = 'xl'
modelo = 'garch'
horizonte = 24
n_simul = 5000
frequencia = 'M'
modelo_vol = 'garch'
modelo_correl = 'ewma'
dist = 'Students'
cen = 'n'

#Configuração da conexão com o banco de dados
# engine = sqlalchemy_connect('sql_TABLE')
data_inicial = datetime.datetime(2019, 4, 1)
data_final = datetime.datetime(2024, 4, 1)

PARAMETERS = {
    "input": "xl",
    "modelo": "garch",
    "horizonte": 24,
    "n_simul": 5000,
    "frequencia": "M",
    "modelo_vol": "garch",
    "modelo_correl": "ewma",
    "dist": "Students",
    "cen": "n",
    "data_inicial": "2019-04-01",
    "data_final": "2024-04-01"
}
