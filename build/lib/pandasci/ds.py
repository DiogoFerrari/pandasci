# from .rutils import rutils
# import pandasci.rutils as rutils
import pandas as pd
import numpy as np
from scipy import stats
import scipy as sp
from dask import dataframe as ddf 
#
import collections.abc
# python 3.10 change aliases for collection; modules, e.g., spss
# that depends on it need this workaround
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
#Now import hyper
import savReaderWriter as spss
import seaborn as sns
import re, os, textwrap, inspect
# dply-like operations
import itertools as it
from plydata.expressions import case_when as case_when_ply
from plydata import define
import zoneinfo
from datetime import datetime
import xlsxwriter
import openpyxl as pxl
from numpy import pi as pi
import textwrap, xlrd, warnings
# 
from scipy.stats import norm as dnorm
from scipy.stats import norm as qnorm
from numpy.random import choice as rsample
from numpy.random import normal as rnorm
from numpy.random import uniform as runif
from numpy.random import seed as set_seed
# 
import gspread                                                      # to read google spreadsheet
from oauth2client.service_account import ServiceAccountCredentials  # to read google spreadsheet
# 
from scipy.stats import t as tdist
from statsmodels.stats.proportion import proportion_effectsize as prop_esize
from statsmodels.stats.proportion import proportions_ztest as prop_ztest
from statsmodels.stats.proportion import proportions_chisquare as prop_chisqtest
from statsmodels.stats.proportion import proportion_confint as ci
#
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import TextBox
import matplotlib.patheffects as pe
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker # to avoid warning about tick location
# 
# 
from datar.tibble import tibble
# 
# Supress warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore")
# Supress warnings from R
# from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
# import logging
# rpy2_logger.setLevel(logging.ERROR)   # will display errors, but not warnings


# * functions
# ** others
def quantile10(x):
    res = np.quantile(x, q=.10) 
    return res
def quantile25(x):
    res = np.quantile(x, q=.25) 
    return res
def quantile75(x):
    res = np.quantile(x, q=.75) 
    return res
def quantile90(x):
    res = np.quantile(x, q=.90) 
    return res
def count_missing(x):
    res = x.isna().sum()
    return res

# ** reading data

def read_data(**kws):
    '''
    fn         filename with path

    sep        the column separator character

    big_data   boolean. If True, use Dask to handle big data

    '''
    fn=kws.get('fn')
    assert fn, "fn (filepath) must be provided."
    # 
    # Big data
    print(f"Loading data '{os.path.basename(fn)}'...", flush=True)
    big_data=kws.get("big_data", False)
    if big_data:
        kws.pop('big_data', 0)
        print("Using dask dataframe for big data...")
        # 
    fn_type=os.path.splitext(fn)[1]
    # 
    if fn_type=='.csv' or fn_type=='.CSV':
        return read_csv(big_data=big_data, **kws)
    # 
    elif fn_type=='.dta' or fn_type=='.DTA':
        return read_dta(big_data=big_data, **kws)
    # 
    elif fn_type=='.sav':
        # return spss_data(**kws)
        return read_spss(**kws)
    # 
    elif (fn_type=='.xls' or fn_type=='.xlsx' or
          fn_type=='.xltx' or fn_type=='.XLTX' or
          fn_type=='.ods' or fn_type=='.ODS' or
          fn_type=='.XLS' or fn_type=='.XLSX'):
        return read_xls(big_data=big_data, **kws)
    elif fn_type=='.tsv':
        return read_tsv(big_data=big_data, **kws)
    elif fn_type=='.txt':
        return read_txt(big_data=big_data, **kws)
    elif fn_type=='.tex':
        return read_tex(big_data=big_data, **kws)
    elif fn_type=='.dat':
        return read_dat(big_data=big_data, **kws)
    elif kws.get('gs_api', None):
        return read_gspread(**kws)
    # 
    else:
        print(f"No reader for file type {fn_type}. If you are trying to read "+
              "a Google spreadsheet, provide the gs_api parameter with path to "+
              "the local API json file for the Google spreadsheet, and the "+
              "parameter sn with the sheet name in the spreadsheet")
        return None
        
def read_csv (big_data, **kws):
    fn=kws.get('fn')
    kws.pop('fn')
    if not kws.get('sep', None):
        kws['sep']=";"
    if not big_data:
        df = pd.read_csv(filepath_or_buffer=fn, **kws)
    else:
        df = read_dask(fn, **kws)
    return eDataFrame(df) if not big_data else df
        
def read_dta (big_data, **kws):
    # 
    fn=kws.get('fn')
    if not big_data:
        df = pd.read_stata(fn)
    else:
        df = read_dask(fn, **kws)
    return eDataFrame(df) if not big_data else df

def read_xls (big_data, **kws):
    fn=kws.get('fn'); kws.pop('fn')
    if not big_data:
        df = eDataFrame(pd.read_excel(io=fn, **kws))
    else:
        df = read_dask(fn, **kws)
    return eDataFrame(df) if not big_data else df
    
def read_xltx(big_data, **kws):
    fn=kws.get('fn'); kws.pop('fn')
    if not big_data:
        df = eDataFrame(pd.read_excel(io=fn, **kws))
    else:
        df = read_dask(fn, **kws)
    return eDataFrame(df) if not big_data else df
    
def read_ods (big_data, **kws):
    fn=kws.get('fn'); kws.pop('fn')
    if not big_data:
        df = eDataFrame(pd.read_excel(io=fn, **kws))
    else:
        df = read_dask(fn, **kws)
    return eDataFrame(df) if not big_data else df
    
def read_tsv (big_data, **kws):
    fn=kws.get('fn')
    kws.pop('fn')
    # 
    kws['sep'] = '\t'
    # 
    if not big_data:
        df = pd.read_csv(filepath_or_buffer=fn, **kws)
    else:
        df = read_dask(fn, **kws)
    return eDataFrame(df) if not big_data else df

def read_txt (big_data, **kws):
    fn=kws.get('fn')
    kws.pop('fn')
    #
    big_data=kws.get("big_data", False)
    kws.pop('big_data', 0)
    # 
    if not big_data:
        df = pd.read_table(filepath_or_buffer=fn, **kws)
    else:
        df = read_dask(fn, **kws)
    return eDataFrame(df) if not big_data else df

def read_tex (big_data, **kws):
    fn = os.path.expanduser(kws['fn'])
    with open(fn) as f:
        content=f.readlines()
    return content

def read_dat (big_data, **kws):
    fn=kws.get('fn')
    kws.pop('fn')
    kws['sep']="\s+"
    if not big_data:
        df = pd.read_csv(fn, **kws)
    else:
        df = read_dask(fn, **kws)
    return eDataFrame(df) if not big_data else df
    
def read_dask(fn, **kws):
    return ddf.read_csv(fn, **kws)

def read_gspread(**kws):
    '''
    Load google spreadsheet
    Note: Remember to share the spreadsheet with e-mail client in json file, 
    which is found under the item "client_email" of the json information. Ex:

	 "client_email": "..."
    
    Input 
    -----
    fn     : filename of the google spreadsheet
    gs_api : json file with API info
    sn     : spreadsheet name
    
    Output 
    ------
    eDataFrame
    '''
    assert kws.get("gs_api", None),"A json file with google spreadsheet API"+\
        "must be provided."
    assert kws.get("sheet_name", None), "The sheet_name must be provided."
    # 
    fn=kws.get("fn", None)
    json_file=kws.get("gs_api", None)
    sheet_name=kws.get("sheet_name", None)
    # credentials (see https://gspread.readthedocs.io/en/latest/oauth2.html)
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    credentials = (
        ServiceAccountCredentials
        .from_json_keyfile_name(json_file, scope)
    )
    # 
    print('Getting credentials...')
    gc = gspread.authorize(credentials)
    # 
    print("Loading worksheet...")
    wks = gc.open(fn).worksheet(sheet_name)
    # 
    
    print(f"File: '{wks.spreadsheet.title}'\nSheet name: '{sheet_name}'!")
    wks = eDataFrame(wks.spreadsheet.sheet1.get_all_records())
    # 
    return wks

def reorderLegend(ax=None,order=None,unique=False):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    return(handles, labels)


# ** simulate data 

def simulate_data(var_groups, n, seed=None):
    '''
    Generate a simulated data set

    Input 
    -----
    var_groups  : a dictionaty with one or more of the following key-value
                  pairs:
                  {"discrete": [
                                {'vars': ['varname1', 'varname2'...],
                                 'min': <integer with minimum value>,
                                 'max': <integer with maximum value>},
                                {'vars': ['another varname1', ...],
                                 'min': <integer with minimum value>,
                                 'max': <integer with maximum value>},
                                 ...
                                ],
                    'continuous' : ['cont. var 1', 'cont. var 2', ...],
                    'categorical': {
                                    'cat. var 1': ['cat 1', 'cat2', ...],
                                    'cat. var 2': ['cat 1', 'cat2', ...],
                                   ...}
    n           : sample size
    seed        : to set the seed. If None, it won't set any seed


    Output 
    ------
    eDataFrame with simulated data


    Details
    -------
    Continuous variables will use a normal distribution with mean zero
    and standard deviation is a randomly selected value between 1 and 2.
    Discrete and categorical variable will sample values with 
    equal probability.

    '''
    def __simulate_data_discrete__(var_groups, n):
        res = {}
        for vars_group in var_groups:
            for var in vars_group['vars']:
                print(f'Generatig data for {var}...', flush=True)
                res[var] = rsample(range(vars_group['min'],
                                         vars_group['max']+1), size=n)
        return res

    def __simulate_data_continuous__(vars, n):
        res = {}
        for var in vars:
            print(f'Generatig data for {var} ...', flush=True)
            res[var] = rnorm(size=n, loc=0, scale=runif(1, 2))
        return res


    def __simulate_data_categorical__(vars, n):
        res = {}
        for var, cats in vars.items():
            print(f'Generatig data for {var} ...', flush=True)
            res[var] = rsample(cats, size=n)
        return res
    res={}
    if seed:
        set_seed(seed)
    for type, info in var_groups.items():
        if type=='discrete':
            res |= __simulate_data_discrete__(info, n)
        elif type=='continuous':
            res |= __simulate_data_continuous__(info, n)
        if type=='categorical':
            res |= __simulate_data_categorical__(info, n)
    res = eDataFrame(res).mutate_type(col2type=None, from_to={"object":'char'})
    return res

# * spss

class spss_data():
    '''
    Class receives a full path of a .sav file and returns a 
    pointer to that file. It requires
    import savReaderWriter as spss
    import pandas as pd
    import re

    Methods can search for variables and load selected vars
    '''
    def __init__(self, fn):
        # df_raw = spss.SavReader(fn, returnHeader=True, rawMode=True)
        self.__fn = fn
        with spss.SavHeaderReader(fn, ioUtf8=True) as header:
            self.__metadata = header.all()


    def search_var(self, regexp='', show_value_labels=False, get_output=False):
        vars_found = {}
        for key, label in self.__metadata.varLabels.items():
            if isinstance(label, bytes):
                label = label.decode("utf-8")
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            if bool(re.search(pattern=regexp, string=label)):
                vars_found[key] = label
        self.__print_vars_found__(vars_found, show_value_labels)
        if get_output:
            return vars_found
        else:
            return None

    def get_var(self, regexp='', show_value_labels=False):
        vars_found = self.search_var(regexp=regexp,
                                     show_value_labels=show_value_labels,
                                     get_output=True)
        return vars_found

    def list_vars(self, show_value_labels=False):
        self.get_var()


    def get_values(self, key, print_values=True):
        # if isinstance(key, str):
        #     key = key.encode()
        try:
            values = self.__metadata.valueLabels[key]
            var_label = self.__metadata.varLabels[key]
        except (KeyError, OSError, IOError) as e:
            return {}
        dic = {}
        for value, label in values.items():
            try:
                value = float(value)
            except (ValueError) as e:
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
            if isinstance(label, bytes):
                label = label.decode("utf-8")
            dic[value] = label
        if print_values:
            self.__print_values__(dic, key, var_label, False)
        return dic

    def values_freq(self, var):
        assert isinstance(var, str), "\n\n'var' must be a string\n\n"
        values = self.get_values(var, print_values=False)
        data = self.load_vars([var], use_labels=False)
        data = data.groupby([var]).size().reset_index(name='count', drop=False)
        data = data.assign(freq = lambda x: 100*x['count']/sum(x['count']))
        data['labels'] = data[var]
        data.replace({'labels':values}, regex=False, inplace=True)
        data = data.filter([var, 'labels', 'count', 'freq'])
        print("\n\n")
        print(data)
        print("\n\n")

    def values_summary(self, var):
        assert isinstance(var, str), "\n\n'var' must be a string\n\n"
        data = self.load_vars([var], use_labels=False)
        print("\n\n")
        print(data.describe())
        print("\n\n")

    def values_print_recode_str(self, key):
        dic = self.get_values(key, False)
        print("rec = {\""+key+"\" : {")
        for k, v in dic.items():
            print(f"    \"{v}\" : ,")
        print("}}")


    def load_vars(self, vars=None, use_labels=True, rec=None, vars_newnames=None):
        '''
        Read the variables in the .sav files and return a DataFrame

        Input:
           vars          a list with the id of the variables to be retrieved 
                         from the .sav file. If 'all', retrieve all variables
           use_labels    bool, if True return the SPSS labels of the variables
           rec           dictionary with the variables and values to be recoded.
                         Format {<varname>: <value1>:<new value>, 
                                            <value2>:<new value>}
           vars_newnames list with the new names of the variables. It muat
                         follow the order of the argument 'vars'

        Output:
            A Data Frame with the variables selected in 'vars'

        '''
        print(vars)
        if vars == 'all':
            vars = list(self.get_var('').keys())
        assert isinstance(vars, list) or vars is None,\
        f"Variable 'vars' must be a list or 'None'"
        assert isinstance(vars_newnames, list) or vars_newnames is None,\
        f"Variable 'vars_newnames' must be a list or 'None'"
        
        if vars:
            vars = self.__toBytes__(vars)
        else:
            vars = self.__metadata.varNames
        print(f"\nLoading values of {len(vars)} variable(s) ...")
        print(vars)
        print("\n\n")
        with spss.SavReader(self.__fn, returnHeader=False, rawMode=True,
                            selectVars = vars) as reader:
            vars_char = self.__toStr__(vars)
            data = pd.DataFrame(reader.all(), columns=vars_char)
        if use_labels:
            for key_bin, var_char in zip(vars, vars_char):
                data = (data
                        .replace(regex=False,
                                 to_replace={var_char :
                                             self.get_values(key_bin,
                                                             print_values=False)}))
        # recode
        if rec:
            self.recode(data, rec)
        # rename
        if vars_newnames:
            self.rename(data, vars, vars_newnames)
        return eDataFrame(data)


    # ancillary functions
    # -------------------
    def recode(self, data, rec):
        for var, rec_values in rec.items():
            data.replace(regex=False,
                         to_replace={var : rec_values},
                         inplace=True)

    def rename(self, data, vars, newnames):
        dic_names = {}
        for old, new in zip(vars, newnames):
            dic_names[old] = new
        data.rename(columns=dic_names, inplace=True)


    def __print_vars_found__(self, vars_found, show_value_labels):
        if not vars_found:
            return
        else:
            print("")
            for key, label in vars_found.items():
                print(f"{key:{13}}- {label}")
                if show_value_labels:
                    self.get_values(key)
            print("")
    

    def __print_values__(self, dic, key, var_label, print_varlabel=True):
        if isinstance(var_label, bytes):
            var_label = var_label.decode('utf-8')
        if isinstance(key, bytes):
            key = key.decode('utf-8')
        if print_varlabel:
            print("")
            print(f"{key:{13}}- {var_label}")
            print("")
        for value, label in dic.items():
            if isinstance(label, bytes):
                label = label.decode('utf-8')
            print(f"  {str(value):{5}} : {label}")
        print("")
        
    def __str__(self):
        return f"\nSPSS data:\n\n{self.__fn}\n\n"


    def __toBytes__(self, vars):
        for i, var in enumerate(vars):
            if not isinstance(var, bytes) and isinstance(var, str):
                vars[i] = var.encode()
        return vars

    def __toStr__(self, vars):
        for i, var in enumerate(vars):
            if isinstance(var, bytes) and not isinstance(var, str):
                vars[i] = var.decode('utf-8')
        return vars

class read_spss():
    '''
    Class receives a full path of a .sav file and returns a 
    pointer to that file. It requires
    import savReaderWriter as spss
    import pandas as pd
    import re

    Methods can search for variables and load selected vars
    '''
    def __init__(self, fn, encoding='utf-8'):
        # df_raw = spss.SavReader(fn, returnHeader=True, rawMode=True)
        fn = os.path.expanduser(fn)
        self.__fn = fn
        self.encoding=encoding
        with spss.SavHeaderReader(fn, ioUtf8=True) as header:
            self.__metadata = header.all()
            self.varLabels = self.__encode_dict__(header.varLabels)
            valuelabels={}
            for k, v in header.valueLabels.items():
                if isinstance(k, bytes) and not isinstance(k, str):
                    k = k.decode(self.encoding)
                valuelabels[k] =  self.__encode_dict__(v)  
            self.__val_labels__ = valuelabels


    def val_labels(self, vars=None):
        if isinstance(vars, str):
            vars=[vars]
        var_labels = {}
        if vars:
            for var in vars:
                var_labels[var] = self.__val_labels__[var]
        else:
            var_labels=self.__val_labels__
        return var_labels
        
    
    def var_list(self):
        self.var_search('.', encoding=self.encoding)
        
    
    def var_search(self, regexp='', show_value_labels=False, get_output=False):
        '''
        Search for variables using regular expression

        Input:
            regexp:      a regular expression to search in the variable names

            show_value_labels:  boolean, if True, display the labels of the 
                                categories of the alongside the variable id and 
                                label
        
            get_output: boolean, if True, return the matches in a dicttionary
                        with the variable id and label

                              
        '''
        vars_found = {}
        for key, label in self.__metadata.varLabels.items():
            if isinstance(label, bytes):
                try:
                    label = label.decode(self.encoding)
                except (OSError, IOError) as e:
                    print(f"Encoding does not work! Try another one!")
            if isinstance(key, bytes):
                key = key.decode(self.encoding)
            if bool(re.search(pattern=regexp, string=label)):
                vars_found[key] = label
        self.__print_vars_found__(vars_found, show_value_labels)
        if get_output:
            return vars_found
        else:
            return None

    def var_values(self, varname, print_values=True, get_output=False):
        '''
        Collect the values of the catorogies of a variable

        Input:
            varname:      a string with the id of the variable or variable name

            pring_values: boolean, if True, display the labels of the 
                          categories for they variable
        
            get_output: boolean, if True, return the matches in a dicttionary
                        with the categories code and label
                              
        '''
        assert isinstance(varname, str), f"'varname' must be a string with the "+\
        "variable name"
        try:
            values = self.__metadata.valueLabels[varname]
            var_label = self.__metadata.varLabels[varname]
        except (KeyError, OSError, IOError) as e:
            return {}
        dic = {}
        for value, label in values.items():
            try:
                value = float(value)
            except (ValueError) as e:
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
            if isinstance(label, bytes):
                label = label.decode("utf-8")
            dic[value] = label
        if print_values:
            self.__print_values__(dic, varname, var_label, False)
        if get_output:
            return dic

    def var_values_freq(self, varname, get_output=True):
        '''
        Collect the frequences for each value catorogies of a variable

        Input:
            varname:      a string with the id of the variable or variable name

            get_output: boolean, if True, returns a data frame with the 
                        values and their frequencies
        
        Output
            A dataframe with the values and their frequencies

                              
        '''
        assert isinstance(varname, str), "\n\n'varname' must be a string\n\n"
        values = self.var_values(varname, print_values=False, get_output=True)
        data = self.vars_load([varname], use_labels=False)
        data = data.groupby([varname]).size().reset_index(name='count', drop=False)
        data = data.assign(freq = lambda x: 100*x['count']/sum(x['count']))
        data['labels'] = data[varname]
        data.replace({'labels':values}, regex=False, inplace=True)
        data = data.filter([varname, 'labels', 'count', 'freq'])
        if self.__val_labels__[varname]:
            data.replace({'labels':self.__val_labels__[varname]}, regex=False,
                         inplace=True)
        print(self.varLabels[varname])
        if get_output:
            return eDataFrame(data)
        else:
            print("\n\n")
            print('\n')
            print(data)
            print("\n\n")
            

    def var_values_summary(self, varname):
        assert isinstance(varname, str), "\n\n'varname' must be a string\n\n"
        data = self.vars_load([varname], use_labels=False)
        print("\n\n")
        print(data.describe())
        print("\n\n")

    def var_values_recode_skeleton(self, varname, use='key', quote_key=False):
        dic = self.__val_labels__[varname]
        print("rec = {\""+varname+"\" : {")
        for k, v in dic.items():
            if use=='labels':
                v = f"\"{v}\""
                print(f"    {v:20} : {k},")
            else:
                if quote_key:
                    print(f"    \"{k}\" : \"{v}\",")
                else:
                    print(f"    {k} : \"{v}\",")
        print("}}")


    def vars_load(self, varnames=None, use_labels=True, rec=None, varsnames_new=None):
        '''
        Read the variables in the .sav files and return a DataFrame

        Input:
           varnames      a list with the id of the variables to be retrieved 
                         from the .sav file
           use_labels    bool, if True return the SPSS labels of the variables
           rec           dictionary with the variables and values to be recoded.
                         Format {<varname>: <value1>:<new value>, 
                                            <value2>:<new value>}
           varsnames_new list with the new names of the variables. It muat
                         follow the order of the argument 'varnamesb'

        Output:
            A Data Frame with the variables selected in 'varnamesb'

        '''
        varnamesb = varnames.copy()
        assert isinstance(varnamesb, list) or varnamesb is None,\
        f"Variable 'varnamesb' must be a list or 'None'"
        assert isinstance(varsnames_new, list) or varsnames_new is None,\
        f"Variable 'varsnames_new' must be a list or 'None'"
        
        if varnamesb:
            varnamesb = self.__toBytes__(varnamesb)
        else:
            varnamesb = self.__metadata.varNames
        print(f"\nLoading values of {len(varnamesb)} variable(s) ...")
        print(varnamesb)
        print("\n\n")
        with spss.SavReader(self.__fn, returnHeader=False, rawMode=True,
                            selectVars = varnamesb) as reader:
            vars_char = self.__toStr__(varnamesb)
            data = pd.DataFrame(reader.all(), columns=vars_char)
        if use_labels:
            for key_bin, var_char in zip(varnamesb, vars_char):
                data = (data
                        .replace(regex=False,
                                 to_replace={var_char :
                                             self.var_values(key_bin,
                                                             print_values=False)}))
        # recode
        if rec:
            self.recode(data, rec)
        # rename
        if varsnames_new:
            self.rename(data, varnamesb, varsnames_new)
        data = eDataFrame(data)
        return data


    # ancillary functions
    # -------------------
    def recode(self, data, rec):
        for varname, rec_values in rec.items():
            data.replace(regex=False,
                         to_replace={varname : rec_values},
                         inplace=True)

    def rename(self, data, vars, newnames):
        dic_names = {}
        for old, new in zip(vars, newnames):
            dic_names[old] = new
        data.rename(columns=dic_names, inplace=True)


    def __print_vars_found__(self, vars_found, show_value_labels):
        if not vars_found:
            return
        else:
            print("")
            for key, label in vars_found.items():
                print(f"{key:{13}}- {label}")
                if show_value_labels:
                    self.var_values(key)
            print("")
    

    def __print_values__(self, dic, key, var_label, print_varlabel=True):
        if isinstance(var_label, bytes):
            var_label = var_label.decode('utf-8')
        if isinstance(key, bytes):
            key = key.decode('utf-8')
        if print_varlabel:
            print("")
            print(f"{key:{13}}- {var_label}")
            print("")
        for value, label in dic.items():
            if isinstance(label, bytes):
                label = label.decode('utf-8')
            print(f"  {str(value):{5}} : {label}")
        print("")
        
    def __str__(self):
        return f"\nSPSS data:\n\n{self.__fn}\n\n"


    def __toBytes__(self, vars):
        for i, varname in enumerate(vars):
            if not isinstance(varname, bytes) and isinstance(varname, str):
                vars[i] = varname.encode()
        return vars

    def __toStr__(self, vars):
        for i, varname in enumerate(vars):
            if isinstance(varname, bytes) and not isinstance(varname, str):
                vars[i] = varname.decode(self.encoding)
        return vars


    def __encode_dict__(self, dict):
        res = {}
        for k, v in dict.items():
            kstr = self.__toStr__([k])[0]
            vstr = self.__toStr__([v])[0]
            res[kstr] = vstr
        return res


# * Extended DataFrame
# ** class
class eDataFrame(pd.DataFrame):
    def __init__(self,  *args, **kwargs):
        # use the __init__ method from DataFrame to ensure
        # that we're inheriting the correct behavior
        super(eDataFrame, self).__init__(*args, **kwargs)
        self.ncol = self.shape[1]
        self.nrow = self.shape[0]
        self.__create_var_labels__()
        self.__create_val_labels__()
        self.set_var_label(vars=kwargs.get("var_labels", None))
        self.set_val_label(vars=kwargs.get("val_labels", None))

    # this method is makes it so our methoeDataFrame return an instance
    # of eDataFrame, instead of a regular DataFrame
    @property
    def _constructor(self):
        return eDataFrame

# ** Properties
    # variables 
    # ---------
    def __create_var_labels__(self):
        self.__var_labels__=None
        self.__var_labels__={}
        for var in self.columns:
            self.__var_labels__[var]=var


    def set_var_label(self, vars=None):
        '''
        Set labels for the variables
        
        Input
           vars  : a dictionary. Keys are the variable names, values are
                   their labels
        '''
        if vars:
            for var, label in vars.items():
                self.__var_labels__[var] = label


    def get_var_label(self, vars=None, regexp=None):
        '''
        Variables have names and labels. This function retrieves the labels
        given the name of a regexp to match the name
        '''
        if isinstance(vars, str):
            vars=[vars]
        res={}
        if regexp:
            res={var:lab for var, lab in self.__var_labels__.items()
                 if bool(re.search(pattern=regexp, string=var))}
        else:
            if not vars:
                vars = list(self.__var_labels__.keys())
            for var in vars:
                if var in list(self.__var_labels__):
                    res[var]=self.__var_labels__[var]
        res = (eDataFrame(res, index=range(1))
               .pivot_longer(id_vars=None, value_vars=list(res.keys()),
                             var_name='var', value_name='label',
                             ignore_index=True))
        return res
            

    def get_var_name(self, labels=None, regexp=None):
        '''
        Variables have names and labels. This function retrieves the name
        given the labels of a regexp to match the label
        '''
        if isinstance(labels, str):
            labels=[labels]
        if not labels:
            labels = list(self.__var_labels__.values())
        res={}
        if regexp:
            for var, lab in self.__var_labels__.items():
                if bool(re.search(pattern=regexp, string=lab)):
                    res[var]=lab
        else:
            for label in labels:
                for var, lab in self.__var_labels__.items():
                    if label == lab:
                        res[var]=lab
        res = (eDataFrame(res, index=range(1))
               .pivot_longer(id_vars=None, value_vars=list(res.keys()),
                             var_name='var',
                             value_name='label', ignore_index=True)
               )
        return res

    def use_var_label(self, vars=None):
        '''
        Replace variable (column) name by their labels
        
        Input
           vars a list or string with the variable (column) names to replace
                with their labels
        '''
        if not vars:
            vars=list(self.__var_labels__.keys())
        if isinstance(vars, str):
            vars=[vars]
        for var in vars:
            var_label = self.__var_labels__.get(var, var)
            self.rename(columns={var:var_label} , inplace=True)


    def use_var_name(self, vars=None):
        if isinstance(vars, str):
            vars=[vars]
        if not vars:
            vars=list(self.__var_labels__.keys())
        for var in vars:
            var_label = self.__var_labels__.get(var, var)
            self.rename(columns={var_label:var} , inplace=True)

    # values 
    # ------
    def __create_val_labels__(self):
        self.__val_labels__=None
        self.__val_labels__={}


    def set_val_label(self, vars=None):
        '''
        Set variable labels

        Input
           vars a dictionary of dictionaries. The first key must be a variable
                 name. The inner dictionary the variable labels
        '''
        if vars:
            for var, label in vars.items():
                self.__val_labels__[var] = label


    def get_val_label(self, var=None):
        '''
        Get variable labels
        '''
        assert isinstance(var, str), "'var' must be a string"
        res={}
        if var not in self.columns:
            print(f"{var} is not in the columns of the DataFrame")
        else:
            res=self.__val_labels__.get(var, {})
            if not res:
                for v, lab in self.__val_labels__.items():
                    if var == lab:
                        res=self.__val_labels__.get(v, {})
                        break
            return {var:res}


    def use_val_label(self, var=None):
        rec=self.get_val_label(var=var)
        self.replace(rec, regex=False, inplace=True)


    def use_val_value(self, var=None):
        if var not in self.columns:
            print(f"{var} is not in the columns of the DataFrame")
        rec={}
        val_labels = self.get_val_label(var=var)
        if not val_labels:
            var=self.get_var_name(labels=var).query(f"var=='{var}'")[var]
            print(var)
            val_labels = self.get_val_label(var=var)
            print(val_labels)
        for value, label in val_labels[var].items():
            rec[label]=value
        self.replace({var:rec}, regex=False, inplace=True)


# ** Data wrangling
    def case_when(self, varname, replace=None):
        '''
        Deprecated. Use mutate_case insead. Kept for backward compatibility.
        It requires duplicating the quotation marks for string and using
        quotes for numbers in the dictionary. 
        Ex:
          {"case match": "'a string'"}
          {"case match": "a number"}
        
        See mutate_case documentation
        '''
        if isinstance(varname, dict):
            res = self
            for var, replace in varname.items():
                res = res.__case_when__(var, replace)
        if isinstance(varname, str):
            res = self.__case_when__(varname, replace)
        return eDataFrame(res)

    def mutate_case(self, varname, replace=None):
        '''
        Recode variable based on matching row values

        Input
        -----
           varname a dict or a string. If a string, the string will be the
                   name of the new variable in the DataFrame. CAUTION: It will 
                   overwrite an existing variable if the name already exists.
                   The replace argument is used only when varname is a string.
                   In that case, replace must be a dictionary with the form:
                  
                    {"(condition A)": <value>}
        
                  or 

                   {"(cond. A) <logical operator> (cond. B)...": <value>}
        
                  The rows that satisfies the conditions will take the value
                  specified in <value>. 

                  If varname is a dict, argument replace is not used, and 
                  varname must take the form

                    {"<new_varname>" : "(condition A)": <value>}
        
                  Cautionary note above applies in this case.
        
                  It is possibl to copy the content of a variable using 
                  "copy(<varname)". Example:


                    {"<new_varname>" : "(condition A)": "copy(<a varname>)"}
    
                  See examples below
        
           Output
           -------
              eDataFrame with the a new variable

           Example
           -------

           df = ds.eDataFrame({"a": ['a', 'b', 'a', 'c'],
                               'b': [1,2,3,4]})

           # Using varname as string
           (
               dft
               .case_when('a_new', {
                   f"(a=='b') ": "B",
                   f"(a!='b') ": "not B"})
               .case_when('b_new', {
                   f"(b<=3) ": "Low",
                   f"(b>3) ": "High"})
           )


           # Using varname as a dict
           (
               dft
               .case_when({
                   'a_new' :{f"(a=='b') ": "B",
                             f"(a=='c') ": "not B"
                             f"(a=='d') ": "copy(a)"
                            },
                   'b_new': {f"(b<=3) ": "Low",
                             f"(b>3) ": "High",
                             True: "copy(a)"
                            }
               })
           )

           # CAUTION: if the var exists, it will overwrite it
           # Using varname as a string
           (
               dft
               .case_when(
                   'a' , {f"(a=='b') ": f"'B'",
                          f"(a!='b') ": f"'not B'"})
           )

            # Using varname as a dict
            (
                dft
                .case_when({
                    'a' :{f"(a=='b') ": f"'B'",
                          f"(a!='b') ": f"'not B'"}
                    })
            )


        '''
        # add quote to all dictionary values (need for the main function)
        if isinstance(varname, dict):
            for k, v in varname.items():
                newv={}
                for vk, vv in v.items():
                    if isinstance(vv, float) or isinstance(vv, int):
                        newv[vk] = f"{vv}"
                    elif not vv:
                        newv[vk] = None
                    elif bool(re.search(pattern='copy\(.*\)', string=vv)):
                        newv[vk] = re.sub(pattern="copy\(|\)", repl='', string=vv)
                    else:
                        newv[vk] = f"\"{vv}\""
                varname[k]=newv
            res = self
            for var, replace in varname.items():
                res = res.__case_when__(var, replace)
                res = res.fill_na(value=np.nan , vars=var)
        if isinstance(varname, str):
            res = self.__case_when__(varname, replace)
            res=res.fill_na(value=np.nan , vars=varname)
        return eDataFrame(res)
                

    def __case_when__(self, varname, replace):
        if varname in self.columns:
            varname_exists = True
            col_names = self.columns
        else:
            varname_exists = False
        # replace using casewhen
        res = (
            self >>
            define(___newvar_name_placeholder___=case_when_ply(replace))
        )
        # remove old var if it existed
        if varname_exists:
            res.drop([varname], axis=1, inplace=True)
            res = (res
                   .rename(columns={"___newvar_name_placeholder___":varname},
                           inplace=False)
                   .filter(col_names)
                   )
        else:
            res.rename(columns={"___newvar_name_placeholder___":varname},
                       inplace=True)
        # set 'nan' to np.nan 
        res=res.replace({varname:{"nan":np.nan}} , regex=False, inplace=False)    
        return eDataFrame(res)

    def pivot_longer(self, id_vars, value_vars=None,
                     var_name=None, value_name='value', ignore_index=True):
        '''
        Reshape the DataFrame to a long format
        
        Input
           id_vars None, string, np.array, or a list of columns in the DataFrame. If None, it will
                   use all variables not in 'value_vars'
        '''
        if isinstance(id_vars, np.ndarray):
            id_vars = id_vars.tolist() 
        assert isinstance(id_vars, list) or isinstance(id_vars, str) or not id_vars, (
            "id_vars must be a list, string, or None"
        )
        if not id_vars:
            value_vars = [value_vars] if isinstance(value_vars, str) else value_vars
            id_vars = (
                self
                .drop(value_vars, axis=1)
                .columns
                .values
                .tolist()
            )
        res = pd.melt(self, id_vars=id_vars, value_vars=value_vars,
                       var_name=var_name, value_name=value_name,
                       ignore_index=ignore_index)
        return eDataFrame(res)
        

    def pivot_wider(self, id_vars=None, cols_from=None, values_from=None,
                    aggfunc='sum', sep="_"):
        '''
        Reshape the DataFrame to a long format
        
        Input
           id_vars None, string, np.array, or a list of columns in the DataFrame. If None, it will
                   use all variables not in 'cols_from' and 'values_from'
        '''
        if isinstance(id_vars, np.ndarray):
            id_vars = id_vars.tolist() 
        assert isinstance(id_vars, list) or isinstance(id_vars, str) or not id_vars, (
            "id_vars must be a list, string, or None"
        )
        if not id_vars:
            id_vars = (
                self
                .drop([cols_from, values_from], axis=1)
                .columns
                .values
                .tolist()
            )
        res = (self
               .pivot_table(values=values_from, index=id_vars,
                            columns=cols_from, aggfunc=aggfunc)
               .reset_index(drop=False)
               )
        res = eDataFrame(res)
        try:
            res = res.flatten_columns()
        except (OSError, IOError, AssertionError) as e:
            pass
        if isinstance(values_from, list) and len(values_from)==1:
            res.columns = [re.sub(pattern=values_from[0]+sep, repl='', string=s)
                           for s in res.columns]
        res.columns.name=''
        return eDataFrame(res)


    def combine(self, cols, colname=None, sep='_', remove=False):
        '''
        Combine columns
        
        Input
           cols    list with the name of the columns to join
           colmane string with the name of the new colum
           sep     string to put between columns merged
           remove  boolean, if True will remove the columns merged
        
        Output
           Extended data frame with columns merged
        '''
        if not colname:
            colname = f"{sep}".join(cols)
        res = (self
               .select_cols(names=cols)
               .fillna("")
               .astype(str)
               .agg(f"{sep}".join, axis=1)
               )
        res = eDataFrame(res)
        res.columns=[colname]
        res = self.bind_col(res, ignore_index=False)
        if remove:
            res=res.drop_cols(names=cols)
        res = res.loc[:, ~res.columns.duplicated(keep='last')]
        return res
        

    def mutate(self, dict=None, var_to_wrap=None, wrap=None, wrap_char="\\n"):
        res = self
        if dict:
            for k, v in dict.items():
                res = res.assign(**{k: v})
                res = res.loc[:, ~res.columns.duplicated(keep='last')]
        if var_to_wrap and wrap:
            res=res.__wrap_var__(var=var_to_wrap, wrap=wrap, wrap_char=wrap_char)
        return eDataFrame(res)

    def mutate_rowwise(self, dict):
        res = self
        for k, v in dict.items():
            res = res.assign(**{k:lambda x: x.apply(v, axis=1)})
            res = res.loc[:, ~res.columns.duplicated(keep='last')]
        return eDataFrame(res)

    def mutate_categorical(self, var, cats=None, ordered=True, wrap=None,
                           wrap_char=None):
        '''
        Change format of string or object to categorical

        Input
        -----
        var  string, list, or dictionary with variable name to set as categorical
             type. If a dictionary is used, the keys must be the variable name
             and the values a list with the categories in the order they should
             follow.
        cats a list with the categories. It None unique variable valies will
             be used and they will be sorted if ordered=True. Ignored if
             var is a dictionary
        ordered boolean, if True it will order the values respecting the
                order provided in 'cats', in increaseing order. If cats=None
                and ordered=True, it sorts the unique values of 'var' first, 
                if it is possible. and use as the categories,
        wrap either None or an integer. If an integer, return category
             labels with string wrapped
        wrap_char If None, it uses '\n' for new line. A specific new line
                  markdown can be uses, such as '<br>'

        Output
        ------
        eDataFrame with categorical variable

        '''
        assert isinstance(var, dict) or \
            isinstance(var, str) or \
            isinstance(var, list) , ("'var' must be a list, string, or a "\
                                        "dictionary")
        vars=None
        if isinstance(var, str):
            vars={var:var}
        if isinstance(var, list):
            vars={v:v for v in var}
        if isinstance(var, dict):
            for vari, cats in var.items():
                self = (
                    self
                    .rename_cols(columns={vari:vari}, tolower=False)
                    .mutate({vari: lambda x:
                             pd.Categorical(x[vari],
                                            categories=cats,
                                            ordered=ordered)})
                )
                if wrap:
                    self=self.__wrap_var__(vari, wrap=wrap, wrap_char=wrap_char)
        else:
            for var, label in vars.items():
                self = (
                    self
                    .rename_cols(columns={var:label}, tolower=False)
                    .mutate({label: lambda x:
                             pd.Categorical(x[label],
                                            categories=cats,
                                            ordered=ordered)})
                )
                if wrap:
                    self=self.__wrap_var__(label, wrap=wrap, wrap_char=wrap_char)

        return eDataFrame(self)
        

    def bind_row(self, df):
        res =  pd.concat([self, df], axis=0, ignore_index=True)
        return eDataFrame(res)


    def bind_col(self, df, ignore_index=True):
        if ignore_index:
            df=df.set_index(self.index)
        res =  pd.concat([self, df], axis=1, ignore_index=ignore_index)
        return eDataFrame(res)
        

    def separate(self, col, into, regexp, keep=False):
        res = self[col].str.extract(regexp)
        if isinstance(into, str):
            into = list(into)
        res.columns = into
        res = self.bind_col(res, ignore_index=False)
        if not keep:
            res = res.drop([col], axis=1)
        return eDataFrame(res)


    def join(self, data, how='left', on=None, left_on=None, right_on=None,
              conflict='keep_all', suffixes=['_x', "_y"], indicator=False):
        '''
        Merge data frames

        Input
           data the data frame to merge
           how (see pd.merge)
           on (see pd.merge)
           on_left (see pd.merge)
           on_right (see pd.merge)
           sufixes (see pd.merge)
           indicator (see pd.merge)
           conflict either 'keep_all', 'keep_x', or 'keep_y'. It defines the
                    bahavior in case of conlfict between columns data frames
                    being merged
                    
        '''
        res = self.merge(data, how=how, on=on ,
                         left_on=left_on, right_on=right_on,
                         suffixes=suffixes, indicator=indicator)
        if conflict=="keep_x":
            res = self.__join_keep__(res, data,
                                     keep_suffix=suffixes[0],
                                     discard_suffix=suffixes[1])
        if conflict=="keep_y":
            res = self.__join_keep__(res, data,
                                     keep_suffix=suffixes[1],
                                     discard_suffix=suffixes[0])
        return eDataFrame(res)


    def __join_keep__(self, res, data, keep_suffix, discard_suffix):
        cols = list(data.columns)
        keep = [col+keep_suffix for col in cols]
        discard = [col+discard_suffix for col in cols]
        # 
        cols_res = list(res.columns)
        cols_res_merged = [col_res for col_res in cols_res if
                           col_res in keep or col_res in discard]
        # check if columns are unique
        if len(cols_res)!=len(set(cols_res)):
            warnings.warn("Columns merged are not not unique! All were kept",
                          UserWarning)
        else:
            for disc in discard:
                if disc in cols_res_merged:
                    res.drop([disc], axis=1, inplace=True)
            for k in keep:
                if k in cols_res_merged:
                    col=re.sub(pattern=keep_suffix, repl="", string=k)
                    res.rename(columns={k:col}, inplace=True)
        return res
        
    
    def exchange(self, data, var, match_on=None,
                 match_on_left=None, match_on_right=None):
        '''
        Exchange values of one data frame from values of another using
        a matching columns with unieuq values
        
        Input
           data a data frame with values to collect 
           var string with the name of the variable whose values will
               be exchnanged
        match_on a list with the variables to match in both data frames
        match_on_left  a list with the variables on left data frame to match
        match_on_right  a list with the variables on right data frame to match
        '''
        assert isinstance(var, str), "'var' must be a string!"
        assert var in self.columns and var in data.columns, ("'var' must "+
                                                             "be in both "+
                                                             "data frames")
        # assert isinstance(match_on_left, list), "'match_on' must be a list!"
        # assert isinstance(match_on, list), "'match_on' must be a list!"
        vars = [var]
        if match_on:
            assert isinstance(match_on, list), "'match_on' must be a list!"
            assert len(self[match_on].drop_duplicates())==self[match_on].nrow,(
                "The matching variable 'match_on' must contain unique values!"
            )
            vars = vars + match_on
        if match_on_left:
            assert isinstance(match_on_left, list), "'match_on_left' must be a list!"
            assert len(self[match_on_left].unique())==self[match_on_left].nrow,(
                "The matching variable 'match_on_left' must contain unique values!"
            )
        if match_on_right:
            assert isinstance(match_on_right, list), "'match_on_right' must be a list!"
            assert len(self[match_on_right].unique())==self[match_on_right].nrow,(
                "The matching variable 'match_on_right' must contain unique values!"
            )
            vars = vars + match_on_right
        #
        # 
        try:
            var_order = '___XXorder_originalXX___'
            self[var_order]=list(range(self.nrow))
            res=(self
                 .join(data[vars], how='inner', on=match_on ,
                       left_on=match_on_left, right_on=match_on_right,
                       conflict="keep_y", suffixes=["_x", "_y"])
                 )
            anti_match=(self
                        .join(data[vars], how='outer', on=match_on ,
                              left_on=match_on_left, right_on=match_on_right,
                              conflict="keep_x", suffixes=["_x", "_y"],
                              indicator=True)
                        .query(f"_merge=='left_only'")
                        .drop(['_merge'], axis=1)
                        )
            res = (res
                   .bind_row(anti_match)
                   .sort_values([var_order], ascending=True)
                   .drop([var_order], axis=1)
                   .reset_index(drop=True)
                   )
        except Exception as e:
            res = self
        return eDataFrame(res)


    def cut(self, var, ncuts=10, labels=None, varname=None,
            robust=False, overwrite=True):
        '''
        Cut a numerical variable into categories

        Input
           var    string with the name of the variable
           ncuts  number of categories computed based on quantiles
           labels list with the labels of the categories
           varname string with the the name of the new variable
           overwrite boolean, if True and variable already exists, overwrite it
           robust Boolean, if True, it computes the cut limits after
                   excluding the outliers
        '''
        if not varname:
            varname=f"{var}_cat"
        res = self[[var]].drop_duplicates().dropna(axis=0)
        if robust:
            idx = res.get_outliers(var).index
            res = res.loc[idx,:]
        q = np.linspace(0, 1, ncuts+1)
        if overwrite:
            if varname in self.columns:
                self = self.drop([varname], axis=1)
        res[varname] = pd.qcut(res[var], q, labels=labels)
        res=self.merge(res, how='left', on=[var])
        return eDataFrame(res)

    def get_dummies(self, vars, keep_original=True,
                    prefix=False, prefix_sep=None, drop_first=False):
        "Deprecated. Use mutate_to_dummy."
        # this is to keep backwards compatibility
        self=self.mutate_to_dummy(vars=vars,
                                  keep_original=keep_original,
                                  prefix=prefix,
                                  prefix_sep=prefix_sep,
                                  drop_first=drop_first
                                  )
        return self

    def mutate_to_dummy(self, vars, keep_original=True,
                        prefix=False, prefix_sep=None, drop_first=False,
                        overwrite=False
                        ):
        '''
        Return dummy version of variables

        Input:
           vars : a list of variable names in the data frame to convert to dummies
           keep_original : boolean. If True, keep the original categorical 
                           variable
           prefix : boolean. If True, use the name of the original variable
                    as prefix
           prefix_sep : string to use as separator when prefix is used
           drop_first : boolean. If True, drop the first category to use as
                        reference

        Output:
            a data frame with columns indicating the categories

        '''
        # dfd = pd.get_dummies(self.filter(vars))
        # res = (self
        #        .bind_col(dfd,  ignore_index=False)
        #        )
        if not prefix or not prefix_sep:
            prefix_sep=['']*len(vars)
        if prefix or not prefix_sep:
            prefix_sep=['_']*len(vars)
        prefix     = prefix if prefix else ['']*len(vars)
        res_dummies = pd .get_dummies(data=self.select_cols(names=vars),
                                      columns=vars,
                                      prefix=prefix,
                                      prefix_sep=prefix_sep,
                                      drop_first=drop_first
                                      )
        res_dummies=eDataFrame(res_dummies)
        # check if new column names already exists
        new_varnames=res_dummies.names()
        current_varnames=self.names()
        common_cols=set.intersection(set(new_varnames), set(current_varnames))
        if common_cols:
            if overwrite:
                self=self.drop_cols(names=common_cols)
            else:
                print(("Note: \nDummy variables could not be created. There are "+
                       "columns already in the data with the names "+
                       "of the new dummy variables. Use either 'overwrite'" +
                       "=True or, to avoid overwriting the existing variables," +
                       "use 'prefix'=True"), flush=True)
                return self
        if keep_original:
            res = self.bind_col(res_dummies, ignore_index=False)
        return eDataFrame(res)


    def mutate_collect(self, into, vars, overwrite=False):
        '''
        This method reverts the "mutate_to_dummy" function (see
        Details)
        
        Input
        -----
        into      string with the name of the new variable
        vars      a list of string with variable names (see Details)
        overwrite boolean, overwrite old variable if it already exists


        Details:
        -------
        It create a variable called <into>. That new variable
        will receive the name of the column listed in the parameter
        'vars' whenever the content of that column is not nil.
        Note that it assumes the list of variables in 'vars'
        is an expansion of a categorical variable, so in each row,
        only one of the columns must be non-nil. Otherwise,
        values can be overwritten in the new value created
        in the order they appear in the list 'vars.'
        
        '''
            
        if into in self.names():
            if overwrite:
                self=self.drop_cols(names=into)
            else:
                res=self
                print(f"\nVariable '{into}' already exists. "+\
                      "New variable not created. "+\
                      "Use overwrite=True to overwrite it.\n", flush=True)
        if overwrite or into not in self.names():
            res=self.copy()
            res[into]=np.nan 
            for var in vars:
                res = (
                    res
                    .mutate_case({
                         into: {
                            f"(pd.isna({var}))": f"copy({into})",
                            True               : var
                               }
                     })
                )
        return eDataFrame(res)


    def reorder(self, var, order):
        res = (self
               .set_index(var, drop=False)
               .reindex(order)
               .reset_index(drop=True)
               )
        return eDataFrame(res)

    def insert_row(self, newrow, index):
        '''
        Insert a new row on position 'index'
        
        Input
           newrow a dict with the new row
           index integer with row position to insert the new row. If None,
                 insert at the end of the DataFrame
        '''
        assert isinstance(newrow, dict), "newrow must be a dict"
        if not index:
            index = self.index.max() + 1
        newrow = pd.DataFrame(newrow, index=[index-.5])
        res = (self
               .append(newrow, ignore_index=False)
               .sort_index()
               .reset_index(drop=True)
               )
        return eDataFrame(res)


    def nest(self, group):
        assert isinstance(group, str) or isinstance(group, list), "'group' "+\
        "must be a list of strings or a string"
        # 
        if isinstance(group, str):
            group = [group]
        # def nest_df(df):
        #     return [df]
        # res = (
        #     self
        #     .groupby(group)
        #     .apply(nest_df)
        #     .reset_index(drop=False, name='data')
        # )
        tmp = (
            self
            .groupby(group)
        )
        tmp = eDataFrame(tmp)
        res = eDataFrame([*tmp[0]])
        res.columns = group
        res['data'] = tmp[1]
        res = res.mutate_rowwise({'data': lambda x: eDataFrame(x['data'])})
        return eDataFrame(res)


    def unnest(self, col, id_vars):
        '''
        Unnest a nested data frame

        Input:
           col     : name of the column that contains the nested data.frame
           id_vars : string or list of strings with the variables in the nested 
                     data frame to use as id in the unnested one
        '''
        assert isinstance(id_vars, str) or isinstance(id_vars, list), (
            "\n\nid_vars must be a list or a string\n\n"
        )
        if isinstance(id_vars, str):
            id_vars = [id_vars]
        res = (
            self
            .mutate_rowwise({col: lambda x: eDataFrame(x[col])})
            )
        if len(id_vars)>1:
            placeholder = "__XXplaceholderXX__"
            placeholder_list=[placeholder]*max(len(id_vars)-1, 1)
            regexp=f"(.*){'(.*)'.join(placeholder_list)}(.*)"
            res = (
                res
                .mutate_rowwise({
                    col: lambda x: eDataFrame(x[col])
                })
                .mutate_rowwise(
                    {col: lambda x:
                     (x[col]
                      .mutate({'id': placeholder.join([str(s) for s in x[id_vars]])})
                      .separate(col="id", into=id_vars, regexp=regexp, keep=False))
                     })
            )
        else:
            res = (
                res
                .mutate_rowwise({
                    col: lambda x: (x[col] .mutate({'___id___': x[id_vars[0]]}))
                })
                
            )
        res = [df for df in res[col]]
        res = pd.concat(res).rename(columns={'___id___':id_vars[0]}, inplace=False)
        return eDataFrame(res)
        

    def recode_skeleton_print(self, var):
        vals = self[var].drop_duplicates().tolist()
        print("rec = {\"<varnew>\" : {")
        for v in vals:
            v = f"\"{v}\""
            print(f"    {v:20} : \"\",")
        print("}}")


    def select(self, vars=None, regex=None):
        '''
        Deprecated. Use select_cols
        '''
        res = self.select_cols(names=vars, regex=regex)
        return eDataFrame(res)
        

    def select_cols(self, names=None, regex=None, positions=None, index=None,
                    range=None, type=None):
        '''
        Select columns of the eDataFrame

        Input
        -----
           names string, list, or dict with the variables to select.
                If a dict, it renames the variables; the keys must be the old
                names of the variables to select, and the values of the dict
                the new names.

           regex a string with a regular expression. It return the variables
                 that match it

           positions a list or a number with the position of the columns
                         to select
                         Note: the first position starts on 1, not 0

           index a list or a number with the "index" of the columns
                         to select. It starts at 0
        
           range a list with two elements indicating the first and 
                 last position to select
                 Note: as for the position argument, this argument counts
                       columns starting on 1 and the end position is
                       inclusive. For instance, range=[1:4] returns the 
                       first column up the the forth.
        
           type a string or list of strings with the type of column to select.
                It accepts all pandas type (int64, float64, object, category,
                etc.) and also a few other entries:

                'char'    returns object and category columns
                'string'  same as char
                'text'    same as char
                'numeric' returns int64, int32, flat32, and float64
                'date'    returns pandas datetime type

        Output
        ------
           a eDataFrame with only the columns selected

        
        Details
        -------
        The order of priority if more than one argument is used is:

            regex   >   names (list/dict/str) >  positions > index > range

        '''
        res = self
        if regex:
            res = self.filter(regex=f"{regex}")
        elif isinstance(names, dict):
            res = (
                self
                .rename(columns=names, inplace=False)
                .filter(list(names.values()))
            )
        elif isinstance(names, str):
            names = [names]
            res = self.filter(names)
        elif isinstance(names, list):
            res = self.filter(names)
        # 
        elif isinstance(positions, list):
            positions = [cp-1 for cp in positions]
            res  = self.iloc[:,positions]
        elif isinstance(positions, int):
            res  = self.iloc[:,[positions-1]]
        # 
        elif isinstance(index, list):
            res  = self.iloc[:,index]
        elif isinstance(index, int):
            res  = self.iloc[:,[index]]
        # 
        elif isinstance(range, list):
            res  = self.iloc[:,range[0]-1:range[1]]
        # 
        res = eDataFrame(res)
        if type:
            res = res.__select_cols_type__(type)
        return res


    def __select_cols_type__(self, type):
        res = eDataFrame()
        tmp = eDataFrame()
        if isinstance(type, str):
            type = [type]
        for tp in type:
            if tp=='numeric':
                tmp = self.select_dtypes(include=['int64', 'float64',
                                                  'int32', 'float32'])
            elif tp=='char' or tp=='string' or tp=='text':
                tmp = self.select_dtypes(include=['category', 'object'])
            elif tp=='date':
                tmp = self.select_dtypes(include=['datetime64'])
            else:
                try:
                    tmp = self.select_dtypes(include=[tp])
                except (OSError, IOError, TypeError) as e:
                    print(f"Error: type {tp} does no exits!")
            if tmp.nrow>0:
                res = res.bind_col(tmp,  ignore_index=False)
        return res


    def mutate_type(self, col2type=None, from_to=None, **kws):
        '''
        Convert column to different types. See pandas astype method.

        Input:
           col2type a dict with {colname:newtype}. See pandas astype method.

           from_to dictionary to convert all columns from one type into another.
                   keys should the old type (type "from"), values should be the
                   new type (type "to")

                   Note: the argument dtype of astype takes {colname:newtype}.
                         from_to takes {from_type:to_type}
                 
                   Note: for the "from" type, it accepts all the types used in 
                         the argument "type" in the method select_cols(). see
                         pandasci.eDataFrame.select_cols()

           **kws see pandas astype (not implemented)

        Output:
            eDataFrame with converted types
        '''
        if col2type:
            for col, type in col2type.items():
                if type=='date':
                    col2type[col] = 'datetime64[ns]'
                elif type=='numeric':
                    col2type[col] = 'float64'
                elif type=='char' or type=='string' or type=='text':
                    col2type[col] = 'category'
            self = eDataFrame(self.astype(dtype=col2type))
        if from_to:
            for from_type, to_type in from_to.items():
                vars = self.select_cols(type=from_type).columns.to_list()
                new_types = {}
                for var in vars:
                    if to_type=='date':
                        totype = 'datetime[ns]'
                        print(totype)
                    elif to_type=='numeric':
                        totype = 'float64'
                    elif to_type=='char' or to_type=='string' or to_type=='text':
                        totype = 'category'
                    else:
                        totype=to_type
                    new_types[var] = totype
                self = eDataFrame(self.astype(dtype=new_types))
        return self


    def select_rows(self, query=None, regex=None, row_numbers=None, index=None,
                    dropna=None, keepna=None):
        '''
        Return eDataFrame with matched rows. See details

        Input
        -----
           - regex string, number, or dict. If string, select all rows that 
                   match the regex. It dict, the key must be the column name and 
                   the value the regex. It will return all rows that match 
                   EITHER one of the regex for the respective column specified. 
                   - Note: one can use select_rows repeatedly to keep only rows 
                           that math ALL regex for each respective column. 
                           See examples.
                    - Note: numbers can be used as values to match, but they are 
                            matched as usual strings.

           - query a string with a text to return the rows that match the criteria
                 exactly. Uses pandas 'query' function. See example.

           - row_numbers a list with the number of the rows to select. It starts 
                       on 1 for the first row (note that it is differnt from 
                       row index, which usually starts at 0 for the first row)
                       contains the maximum value of column 'a'

           index a list with values of the row index to select
           drop_na a string or list of column names to drop NA. If a boolean value 
                   "True" is used, it removes rows with NA considering all columns

        Output
        ------

        Details
        -------
        If all parameters are provided, only the one with the highest priority will
        be considered. The priority order is 

                     regex   >   query   >   row_numbers   >   index  > value

        The drop_na is always considered. To remove rows with NAs, it uses the 
        method dropna.


        Examples
        --------
        TBD

        '''
        res = self
        if regex:
            assert (isinstance(regex, dict) or
                    isinstance(regex, str) or
                    isinstance(regex, float) or
                    isinstance(regex, int)), "regex must be either a string, "+\
            "a number, or a dictionary"
            if isinstance(regex, dict):
                idx = np.array([False]*self.nrow)
                for var, var_regex in regex.items():
                    var = str(var)
                    idxi = self[var].astype(str).str.contains(f"{var_regex}").values
                    idx  = (idx | idxi) 
                res = self.loc[idx]
            else:
                if isinstance(regex, float) or isinstance(regex, int):
                    regex = str(regex)
                idx = np.array([False]*self.nrow)
                f = lambda x: x.astype(str).str.contains(regex, na=False, case=False)
                res = self.astype(str)
                res = self[res.apply(f).any(axis=1)]
        elif query:
            res = self.query(expr=query)
        elif row_numbers:
            assert isinstance(row_numbers, list), 'row_numbers must be a list'
            row_numbers = [rn-1 for rn in row_numbers]
            res = self.iloc[row_numbers]
        elif index:
            assert isinstance(index, list), 'index must be a list'
            res = self.loc[index]
        if dropna:
            assert (
                isinstance(dropna, list) or
                isinstance(dropna, str) or
                isinstance(dropna, bool)
            ), "dropna must be None, a list, a boolean, or a string"
            if not dropna or (isinstance(dropna, bool) and dropna):
                dropna = None 
            if dropna and isinstance(dropna, str):
                dropna = [dropna]
            if res.nrow>0:
                res = res.dropna(subset=dropna, axis=0)
        if keepna:
            assert (
                isinstance(keepna, list) or
                isinstance(keepna, str) or
                isinstance(keepna, bool)
            ), "keepna must be None, a list, a boolean, or a string"
            if not keepna or (isinstance(keepna, bool) and keepna):
                keepna = res.names() 
            if keepna and isinstance(keepna, str):
                keepna = [keepna]
            if res.nrow>0:
                res=res[res[keepna].isnull().any(axis=1)]
        return eDataFrame(res)


            
    def drop_cols(self, names=None, regex=None, **kws):
        '''
        
        
        Input:
           names string or list with the variables to drop.
           regex a string with a regular expression. It drops the variables
                 that match it. If provided, 'names' is ignored
           kws check pandas drop method (works only when regex is not provided)

        Output:
            a eDataFrame without the columns selected to drop

        '''
        if regex:
            res = self.select(regex=f"^((?!{regex}).)*$")
        else:
            if isinstance(names, str):
                names = [names]
            kws["axis"] = 1
            res = self.drop(names, **kws)
        return eDataFrame(res)


    def drop_rows(self, query=None, regex=None, row_numbers=None, index=None,
                    dropna=None, keepna=None):
        '''
        Revove rows. See select_rows for parameters.
        '''
        self=self.reset_index(drop=True)
        idx = self.index
        if query or regex or row_numbers or index:
            idx_remove = self.select_rows(query=query, regex=regex,
                                          row_numbers=row_numbers, index=index,
                                          dropna=None).index
            idx_keep = list(np.setdiff1d(idx, idx_remove))
            res = self.loc[idx_keep]
        else:
            res = self
        if dropna:
            res = res.select_rows(dropna=dropna)
        # if len(idx_remove)>0:
        #     idx_keep = list(np.setdiff1d(idx, idx_remove))
        #     res = self.loc[idx_keep]
        # else:
        #     res = self
        if keepna:
            res = res.select_rows(keepna=keepna)
        return eDataFrame(res)


    def fill_na(self, value=None, vars=None, **kws):
        '''
        Fill NA values with the provided 'value'
        
        Input
           value the value to fill NAs with
           vars a string of list with the variable to fill in. If None,
                fill all variables
           kws see pd.fillna
        
        Output
           eDataFrame with NAs filled
        '''
        vars = [vars] if isinstance(vars, str) else vars
        assert not vars or isinstance(vars, list), 'vars must be a string or a list'
        inplace=kws.get("inplace", False)
        kws['inplace']=inplace
        var_dict={}
        if vars:
            for var in vars:
                var_dict[var]=value
            if not inplace:
                res=self.fillna(value=var_dict, **kws)
            else:
                self.fillna(value=var_dict, **kws)
                res=self
        else:
            if not inplace:
                res=self.fillna(value=var_dict, **kws)
            else:
                self.fillna(value=var_dict, **kws)
                res=self
        return eDataFrame(res)

    def rename_cols(self, columns=None, regex=None, tolower=False, inplace=False,
                    cols2wrap=None, wrap=None, **kws):
        '''
        Rename columns of the eDataFrame

        Input
        -----
           columns  : dict. Keys are the old name, values are the new one.
           regex    : dict. When the key is the name of the old variable,
                      the value must be another dict with 
                      {regex:replacement string}
                      Otherwise, it can receive a dict with
                      {regex:replacement string}
                      only, and it will replace the regular expression match in
                      all columns
           tolower  : change capital letters to lower case
           cols2wrap: list of strings with names of columns to wrap
                      If None and wrap is not None, wrap all columns
           wrap     : integer to wrap the column name
        
           **kws other arguments (see pd.DataFrame.rename)

        Output
        ------
            eDataFrame with columns renamed

        Example
        -------

        '''
        assert not (columns and regex), "Use either columns or regex, but not both"
        if tolower:
            self.columns=self.columns.str.lower()
        res=self
        if columns:
            res = res.rename(columns=columns, inplace=False)
        if regex:
            for k, v in regex.items():
                if isinstance(v, dict):
                    from_regex, to_str=*v.keys(), *v.values()
                    res = res.rename(columns={k:re.sub(from_regex,to_str, k)},
                                     inplace=False)
                else:
                    res = res.rename(columns=lambda x: re.sub(k,v,x),
                                     inplace=False)
        if wrap:
            if not cols2wrap:
                cols2wrap=res.columns
            cols2wrap_raw=res.select_cols(names=cols2wrap).columns
            cols2wrap_wrapped=(
                res
                .select_cols(names=cols2wrap)
                .columns
                .str
                .wrap(wrap)
            )
            cols2wrap={old:new for old, new in zip(cols2wrap_raw,
                                                   cols2wrap_wrapped)}
            res=res.rename(columns=cols2wrap)
        # 
        if inplace:
            self=res
        else:
            return eDataFrame(res)


    def scale(self, vars=None, exclude=None, centralize=True, center='mean',
              group=None, newnames=None):
        '''
        Rescale variables by dividing it by its standard deviation

        Input:
            vars    : string, dict, or list with variables to scale. If 'None', 
                      it will scale all numerical variables. If using a dict,
                      the keys will be the names of the new rescale variables
            exclude : string or list with variables NOT to scale. If 'None', it will
                      scale all numerical variables in the argument 'vars'
            centralize: boolean, if True will recenter the variables
            center    : either 'mean' or a number to use as the center
            newnames  : string or list with name of the new rescaled variable.
                        if None, overwrite the original variable. It has no
                        effect when 'vars' is a dict
        Output
            dataframe with variables rescaled
        '''
        if isinstance(vars, str):
            vars = [vars]
        elif isinstance(vars, dict):
           newnames = list(vars.keys()) 
           vars = list(vars.values()) 
        else:
            if isinstance(newnames, str):
                newnames = [newnames]
        if isinstance(exclude, str):
            exclude = [exclude]
        if isinstance(newnames, list):
            for var, newvar in zip(vars, newnames):
                self = self.mutate({newvar: lambda x: x[var]})
                vars = newnames
        self = self.__rescale__(vars=vars, exclude=exclude,
                                centralize=centralize, scale=True,
                                center=center, group=group)
        return self

    def centralize(self, vars=None, exclude=None, center='mean'):
        '''
        Centralize variables by subtract it from the average or a center
        specified in the parameter 'center'

        Input:
            vars    : list with variables to centralize. If 'None', it will
                      centralize all numerical variables
            exclude : list with variables NOT to centralize. If 'None', it will
                      centralize all numerical variables in the argument 'vars'
            center  : either 'mean' or a number to use as the center
        
        Output
            dataframe with variables rescaled
        '''
        self = self.__rescale__(vars=vars, exclude=exclude,
                                centralize=True, scale=False, center=center,
                                group=None)
        return self

    def __rescale__(self, vars=None, exclude=None, centralize=False,
                    scale=False, center='mean', group=None):
        assert (center=='mean' or isinstance(center, float) or
                isinstance(center, int)), ("'center' must be either the string "+
                                           "'mean' or a number")
        if not vars:
            vars=self.select_dtypes(include=['int64','float64']).columns
        else:
            vars=self[vars].select_dtypes(include=['int64','float64']).columns
        if exclude:
            assert isinstance(exclude, list), "'exclude' must be a list"
            vars=[var for var in vars if var not in exclude]
        for var in vars:
            x = self.filter([var])
            self=self.drop([var], axis=1)
            if centralize:
                if center=='mean':
                    center_value=x.mean()
                else:
                    center_value=center
                x = x-center_value
            if scale:
                x=x/x.std()
            self[var]=x
        return self


    # def reset_index(self, name=None, drop=False):
    #     return eDataFrame(self.reset_index(drop=drop, name=name))

# ** group by
    def groupby(self, group, *args, **kwargs):
        res = egroupby(self, group)
        return res
        

# ** Statistics

    def get_outliers(self, var):
        Q1 = self[var].quantile(0.25)
        Q3 = self[var].quantile(0.75)
        IQR = Q3 - Q1
        idx = (self[var] < (Q1 - 1.5 * IQR)) | (self[var] > (Q3 + 1.5 * IQR))
        return self.loc[idx,:]


    def summary(self, vars=None, funs={'N':'count', "Missing":count_missing,
                                       'Mean':'mean',
                                       'Std.Dev':'std',
                                       # "Q25":quantile25,
                                       'Median':'median',
                                       # "Q75":quantile75,
                                       'Min':'min', "Max":'max'},
                groups=None, wide_format=None, ci=False, ci_level=0.95):
        '''
        Compute summary of numeric variables

        Input
           vars a string, a dict, or a list with columns to compute the summary. 
                If None, compute summary for all numerical variables.
                If dict, it must be {<colname>:<new name>}
           funs a dictionary of function labels (key) and functions (value)
                to use for the summary. Default values use some common
                summary functions (mean, median, etc.)
           groups a string or list with variable names. If provided, compute
                  the summaries per group
           wide_format if True, return results in a wide_format
           ci   boolean. If true, includes a gaussian conficence
                interval around the average using 'ci_level' as
                the confidence level
           ci_level a number between 0 and 1 indicating the
                    confidence level
        
        Output
          eDataFrame with summary
        '''
        vars_newname =None # to use when dict is used for vars
        if not vars:
            vars = self.select_dtypes(exclude = ['object']).columns.values.tolist()
        if groups and not isinstance(groups, list):
            groups = [groups]
        if vars and isinstance(vars, str):
            vars = [vars]
        if vars and isinstance(vars, dict):
            vars_oldname = list(vars.keys())
            vars_newname = list(vars.values())
            vars = vars_oldname 
        if groups:
            assert not 'variable' in groups,\
            ("Please, rename the variable called 'variable'. A group variable "+
             "called 'variable' is not allowed")
        assert isinstance(funs, list) or isinstance(funs, dict),\
        ("'funs' must be a list or a dictionary of functions")
        assert isinstance(vars, list), "'vars' must be a list of variable names"
        funs_labels={}
        # 
        if isinstance(funs, dict):
            funs_names = list(funs.values())
            funs = {fun:fun_name for fun_name, fun in funs.items()}
        else:
            funs_names = funs
            funs = {fun:fun_name for fun_name, fun in zip(funs, funs)}
        for fun_name, fun_label in funs.items():
            if callable(fun_name):
                funs_labels[fun_name.__name__] = fun_label
            else:
                funs_labels[fun_name] = fun_label
        # 
        # 
        if groups:
            res = self.__summary_group__(vars, funs_names, groups, wide_format,
                                         vars_newname)
        else:
            res = self.__summary__(vars, funs_names)
        # 
        res.rename(columns=funs_labels, inplace=True)
        cols = list(res.columns)
        cols = [col for col in cols if col not in list(funs_labels.keys())]
        res = res.filter(cols+list(funs_labels.keys()))
        # 
        if 'Missing' in res.columns and 'N' in res.columns:
            res = eDataFrame(res)
            vars = ['variable', 'N', 'Missing', "Missing (%)", "Mean",
                    "Std.Dev", "Median", "Min", "Max"] 
            if not groups:
                vars = vars
            else:
                vars = groups+vars
            res = (
                res
                .mutate({"Missing (%)": lambda x: x['Missing']/x['N'] })
                .select(vars+ list(res.columns))
            )
            res = res.loc[:,~res.columns.duplicated()]

        if vars_newname:
            newnames = {old:new for old, new in zip(vars_oldname, vars_newname)}
            res=res.replace({"variable":newnames}, regex=False, inplace=False)
        try:
            res=eDataFrame(res)
            res=res.mutate_type(col2type={"N":int}, from_to=None)
        except (KeyError) as e:
            pass

        if ci:
            ci_level= ci_level + (1-ci_level)/2
            z = qnorm.ppf(ci_level, loc=0, scale=1)
            res = (
                res
                .mutate_rowwise({
                    'lower': lambda col: col['Mean'] - z*col['Std.Dev'],
                    'upper': lambda col: col['Mean'] + z*col['Std.Dev'],
                })
            )
        # 
        if 'N' in res.columns:
            res = (
                res
                .mutate_type(col2type={"N":"int"}  )
            )
        return eDataFrame(res)


    def __summary__(self, vars, funs):
        res = (self
               .filter(vars)
               .apply(funs)
               .transpose()
               .reset_index(drop=False)
               .rename(columns={'index':"variable"}, inplace=False)
        )
        return res


    def __summary_group__(self, vars, funs, groups=None, wide_format=None,
                          vars_newname=None):
        funs_name=[]
        for f in funs:
            if hasattr(f, '__call__'):
                funs_name.append(f.__name__)
            else:
                funs_name.append(f)
        
        res=(self
             .filter(list(set(vars+groups)))
             .groupby(groups)
             .agg(funs)
             .reset_index(drop=False)
        )
        colnames_new = []
        for i, tup in enumerate(res.columns):
            l = list(tup)
            if l[1]:
                col = l[0]+"_"+l[1]
                colnames_new.append((l[0]+"_"+l[1]))
            else:
                colnames_new.append((l[0]))
        res.columns = colnames_new
        #
        regexp="".join([f"{v}|" for v in funs_name])
        regexp=re.sub(pattern="\|$", repl="", string=regexp)
        regexp="("+regexp+")"
        # to keep missing on the df
        regexp=re.sub(pattern="count_missing", repl='cmissing', string=regexp)
        res = (res
                .melt(id_vars=groups, value_vars=None,
                      var_name='variable', value_name='value', col_level=None)
               )
        cols = (res
                .variable
                # this transformation keep 'missing' stat on the df
                .replace({"count_missing":'cmissing'}, regex=True, inplace=False)
                .str
                .extract("^(.*)_"+regexp+"$")
                .replace({'cmissing':"count_missing"}, regex=False, inplace=False)
        )
        cols.columns = ["variable", 'stat']
        res[["variable", 'stat']] = cols[["variable", 'stat']]
        res = (res
               .filter(groups+['variable', "stat", "value"])
               .assign(stat=lambda x: [f"variable_{v}" for v in x.stat])
               .pivot_table(values='value', index=groups+['variable'], columns='stat', aggfunc="sum")
               .reset_index( drop=False))
        # wide format
        if wide_format:
            values = [f"variable_{fun}" for fun in funs_name]
            res = (res
                   .pivot_table(values=values, index='variable',
                                columns=groups, aggfunc="sum")
                   .reset_index( drop=False)
            )
            prefix = "_".join(list(res.columns.names))
            suffix = ["_".join([str(s) for s in v]) for v in res.columns]
            res.columns=[f"{prefix}_{suf}" for suf in suffix]
            res.columns= [re.sub(pattern="^stat_|", repl="", string=s) for s in res.columns]
            res.columns= [re.sub(pattern="variable__$", repl="variable", string=s) for s in res.columns]
            res.columns= [re.sub(pattern="_variable_", repl="_", string=s) for s in res.columns]
            if vars_newname:
                res=res.replace({old:new for old, new in zip(vars, vars_newname)},
                                regex=False, inplace=False)
            # 
        res = eDataFrame(res)
        col_names = ['variable_'+fun for fun in funs_name]
        for col_name, fun in zip(col_names, funs_name):
            res.rename(columns={col_name:fun}, inplace=True)
        return res


    def freq(self, vars=None, groups=None, condition_on=None,
             include_na=False, output='print'):
        '''
        Compute frequency and marginal frequence (conditional on)

        Input
        -----
           vars : a list of string with variable names to return values frequencies
           groups   : a list of strings with 
                       variable names to condition marginal frequencies on. 
                       If 'groups' are used, 'condition_on' is ignored
           condition_on (deprecated): see 'groups'

        Output
        ------
            DataFrame with frequencies

        '''
        assert vars, "Parameter 'vars' not informed."
        if not isinstance(vars, list):
            vars = [vars]
        # 
        if include_na:
            res=self.copy()
            for var in vars:
                if pd.api.types.is_categorical_dtype(res[var]):
                    res[var] = res[var].cat.add_categories("Missing")
                res=res.fillna(value={var:'Missing'}, inplace=False)
        else:
            res=self
        # 
        if condition_on and not isinstance(condition_on, list):
            condition_on = [condition_on]
        if groups and not isinstance(groups, list):
            condition_on = [groups]
        if groups and isinstance(groups, list):
            condition_on = groups
        if condition_on:
            if not all([cond_on in vars for cond_on in condition_on]):
                for cond_on in condition_on:
                    if cond_on not in vars:
                        vars.append(cond_on)

        if not condition_on:
            def compute_stdev(df):
                n = df['n_tot']
                p = df['freq']/100
                df['stdev'] = 100*np.sqrt(p*(1-p)/n)
                return eDataFrame(df)
            res=(res
                 .groupby(vars)
                 .size()
                 .reset_index(name='n', drop=False)
                 .assign(freq=lambda x: 100*x.n/x.n.sum(),
                         n_tot = lambda x: x.n.sum())
                 .groupby(vars)
                 .apply(compute_stdev)
                 .drop(['n_tot'], axis=1)
            )
        else:
            def compute_freq(df):
                df['freq'] = 100*df['n']/sum(df['n'])
                return eDataFrame(df)
            def compute_stdev(df):
                n = sum(df['n'])
                p = df['freq']/100
                df['stdev'] = 100*np.sqrt(p*(1-p)/n)
                return eDataFrame(df)
            res = (res
                   .groupby(vars)
                   .size()
                   .reset_index(name='n', drop=False)
                   .groupby(condition_on)
                   .apply(compute_freq)
                   .reset_index(drop=True)
                   .groupby(condition_on)
                   .apply(compute_stdev)
                   .reset_index(drop=True)
                   .sort_values(by=(condition_on+vars),  ascending=True)
            )
        res=eDataFrame(res)
        # confidence intervals
        res = (
            res
            .mutate({"lo": lambda x: x['freq']-1.96*x['stdev']})
            .mutate({"hi": lambda x: x['freq']+1.96*x['stdev']})
        )
        if output=='print':
            print(eDataFrame(res).to_string())
        else:
            return eDataFrame(res)


    def quantiles(self, var, nq=10, labels=None, silent=False):
        res = (
            self
            .mutate({var: lambda col: pd.qcut(col[var], q=nq, labels=labels)})
            .freq(vars=var, groups=None, include_na=False)
            .drop_cols(names=['lo', 'hi', 'stdev'])
        )
        if not silent:
            print(res.to_string(index=False))
        return eDataFrame(res)


    def ci_t(self, var, alpha=.95):
        x = self[var]
        ci = st.t.interval(loc=x.mean(), scale=st.sem(x), alpha=alpha,
                           df=len(x)-1)
        return ci
        

    def ci_norm(self, var, alpha=.95):
        x = self[var]
        ci = st.norm.interval(loc=x.mean(), scale=st.sem(x), alpha=alpha)
        return ci


    def corr_pairwise(self, vars, long_format=True, lower_tri=False):
        assert isinstance(vars, list), "'vars' need to be a list"
        res = (self
               .filter(vars)
               .corr()
        )
        if long_format:
            res.values[np.triu_indices_from(res, 0)] = np.nan
            res = res.unstack().reset_index( drop=False).dropna()
            res.columns = ['v1', 'v2', 'corr']
        elif lower_tri:
            res.values[np.triu_indices_from(res, 0)] = np.nan
        return res


    def pearsonr(self, vars, alpha=0.05):
        ''' 
        Calculate Pearson correlation along with the confidence interval 
        using scipy and numpy

        Parameters
        ----------
        x, y : iterable object such as a list or np.array
          Input for correlation calculation
        alpha : float
          Significance level. 0.05 by default
        Returns
        -------
        r : float
          Pearson's correlation coefficient
        pval : float
          The corresponding p value
        lo, hi : float
          The lower and upper bound of confidence intervals
        '''
        assert isinstance(vars, list), "'vars' must be a list"
        # assert len(vars)==2, "'vars' must be a string of size 2"
        res={'var1':[],
             'var2':[],
             'corr':[],
             'p-value':[],
             'low':[],
             'high':[],
             }
        vars1=vars[:-1]
        vars2=vars[1:]
        for var1 in vars1:
            for var2 in vars2:
                if var1!=var2:
                    res["var1"]+=[var1]
                    res["var2"]+=[var2]
                    # 
                    subset = self[[var1,var2]].dropna(subset=None, axis=0)
                    x = np.array(subset[var1])
                    y = np.array(subset[var2])
                    r, p = stats.pearsonr(x,y)
                    r_z = np.arctanh(r)
                    se = 1/np.sqrt(x.size-3)
                    z = stats.norm.ppf(1-alpha/2)
                    lo_z, hi_z = r_z-z*se, r_z+z*se
                    lo, hi = np.tanh((lo_z, hi_z))
                    # 
                    res['corr']+=[r]
                    res['p-value']+=[p]
                    res['low']+=[lo]
                    res["high"]+=[hi]
        return eDataFrame(res)
    

    def prop_test(self, var, var_value,
                  treat, treat_value, control_value,
                  group_regexp=None,
                  alpha=.05,
                  test = ['chisq', 'z', 't'],
                  tail='two'):
        '''
        Compute test statistics of difference in proportions

        Inputs:
            var  : string with the outcome variable name to compute proporitons
            var_value : string/numeric value with value of the outcome 
                        variable to compute difference between treatment
                        and control groups
            treat : string with the name of the treatment variable
            treat_value : value of the treatment in the var 'treat'
            control_value : value of the control in the var 'treat'
            group_regexp  : string for subset the data set using 'query'
                            function
            alpha : alpha-level of the test
            test : string with the test statistics to use (chisq, z, t)
            tail : string with either 'one' or 'two' 
            
        '''
        assert tail in ['one', 'two'], "'tail' must be either 'one' or 'two'"
        if group_regexp:
            tmp = self.query(group_regexp)
        else:
            tmp = self
        tmp = (tmp
               .query(f"{treat}==['{treat_value}', '{control_value}']")
               .case_when("y", {
                   f"{var}=='{var_value}'": f"1",
                   f"{var}!='{var_value}'": f"0",
               })
               .case_when('t', {
                   f"{treat}=='{treat_value}'": f"1",
                   f"{treat}=='{control_value}'": f"0",
               })
               .filter([var, "y", treat, "t"])
               .groupby([ "t", "y"])
               .size()
               .reset_index(name="n",drop=False)
        )
        y1 = float(tmp.loc[(tmp.t==1) & (tmp.y==1), 'n'])
        n1 = float(tmp.loc[(tmp.t==1) & (tmp.y==0), 'n']) + y1
        y2 = float(tmp.loc[(tmp.t==0) & (tmp.y==1), 'n'])
        n2 = float(tmp.loc[(tmp.t==0) & (tmp.y==0), 'n']) + y2
        p1 = y1/n1
        p2 = y2/n2
        sd = np.sqrt( p1*(1-p1)/n1 + p2*(1-p2)/n2 )
        ATE      = p1 - p2
        if tail=='two':
            t = dnorm.ppf(q=1-alpha/2, loc=0, scale=1)
            # t =  tdist(df).ppf()
        if tail=='one':
            t = dnorm.ppf(q=1-alpha, loc=0, scale=1)
            # t = tdist(df).ppf(1-alpha)
        # 
        ATE_low  = ATE - t*sd
        ATE_high = ATE + t*sd
        if isinstance(test, list):
            test = 'chisq'
        if test == 'chisq':
            stat_all = prop_chisqtest([y1, y2], [n1, n2])
            stat   = stat_all[0]
            pvalue = stat_all[1]
        return pd.DataFrame({"variable": [var], "variable_value": [var_value],
                             "group":group_regexp,
                             'treat_value':[treat_value],
                             'control_value':[control_value],
                             "p1":[p1], "p2": [p2],
                             "ATE":[ATE],
                             "ATE_low":[ATE_low], "ATE_high":[ATE_high],
                             "stat": [stat],
                             'pvalue':[pvalue], 'test':[test]})
    

    def tab(self, row, col, groups=None,
            margins=True, normalize='all',#row/columns
            margins_name='Total', report_format=True, digits=2):
        '''
        Create a 2x2 table

        Input
           row   string with variable name to go to rows
           col   string with variable name to go to columns
           groups string or list with variable names to use as grouping variables.
                  It will generate one 2x2 table per groups
           margins bool. It True, return the rows and column totals
           normalize Either 'all', 'row', or 'columns'. Indicate how to 
                     compute the marginal percentages in each cell
           margins_name string with the name of the row and column totals
           report_format bool. If True, return a table with cells that display
                               the percentages followed by the number of 
                               cases in each cell.
        digits integer with the number of digits to use
        '''
        vars_row = row
        vars_col = col
        tab = self.copy()
        tab[vars_row] = tab[vars_row].astype(str)
        tab[vars_col] = tab[vars_col].astype(str)
        if groups:
            groups = [groups] if isinstance(groups, str) else groups
            ngroups=len(groups)
            resn = tab.__tab_groups__(vars_row, vars_col, normalize=False,
                                       margins=margins, margins_name=margins_name,
                                       groups=groups)
            resp = tab.__tab_groups__(vars_row, vars_col, normalize,
                                       margins, margins_name, groups)
        else:
            ngroups=0
            resn = tab.__tab__(vars_row, vars_col, normalize=False,
                                margins=margins, margins_name=margins_name)
            resp = tab.__tab__(vars_row, vars_col, normalize=normalize,
                                margins=margins, margins_name=margins_name)
        colsn=resn.columns[ngroups+1:]
        colsp=resp.columns[ngroups+1:]
        res=eDataFrame(resp.iloc[:,0:ngroups+1])
        if report_format:
            for coln, colp in zip(colsn, colsp):
                col = [f"{round(100*p, digits)} % ({n})" for p,n
                       in zip(resp[colp], resn[coln])]
                res = res.mutate({coln:col})
        else:
            for coln, colp in zip(colsn, colsp):
                res[coln]=resn[coln]
                res[str(colp)+"_freq"]=100*resp[colp]
        # Group columns using varname as label
        ncat = len(tab[vars_col].unique())
        ngroups = 0 if not groups else len(groups)
        col_groups = ['']*(ngroups+1) + [vars_col]*ncat+['']
        col_ix = pd.MultiIndex.from_arrays([col_groups, res.columns])
        res.columns = col_ix
        res.columns.names = ['', '']
        res.columns.name = ''
        return eDataFrame(res)


    def __tab_groups__(self, vars_row, vars_col, normalize,
                       margins, margins_name, groups):
        res = (self.
               groupby(groups)
               .apply(eDataFrame.__tab__,
                      vars_row, vars_col, normalize, margins, margins_name)
               .reset_index(drop=False)
        )
        cols = [col for cidx, col in enumerate(list(res.columns) ) if
                not bool(re.search(pattern='^level_[0-9]$', string=col))]
        res=res.filter(cols)
        return eDataFrame(res)
        

    def tabn(self, vars_row, vars_col, normalize=False, margins=True,
             margins_name='Total'):
        res = self.__tab__(vars_row, vars_col, normalize=normalize,
                           margins=margins, margins_name=margins_name)
        return eDataFrame(res)


    def tabp(self, vars_row, vars_col, normalize="all", margins=True,
             margins_name='Total'):
        res = self.__tab__(vars_row, vars_col, normalize=normalize,
                           margins=margins, margins_name=margins_name)
        return eDataFrame(res)


    def __tab__(self, vars_row, vars_col, normalize='all', margins=True,
                margins_name='Total'):
        if normalize=='row':
            normalize='index'
        if normalize=='column' or normalize=='col':
            normalize='columns'
        res = pd.crosstab(index=[self[vars_row]],
                          columns=[self[vars_col]],
                          margins=margins, margins_name=margins_name,
                          normalize=normalize)
        res = res.reset_index(drop=False)
        return res



    def balance(self, vars, balance_on, include_summary=True,
                concise=True, estimand="ATT"):
        '''
        Compute balance of variables given the treatment

        Input 
        -----
        vars  : str, list or dict with the name of the variables to check
                balance. See pandasci.eDataFrame.select_cols

        balance_on : string with the name of the variable to compare
                     if the variables specified in 'vars' are balanced
                     in the levels of the variable specified in balance_on 
        
        estimand : 'ATT' of 'ATE'. See MatchIt help
        
        include_summary : boolean. If True, include overall summary of the 
                          variables in 'vars'
        
        concise  : boolean. If True, omit some balance statistics (e.g.
                   Var Ratio, eCDF Mean, etc.)

        Output 
        ------
        eDataFrame with balance summaries

        '''
        # Temporarily using R to compute balance
        import rpy2.robjects as robj
        from rpy2.robjects import pandas2ri, r
        from rpy2.robjects.packages import importr
        pandas2ri.activate()
        base = importr("base")
        match = importr("MatchIt")

        # 
        if isinstance(vars, str):
            vars = [vars]
        if isinstance(vars, list):
            X = [f"`{x}`" for x in vars]
            X = " + ".join(X)
            vars = vars + [balance_on]
        elif isinstance(vars, dict) and isinstance(balance_on, str):
            X = [f"`{x}`" for x in list(vars.values())]
            X = " + ".join(X)
            vars = vars | {balance_on:balance_on}
        else:
            raise TypeError("Check accepted formats for parameter 'vars'")
        # 
        tab = self.select_cols(names=vars).drop_rows(dropna=True)
        # 
        f = robj.Formula(f"`{balance_on}` ~ {X}")
        mat = match.matchit(f, data=tab,
                            distance='mahalanobis',
                            method = robj.NULL,
                            exact = robj.NULL,
                            estimand = estimand,
                            caliper = robj.NULL,
                            std_caliper = True)
        res = r.summary(mat)
        res = res.rx['sum.all']
        res = base.data_frame(res)
        # res = base.cbind(base.rownames(res), base.as_data_frame(res, row_names=robj.NULL))
        ru=rutils.rutils()
        res = ru.df2pandas(res).reset_index(drop=False)
        res = (
            res
            .rename_cols(regex={'sum.all.':'',
                                '\.\.':'.',
                                '\.*$':'',
                                '\.':' '})
            .rename_cols(columns={'index':'variable'})
        )
        if include_summary:
            res = self.summary(vars=vars).join(res, how='left')
        if concise:
            res= res.drop_cols(names=['Missing', 'Var Ratio', 'eCDF Mean',
                                      'eCDF Max', 'Std Pair Dist'])
        return res



# ** Utilities
    def names(self, regexp=None, print_long=False):
        names = list(self)
        if regexp:
            names = [nm for nm in names if
                     bool(re.search(pattern=regexp, string=nm))]
        if print_long:
            for col in names:
                print(f"\"{col}\",")
        if not names:
            print("\nNo column name matches the regexp!\n")
        return names
            

    def to_org(self, round=4):
        res = self
        if round:
            res = res.round(round)
        s = res.to_csv(sep='|', index=False).replace('\n', ' | \n | ')
        s = re.sub(pattern="^", repl="|", string=s)
        s = re.sub(pattern=".$", repl="", string=s)
        print(s)

    
    def flatten_columns(self, sep='_'):
        'Flatten a hierarchical index'
        assert isinstance(self.columns, pd.MultiIndex), "Not a multiIndex"
        self = (self.reset_index(drop=True))
        def _remove_empty(column_name):
            return tuple(element for element in column_name if element)
        def _join(column_name):
            column_name = [str(col) for col in column_name]
            return sep.join(column_name)
        new_columns = [_join(_remove_empty(column)) for column in
                       self.columns.values]
        self.columns = new_columns
        return self


    def flatten(self, drop=True):
        ngroups=len(self.index.codes)
        idx = self.index.codes[ngroups-1]
        res=self.reset_index(drop=drop).set_index(idx).sort_index()
        return res


    def tolatex(self, fn=None, na_rep="", table_env=True,
                align=None,
                add_hline=None,
                add_blank_row=None,
                add_row_group=None,
                add_row_group_hspace=0.5,
                # 
                add_col_group=None,
                # 
                escape=True,
                index=False,
                digits=2,
                # 
                float_format=None,
                rotate=False,
                position='th!',
                longtable=None,
                encoding=None,
                decimal='.',
                # 
                caption=None,
                label=None,
                **kws):
        '''
        Extension of to_latex pandas' function
        
        Input
           fn : path to save the .tex table
           table_env : boolean, if True uses latex table environment. Otherwise
                       uses only tabular
           align : string with align features (e.g., llp{7cm})
           add_hline  : a list with the line number to draw a horizontal line 
                        in the table
           add_blank_row : a list with the line number to add a blank in 
                           the table
           add_row_group : a dict with the line number (key) and the text (value) 
                           to add on that line as a multiline text. 
                           Useful to group rows.
           add_row_group_hspace : size of the horizontal space (in em)
                                   to add to the rows in the groups
           add_col_group : a list of dictinaries. Each dictionary must have 
                           the column labels to add (keys) and tuples of
                           integers indicating "(initial column, end column, <l/r/c alignment>)."
                           It will add a multicolumn text at the top of the table, 
                           above the heading. 
                           If there is more than one dictionary in the list, 
                           the multicolumn lines are added in the order they 
                           appear in the list. Useful to group columns.
           rotate : boolean. If True, rotate the table 90 degrees. Default False
        '''
        pdcolw = pd.get_option('display.max_colwidth')
        pd.set_option('display.max_colwidth', None)
        # 
        # self = self.reset_index(drop=True)
        self.index = range(1, self.nrow+1)
        if add_hline:
            add_hline = [l+.5 for l in add_hline]
            d = self.iloc[0].to_dict()
            d = {k:np.nan  for k in d.keys()}
            d[self.columns[0]] = '\midrule'
            d = eDataFrame(d, index=add_hline)
            self = (
                self
                .append(d, ignore_index=False)
                .sort_index()
            )
            escape=False
        if add_blank_row:
            add_blank_row = [l+.6 for l in add_blank_row]
            d = self.iloc[0].to_dict()
            d = {k:np.nan  for k in d.keys()}
            d = eDataFrame(d, index=add_blank_row)
            self = (
                self
                .append(d, ignore_index=False)
                .sort_index()
            )
            escape=False
        if add_row_group:
            dres = eDataFrame()
            ncol = self.ncol
            idx = []
            # 
            # add horizontal space in the groups
            first_col = self.names(".")[0]
            self=(
                self
                .mutate({first_col: lambda col:
                         [f"\hspace{{{add_row_group_hspace}em}}"+str(s)
                          for s in col[first_col]]})
            )
            # 
            for row, text in add_row_group.items():
                row_idx = row-.1
                idx += [row_idx]
                d = self.iloc[0].to_dict()
                d = {k:np.nan  for k in d.keys()}
                d[self.columns[0]] = f"\multicolumn{{{ncol}}}{{l}}{{\textbf{{{text}}}}}"
                d = eDataFrame(d, index=[row_idx])
                dres = dres.bind_row(d)
            dres.index = idx
            self = (
                self
                .append(dres, ignore_index=False)
                .sort_index()
            )
            # must be false of the row commands get messed up
            escape=False
            # escape all % in columns or rows
            self = (
                self
                .replace({"%":"\%"} , regex=True, inplace=False)
                .rename_cols(regex={"%":"\%"}, tolower=False)
            )
        tab = (
            self
            .round(digits)
            .to_latex(na_rep=na_rep, index=index, escape=escape,
                      caption=caption, label=label, position=position, **kws)
        )
        if align:
            tab = tab.replace('begin\{tabular\}\{.*\}',
                              f'{{begin{{tabular}}{align}}}')
            tab = re.sub(pattern="begin{tabular}\{.*\}",
                         repl=f'begin{{tabular}}{{{align}}}',
                         string=tab)
        if add_hline:
            tab = re.sub(pattern='\\\\midrule.*', repl='\\\\midrule', string=tab)
        if add_row_group:
            tab = tab.split("\\\\\n")
            tab = [re.sub(pattern='&', repl="", string=txti) if
                   bool(re.search(pattern='\\\\multicolumn', string=txti)) else
                   txti for txti in tab ]
            tab = "\\\\\n".join(tab)
        # 
        # 
        if not table_env and not caption and not label:
            tab = tab.replace('\\begin{table}','')
            tab = tab.replace('\\end{table}','')
        if table_env and not caption and not label and not \
           bool(re.search(pattern='\\\\begin\{table\}', string=tab)):
            tab = "\\begin{table}\n" + tab + "\\end{table}"
        # 
        # 
        if rotate:
            tab = tab.replace("\\begin{table}","")
            tab = tab.replace("\\end{table}","")
            tab = "\\begin{sidewaystable}" + tab + "\\end{sidewaystable}"
        # --------------------------
        if add_col_group:
            assert isinstance(add_col_group, list), ("'add_col_group' must be "+
                                                     "a list of dictionaries.")
            # getting text to add
            # add_col_group.reverse()
            txt=''
            for d in add_col_group:
                col_pos=1
                midrule=""
                for col_label, pos in d.items():
                    assert len(pos)==3, ("Dictionary values of add_col_group "+\
                                         "must be a tuple with three values: "+\
                                         "(initial column, end column, <l/r/c alignment>"
                                         )
                    for current_pos in range(col_pos, pos[0]+1):
                        if current_pos<pos[0]:
                            txt+=" & "
                        else:
                            txt+=(f" \\multicolumn{{{pos[1]-pos[0]+1}}}"+
                                  f"{{{pos[2]}}}{{{col_label}}} ")
                        col_pos+=1
                    col_pos+=1
                    midrule+=f'\\cmidrule(lr){{{pos[0]}-{pos[1]}}}'
                txt+=f'\\\\\n {midrule} \n'
            # adding text to the table
            tab=tab.replace('\\toprule\n', f'\\toprule\n {txt}')
        # --------------------------
        # saving
        if fn:
            with pd.option_context('display.max_colwidth', None):
                with open(fn, "w+") as f:
                    f.write(tab)
        pd.set_option('display.max_colwidth', pdcolw)
        return tab

    def print(self, digits=2, *args, **kwargs):
        kwargs['index']=False
        format=f'{{:0.{digits}f}}'
        with pd.option_context('display.float_format', format.format):
            print(self.to_string(*args, **kwargs), flush=True)
        
    def t(self):
        res = self.transpose().reset_index(drop=False)
        res.columns = res.iloc[0].values
        res = res.iloc[1:,].reset_index(drop=True)
        return eDataFrame(res)


    def toexcel(self, fn=None, sheet_name=None, index=False, **kws):
        '''
        Save eDataFrame to excel

        Input
    	-----
           sheet_name  : string (default None) with the name of the sheet
                         to save the content. If None, use the first
                         sheet and overwrite the file
           index  : boolean, if False, do not save the row index in a column
           **kws  : see pd.to_excel 

        '''
        fn = os.path.expanduser(fn)
        if not sheet_name:
            self.to_excel(excel_writer=fn, index=index, **kws)
        else:
            # with pd.ExcelWriter(path=fn, engine='xlsxwriter') as writer:
            #     self.to_excel(writer,
            #                   sheet_name=sheet_name,
            #                   index=index,
            #                   **kws)
            #     # writer.save()
            excel_book = pxl.load_workbook(fn)
            with pd.ExcelWriter(fn, engine='openpyxl') as writer:
                writer.book = excel_book
                writer.sheets = {worksheet.title: worksheet for worksheet in
                                 excel_book.worksheets}
                self.to_excel(writer,
                              sheet_name=sheet_name,
                              index=index,
                              **kws)
                writer.save()


# ** Utilities (ancillary)
    def __wrap_var__(self, var, wrap=None, wrap_char=None):
        # 
        wascat=False
        if wrap:
            if self[var].values.dtype== 'category':
                wascat=True
                ordered=self[var].cat.ordered
                cats = self[var].cat.categories.values
                cats = [textwrap.fill(cat, wrap)  for cat in cats]
            self=self.mutate({var: lambda x: x[var].str.wrap(wrap)})
            if wrap_char:
                self=self.replace({var:{"\\n":wrap_char}} , regex=True)
            if wascat:
                if wrap_char:
                    cats = [re.sub(pattern='\\n', repl=wrap_char, string=c) for
                            c in cats]
                self=(
                   self
                   .mutate({var: lambda x: pd.Categorical(x[var],
                                                          categories=cats,
                                                          ordered=ordered
                                                          )})) 
        return self



# ** Plots
# *** Main

    def plot(self, *args, **kws):
        tab=pd.DataFrame(self.copy())
        return tab.plot(*args, **kws)
        
    
# *** Scatter plot
    def plot_scatter(self, x, y, **kwargs):
        ax, axl = self.plot_line(x, y, kind='scatter', pts_show=True, **kwargs)
        plt.tight_layout()
        return ax, axl
        
# *** Line Plot
    def plot_line(self, x, y,
                  type='line',
                  # facets
                  facet_x=None, facet_x_dist=3,
                  facet_y=None, facet_y_dist=1,
                  facet_title_loc='left',
                  facet_title_sep=',',
                  facet_fontsize=15,
                  facet_ncol=None,
                  facet_sharex=True,
                  facet_sharey=True,
                  # title 
                  # -----
                  title=None,
                  subtitle=None,
                  title_ycoord=1.1,
                  title_xcoord=-.04,
                  title_yoffset=.05,
                  title_size=15,
                  subtitle_size=10,
                  title_alpha=.6,
                  subtitle_alpha=.6,
                  # legend
                  leg_pos='top',
                  leg_title=None,
                  leg_fontsize=10,
                  leg_xpos=None,
                  leg_ypos=None,
                  leg_ncol=3,
                  figsize=[10, 6], tight_layout=True,
                  **kwargs
                  ):
        kind = kwargs.get('kind', 'line')
        # line style
        linetype = kwargs.get('linetype', None)
        colors = kwargs.get('colors', None)
        # markers
        pts_show = kwargs.get("pts_show", False)
        pts_style=kwargs.get('pts_style', False)
        grid = kwargs.get('grid', True)
        size = kwargs.get('size', 20)
        if not linetype and colors and pts_show and pts_style:
            linetype=colors
            pts_show= pts_style
            pts_style=None
            print(f"\nNote: using 'colors' and 'pty_style' together "+\
                  "forces the plot to use different linetypes\n")
        else:
            if pts_show and not linetype and not pts_style:
                pts_style=kwargs.get('pts_style', 'o')
            elif pts_show and not linetype and pts_style:
                pts_show=pts_style
            elif pts_show and linetype and pts_style:
                pts_show= pts_style
                pts_style=None
            elif pts_show and not linetype and not pts_style:
                pts_show=True
                pts_style=None
            elif pts_show and linetype and not pts_style:
                pts_show=True
                pts_style=None
            else:
                pts_show=None
                pts_style=None
        # others
        palette = kwargs.get('palette', None)

        ax = sns.relplot(x=x, y=y, 
                         kind=kind,
                         # linestype
                         hue=colors,
                         style=linetype,
                         # markers
                         markers=pts_show,
                         marker=pts_style,
                         # facet
                         row=facet_y,
                         col=facet_x,
                         col_wrap=facet_ncol,
                         facet_kws={'sharey': facet_sharey,
                                    'sharex': facet_sharey},
                         palette=palette,
                         data=self,
                         )
        # Legend
        # ------
        if colors or linetype:
            # if leg_pos=='left':
            #     leg = ax._legend
            #     leg.set_bbox_to_anchor([1,1])
            # else:
            if leg_pos=='top':
                leg_pos='upper left'
                if not leg_ypos:
                    leg_ypos = 1.1
                if not leg_xpos:
                    leg_xpos = 0
            ax.legend.remove()
            # this line is giving error when using facet_y, colors !!!
            axc=ax.axes[0][0]
            leg = axc.legend(loc=leg_pos, bbox_to_anchor=(leg_xpos, leg_ypos),
                             handlelength=1.5,
                             title=leg_title,
                             handletextpad=.3, prop={'size':leg_fontsize},
                             # pad between the legend handle and text
                             labelspacing=.2, # vert sp between the leg ent.
                             columnspacing=1, # spacing between columns
                             ncol=leg_ncol, mode=None, frameon=True, fancybox=True,
                             framealpha=0.5, facecolor='white')
            leg._legend_box.align = "left"
        
        # Facet titles
        # ------------
        if facet_x or facet_y:
            if facet_title_loc!='center':
                ax.set_titles("", loc='center')
            if facet_x and facet_y:
                facet_titles="{col_name}"+f"{facet_title_sep} "+"{row_name}"
            if facet_x and not facet_y:
                facet_titles="{col_name}"
            if not facet_x and facet_y:
                facet_titles="{row_name}"
            ax.set_titles(facet_titles, loc=facet_title_loc,
                          size=facet_fontsize)
        elif title or subtitle:
            axc=ax.axes[0][0]
            xcoord=title_xcoord
            ycoord=title_ycoord if subtitle else title_ycoord -.05
            yoffset=title_yoffset if subtitle else 0
            axc.annotate(title, xy=(xcoord, ycoord),
                        xytext=(xcoord, ycoord),
                        xycoords='axes fraction', size=title_size,
                        alpha=title_alpha)
            axc.annotate(subtitle, xy=(xcoord, ycoord-yoffset),
                        xytext=(xcoord, ycoord-yoffset),
                        xycoords='axes fraction', size=subtitle_size,
                        alpha=subtitle_alpha)

        for axc in it.chain(*ax.axes):
            if grid:
                self.__plot_grid__(ax=axc, **kwargs)
            # self.__plot_border__(axs=axc)



        # adjustments 
        # -----------
        ax.fig.set_size_inches(figsize[0], figsize[1])
        ax.tight_layout(w_pad=facet_y_dist, h_pad=facet_y_dist)


        # Message 
        # -------
        print(f"\nFor easy manipulation, this method returns two elements: \n"+\
              "1. The axis of the entire figure\n"+\
              "2. The axes for each facet in a list\n\n")
        plt.tight_layout()
        # return ax, it.chain(*ax.axes)
        return ax, ax.axes.flatten()


# *** Polar plot
    def plot_polar(self, vars, group=None, func='mean', facet=None, **kws):
        '''
        Plot a summary value of variables using polar coordinate
        
        Input:
           vars  a list or a dictionary of lists of variables in the DataFrame. 
                 if a dictionary is used, the list of variables under each 
                 key is plotted together, a shaded area is added to the plot
                 based on the group, and the key is used as label for the group.
           group a string with the name of the variable in the DataFrame to 
                 use as group of the lines in the plot. Use 'group_linestyles'
                 to define line styles for each group.
                 Ex: group_linestyles=['-', '--']
           facet a string with the name of the variable in the DataFrame to
                 use as facet
           func a string with the function to compute the summary of the 
                variables
           kws:
             - labels
                - labels      : a list of variables in the same order of 'vars'.
                                The labels are used instead of the var names in the plot
                - labels_wrap : size to wrap the labels
                - labels_size : float, size of the labels
                - labels_dist : float, distance between the labels and the plot
             - legend keywords:
               - legend_title      : string
               - legend_title_size : inteter
               - legend_show       : boolean
               - legend_frame      : boolean, to use a frame in the legend
               - legend_items_size : size of the font of the items
               - legend_ncol       : integer, number of columns in the legend
             - threshold keywords:
               - thresholds          : integers in a list, value to draw circles 
                                       to represent thresholds
               - thresholds_color    : list of colors of the thresholds
               - thresholds_size     : list of sizes of the lines
               - thresholds_linetype : list of linetypes
             - facet
               - ycoord              :kws.get('facet_title_ypos', 1.05)
               - facet_title_size    :kws.get('facet_title_size', 15)
             - Shaded regions
               - shade_colors  : a list of colors 
               - shade_alpha   : a list of alpha values
               - shade_ymin    : float, the min value for the shaded area
               - shade_ymax    : float, the max value for the shaded area
               - grid_which    : string, which grid to draw (minor, major, both)
               - grid_axis     : string, axis to draw (x, y, both)
               - grid_linetype : grid linetype
               - grid_alpha    : float between 0 and 1, grid line transparency
              Output
           A polar plot. It return the an axis object, or a list of axes if
           facets are used
        '''
        labels=kws.get('labels', None)
        legend_show=kws.get('legend_show', True)
        # table
        # -----
        tab, radian_pos = self.__plot_polar__get_tab_main__(vars, group,
                                                            func, facet)
        # Plot
        # ----
        # No Groups
        if not group and not facet:
            fig, ax=self.__create_figure__(nrows=1, ncols=1, polar=True, **kws)
            self.__plot_polar_simple__(tab, func, ax, **kws)
        if not group and facet:
            pass
        # Groups
        if group and not facet:
            fig, ax=self.__create_figure__(nrows=1, ncols=1, polar=True, **kws)
            self.__plot_polar_groups__(tab=tab, func=func, ax=ax,
                                       group=group, **kws)
        if group and facet:
            nplots=len(tab[facet].unique())
            ncols = kws.get('facet_ncols', nplots)
            nrows = kws.get('facet_nrows', 1)
            fig, ax = self.__create_figure__(nrows=nrows, ncols=ncols,
                                             polar=True, **kws)
            if ncols==1 or nrows==1:
                ax=list(it.chain(ax))
            else:
                ax=list(it.chain(*ax))
            self.__plot_polar_facets__(tab, group, func, ax, facet, **kws)
        # Plot aesthetic elements (grid, labels, legend, etc)
        if not facet:
            ax = [ax]
        for axc in ax:
            self.__plot_polar_add_thresholds__(axc, **kws)
            self.__plot_polar_addlabels__(axc, tab=tab, vars=vars,
                                          radian_pos=radian_pos, **kws)
            if 'variable_group' in tab.columns:
                self.__plot_polar_vars_group__(ax=axc, tab=tab,
                                               func=func, **kws)
            self.__plot_grid__(ax=axc, **kws)
            self.__plot_yticks__(ax=axc, **kws)
        if legend_show:
            self.__plot_polar_add_legend__(ax[0], **kws)
        return ax
        


    def __plot_polar_add_legend__(self, ax, **kws):
        legend_title=kws.get("legend_title", '')
        legend_frame=kws.get("legend_frame", True)
        legend_items_size=kws.get('legend_items_size', 10)
        legend_title_size=kws.get('legend_title_size', 12)
        legend_ncol=kws.get('legend_ncol', 1)
        # 
        leg = ax.legend(loc='lower left',
                        bbox_to_anchor=kws.get('legend_pos', (0, 1.01)),
                        handlelength=2,
                        title=legend_title,
                        handletextpad=.3, prop={'size':legend_items_size},
                        # pad between the legend handle and text
                        labelspacing=.2, #vertical space between the leg entries
                        columnspacing=1, # spacing between columns
                        # handlelength=1, #  length of the legend handles
                        ncol=legend_ncol, frameon=legend_frame, fancybox=True,
                        framealpha=0.5, facecolor='white')
        leg._legend_box.align = "left"
        leg.get_title().set_fontsize(legend_title_size) #legend 'Title' fontsize

    def __plot_polar_add_thresholds__(self, ax, **kws):
        thresholds=kws.get('thresholds', False)
        thresholds_color=kws.get('thresholds_color', ['red'])
        thresholds_size=kws.get('thresholds_size', [2])
        thresholds_linetype=kws.get('thresholds_linetype', ['--'])
        rads = np.arange(0, (2*np.pi), 0.1)
        if thresholds is not False:
            assert isinstance(thresholds, list), "'thresholds' must be a list!"
            for trh,trhc,trhs,trhl in zip(thresholds,
                                          it.cycle(thresholds_color),
                                          it.cycle(thresholds_size),
                                          it.cycle(thresholds_linetype)):
                ax.plot(rads,[trh]*len(rads),
                        color=trhc,
                        linestyle=trhl,
                        linewidth=trhs,
                        zorder=0,
                        alpha=.4) 
        

    def __plot_polar_addlabels__(self, ax, **kws):
        tab=kws.get('tab')
        labels=kws.get('labels', None)
        if not labels:
            labels=(tab
                    .filter(['variable', 'pos'])
                    .drop_duplicates()
                    .sort_values(['pos'], ascending=True)
                    ['variable']
                    .tolist())
        labels_wrap = kws.get('labels_wrap', None)
        labels_size=kws.get('labels_size', 8)
        labels_dist=kws.get('labels_dist', 20)
        radian_pos=kws.get('radian_pos')
        if labels_wrap:
            labels=["\n".join(textwrap.wrap(label, labels_wrap)) 
                    for label in labels]
        vars  = kws.get('vars')
        # this line is to avoid warning about tick location
        ax.xaxis.set_major_locator(mticker.FixedLocator(radian_pos))
        # ----
        ax.set_xticklabels(labels=labels, color='black', fontsize=labels_size)
        ax.set_xticks(ticks=radian_pos, minor=[])
        ax.tick_params(axis='x', pad=labels_dist)



    def __plot_polar_facets__(self, tab, group, func, ax, facet, **kws):
        for fct, axc in zip(np.sort(tab[facet].unique()),ax):
            tabt=tab.query(f"{facet}=={fct}")
            self.__plot_polar_groups__(tabt, group, func, axc, **kws)
            # -----
            # Title
            # -----
            plt.subplots_adjust(top=.78)
            xcoord=-.07
            ycoord=kws.get('facet_title_ypos', 1.05)
            yoffset=.07
            facet_title_size=kws.get('facet_title_size', 15)
            axc.annotate(f"{facet}: {fct}",
                         xy=(xcoord, ycoord),
                         xytext=(xcoord, ycoord),
                         xycoords='axes fraction',
                         size=facet_title_size, alpha=.6)


    def __plot_polar_groups__(self, tab, group, func, ax, **kws):
        linestyles=kws.get('group_linestyles', '-')
        groups=np.sort(tab[group].unique())
        for gr, ls in zip(groups, it.cycle(linestyles)):
            kws['linestyle']=ls
            tab[group] = tab[group].astype(str)
            tabt=tab.query(f"{group}=='{gr}'")
            self.__plot_polar_simple__(tabt, func, ax, label=gr, **kws)


    def __plot_polar_simple__(self, tab, func, ax, **kws):
        label=kws.get('label', None)
        color=kws.get("color", 'black')
        ls=kws.get('linestyle', '-')
        linesize=kws.get('linesize', 1)
        alpha=kws.get('alpha', 1)
        ylim=kws.get('ylim', None)
        x, y = list(tab['pos']), list(tab[func])
        x.append(x[0]); y.append(y[0])# to close the circle
        ax.plot(x, y, color=color,
                linewidth=linesize,
                linestyle=ls, alpha=alpha,
                label=label
                )
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        return ax


    def __plot_polar__get_tab_main__(self, vars, group, func, facet):
        if facet:
            tab=eDataFrame()
            for fct in self[facet].unique():
                tabt, radian_pos = (self
                                    .query(f"{facet}=='{fct}'")
                                    .__plot_polar__get_tab__(vars, group, func,
                                                             {'var':facet,
                                                              'value':fct})
                        )
                tab=tab.append(tabt)
        else:
            tab, radian_pos  = self.__plot_polar__get_tab__(vars=vars,
                                                            group=group,
                                                            func=func)
        return tab, radian_pos

    def __plot_polar__get_tab__(self, vars, group, func, facet=None):
        vars_group=[]
        vars_list=[]
        vars_group_flag=False
        if isinstance(vars, str):
            vars = [vars]
        if isinstance(vars, dict):
            for key, values in vars.items():
                for value in values:
                    vars_group.append(key)
                    vars_list.append(value)
            vars=vars_list
            vars_group=pd.DataFrame(dict(variable=vars,
                                         variable_group=vars_group))
            vars_group_flag=True
        if group:
            group=[group]
            vars_to_select=vars+group
        else:
            vars_to_select=vars
        # 
        radian_interval = 2*pi/(len(vars))
        radian_pos = np.arange(0, 2*pi, radian_interval)
        tab = (self
               .filter(vars_to_select)
               .summary(vars, [func], group)
               .merge(pd.DataFrame(zip(vars, radian_pos),
                                   columns=['variable', 'pos']),
                      how='left', on=['variable'])
               .sort_values(['pos'], ascending=True)
               )
        # 
        if vars_group_flag:
            tab=tab.merge(vars_group, how='left', on=['variable'])
        if facet:
            tab[facet['var']]=facet['value']
        return tab, radian_pos 


    def __plot_polar_vars_group__(self, ax, tab, func, **kws):
        groups=tab['variable_group'].unique()
        shade_colors=kws.get('shade_colors', ['grey'])
        shade_alpha=kws.get('shade_alpha', False)
        if not shade_alpha:
            shade_alpha = np.linspace(1, .3, len(groups))
        for group, color, alpha in zip(groups,
                                       it.cycle(shade_colors),
                                       it.cycle(shade_alpha)
                                       ):
            tabt = (tab
                    .filter(["variable_group",'pos', func])
                    .query(f"variable_group=='{group}'")
                    )
            ymin=kws.get('shade_ymin', min(tab[func]))
            ymax=kws.get('shade_ymax', max(tab[func]))
            xmin=min(tabt["pos"])
            xmax=max(tabt["pos"])
            ax.fill_between(x=np.linspace(xmin, xmax, 100),
                            y1=ymin, y2=ymax,
                            color=color,
                            alpha=alpha,
                            zorder=0,
                            label=group)

# *** Histogram
    def plot_hist(self, var, groups=None, discrete=False,
                  xtickslabel_wrap=None, 
                  xtickslabel_fontsize=None,
                  ytickslabel_wrap=None, 
                  ytickslabel_fontsize=None,
                  facet=None, ax=None, **kws):
        '''
        Plot a histogram 
        
        Input:
           vars  a list of variables in the DataFrame. 
           group a string with the name of the variable in the DataFrame to 
                 use as group of the lines in the plot
           facet a string with the name of the variable in the DataFrame to
                 use as facet
           ax    a matplotlib subplot object to plot. If none, the function
                 creates a fugure and the subplots
           multiple used when groups are used. See sns.histplot
           # kws:
           #   - legend keywords:
           #     - legend_title      : string
           #     - legend_title_size : inteter
           #     - legend_show       : boolean
           #     - legend_frame      : boolean, to use a frame in the legend
           #     - legend_items_size : size of the font of the items
           #     - legend_ncol       : integer, number of columns in the legend
           #   - facet
           #     - ycoord              :kws.get('facet_title_ypos', 1.05)
           #     - facet_title_size    :kws.get('facet_title_size', 15)
           #    Output
           # A histogram plot. It return the an axis object, or a list of axes if
           # facets are used
        '''
        bin_labels=kws.get('bin_labels', True)
        if xtickslabel_wrap:
            self=self.__wrap_var__(var, wrap=xtickslabel_wrap)
        if not facet:
            if not ax:
                fig, ax = self.__create_figure__(nrow=1, ncol=1, **kws)
            self.__plot_hist_main__(ax, var, discrete=discrete, groups=groups,
                                    **kws)
        if bin_labels:
            self.__plot_hist_bin_labels__(ax, **kws)
        grid_which=kws.get("grid_which", 'major')
        grid_axis = kws.get("grid_axis", 'y')
        grid_linetype = kws.get("grid_linetype", '-')
        grid_alpha = kws.get("grid_alpha", .3)
        self.__plot_grid__(ax,
                           grid_which=grid_which, grid_axis= grid_axis,
                           grid_linetype= grid_linetype,
                           grid_alpha= grid_alpha)
        self.__plot_yticks__(ax)
        self.__plot_border__([ax])
        if xtickslabel_fontsize:
            ax.xaxis.set_tick_params(labelsize=xtickslabel_fontsize)
        return ax
            
    
        
# *** Density plot 
    # ------------
    # def plot_density(self, var, ax=None, **kws):
    #     if not ax:
    #         fig, ax = self.__create_figure__(nrow=1, ncol=1, **kws)
    #     sns.kdeplot(data=self, x=var, ax=ax,
    #                 fill=True, common_norm=False, palette="crest",
    #                 alpha=.5, linewidth=0, **kws)
    #     # grid
    #     ax.grid(b=None, which='major', axis='both', linestyle='-', alpha=.3)
    #     ax.set_axisbelow(True) # to put the grid below the plot
    #     # -------
    #     # Splines (axes lines)
    #     # -------
    #     ax.spines['bottom'].set_visible(True)
    #     ax.spines['left'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     # ax.spines['bottom'].set_position(('data', 0))
    #     # ax.spines['bottom'].set_linestyle('-')
    #     # ax.spines['bottom'].set_linewidth(3)
    #     # ax.spines['bottom'].set_smart_bounds(True)
    #     return ax
    def plot_density(self, var=None, group=None,
                     facet={'col':None, 'row':None},
                     cmap='dark',
                     leg_title=None, leg_title_size=13, leg_ncol=None,
                     # 
                     rug=True, info=True, info_reduced=True,
                     fill_range=None,
                     grid=True,
                     # 
                     title_size=13,
                     # 
                     height=5, width=2.5,
                     kws_kde=None, kws_fill=None,
                     kws_grid=None,
                     kws=None
                     ):
        '''
        Density plot

        Input
        -----
           var    : string with column to plot
           group  : string with the column with groups to plot separate densities
           facet  : dict {"col":<col name>, "row":<row names>}. One or both
                    can be omitted. Ex: it accepts {'col':<col name>}

           cmap : color map (see seaborn color map)

           leg_title      : string, title of the legend
           leg_title_size : integer, size of the title
           leg_ncol       : integer, number of columns of the legend

           rug    : bool, to plot or not the rug
           info   : bool, to plot or not text info
           height : height of the plot
           width  : width of the plot

           kws_kde : dict, kde kws (see seaborn kde)
           kws_fill : dict, infor to fill the plot (color)

        Output
        ------
           Returns a seaborn facet object with axes
        '''

        assert var,"You must inform the 'var'."

        aspect=height/width
        # 
        kws_kde=({"alpha": 0.3, "linewidth": 0, "bw_adjust": 3,
                  "fill":True} if not kws_kde else kws_kde)
        #80d4ff
        kws_fill = {
            "color": "grey",
            "alpha": 0.5} if not kws_fill else kws_fill
        # 
        kws_grid = {"grid_axis":'both', 'grid_linetype':':',
                    "grid_alpha":.5} if not kws_grid else kws_grid
        facet=self.__get_facet__(facet)
        #
        axs=sns.displot(
            x=var,
            data=self,
            hue=group,
            kind="kde",
            col=facet['col'],
            row=facet['row'],
            rug=rug,
            height=height,
            aspect=aspect,
            palette=cmap,
            **kws_kde|kws_fill,
            # legend=False,
            # rug_kws=rug_kws,
        )
        ## ------
        ## legend
        ## ------
        if group:
            leg_ncol=len(self[group].unique()) if not leg_ncol else leg_ncol
            # xpos=0.075 - 0.02*(nfacets_col-1)
            # ypos=.83   + 0.038*(nfacets_row-1)
            # xpos=1
            # ypos=.7
            sns.move_legend(axs, #loc='lower left',#bbox_to_anchor=(xpos, ypos),
                            loc='upper right',
                            handlelength=2,
                            title=leg_title,
                            title_fontsize=leg_title_size,
                            handletextpad=.3, prop={'size':12},
                            labelspacing=.2, #  vertical space between the legend entries.
                            columnspacing=1, # spacing between columns
                            ncol=leg_ncol, mode=None, frameon=False, fancybox=True,
                            framealpha=0.5, facecolor='white'
                            )
            leg=axs.legend
            leg._legend_box.align = "left"
        axs.data=self.__plot_get_data_facet__(axs=axs, facet=facet)
        axsf=axs.axes.flatten()
        # ----
        # Grid 
        # ----
        if grid:
            for axc in axsf:
                if not kws_grid:
                    kws_grid={"grid_alpha":.2}
                self.__plot_grid__(ax=axc, **kws_grid)
        ## ----
        ## Info
        ## ---- 
        if info:
            self.__plot_density_info__(axs=axs, var=var, group=group,
                                       cmap=cmap,
                                       info_reduced=info_reduced,
                                       leg_title=leg_title,
                                       leg_title_size=leg_title_size,
                                       leg_ncol=leg_ncol)
        ## -----
        ## title
        ## ----- 
        axs.set_titles('',loc='center')
        axs.set_titles(row_template = '{row_name}',
                      col_template = '{col_name}',
                      loc='left',
                      weight='bold',
                      size=title_size,
                      )
        self.__plot_border__(axsf)
        plt.tight_layout()
        return  axs

    def __plot_density_info__(self, axs=None,
                              var=None,
                              group=None,
                              cmap=None,
                              info_reduced=None,
                              leg_title=None,
                              leg_title_size=13,
                              leg_ncol=None,
                              ):
        nfacets=len(axs.data.items())
        nfacets_row, nfacets_col=axs.axes.shape
        # 
        group_flag=True if group else False
        for facet, info in axs.data.items():
            if not group_flag:
                group=None
            tab=info['data']
            axc=info['ax']
            # 
            if var=='group':
                tab=tab.rename_cols(columns={var:'var'})
                var='var'
            if not group_flag:
                tab=tab.mutate({"group": var})
                group='group'
            for i, gi in enumerate(tab[group].unique()):
                tab=tab.mutate({group: lambda x: x[group].astype(str)})
                xi, n = tab.__density_get_group_data__(var,
                                                       group_var=group,
                                                       group_value=gi)
                n=self.nrow/nfacets
                prop_gi = len(xi)/n/nfacets
                scale_density=prop_gi
                # 
                mean = float(np.mean(xi))
                median=float(np.median(xi))
                # print(group, flush=True)
                # print(gi, flush=True)
                # print(xi, flush=True)
                # print("ok", flush=True)
                # print(np.min(xi), flush=True)
                min = float(np.min(xi))
                max = float(np.max(xi))
                std = float(sp.stats.tstd(xi))
                sk=float(sp.stats.skew(xi))
                kt=float(sp.stats.kurtosis(xi))
                # 
                label=f"Mean ($\mu$)" if i==0 else None
                ymax=sp.stats.gaussian_kde(xi).pdf(mean)*scale_density
                axc.vlines(
                    x=mean,
                    ymin=0,
                    ymax=ymax,
                    ls="solid",
                    color=sns.color_palette(cmap)[i],
                    alpha=1,
                    lw=1.5,
                    label=label,
                )
                if not info_reduced:
                    label=f"Median" if i==0 else None
                    ymax=sp.stats.gaussian_kde(xi).pdf(median)*scale_density
                    axc.vlines(
                        x=median,
                        ymin=0,
                        ymax=ymax,
                        ls="--",
                        color=sns.color_palette(cmap)[i],
                        alpha=1,
                        lw=1.5,
                        label=label,
                    )
                    ymax=[sp.stats.gaussian_kde(xi).pdf(mean-std)*scale_density,
                          sp.stats.gaussian_kde(xi).pdf(mean+std)*scale_density]
                    axc.vlines(
                        x=[mean - std, mean + std],
                        ymin=0,
                        ymax=ymax,
                        ls=":",
                        color=sns.color_palette(cmap)[i],
                        label="\u03BC \u00B1 \u03C3" if i==0 else None,
                    )
                # ----------------------
                # Annotations and legend
                # ----------------------
                xpos= 0.017
                ypos= 0.95#-.1*(leg_ncol-1)
                xpos_col_space=8
                xpos_col_space_additional=1
                axc.text(
                    xpos*(xpos_col_space*i+xpos_col_space_additional),
                    ypos,
                    f"Mean: {mean:.2f}",
                    # fontdict=font_kws,
                    color=sns.color_palette(cmap)[i],
                    transform=axc.transAxes,
                )
                axc.text(
                    xpos*(xpos_col_space*i+xpos_col_space_additional),
                    ypos-.05,
                    f"Std. dev: {std:.2f}",
                    # fontdict=font_kws,
                    color=sns.color_palette(cmap)[i],
                    transform=axc.transAxes,
                )
                axc.text(
                    xpos*(xpos_col_space*i+xpos_col_space_additional),
                    ypos-.1,
                    f"Median: {median:.2f}",
                    # fontdict=font_kws,
                    color=sns.color_palette(cmap)[i],
                    transform=axc.transAxes,
                )
                axc.text(
                    xpos*(xpos_col_space*i+xpos_col_space_additional),
                    ypos-.15,
                    f"Min.: {min:.2f}",
                    # fontdict=font_kws,
                    color=sns.color_palette(cmap)[i],
                    transform=axc.transAxes,
                )
                axc.text(
                    xpos*(xpos_col_space*i+xpos_col_space_additional),
                    ypos-.2,
                    f"Max.: {max:.2f}",
                    # fontdict=font_kws,
                    color=sns.color_palette(cmap)[i],
                    transform=axc.transAxes,
                )
                axc.text(
                    xpos*(xpos_col_space*i+xpos_col_space_additional),
                    ypos-.25,
                    f"Skew: {sk:.2f}",
                    # fontdict=font_kws,
                    color=sns.color_palette(cmap)[i],
                    transform=axc.transAxes,
                )
                axc.text(
                    xpos*(xpos_col_space*i+xpos_col_space_additional),
                    ypos-.3,
                    f"Kurtosis: {kt:.2f}",  # Excess Kurtosis
                    # fontdict=font_kws,
                    color=sns.color_palette(cmap)[i],
                    transform=axc.transAxes,
                )
                axc.text(
                    xpos*(xpos_col_space*i+xpos_col_space_additional),
                    ypos-.35,
                    f"Counts: {len(xi)}",
                    # fontdict=font_kws,
                    color=sns.color_palette(cmap)[i],
                    transform=axc.transAxes,
                )
                leg=axc.legend(loc="lower left")
                leg._legend_box.align = "left"


    def __plot_get_data_facet__(self, axs, facet):
        res={}
        if not axs.axes_dict:
            res[None]={'data':self, 'ax':axs.axes[0, 0]}
        else:
            for facet_cat, axc in axs.axes_dict.items():
                if isinstance(facet_cat, tuple):
                    row=facet_cat[0]
                    col=facet_cat[1]
                    cases=f"({facet['col']}=='{col}') and ({facet['row']}=='{row}')"
                    tab= self.select_rows(query=cases)
                else:
                    if facet['col']:
                        cases=f"({facet['col']}=='{facet_cat}')"
                    else:
                        cases=f"({facet['row']}=='{facet_cat}')"
                    tab= self.select_rows(query=cases)
                res[facet_cat]={'data':tab, 'ax':axc}
        return res


    def __density_get_group_data__(self, var, group_var, group_value):
        xi=(
            self
            .select_rows(query=f"{group_var}=='{group_value}'")
            .select_cols(names=var)
            .values
            .flatten()
        )
        n=self.nrow
        return xi, n

    def __get_facet__(self, facets):
        facets_res = {'col':None, 'row':None}
        if facets:
            for k, v in facets.items():
                facets_res[k]=v
        return facets_res




    def __plot_hist_bin_labels__(self, ax, **kws):
        bin_labels_color=kws.get('bin_labels_color', 'black')
        bin_labels_ha=kws.get('bin_labels_ha', 'center')
        bin_labels_va=kws.get('bin_labels_va', 'bottom')
        bin_labels_fontsize=kws.get('bin_labels_fontsize', None)
        bin_labels_round=kws.get('bin_labels_round', 2)
        s=0
        for p in ax.patches:
            s+= p.get_height()
        for p in ax.patches: 
            if p.get_height()>0:
                label=f'{round(p.get_height()/s, bin_labels_round)}'
                ax.text(p.get_x() + p.get_width()/2.,
                        p.get_height(),
                        label, 
                        fontsize=bin_labels_fontsize,
                        color=bin_labels_color,
                        ha=bin_labels_ha,
                        va=bin_labels_va,
                        zorder=1,
                        )

    def __plot_hist_main__(self, ax, var, discrete, **kws):
        border_color=kws.get('border_color', 'white')
        alpha=kws.get('alpha', .7)
        color=kws.get('color', None)
        kde=kws.get('density', False)
        kde_kws=kws.get('kde_kws', None)
        linewidth=kws.get('linewidth', 2)
        stat=kws.get('stat', 'probability')
        ylim = kws.get("ylim", None)
        bins = kws.get("bins", 'auto')
        groups=kws.get("groups", None)
        multiple=kws.get("multiple", None)
        #
        if groups and not multiple:
            multiple='dodge'
        elif not groups and not multiple:
            multiple='layer'
        sns.histplot(data=self.drop_rows(dropna=var),
                        x=var,
                        alpha=alpha,
                        bins=bins,
                        edgecolor=border_color,
                        color=color,
                        linewidth=linewidth,
                        hue=groups,
                        multiple=multiple,
                        stat=stat,
                        discrete=discrete,
                        kde=kde,
                        kde_kws=kde_kws,
                        zorder=1,
                        ax=ax)
        if ylim:
            ax.set_ylim(ylim)


# *** Plot table
    def plot_table(self, ax=None, **kws):
        '''
        Plot table
        
        Input
           col_withs either a float or a list of floats between 0 and 1. If a 
                     float, use the provided value to set the size of all 
                     columns. If a list, set the value of the columns 
                     based on the list
        '''
        row_labels=kws.get('row_labels', None)
        col_widths = kws.get("col_widths", None)
        row_scale = kws.get("row_scale", 1)
        col_scale = kws.get("col_scale", 1)
        va = kws.get("va", 'center')
        ha = kws.get("ha", 'center')
        loc = kws.get("loc", 'top')
        borders = kws.get("borders", 'closed')
        header_colors = kws.get('header_colors',
                          plt.cm.BuPu(np.full(len(self.columns), 0.1)))
        fontsize = kws.get("fontsize", 10)
        if isinstance(col_widths, float):
            col_widths = [col_widths]*len(tab.columns)
        if not ax:
            fig, ax = self.__create_figure__(nrow=1, ncol=1, **kws)
        gtab=ax.table(cellText=self.values,
                      rowLabels=row_labels,
                      colWidths = col_widths ,
                      colLabels=self.columns,
                      cellLoc = ha,
                      rowLoc = va,
                      colColours=header_colors,
                      edges=borders,
                      loc=loc,
                      )
        gtab.scale( col_scale, row_scale)
        gtab.set_fontsize(fontsize)
        return ax

    # Plot correlation
    # ----------------
    def plot_corr(self, vars=None, sig=False, legend=False, cmap='coolwarm',
                  size=8, ax=None):
        '''
        Plot correlation matrix

        Input
    	-----
           vars  : list with variables to plot. If None, plot all variables.
           sig  : boolean, if True plot stars with significance-level (default False)
           legend  : If True, plot legend (default False)
           cmap    : color map to use
           size    : font size

        Output
    	------
           Plot with correlation matrix
        '''
        if not vars:
            vars = self.names()
        cor=self.corr_pairwise(vars, long_format=False)
        if sig:
            pval = (
                self
                .select_cols(names=vars)
                .corr(method=lambda x, y:
                      stats.pearsonr(x, y)[1]) - np.eye(*cor.shape)
            )
            p = pval.applymap(lambda x: ''.join(['*' for t in [0.01, 0.05, 0.1]
                                                 if x<=t]))
            cor = cor.round(2).astype(str) + "\n" + p
        mask = np.zeros_like(cor, dtype=bool)
        mask[np.triu_indices_from(mask)]= True
        if not ax:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 6], tight_layout=True)
        heatmap = sns.heatmap(cor, 
                              mask = mask,
                              # square = True,
                              linewidths = .5,
                              cbar = False,
                              cmap = cmap,
                              cbar_kws = {'shrink': .6, 
                                          'ticks' : [-1, -.5, 0, 0.5, 1]},
                              vmin = -1, 
                              vmax = 1,
                              annot = True,
                              annot_kws = {"size": size},
                              ax=ax
                              )
        return heatmap



    # plot statistics with std. error 
    # -------------------------------
    def plot_coef(self, x, y, se=None, ci=None, model_id=None, 
                  xlab=None, ylab=None,
                  title=None,
                  title_posx=0.1,
                  title_posy=0.98,
                  remove_intercept='Intercept',
                  colors=None,
                  shapes=None,
                  size=None,
                  dodge=0.05,
                  # h/vlines
                  # --------
                  hline=0,
                  hline_style='--',
                  hline_color='red',
                  hline_width=1,
                  # 
                  vline=0,
                  vline_style='--',
                  vline_color='red',
                  vline_width=1,
                  # 
                  # facet 
                  # -----
                  facet_x=None,
                  facet_sharex=True,
                  facet_y=None,
                  facet_sharey=True,
                  # 
                  facet_ncols=None,
                  facet_nrows=None,
                  facet_fontsize=11,
                  facet_alpha=.7,
                  facet_fontweight='bold',
                  facet_yoffset=0,
                  # 
                  # legend 
                  # ------
                  leg_pos='top',
                  leg_pos_manual=None,
                  leg_ncol=1,
                  leg_fontsize=10,
                  leg_title=None,
                  leg_facet=0,
                  # 
                  kws_grid={},
                  kws_border={},
                  # 
                  figsize=None,
                  fig_layout=None,
                  fig_widths=None,
                  fig_heights=None,
                  fig_margins=None,
                  tight_layout=True,
                  ax=None):
        '''
        
        Input
        -----
        se         : a string with the column that contains the standard
                     error of the estimate. If None, 'ci' must be provided.
        ci         : a tuple of strings containing the name of the columns
                     with the lower and upper bound of the confidence interval.
                     If None, se must be provided.
        leg_facet  : integer, the facet to plot the legend. In case of 
                     multiple rows and cols, facets are ordered left-right and 
                     top-bottom starting at zero. If None, all facets
                     will display their legend
                  
        figsize    : tuple (width, height) for the whole figure
        fig_layout : matrix with the layout of the figure. Ex:
                     [['a', 'b', 'c'],
                      ['A', 'B', 'c']]
                     will draw a figure with three columns. The first and 
                     second columns will have two rows, the third will have one
        fig_widths : a list with the ratio for each column of plots
        fig_height : a list with the ratio for each row of plots
        fig_margins: a dict(top= , right=, left=, bottom=) with the size of the
                     margins. Ex;
                        dict(top=.85, bottom=.15, right=.99, left=.1)
        
        '''
        if facet_x:
            if self[facet_x].dtype=='category':
                cats_x=self[facet_x].cat.categories.values
            else:
                cats_x=self[facet_x].unique()
            ncats_x=len(cats_x)
            if not facet_ncols:
                facet_ncols=len(self[facet_x].unique())
            elif facet_ncols>ncats_x:
                facet_ncols=ncats_x
            elif facet_ncols<ncats_x and not facet_nrows:
                facet_nrows=np.ceil(ncats_x/facet_ncols)
        # 
        if facet_y:
            if self[facet_y].dtype=='category':
                cats_y=self[facet_y].cat.categories.values
            else:
                cats_y=self[facet_y].unique()
            ncats_y=len(cats_y)
            if not facet_nrows:
                facet_nrows=len(self[facet_y].unique())
            elif facet_nrows>ncats_y:
                facet_nrows=ncats_y
            elif facet_nrows<ncats_y and not facet_ncols:
                facet_ncols=int(np.ceil(ncats_y/facet_nrows))
        # 
        if facet_x and facet_y:
            facet_nrows=ncats_y
            facet_ncols=ncats_x
        # 
        # check if inverted or not
        covars_on_xaxis=False if dict(self.dtypes)[x]==float else True
        ncovars=len(self[x].unique()) if covars_on_xaxis else len(self[y].unique())
        # 
        if not figsize:
            if covars_on_xaxis:
                figsize=[10, 6]
            else:
                figsize=[9, 10]
        if not ax:
            if leg_pos_manual:
                tight_layout=False
            if fig_layout:
                fig, ax = self.__create_figure__(
                    fig_layout=fig_layout,
                    figsize=figsize,
                    fig_widths=fig_widths,
                    fig_heights= fig_heights,
                    sharex=facet_sharex,
                    sharey=facet_sharey,
                )
            else:
                fig, ax = self.__create_figure__(
                    figsize=figsize,
                    nrows=facet_nrows if facet_nrows else 1, 
                    ncols=facet_ncols if facet_ncols else 1,
                    tight_layout=tight_layout,
                    sharex=facet_sharex,
                    sharey=facet_sharey,
                )
        # 
        args=locals()
        args.pop('self')
        args.pop('ax')
        args["covars_on_xaxis"]=covars_on_xaxis
        args["ncovars"]=ncovars
        # 
        if not facet_x and not facet_y:
            self.__plot_coef_main__(**args, ax=ax, facet=0)
        # 
        # facets 
        # ------
        xcoord=0
        ycoord=1.13+facet_yoffset
        yoffset=.07
        if facet_x and not facet_y:
            for i, cat_x in enumerate(cats_x):
                axc=ax[i]
                tab=self.select_rows(query=f"{facet_x}=='{cat_x}'")
                axc=tab.__plot_coef_main__(**args, ax=axc, facet=i)
                plt.subplots_adjust(top=.78)
                axc.annotate(cat_x, xy=(xcoord, ycoord),
                             xytext=(xcoord, ycoord),
                             xycoords='axes fraction',
                             # 
                             fontweight=facet_fontweight,
                	     size=facet_fontsize,
                             alpha=facet_alpha
                             )
        # 
        if not facet_x and facet_y:
            for i, cat_y in enumerate(cats_y):
                axc=ax[i]
                tab=self.select_rows(query=f"{facet_y}=='{cat_y}'")
                axc=tab.__plot_coef_main__(**args, ax=axc, facet=i)
                plt.subplots_adjust(top=.78)
                axc.annotate(cat_y, xy=(xcoord, ycoord),
                             xytext=(xcoord, ycoord), xycoords='axes fraction',
                             # 
                             fontweight=facet_fontweight,
                	     size=facet_fontsize,
                             alpha=facet_alpha
                             )
        # 
        if facet_x and facet_y:
            faceti=0
            for j, cat_y in enumerate(cats_y):
                for i, cat_x in enumerate(cats_x):
                    axc=ax[faceti]
                    query=f"({facet_x}=='{cat_x}') and ({facet_y}=='{cat_y}')"
                    tab=self.select_rows(query=query)
                    axc=tab.__plot_coef_main__(**args, ax=axc, facet=faceti)
                    if j==0:
                        cat=f"{cat_x}"
                        plt.subplots_adjust(top=.78)
                        axc.annotate(cat, xy=(xcoord, ycoord),
                                     xytext=(xcoord, ycoord),
                                     xycoords='axes fraction',
                                     # 
                                     fontweight=facet_fontweight,
                                     size=facet_fontsize,
                                     alpha=facet_alpha
                                     )
                    if i==0:
                        axc.set_ylabel(cat_y, weight=facet_fontweight,
                                       size=facet_fontsize,
                                       alpha=facet_alpha)
                        # cat=f"{cat_x}"
                        # axc.annotate(cat, xy=(xcoord, ycoord),
                        #              xytext=(xcoord, ycoord),
                        #              xycoords='axes fraction',
                        #              # 
                        #              fontweight=facet_fontweight,
                	#              size=facet_fontsize,
                        #              alpha=facet_alpha
                        #              )
                    faceti+=1
        # 
        if fig_margins:
            plt.subplots_adjust(**fig_margins)
        fig.suptitle(title, x=title_posx, y=title_posy, ha='left')
        fig.supxlabel(xlab, y=0.01, ha='center', va='bottom')
        fig.supylabel(ylab, x=0.0, va='center')
        # fig.text(0.5, 0, xlab, ha='center')
        # fig.text(0.0,.5, ylab, va='center', rotation='vertical')
        return ax



    def __plot_coef_main__(self, **kws):
        ax=kws.get("ax")
        width=0.25
        x= kws.get("x")
        y= kws.get("y")
        model_id= kws.get("model_id")
        se=kws.get("se", None)
        ci=kws.get("ci", None)
        remove_intercept= kws.get("remove_intercept")
        xlab= kws.get("xlab")
        ylab= kws.get("ylab")
        hline= kws.get("hline")
        hline_style= kws.get("hline_style")
        hline_color= kws.get("hline_color")
        hline_width= kws.get("hline_width")
        vline= kws.get("vline")
        vline_style= kws.get("vline_style")
        vline_color= kws.get("vline_color")
        vline_width= kws.get("vline_width")
        shapes= kws.get("shapes")
        colors= kws.get("colors")
        size= kws.get("size", 7)
        dodge= kws.get("dodge")
        leg_pos=kws.get("leg_pos")
        leg_pos_manual=kws.get("leg_pos_manual")
        leg_ncol=kws.get("leg_ncol")
        leg_fontsize=kws.get("leg_fontsize")
        leg_title=kws.get("leg_title")
        leg_facet=kws.get("leg_facet")
        # 
        covars_on_xaxis=kws.get("covars_on_xaxis")
        ncovars=kws.get("ncovars")
        # 
        kws_grid=kws.get("kws_grid")
        kws_border=kws.get("kws_grid")
        #
        facet=kws.get("facet")

        # 
        assert se or ci, ('Either the standard error (se) or '+
                          ' the confidence interval (ci) must be provided!')

        use_ci=False
        if ci and not se:
            use_ci=True
            assert len(ci)==2, ("ci must be a tuple with two strings, "+
                                "the first indicating the lower bound, "+
                                "the second the upper bound of the CI")
            assert np.all(self[ci[0]]<self[ci[1]]), ("There are observations "+
                                                     "whose lower bound of the "+
                                                     "confidence interval is "+
                                                     "larger than the upper "+
                                                     "bound. Check ci."
                                                     )
            muhat = y if covars_on_xaxis else x
            self=(
                self
                .mutate({"se": lambda x: (x[ci[0]]-x[muhat])/(-1.96) })
                # .mutate({"se": lambda x: [x[ci[0]], x[ci[1]]] })
            )
            se='se'
        # 
        # to handle more than one model
        model_id_colname="___model_id___"
        if not model_id:
            model_id_informed=False
            self=self.mutate({model_id_colname: model_id_colname})
        else:
            model_id_informed=True
            self=self.rename_cols(columns={model_id:model_id_colname})

        if remove_intercept:
            self=self.drop_rows(regex=remove_intercept)

        # 
        if not leg_facet and not isinstance(leg_facet, int):
            leg_facet=facet

        mods = self.sort_values([model_id_colname], ascending=True)[model_id_colname].unique()
        nmods = len(mods)
        #
        # dogdge
        dodges=np.linspace(0-dodge*(nmods-1), 0+dodge*(nmods-1), nmods)
        # 

        shapes_used=True if shapes else False
        shapes = ['o']*nmods if not shapes else shapes
        shapes = [shapes]*nmods if not isinstance(shapes, list) else shapes

        colors_used=True
        if not colors:
            colors_used=False
            colors=[col for name, col in mcolors.TABLEAU_COLORS.items()]
        colors = [colors]*nmods if not isinstance(colors, list) else colors
        # 
        leg_elements={'labels':[],
                      'shapes':[],
                      'colors':[]}
        for i, mod in enumerate(mods):
            modi = self.select_rows(query=f"{model_id_colname}=='{mod}'")
            # modi = modi.set_index(x).reindex(self[x].unique())

            if covars_on_xaxis:
                # if use_ci:
                #     lo=modi.loc[:,ci[0]]
                #     up=modi.loc[:,ci[1]]
                #     yerror=[lo, up]
                # else:
                #     yerror=modi[se]
                yerror=modi[se]
                xerror=None
                trans = Affine2D().translate(dodges[i], 0.0) + ax.transData
            else:
                # if use_ci:
                #     lo=modi[ci[0]]
                #     up=modi[ci[1]]
                #     xerror=[lo, up]
                # else:
                #     xerror=modi[se]
                xerror=modi[se]
                yerror=None
                trans = Affine2D().translate(0.0, dodges[i]) + ax.transData
            if use_ci:
                modi=modi.sort_values([x], ascending=True)
                for (idx_row, row) in modi.iterrows():
                    ax.scatter(row[x], row[y],
                               color=colors[i], 
                               marker=shapes[i],
                               transform=trans
                               )
                    if covars_on_xaxis:
                        ax.plot([row[x], row[x]],
                                [row[ci[0]], row[ci[1]]],
                                color=colors[i], 
                                transform=trans
                                )
                    else:
                        ax.plot([row[ci[0]], row[ci[1]]],
                                [row[y], row[y]],
                                color=colors[i], 
                                transform=trans
                                )
            else:
                ax.errorbar(x=modi[x], y=modi[y],
                        # yerr=[modi[ci[0]], modi[ci[1]]], xerr=xerror,
                        yerr=yerror, xerr=xerror,
                        ls='none',
                        color=colors[i],
                        # markers
                        marker=shapes[i],
                        markersize=size,
                        # 
                        transform=trans
                        )
            # ax.set_xlabel(xlab)
            # ax.set_ylabel(ylab)
            # 
            if ((isinstance(hline, int) or isinstance(hline, float)) and
                covars_on_xaxis):
                ax.axhline(y=hline,
                           linestyle=hline_style,
                           color=hline_color, linewidth=hline_width)
            if ((isinstance(vline, int) or isinstance(vline, float)) and
                not covars_on_xaxis):
                ax.axvline(x=vline,
                           linestyle=vline_style,
                           color=vline_color, linewidth=vline_width)

            leg_elements['labels']+=[mod]
            leg_elements['shapes']+=[shapes[i]]
            leg_elements['colors']+=[colors[i]]

        if (shapes_used or colors_used or model_id_informed) and leg_facet==facet:
            if leg_pos=='top' and not leg_pos_manual:
                leg_pos_manual=(0.5, 1.01)
            # ## finally, build customized legend
            legend_elements = [Line2D([0], [0],
                                      marker=leg_elements['shapes'][i],
                                      label=leg_elements['labels'][i],
                                      color = leg_elements['colors'][i],
                                      markersize=8) for i, m in enumerate(shapes)]
            # _ = ax.legend(handles=legend_elements, loc=2,
            #               prop={'size': 15},
            #               labelspacing=1.2)
            # handles, labels = ax.get_legend_handles_labels()
            if leg_pos_manual:
                leg = ax.legend(loc='lower left',
                                handles=legend_elements, 
                                bbox_to_anchor=leg_pos_manual, handlelength=2,
                                title=leg_title,
                                handletextpad=.3, prop={'size':leg_fontsize},
                                # pad between the legend handle and text
                                labelspacing=.3, #  vertical space between the legend entries.
                                columnspacing=1, # spacing between columns
                                # handlelength=1, #  length of the legend handles
                                ncol=leg_ncol, mode=None, frameon=False, fancybox=True,
                                framealpha=0.5, facecolor='white')
            else:
                leg = ax.legend(loc=leg_pos, handlelength=2,
                                handles=legend_elements, 
                                title=leg_title,
                                handletextpad=.3, prop={'size':leg_fontsize},
                                # pad between the legend handle and text
                                labelspacing=.3, #  vertical space between the legend entries.
                                columnspacing=1, # spacing between columns
                                # handlelength=1, #  length of the legend handles
                                ncol=leg_ncol, mode=None, frameon=False, fancybox=True,
                                framealpha=0.5, facecolor='white')
            leg._legend_box.align = "left"
        # Annotation
        # ----------
        #     fs = 16
        #     ax.annotate('Control', xy=(0.3, -0.2), xytext=(0.3, -0.35), 
        #                 xycoords='axes fraction', 
        #                 textcoords='axes fraction', 
        #                 fontsize=fs, ha='center', va='bottom',
        #                 bbox=dict(boxstyle='square', fc='white', ec='black'),
        #                 arrowprops=dict(arrowstyle='-[, widthB=6.5, lengthB=1.2', lw=2.0, color='black'))

        #     ax.annotate('Study', xy=(0.8, -0.2), xytext=(0.8, -0.35), 
        #                 xycoords='axes fraction', 
        #                 textcoords='axes fraction', 
        #                 fontsize=fs, ha='center', va='bottom',
        #                 bbox=dict(boxstyle='square', fc='white', ec='black'),
        #                 arrowprops=dict(arrowstyle='-[, widthB=3.5, lengthB=1.2', lw=2.0, color='black'))
        ax.tick_params(top=None, bottom=None, left=None, right=None)
        if not kws_grid or not kws_grid.get("grid_axis", None):
            kws_grid['grid_axis']='y' if covars_on_xaxis else 'x'
        self.__plot_grid__(ax, **kws_grid)
        self.__plot_border__([ax], **kws_border)

        return ax

# *** Plot ancillary functions
    def __plot_border__(self, axs, **kws):
        for axc in axs:
            axc.spines['bottom'].set_visible(True)
            axc.spines['left'].set_visible(False)
            axc.spines['right'].set_visible(False)
            axc.spines['top'].set_visible(False)
        
    def __create_figure__(self, **kws):
        nrows = kws.get("nrows", 1)
        ncols = kws.get("ncols", 1)
        sharex = kws.get("sharex", False)
        sharey = kws.get("sharey", False)
        figsize=kws.get('figsize', [10, 6])
        fig_layout=kws.get('fig_layout', None)
        if not figsize:
            figsize=[10, 6]
        tight_layout=kws.get("tight_layout", True)
        polar=kws.get("polar", False)
        if fig_layout:
            fig_widths=kws.get("fig_widths", None)
            fig_heights=kws.get("fig_heights", None)
            fig_ratio=dict(width_ratios= fig_widths,
                           height_ratios=fig_heights)
            fig, axs = plt.subplot_mosaic(fig_layout,
                                          gridspec_kw=fig_ratio,
                                          figsize=figsize,
                                          sharex=sharex,
                                          sharey=sharey,
                                          constrained_layout=True)
            ax=np.array([axc for k, axc in axs.items()])
        else:
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                                   tight_layout=tight_layout,
                                   subplot_kw=dict(polar=polar),
                                   sharex=sharex,
                                   sharey=sharey
                                   )
        if nrows>1 or ncols>1:
            ax=ax.flatten()
        return fig, ax

    def __plot_yticks__(self, ax, **kws):
        ytick_size = kws.get("ytick_size", None)
        ax.tick_params(axis="y",
                       labelsize= ytick_size
                       # top=None, bottom=True, left=True, right=None, 
                       # labeltop=None, labelbottom=True,
                       # labelleft=True, labelright=None,
                       # which='major', direction='out', color='black', pad=3,
                       # grid_color='lightgrey', grid_linestyle='--',
                       # grid_linewidth=.5,
                       # labelcolor='black'
                        )
        

    def __plot_grid__(self, ax, **kws):
        grid_which=kws.get("grid_which", 'major')
        grid_axis = kws.get("grid_axis", 'both')
        grid_linetype = kws.get("grid_linetype", '-')
        grid_alpha = kws.get("grid_alpha", .4)
        ax.grid(b=None, which=grid_which, axis=grid_axis,
                linestyle=grid_linetype, alpha=grid_alpha)
        ax.set_axisbelow(True) # to put the grid below the plot

# * egroupby class

class egroupby(pd.core.groupby.DataFrameGroupBy):
    '''
    Extends the functionalities of pandas groupby class
    '''
    def __init__(self,  *args, **kwargs):
        super(egroupby, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return egroupby

    def mutate(self, dict, flatten=True):
        res=self.apply(lambda x: x.mutate(dict))
        if flatten:
            try:
                res = res.flatten()
            except (OSError, IOError, AttributeError) as e:
                pass
        return res

    def mutate_case(self, dict, replace=None):
        res = self.apply(lambda x: x.mutate_case(dict))
        return res

    def scale(self, *args, **kws):
        res=self.apply(lambda x: x.scale(*args, **kws))
        return res

    def pivot_longer(self, flatten=True, *args, **kws):
        res=self.apply(lambda x: x.pivot_longer(*args, **kws))
        if flatten:
            res = res.flatten(drop=True)
        return res
        

    def freq(self, *args, **kws):
        res=(self.apply(lambda x: x.freq(*args, **kws))
             .reset_index(drop=False)
             # .filter(regex=f"^[level_[0-9]*]")
        )
        return res
        

# * Plots
# ** Extended Slider

class eSlider(Slider):
    def __init__(self, left, bottom, width, height,
                 lower, upper, init, *args, **kwargs):
        # use the __init__ method from Slider to ensure
        # that we're inheriting the correct behavior
        self.axes_marker = plt.axes([left, bottom, width, height])
        super(eSlider, self).__init__(
            ax=self.axes_marker,
            valmin=lower, valmax=upper,
            valinit=init,
            visible=False, *args, **kwargs)
        self.slider()

    # this method is makes it so our method eSlider return an instance
    # of eSlider, instead of a regular Slider
    @property
    def _constructor(self):
        return eSlider

    def on_update(self, func):
        self.func = func
        def func_extension(func):
            self.func()
            self.update_slider_marker()
        self.on_changed(func_extension)
        

    def slider_marker(self, pos, y, color='white', edgecolor='black', s=100,
                      linewidth=.3):
        self.marker = self.ax.scatter(pos, y, color=color, edgecolor=edgecolor,
                                      s=s, linewidth=linewidth, zorder=100,
                                      marker='o') 

    def slider(self, markercolor='white', markeredgecolor='black',
               markersize=100, markerlinewidth=.3, 
               barcolor='lightgrey', barstyle='-',barwidth=4,
               **kwds):
        # pos, .5, color, edgecolor, s, linewidth, zorder (marker)
        # facecolor='white', edgecolor='red', hatch=None, (rectable)
        # 
        self.vline.set_xdata(-np.Inf) # to remove the initial marker of valinit
        self.vline.set_xdata(-10000) # to remove the initial marker of valinit
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_linewidth(3)
        self.ax.tick_params(top=None, bottom=None, left=None, right=None, 
                            labeltop=None, labelbottom=None, labelleft=None,
                            labelright=None,
                            which='major', direction='out', color='black', pad=3,
                            # grid_color='lightgrey', grid_linestyle='--',
                            # grid_linewidth=.5,
                            # labelsize=10, labelcolor='black'
                            )
        # bar
        # self.ax.plot([.01, .98], [.5, .5],
        self.ax.axhline(.5, xmin=0.01, xmax=.99,
                     color=barcolor,
                     linestyle=barstyle,
                     path_effects=[pe.Stroke(linewidth=4,
                                             foreground='black'),
                                      pe.Normal()],
                     # transform=plt.gca().transAxes,
                     linewidth=barwidth, solid_capstyle='round')
        
        # marker
        self.slider_marker(pos=self.val, y=.5, color=markercolor,
                           edgecolor=markeredgecolor,
                           s=markersize, linewidth=markerlinewidth)
    
    # 
    def update_slider_marker(self):
        self.marker.remove()
        self.slider_marker(pos=self.val, y=.5)


# ** Extended TextBox

class eTextBox(TextBox):
    def __init__(self, left, bottom, width, height, hovercolor="whitesmoke",
                 *args, **kwargs):
        # use the __init__ method from TextBox to ensure
        # that we're inheriting the correct behavior
        self.axes_box = plt.axes([left, bottom, width, height])
        super(eTextBox, self).__init__(ax=self.axes_box, hovercolor=hovercolor,
                                       *args, **kwargs)
        self.box()

    # this method is makes it so our method eTextBox return an instance
    # of eTextBox, instead of a regular TextBox
    @property
    def _constructor(self):
        return eTextBox

    def box(self, bottom=True, left=False,
            right=False, top=False, linestyle="-",
            linewidth=1):
        self.ax.spines['bottom'].set_visible(bottom)
        self.ax.spines['left'].set_visible(left)
        self.ax.spines['right'].set_visible(right)
        self.ax.spines['top'].set_visible(top)
        self.ax.spines['bottom'].set_linestyle(linestyle)
        self.ax.spines['bottom'].set_linewidth(linewidth)



