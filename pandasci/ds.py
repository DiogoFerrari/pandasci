import pandas as pd
import numpy as np
from scipy import stats
import savReaderWriter as spss
import seaborn as sns
import re
import os
import inspect
# dply-like operations
import itertools as it
from plydata.expressions import case_when
from plydata import define
from datetime import datetime
# 
from scipy.stats import norm as dnorm
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
import xlrd

# {{{ spss                  }}}

# will be deprecated soon due to methods naming (old class)
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
                         from the .sav file
           use_labels    bool, if True return the SPSS labels of the variables
           rec           dictionary with the variables and values to be recoded.
                         Format {<varname>: <value1>:<new value>, 
                                            <value2>:<new value>}
           vars_newnames list with the new names of the variables. It muat
                         follow the order of the argument 'vars'

        Output:
            A Data Frame with the variables selected in 'vars'

        '''

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
    def __init__(self, fn):
        # df_raw = spss.SavReader(fn, returnHeader=True, rawMode=True)
        self.__fn = fn
        with spss.SavHeaderReader(fn, ioUtf8=True) as header:
            self.__metadata = header.all()


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
        if get_output:
            return data
        else:
            print("\n\n")
            print(data)
            print("\n\n")
            

    def var_values_summary(self, varname):
        assert isinstance(varname, str), "\n\n'varname' must be a string\n\n"
        data = self.vars_load([varname], use_labels=False)
        print("\n\n")
        print(data.describe())
        print("\n\n")

    def var_values_recode_skeleton(self, varname):
        dic = self.var_values(varname, False)
        print("rec = {\""+varname+"\" : {")
        for k, v in dic.items():
            print(f"    \"{v}\" : ,")
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
                         follow the order of the argument 'varnames'

        Output:
            A Data Frame with the variables selected in 'varnames'

        '''

        assert isinstance(varnames, list) or varnames is None,\
        f"Variable 'varnames' must be a list or 'None'"
        assert isinstance(varsnames_new, list) or varsnames_new is None,\
        f"Variable 'varsnames_new' must be a list or 'None'"
        
        if varnames:
            varnames = self.__toBytes__(varnames)
        else:
            varnames = self.__metadata.varNames
        print(f"\nLoading values of {len(varnames)} variable(s) ...")
        print(varnames)
        print("\n\n")
        with spss.SavReader(self.__fn, returnHeader=False, rawMode=True,
                            selectVars = varnames) as reader:
            vars_char = self.__toStr__(varnames)
            data = pd.DataFrame(reader.all(), columns=vars_char)
        if use_labels:
            for key_bin, var_char in zip(varnames, vars_char):
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
            self.rename(data, varnames, varsnames_new)
        return eDataFrame(data)


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
                vars[i] = varname.decode('utf-8')
        return vars


# }}}
# {{{ Extended DataFrame    }}}

class eDataFrame(pd.DataFrame):
    def __init__(self,  *args, **kwargs):
        # use the __init__ method from DataFrame to ensure
        # that we're inheriting the correct behavior
        super(eDataFrame, self).__init__(*args, **kwargs)

    # this method is makes it so our methoeDataFrame return an instance
    # of eDataFrame, instead of a regular DataFrame
    @property
    def _constructor(self):
        return eDataFrame

    # =====================================================
    # Data wrangling
    # =====================================================
    def case_when(self, varname, replace):
        if varname in self.columns:
            varname_exists = True
            col_names = self.columns
        else:
            varname_exists = False
        # replace using casewhen
        res = (
            self >>
            define(___newvar_name_placeholder___=case_when(replace))
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
        return eDataFrame(res)

    def pivot_longer(self, id_vars, value_vars=None,
                     var_name=None, value_name='value', ignore_index=True):
        res = pd.melt(self, id_vars=id_vars, value_vars=value_vars,
                       var_name=var_name, value_name=value_name,
                       ignore_index=ignore_index)
        return eDataFrame(res)
        
    def pivot_wider(self, id_vars=None, cols_from=None, values_from=None,
                    aggfunc='sum'):
        res = (self
               .pivot_table(values=values_from, index=id_vars,
                            columns=cols_from, aggfunc=aggfunc)
               .reset_index(drop=False)
               )
        res.columns.name=''
        return eDataFrame(res)


    # =====================================================
    # Statistics
    # =====================================================

    def summary(self, vars, funs, groups=None, wide_format=None):
        if groups:
            res = self.__summary_group__(vars, funs, groups, wide_format)
        else:
            res = self.__summary__(vars, funs)
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


    def __summary_group__(self, vars, funs, groups=None, wide_format=None):
        res=(self
             .filter(vars+groups)
             .groupby(groups)
             .agg(funs)
             .reset_index( drop=False)
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
        regexp="".join([f"{v}|" for v in funs])
        regexp=re.sub(pattern="\|$", repl="", string=regexp)
        regexp="("+regexp+")"
        # 
        res = (res
                .melt(id_vars=groups, value_vars=None,
                      var_name='variable', value_name='value', col_level=None))
        cols = (res
                .variable
                .str
                .extract("(.*)_"+regexp)
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
            values = [f"variable_{fun}" for fun in funs]
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
            # 
        col_names = ['variable_'+fun for fun in funs]
        for col_name, fun in zip(col_names, funs):
            res.rename(columns={col_name:fun}, inplace=True)
        return res



    def freq(self, vars, condition_on=None):
        '''
        Compute frequency and marginal frequence (conditional on)

        Input:
           vars a list of string with variable names to return values frequencies
           condition_on a list of strings with variable names to condition 
                        marginal frequencies on. They must be also in the 'vars'

        Output:
            DataFrame with frequencies

        '''
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
            res=(self
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
            res = (self
                   .groupby(vars)
                   .size()
                   .reset_index(name='n', drop=False)
                   .groupby(condition_on)
                   .apply(compute_freq)
                   .groupby(condition_on)
                   .apply(compute_stdev)
                   .sort_values(by=(condition_on+['freq']),  ascending=True)
            )
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
        assert len(vars)==2, "'vars' must be a string of size 2"
        subset = self[vars].dropna(subset=None, axis=0)
        x = subset[vars[0]]
        y = subset[vars[1]]
        r, p = stats.pearsonr(x,y)
        r_z = np.arctanh(r)
        se = 1/np.sqrt(x.size-3)
        z = stats.norm.ppf(1-alpha/2)
        lo_z, hi_z = r_z-z*se, r_z+z*se
        lo, hi = np.tanh((lo_z, hi_z))
        return {'cor':r, 'p-value':p, 'low':lo, "high":hi}
    

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
    

    def names(self, print_long=False):
        if print_long:
            for col in list(self):
                print(col)
        else:
            print(print(list(self)))

    # =====================================================
    # Utilities
    # =====================================================
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
        self = (self.reset_index(drop=False))
        def _remove_empty(column_name):
            return tuple(element for element in column_name if element)
        def _join(column_name):
            return sep.join(column_name)
        new_columns = [_join(_remove_empty(column)) for column in
                       self.columns.values]
        self.columns = new_columns
        return self

    def scale(self, vars=None, exclude=None, centralize=True, center='mean'):
        '''
        Rescale variables by dividing it by its standard deviation

        Input:
            vars    : list with variables to scale. If 'None', it will
                      scale all numerical variables
            exclude : list with variables NOT to scale. If 'None', it will
                      scale all numerical variables in the argument 'vars'
            centralize: boolean, if True will recenter the variables
            center    : either 'mean' or a number to use as the center
        
        Output
            dataframe with variables rescaled
        '''
        self = self.__rescale__(vars=vars, exclude=exclude,
                                centralize=centralize, scale=True,
                                center=center)
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
                                centralize=True, scale=False, center=center)
        return self

    def __rescale__(self, vars=None, exclude=None, centralize=False,
                    scale=False, center='mean'):
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


    # =====================================================
    # Plots
    # =====================================================
    def __create_figure__(self, **kws):
        nrows = kws.get("nrows", 1)
        ncols = kws.get("ncols", 1)
        figsize=kws.get('figsize', [10, 6])
        tight_layout=kws.get("tight_layout", True)
        polar=kws.get("polar", False)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                               tight_layout=tight_layout,
                               subplot_kw=dict(polar=polar))
        return fig, ax


    # =====================================================
    # Scatter plot
    # =====================================================
    def plot_scatter(self, x, y, **kwargs):
        self.plot_line(x, y, kind='scatter', pts_show=True, **kwargs)
        
    # =====================================================
    # Line Plot
    # =====================================================
    def plot_line(self, x, y,
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
                  # legend
                  leg_pos='left',
                  leg_title=None,
                  leg_fontsize=13,
                  leg_xpos=0, leg_ypos=1,
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
                         data=self)
        # Legend
        # ------
        if colors or linetype:
            if leg_pos=='left':
                leg = ax._legend
                leg.set_bbox_to_anchor([1,.95])
            else:
                ax.legend.remove()
                axc=ax.axes[0][0]
                leg = axc.legend(loc='lower left', bbox_to_anchor=(leg_xpos,
                                                                   leg_ypos),
                                 handlelength=1.5,
                                 title=leg_title,
                                 handletextpad=.3, prop={'size':leg_fontsize},
                                 # pad between the legend handle and text
                                 labelspacing=.2, # vert sp between the leg ent.
                                 columnspacing=1, # spacing between columns
                                 ncol=3, mode=None, frameon=True, fancybox=True,
                                 framealpha=0.5, facecolor='white')
                leg._legend_box.align = "left"
                ax.fig.tight_layout()
        
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


        self.__plot_grid__(it.chain(*ax.axes))
        self.__plot_border__(it.chain(*ax.axes))



        # adjustments 
        # -----------
        ax.fig.set_size_inches(figsize[0], figsize[1])
        ax.tight_layout(w_pad=facet_y_dist, h_pad=facet_y_dist)


        # Message 
        # -------
        print(f"\nFor easy manipulation, this method returns two elements: \n"+\
              "1. The axis of the entire figure\n"+\
              "2. The axes for each facet in a list\n\n")
        return ax, it.chain(*ax.axes)


    # =====================================================
    # Polar plot
    # =====================================================
    def plot_polar(self, vars, group=None, func='mean', facet=None, **kws):
        '''
        Plot a summary value of variables using polar coordinate
        
        Input:
           vars  a list or a dictionary of lists of variables in the DataFrame. 
                 if a dictionary is used, the list of variables under each 
                 key is plotted together, a shaded area is added to the plot
                 based on the group, and the key is used as label for the group.
           group a string with the name of the variable in the DataFrame to 
                 use as group of the lines in the plot
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
                ax=list(itertools.chain(ax))
            else:
                ax=list(itertools.chain(*ax))
            self.__plot_polar_facets__(tab, group, func, ax, facet, **kws)
        if facet:
            for axc in ax:
                self.__plot_polar_add_thresholds__(axc, **kws)
                self.__plot_polar_addlabels__(axc, tab=tab, vars=vars,
                                              radian_pos=radian_pos, **kws)
                if 'variable_group' in tab.columns:
                    self.__plot_polar_vars_group__(ax=axc, tab=tab,
                                                   func=func, **kws)
            if legend_show:
                self.__plot_polar_add_legend__(ax[0], **kws)
        else:    
            self.__plot_polar_add_thresholds__(ax, **kws)
            self.__plot_polar_addlabels__(ax, tab=tab, vars=vars,
                                          radian_pos=radian_pos, **kws)
            if 'variable_group' in tab.columns:
                self.__plot_polar_vars_group__(ax=ax, tab=tab, func=func, **kws)
            if legend_show:
                self.__plot_polar_add_legend__(ax, **kws)
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
                                          itertools.cycle(thresholds_color),
                                          itertools.cycle(thresholds_size),
                                          itertools.cycle(thresholds_linetype)):
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
        for gr, ls in zip(groups, itertools.cycle(linestyles)):
            kws['linestyle']=ls
            tabt=tab.query(f"{group}=={gr}")
            self.__plot_polar_simple__(tabt, func, ax, label=gr, **kws)


    def __plot_polar_simple__(self, tab, func, ax, **kws):
        label=kws.get('label', None)
        color=kws.get("color", 'black')
        ls=kws.get('linestyle', '-')
        linesize=kws.get('linesize', 1)
        alpha=kws.get('alpha', 1)
        x, y = list(tab['pos']), list(tab[func])
        x.append(x[0]); y.append(y[0])# to close the circle
        ax.plot(x, y, color=color,
                linewidth=linesize,
                linestyle=ls, alpha=alpha,
                label=label
                )
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
                                       itertools.cycle(shade_colors),
                                       itertools.cycle(shade_alpha)
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




    ## ------------------------
    ## Plot ancillary functions
    ## ------------------------ 
    def __plot_grid__(self, axs):
        for axc in axs:
            # grid
            axc.grid(b=None, which='major', axis='y', linestyle='-', alpha=.3)
            axc.set_axisbelow(True) # to put the grid below the plot
        
    def __plot_border__(self, axs):
        for axc in axs:
            axc.spines['bottom'].set_visible(True)
            axc.spines['left'].set_visible(False)
            axc.spines['right'].set_visible(False)
            axc.spines['top'].set_visible(False)
        


# }}}
# {{{ Extended Slider }}}


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
    

    

# }}}
# {{{ Extended TextBox }}}

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



# }}}
# {{{ functions }}}

def read_data(**kwargs):
    fn=kwargs.get('fn')
    assert fn, "fn (filepath) must be provided."
    fn_type=os.path.splitext(fn)[1]
    # 
    if fn_type=='.csv' or fn_type=='.CSV':
        return read_csv(**kwargs)
    # 
    elif fn_type=='.dta' or fn_type=='.DTA':
        return read_dta(**kwargs)
    # 
    elif fn_type=='.sav':
        return spss_data(**kwargs)
    # 
    elif (fn_type=='.xls' or fn_type=='.xlsx' or
          fn_type=='.XLS' or fn_type=='.XLSX'):
        return read_xls(**kwargs)
    # 
    else:
        print(f"No reader for file type {fn_type}")
        return None
        

def read_csv(**kwargs):
    df = pd.read_csv(filepath_or_buffer=kwargs.get('fn'),
                     sep=kwargs.get('sep', ';'),
                     index_col=kwargs.get('index_col'),
                     decimal=kwargs.get('decimal', '.')
                     )
    return eDataFrame(df)
        
def read_dta(**kwargs):
    fn=kwargs.get('fn')
    return eDataFrame(pd.read_stata(fn))

def read_xls(**kwargs):
    fn=kwargs.get('fn'); kwargs.pop('fn')
    df = eDataFrame(pd.read_excel(io=fn, **kwargs))
    # 
    print(f"\nFunction arguments:\n")
    print(inspect.signature(pd.read_excel))
    print(f"\nFor details, run help(pd.read_excel)\n")
    print(f"Data set loaded!")
    # 
    return df
    
    
def reorderLegend(ax=None,order=None,unique=False):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    return(handles, labels)

# }}}
