import pandas as pd
import numpy as np
from scipy import stats
import savReaderWriter as spss
import seaborn as sns
import re
import os
# dply-like operations
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

# {{{ spss                  }}}

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
        return res


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
            assert all([cond_on in vars for cond_on in condition_on]), \
            "\n\nVariables in list 'condition_on' must all"+\
            " be in 'vars' or be 'None'\n\n"
        

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
                return df
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
                            labeltop=None, labelbottom=None, labelleft=None, labelright=None,
                            which='major', direction='out', color='black', pad=3,
                            # grid_color='lightgrey', grid_linestyle='--', grid_linewidth=.5,
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
        self.marker = self.slider_marker(pos=self.val, y=.5, color=markercolor,
                                         edgecolor=markeredgecolor,
                                         s=markersize, linewidth=markerlinewidth)
    
    def slider_marker(self, pos, y, color='white', edgecolor='black', s=100, linewidth=.3):
        self.marker = self.ax.scatter(pos, y, color=color, edgecolor=edgecolor,
                                      s=s, linewidth=linewidth, zorder=100, marker='o') 
    # 
    def update_slider_marker(self):
        self.marker.remove()
        self.marker = slider_marker(self, pos=self.val, y=.5)
    

    

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
