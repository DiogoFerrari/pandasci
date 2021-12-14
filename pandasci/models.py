from .ds import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import prince # for PCA
from statsmodels.formula.api import glm as glm
from statsmodels.formula.api import logit as logit
from statsmodels.stats import proportion as pwr2prop
from statsmodels.api import families as family
import warnings
import itertools
from statsmodels.iolib.summary2 import summary_col


# {{{ Power and Sample size }}}

class power():
    def __init__(self, es, alpha=0.05, power=0.8, two_tail=True,
                 ratio=1,
                 type=None):
        '''
        Compute the sample size

        Input 
        -----
        es       : effect size
        alpha    : significance level
        power    : power of the test
        two_tail : boolean. If True, use two-tail test
        ratio    : Sample size ratio, nobs2 = ratio * nobs1. Use 1 if the
                   sample size is the same in the two groups
        type     : sting with the type of test. Options are:

                   2prop :  used when you have two groups and
                            you want to know if the proportions of each group are
                            different from one another.
                            Examples: 
                            -------
                            a. Is the proportion of men in favor of gender equality
                               different from the proportion of woman in favor of
                               gender equality
                            b. Is proportion of cases with outcome "A" different 
                               different in the treatment and control group?

        '''
        self.es=es
        self.alpha=alpha
        self.power=power
        self.ratio=ratio
        self.two_tail=two_tail
        self.type=type
        if type=='2prop':
            self.data = self.__2prop__(es, alpha, power, two_tail, type, ratio)

    def __2prop__(self, es, alpha, power, two_tail, type, ratio):
        if isinstance(es, float):
            es = [es]
        # 
        prop1s = np.linspace(0, 1, 21)
        res = eDataFrame()
        # 
        if two_tail:
            tail='two-sided'
        for esi in es:
            prop2s = prop1s + esi
            for prop1, prop2 in zip(prop1s, prop2s):
                if 0 < prop2 <1:
                    nobs1 = pwr2prop.samplesize_proportions_2indep_onetail(
                        diff=esi,
                        alpha=alpha,
                        prop2=prop2,
                        power=power,
                        ratio=ratio,
                        alternative=tail
                    )
                    tmp = eDataFrame({
                        'prop1'              : [prop1],
                        'prop2'              : [prop2],
                        'es'                 : [esi],
                        'sample_size_group1' : [nobs1],
                        'sample_size_group2' : [nobs1*ratio],
                        'pwr'                : [power],
                        'alpha'              : [alpha],
                        'two-sided'          : [tail],
                    })
                    res = res.bind_row(tmp)
        return res

    def plot(self):
        if self.type=='2prop':
            ax = self.__plot_2prop__()
        return ax

    def __plot_2prop__(self):
        tab = (
            self
            .data
            .pivot_longer(id_vars=None, value_vars=['sample_size_group1',
                                                    'sample_size_group2'],
                          var_name='Group', value_name='Sample size', ignore_index=True)
            .replace({'Group':{'sample_size_group1':"Reference (e.g. control group)",
                               'sample_size_group2':'Interest (e.g. treatment group)'}} ,
                     regex=False, inplace=False)
            .rename(columns={'es':'Effect size'}, inplace=False)
        )
        # 
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 6], tight_layout=True)
        sns.scatterplot(x='prop2', y='Sample size', data=tab,
                        style='Group', hue='Effect size', palette='RdBu',s=70,
                        ax=ax)
        sns.lineplot(x='prop2', y='Sample size', data=tab,
                        style='Group', hue='Effect size', palette='RdBu',
                        ax=ax, alpha=.3, legend=False)
        # 
        ax.set_xlabel("Proportion of 'positive' outcome cases (Y=1) "+\
                      'in the group of interest')
        ax.set_ylabel('Sample size required do detect the effect size')
        # 
        # grid
        ax.grid(b=None, which='major', axis='both', linestyle='-', alpha=.3)
        ax.set_axisbelow(True) # to put the grid below the plot
        # legend
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(loc='upper right')
        leg._legend_box.align = "left"
        # -------
        # Splines (axes lines)
        # -------
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        title = "Sample size calculation"
        subtitle = ("Info:  "+
                    f"$\\alpha$: {self.alpha}; "+
                    f"Power ($\\beta$): {self.power}; "+
                    f"Test : {self.type}; "+
                    f"Two-sided : {self.two_tail}; "+
                    f"Ratio between sample sizes in each group : {self.ratio}"
                    )
        # -----
        # Title
        # -----
        plt.subplots_adjust(top=.78)
        xcoord=-.07
        ycoord=1.13
        yoffset=.07
        ax.annotate(title, xy=(xcoord, ycoord),
                   xytext=(xcoord, ycoord), xycoords='axes fraction',
                           size=15, alpha=.6)
        ax.annotate(subtitle, xy=(xcoord, ycoord-yoffset),
                   xytext=(xcoord, ycoord-yoffset), xycoords='axes fraction',
                           size=11, alpha=.6)
        # maximum value
        max_sample_sizes = (
            tab
            .groupby(['Effect size', 'Group'])
            .apply(lambda x: x.nlargest(1, columns=['Sample size']))
        )
        for idx, row in max_sample_sizes.iterrows():
            x = row['prop2']
            y = row['Sample size']
            group = row['Group']
            # 
            va = 'bottom' if 'Reference' in group else 'top'
            txt = f"Max: {int(row['Sample size']):,}\n({group})"
            txt = f'\n{txt}' if va == 'top' else f"{txt}\n"
            color='red' if va == 'top' else 'green'
            # 
            ax.scatter(x, y, s=60, color=color)
            ax.text(x, y, s=txt,
                    ha='center', va=va, ma='center',
                    fontdict=dict(weight='normal', style='italic',
                                  color=color, fontsize=10, alpha=1))
        return ax
        

    # =====================================================
    # default methods
    # =====================================================
    def __str__(self):
        print(self.data, flush=True)
        return None

    def __repr__(self):
        print(self.data, flush=True)
        return ''

# }}}
# {{{ regression models   }}}

class regression():
    def __init__(self, formula, data, family, na='omit',
                 engine='python', *args, **kws):
        '''
        Run regression models
        
        Input
           formula  : a string of a dictionary with labels for the models (key) 
                      and regression formulas (values)
           family   : string 'gaussian,' 'binomial,' etc
           engine   : 'python' or 'r.' Defines the software to use to run the 
                       regressions
        '''
        assert isinstance(formula, str) or isinstance(formula, dict),\
            "'formula' must be a string or dictionary"
        # 
        formula = {"Model 1": formula} if isinstance(formula, str) else formula
        # 
        self.data=eDataFrame(data)
        self.regression = self.__regression__(formula, family, na, engine,
                                              *args, **kws)
    # =====================================================
    # Methods
    # =====================================================
    def get_info(self, model=None):
        if model:
            self.__get_info__(model)
        else:
            for model in self.regression.keys():
                self.__get_info__(model)
        
    def print_omitted(self, summary=True, get=False):
        res = {}
        for label, info in self.regression.items():
            vars = self.__get_variables__(info['formula'])
            self.get_info(model=label)
            idx = info['omitted']
            resi=None
            if idx:
                resi=self.data.loc[idx, vars]
                resi=resi.summary(vars) if summary else resi
            if get:
                res[label]=resi
            else:
                print(resi)
        return res if res else None

    ## -------
    ## summary
    ## -------
    def summary_default(self, model=None):
        if model:
            print(self.regression[model]['sumamry_default'])
        else:
            for model in self.regression.keys():
                print(self.regression[model]['summary_default'])
            
    def summary_tidy(self, model=None, join=True, get=True):
        res = eDataFrame() if join or model else {}
        if model:
            res = self.regression[model]['sumamry_tidy']
            res['model']=model
        else:
            for model in self.regression.keys():
                resi=self.regression[model]['summary_tidy']
                resi['model']=model
                if join:
                    res = res.bind_row(resi)
                else:
                    res[model]=resi
        return res if get else print(res)

    ## ----
    ## plot
    ## ---- 
    def plot(self,
             model=None,
             terms=None,
             which='coef',
             title=None,
             sort=True,
             # 
             # Model parameters
             coefs=None,
             model_names=None,
             show_p = True,
             scale=False,
             ci_level=.95,
             # 
             show_data=False,
             grid=False,
             jitter=0,
             # 
             # colors
             palette='Set1',
             # 
             legend_ncol=1,
             legend_position = 'top',
             legend_direction= 'vertical',
             legend_title="",
             legend_title_position = 'top',
             legend_ha=0,
             legend_va=0,
             # 
             xlab=None,
             ylab=None,
             ylab_wrap=1000,
             xlab_wrap=1000,
             coord_flip=False,
             wrap_title = 50,
             wrap_labels = 25,
             #
             theme='bw',
             # 
             # Plot devide
             height=None,
             width=None,
             fn=None
             #
             ):
        '''
        Input
           which : a string with 'coef' for coefficient plot or 'pred' for
                   predictive plot
        '''
        if height or width:
            height = height if height else 7
            width = width if width else 8
            grd.dev_new(height=height, width=width)
        args=locals()
        if   self.engine=='r' and which=='coef':
            g= self.__plot_coefr__(args)
        elif self.engine=='r' and which=='pred':
            g= self.__plot_predr__(args)
        # 
        elif self.engine=='python':
            g= print("To be implemented")
        return g


    def dev_off(self):
        grd.dev_off()

    # =====================================================
    # Ancillary
    # =====================================================
    def __regression__(self, formula, family, na, engine, *args, **kws):
        self.engine=engine
        if engine=='python':
            res = self.__dispatcher__(formula, family, na, *args, **kws)
        else:
            res = self.__dispatcherr__(formula, family, na, *args, **kws)
        return res

    def __dispatcher__(self, formula, family, na, *args, **kws):
        if family=='gaussian':
            return self.__run_gaussian__(formula, family, na, *args, **kws)
        if family=='binomial':
            return self.__run_binomial__(formula, family, na, *args, **kws)
        if family=='multinomial':
            return self.__run_multinomial__(formula, family, na, *args, **kws)


    def __dispatcherr__(self, formula, family, na, *args, **kws):
        if family=='gaussian':
            return self.__run_gaussianr__(formula, family, na, *args, **kws)
        if family=='binomial':
            return self.__run_binomialr__(formula, family, na, *args, **kws)
        if family=='multinomial':
            return self.__run_multinomialr__(formula, family, na, *args, **kws)


    def __repr__(self):
        if self.engine == 'python':
            print("to be implemented")
        if self.engine == 'r':
            self.__reprr__()
        return ""

    def __get_variables__(self, formula):
        formula=re.sub(pattern=' ', repl='', string=formula)
        vars = re.split("~|\+|\||\*", formula)
        vars = [v for v in vars if v!='']
        return vars


    def __get_data__(self, formula, na):
        vars = self.__get_variables__(formula)
        if na=='omit':
            res = self.data[vars].dropna(subset=None, axis=0)
            omitted = [index for index, row in
                       self.data[vars].iterrows() if row.isnull().any()]
        return res, omitted


    def __get_info__(self, model):
        info = self.regression[model]
        print(f"\nRegression: {model}")
        print(f"Formula: {info['formula']}")
        print(f"Family: {info['family']}")
        print(f"Function: {info['function']}")


    def __collect_summary__(self, fit, formula, family, function,
                            summary_tidy=None,
                            default_summary=None,
                            summary_default=None,
                            omitted=None):
        dic = {"formula":formula,
               'family':family,
               "function":function,
               "fit":fit,
               'omitted':omitted,
               "summary_tidy":summary_tidy,
               "summary_default":summary_default,
               'Obs':None,
               'aic':None,
               'bic':None,
               'r2':None,#1-(fit.deviance/ fit.null_deviance),
               'rmse':None
               }
        return dic


    def __get_fit_list__(self, models):
        if isinstance(models, str):
            models = [models]
        if not models:
            mods=[]
            for k, info in self.regression.items():
                mods.append(info['fit'])
        else:
            for model in models:
                mods = self.regression[model]['fit']
        return mods

    # =====================================================
    # Python Engine
    # =====================================================
    def __run_gaussian__(self, *args, **kws):
        print("Gaussian regression not implemented yet")

    def __run_multinomial__(self, *args, **kws):
        print("Multinomial regression not implemented yet")

    def __run_binomial__(self, *args, **kws):
        formulas = kws.get('formulas', None)
        if not formulas:
            formulas = {'Model 1':kws.get('formula', None)}
        tab=pd.DataFrame()
        for idx, (label, formula) in enumerate(formulas.items()):
            mod = glm(formula, data=self.data, family=family.Binomial())
            fit = mod.fit()
            tmp = pd.DataFrame({'idx':idx,
                                'label':label,
                                "formula":formula,
                                'family':kws.get("family", 'gaussian'),
                                "mod":[mod],
                                "fit":[fit],
                                "summ_tidy":[self.__get_summary1__(fit)],
                                "summ_default":[self.__get_summary2__(fit)],
                                "summ3":[self.__get_summary3__(mod, fit)],
                                'Obs':fit.nobs,
                                'aic':fit.aic,
                                'bic':fit.bic,
                                'r2':1-(fit.deviance/ fit.null_deviance),
                                # 'rmse':np.sqrt(np.mean((self['y']-fit.predict())**2))
                                })
            tab=pd.concat([tab, tmp], axis=0, ignore_index=True)
        return tab
        

    # =====================================================
    # Summary
    # =====================================================
    def __get_summary1__(self, fit):
        tab=fit.summary2().tables[1].reset_index( drop=False)
        tab.rename(columns={'index':'term'}, inplace=True)
        return eDataFrame(tab)


    def  __get_summary2__(self, fit):
        tab=summary_col(fit, stars=True).tables[0]
        tab = pd.DataFrame(tab).reset_index( drop=False)
        tab.rename(columns={'index':'term'}, inplace=True)
        return eDataFrame(tab)


    def  __get_summary3__(self, mod, fit, digits=4):
        summ = self.__get_summary1__(fit)
        depvar=mod.endog_names
        res = (summ
               .rename(columns={'P>|z|':'pvalue'}, inplace=False)
               .mutate({'Coef': lambda x: round(x['Coef.'], digits).astype(str)})
               .case_when('Coef.', {
                   f"(pvalue<0.001)": f"Coef+'***'",
                   f"(pvalue<0.01)": f"Coef+'**'",
                   f"(pvalue<0.05)": f"Coef+'*'",
                   f"(pvalue<0.1)": f"Coef+'.'",
                   f"(pvalue>0.1)": f"Coef",
               })
               .mutate({'ci': lambda x: [f"({round(xlower, digits)}, {round(xupper, digits)})" for
                                         xlower, xupper in zip(x['[0.025'], x['0.975]'])]})
               .filter(['term', 'Coef.', 'ci'])
               .pivot_longer(id_vars='term', value_vars=None,
                             var_name='stat', value_name=depvar,
                             ignore_index=False)
               .reset_index(drop=False)
               .sort_values(['index', 'stat'], ascending=True)
               .case_when('term', {
                   f"(stat=='ci')": f"''",
                   f"(stat=='Coef.')": f"term",
               })
               .reset_index(drop=True)
               .drop(['index', 'stat'], axis=1)
               )
        return eDataFrame(res)

        

    def table(self, model_names='label', ci=True,
              include_stats=['r2', 'bic', 'Obs'],
              stars=True, labels=None):
        '''
        Create table with all models
        
        Input
           model_names a string with 'label', 'depvar', 'both'
        '''
        tab_final = pd.DataFrame()
        for i, (idx, row) in enumerate(self.reg.iterrows()):
            if ci:
                tab = row.summ3
            else:
                tab = row.summ_default
            depvar=row['mod'].endog_names
            info=f"{row.label}"
            tab = (tab
                   .replace({'term':{'':np.nan }}, regex=False)
                   .fillna(method='ffill')
                   .assign(estimate=['mean', 'se']*(int(tab.shape[0]/2)))
                   # .append({'term':"RMSE",
                   #          'estimate':'zz',
                   #          depvar:row.rmse}, ignore_index=True)
                   .rename(columns={depvar:info}, inplace=False)
                   )
            if i==0:
                tab_final=tab
            else:
                tab_final = (tab_final
                             .merge(tab, how='outer', on=['term', 'estimate'])
                             )
        terms = []
        for term_before, term_after in zip(tab_final.term[0:-1],
                                           tab_final.term[1:]):
            if term_before==term_after:
                terms.append(term_after)
            else:
                terms.append("")
        terms.append("")
        tab_final['term'] = terms
        # 
        # get stats
        if include_stats:
            assert all([stat in self.reg.columns for stat in include_stats]),(
                "Stats in 'include_stat' was not computed"
            )
            stats = (self.reg
                 .filter(['label']+include_stats)
                 .melt(id_vars='label', value_vars=None,
                       var_name=None, value_name='value', col_level=None)
                 .pivot_table(values='value', index='variable', columns='label', aggfunc="sum")
                 .reset_index( drop=False)
                 .rename(columns={'variable':'term'}, inplace=False)
                 )
            tab_final=pd.concat([tab_final, stats], axis=0)
                
        if model_names=='label':
            info=f"{row.label}"
        elif model_names=='depvar':
            info=f"{depvar}"
        elif model_names=='both':
            info=f"{row.label} ({depvar})"
        tab_final=(tab_final
                   .drop(['estimate'], axis=1))
        if labels:
            tab_final = (tab_final
                         .replace({'term':labels} , regex=True, inplace=False)
                         .fillna("")
                         )
        #
        if not stars:
           tab_final = tab_final.replace({"\**":""} , regex=True, inplace=False) 
        return eDataFrame(tab_final)

    def pull(self, model_index=None):
        res = eDataFrame()
        if not model_index:
            model_index = list(range(self.reg.nrow))
        for idx in model_index:
            rest = self.reg.loc[idx, 'summ_tidy']
            rest['label'] = self.reg.loc[idx, 'label']
            res = res.bind_row(rest)
        return res

    def prediction(self, newdata, model_index=1):
        '''
        Get predicted values

        Input:
           newdata data frame with the same columns used to run the regression

        Output:
            DataFrame with predicted values

        '''
        fit = self.reg["fit"][model_index-1]
        pred = fit.get_prediction(newdata)
        pred_mean = eDataFrame({"pred":pred.predicted_mean})
        ci = eDataFrame(pred.conf_int(), columns=['pred_lower',
                                                  'pred_upper'])
        pred=(newdata
              .bind_col(pred_mean,  ignore_index=False)
              .bind_col(ci,  ignore_index=False) )
        return eDataFrame(pred)
        



            
    # =====================================================
    # Plots
    # =====================================================
    def __plot_coef_python__(self, model_index=1, sort=True, title=None, labels=None):
        '''
        Plot regression coefficients

        Input
           model_index the index of the info in the regression object
           sort boolean, if true the values are sorted in the plot
           title string, the title of the plot
           labels dict, the labels of the variables. Accepts regular expression.
        '''
        tab = self.reg.summ_tidy[model_index-1].loc[lambda x:
                                              ~x['term'].str.contains(".tercept")]
        if sort:
            tab = tab.sort_values(['Coef.'], ascending=True)
        if labels:
            tab = tab.replace({'term':labels} , regex=True, inplace=False)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 6], tight_layout=True)
        #
        ax.errorbar(tab['Coef.'], tab['term'], xerr=tab['Std.Err.'], fmt='.k')
        ax.axvline(0, ymin=0, ymax=1, color='red', linestyle='--', linewidth=1)
        # grid
        ax.grid(b=None, which='major', axis='both', linestyle='-', alpha=.3)
        ax.set_axisbelow(True) # to put the grid below the plot
        #
        plt.subplots_adjust(top=.78)
        xcoord=-.0
        ycoord=1.01
        yoffset=.07
        ax.annotate(title, xy=(xcoord, ycoord),  xytext=(xcoord, ycoord),
                    xycoords='axes fraction', size=10, alpha=.6)
        plt.ion()
        plt.show()
        return ax


    def __plot_pred_python__(self, x, newdata=None, model_index=1,
                  linecolor='black', show_pts=True, pts_color='black',
                  label=None, grid=True, legend=True,
                  ci_linetype="--", ci_color='grey',
                  figsize=[10, 6], ax=None):
        '''
        Plot predicted values
        '''
        # 
        if not ax:
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=figsize, tight_layout=True)
        #
        pred = self.prediction(newdata, model_index=model_index)
        xpred = pred[x]
        ax.plot(xpred, pred.pred, color=linecolor, label=label)
        ax.plot(xpred, pred.pred_lower, color=ci_color, linestyle=ci_linetype)
        ax.plot(xpred, pred.pred_upper, color=ci_color, linestyle=ci_linetype)
        ax.fill_between(x=xpred, y1=pred.pred_lower, y2=pred.pred_upper,
                        color=ci_color, alpha=.3)
        if show_pts:
            xpts = self.data[x]
            y = self.reg['mod'][model_index-1].endog_names
            ypts = self.data[y]
            ax.scatter(xpts, ypts, color=pts_color, alpha=.4)
        # -------
        # Splines (axes lines)
        # -------
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if grid:
            # grid
            ax.grid(b=None, which='major', axis='both',
                    linestyle='-', alpha=.3)
            ax.set_axisbelow(True) # to put the grid below the plot
        if legend:
            ax.legend()

        return ax

# }}}
# {{{ PCA                 }}}

class pca():
    def __init__(self, data, *args, **kws):
        '''
        Fit a PCA using prince module 

        Inputs:
            data    : pandas DataFrame with the variables to compute the PCAs
            ncomp   : number of PCA components. Default equals the number of 
                      variables
            niter   : The number of iterations used for computing the SVD.
            inplace : Whether to perform the computations inplace or not. 
                        Default is True
            seed    : seed for the random state
            invert  : list with indexes of the principal components to 
                      invert the axis (multiply by -1)
        Outputs:
            PCA object with fitted info and scores
        '''
        # arguments 
        ncomp=kws.get("ncomp", data.shape[1])
        niter=kws.get('niter', 10)
        inplace=kws.get('inplace', True)
        seed=kws.get('seed', 666)
        invert=kws.get('invert', None)
        # pca
        info = prince.PCA(
            n_components=ncomp,
            n_iter=niter,
            rescale_with_mean=True,
            rescale_with_std=True,
            copy=inplace,
            check_input=True,
            engine='sklearn',
            random_state=seed
        )
        self.data=data
        self.invert=invert
        self.fit = info.fit(data)
        self.scores = self.fit.fit_transform(data)
        self.scores.columns=[f"Comp{str(i)}" for i in
                             range(1,self.scores.shape[1]+1)]
        if invert:
            invvalues=all([x==1 or x==-1 for x in invert])
            assert isinstance(invert, list), "'invert' must be a list"
            assert invvalues, "Value in 'invert' must be either 1 or -1"
            assert len(invert)==self.scores.shape[1], "'invest' must have as "+\
            f"many elements as the components of the PCAs computed: "+\
            f"{self.scores.shape[1]}"
            for i, mult in enumerate(invert):
                self.scores[f"Comp{i+1}"]=mult*self.scores[f"Comp{i+1}"] 
        self.scores = eDataFrame(self.scores)
        self.correlations = self.fit.column_correlations(self.data)
        self.correlations.columns = [f"Comp{i+1}" for i in range(self.scores.ncol)]
        self.correlations = eDataFrame(self.correlations)

    # =====================================================
    # plots
    # =====================================================
    def plot(self, *args, **kws):
        '''
        Plot PCA results

        Input
           fn filename with path to save the plot. If omitted, plot will be 
              displayed but not saved
        
        Output
           A plot with PCA summary results. If
        '''
        assert self.fit.n_components>1, ("You need at least 2 components to "+\
                                         "generate the plots")
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=[10, 6], tight_layout=True)
        axc=ax[0][0]
        self.plot_eigenvalue(axc, **kws)
        axc=ax[0][1]
        self.plot_propvar(axc, **kws)
        axc=ax[1][0]
        self.plot_corr(axc, **kws)
        axc=ax[1][1]
        self.plot_scores(axc, **kws)
        # saving
        fn = kws.get("fn", None)
        if fn:
            plt.savefig(fn)
            print(f'File {fn} saved!', flush=True)
    

    def plot_eigenvalue(self, ax, *args, **kws):
        y=self.fit.eigenvalues_
        x=[str(i) for i in range(1, len(y)+1)]
        ax.scatter(x, y, color="black", alpha=.4)
        ax.plot(x, y, color="black", alpha=.6)
        ax.axhline(1, xmin=0, xmax=1, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Eigenvalue")
        self.plot_asthetics(ax)

    def plot_propvar(self, ax, *args, **kws):
        y=self.fit.explained_inertia_
        x=[str(i) for i in range(1, len(y)+1)]
        g = sns.barplot(x=x, y=y, ax=ax, color='grey')
        for p in g.patches:
            g.annotate('{:.2f}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()),
                       ha='center', va='bottom',
                       color= 'black')
        ax.set_ylabel('Proportion of Variance')
        ax.set_xlabel('Principal Component')
        self.plot_asthetics(ax)

    def plot_asthetics(self, ax, *args, **kws):
        ax.grid(b=None, which='major', axis='y', linestyle=':', alpha=.3)
        ax.set_axisbelow(True) # to put the grid below the plot
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


    def plot_corr(self, ax, *args, **kws):
        '''
        Plot correlation between components and original variables.

        Inputs:
            x  integer with component to plot on x-axis
            y  integer with component to plot on y-axis
        '''
        x=kws.get('x', 1)
        y=kws.get('y', 2)
        x-=1
        y-=1
        # 
        corrmat = self.fit.column_correlations(self.data)
        if self.invert:
            for i, mult in enumerate(self.invert):
                corrmat.iloc[:,i]=mult*corrmat.iloc[:,i]
        corrmat=corrmat.reset_index(drop=False)
        ax.scatter(x=corrmat[x], y=corrmat[y], s=10, color='black', alpha=.4)
        ax.axvline(0, ymin=0, ymax=1, color='red', linestyle='--', linewidth=1)
        ax.axhline(0, xmin=0, xmax=1, color='red', linestyle='--', linewidth=1)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        for i, row in corrmat.iterrows():
            ax.text(row[x], row[y], str(row['index']), ha='left', va='bottom')
        # 
        variances=self.fit.explained_inertia_
        xperc=round(100*variances[x], 2)
        yperc=round(100*variances[y], 2)
        ax.set_xlabel(f'Component {x+1}\n(Captures {xperc}% of variance)')
        ax.set_ylabel(f'Component {y+1}\n(Captures {yperc}% of variance)')
        ax.set_title("Correlations with Original Variables")
        self.plot_asthetics(ax)

    def plot_scores(self, ax, *args, **kws):
        x=kws.get('x', 1)
        y=kws.get('y', 2)
        color_labels=kws.get('color_labels', None)
        color=kws.get('color','black')
        size=kws.get('size', 13)
        alpha=kws.get('alpha', .4)
        data=kws.get('data', self.scores)
        x-=1
        y-=1
        if color_labels:
            # not sure how this plot is workint in the prince package
            pass
            # labels=data[color_labels]
            # data=data.drop([color_labels], axis=1)
            # self.fit.plot_row_coordinates(
            #     X=data,
            #     color_labels=labels,
            #     x_component=x,
            #     y_component=y,
            #     labels=None,
            #     ellipse_outline=True,
            #     ellipse_fill=True,
            #     show_points=True,
            #     s=size,
            #     alpha=alpha,
            #     ax=ax,
            # )
            # ax.legend()
        ax.scatter(data.iloc[:,x],
                   data.iloc[:,y],
                   color=color,
                   s=size, alpha=alpha)
        # grid
        ax.grid(b=None, which='major', axis='x', linestyle='-', alpha=.3)
        ax.set_axisbelow(True) # to put the grid below the plot
        variances=self.fit.explained_inertia_
        xperc=round(100*variances[x], 2)
        yperc=round(100*variances[y], 2)
        ax.set_xlabel(f'Component {x+1}\n(Captures {xperc}% of variance)')
        ax.set_ylabel(f'Component {y+1}\n(Captures {yperc}% of variance)')
        ax.set_title("")
        self.plot_asthetics(ax)


# }}}
