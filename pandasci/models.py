from .ds import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import prince # for PCA
from statsmodels.formula.api import glm as glm
from statsmodels.formula.api import logit as logit
from statsmodels.api import families as family
import warnings
import itertools
from statsmodels.iolib.summary2 import summary_col


# {{{ regression models }}}

class regression():
    def __init__(self, data, *args, **kws):
        '''
        Run regression models
        
        Input
           depvar      : string with dependent variable
           indvars     : list with independent variables
           formula     : string with formula
           formulas    : a dictionary with label of the models (key) and 
                         regression formulas (values)
           family      : string, 'gaussian', 'binomial'
        '''
        depvar   = kws.get('depvar', False)
        indvars  = kws.get('indvars', False)
        formula  = kws.get('formula', False)
        formulas = kws.get('formulas', False)
        # 
        if not formula and not formulas:
            assert depvar and indvars, ("Either formula(s) or 'depvars' and "+\
                                        "'indvar' must must be provided")
        if formula and formulas:
            warnings.warn("\n\nWhen both 'fomula' and 'formulas' "+\
                          "are provided, 'formula' is ignored and 'formulas' is "+\
                          'used.', RuntimeWarning)
        
            
        if (depvar and indvars and (formula or formulas)):
            warnings.warn("\n\nWhen both 'indvars' and 'depvar' "+\
                          "are provided, formula(s) are ignored and variables "+\
                          'are used instead.', RuntimeWarning)
                                                                   

        self.data=data
        self.reg = eDataFrame(self.__run_regresion__(*args, **kws))
    

    def __run_regresion__(self, *args, **kws):
        depvar   = kws.get('depvar', False)
        indvars  = kws.get('indvars', False)
        formula  = kws.get('formula', False)
        formulas = kws.get('formulas', False)
        if indvars and depvar:
            formulas = self.__get_formulas__(depvar=depvar, indvars=indvars)
        elif not formulas:
            formulas = {'Model 1':formula}
        # 
        family=kws.get("family", 'gaussian')
        if family=='gaussian':
            return self.__run_gaussian__(formulas, *args, **kws)
        if family=='binomial':
            return self.__run_binomial__(*args, **kws)
            

    def __run_gaussian__(self, *args, **kws):
        print("Gaussian regression not implemented yet")


    def __run_binomial__(self, *args, **kws):
        formulas = kws.get('formulas', None)
        if not formulas:
            formulas = {'Model 1':kws.get('formula', None)}
        tab=pd.DataFrame()
        for label, formula in formulas.items():
            mod = glm(formula, data=self.data, family=family.Binomial())
            fit = mod.fit()
            tmp = pd.DataFrame({'label':label,
                                "formula":formula,
                                'family':kws.get("family", 'gaussian'),
                                "mod":[mod],
                                "fit":[fit],
                                "summ1":[self.__get_summary1__(fit)],
                                "summ2":[self.__get_summary2__(fit)],
                                "summ3":[self.__get_summary3__(mod, fit)],
                                'Obs':fit.nobs,
                                'aic':fit.aic,
                                'bic':fit.bic,
                                'r2':1-(fit.deviance/ fit.null_deviance),
                                # 'rmse':np.sqrt(np.mean((self['y']-fit.predict())**2))
                                })
            tab=pd.concat([tab, tmp], axis=0, ignore_index=True)
        return tab


    def __get_formulas__(self, depvar, indvars):
        indvars_combinations = [itertools.combinations(indvars, i) for i in
                                range(1, len(indvars)+1)]
        formulas={}
        i=1
        for k_vars in indvars_combinations :
            for vars in k_vars:
                if len(vars)>1:
                    indvars = " + ".join(list([str(v) for v in vars]))
                    formula=f"{depvar} ~ {indvars}"
                else:
                    formula=f"{depvar} ~ {vars[0]}"
                formulas[f'Model {i}']=formula
                i+=1
        return formulas
        

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
                tab = row.summ2
            depvar=row['mod'].endog_names
            model=f"{row.label}"
            tab = (tab
                   .replace({'term':{'':np.nan }}, regex=False)
                   .fillna(method='ffill')
                   .assign(estimate=['mean', 'se']*(int(tab.shape[0]/2)))
                   # .append({'term':"RMSE",
                   #          'estimate':'zz',
                   #          depvar:row.rmse}, ignore_index=True)
                   .rename(columns={depvar:model}, inplace=False)
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
            model=f"{row.label}"
        elif model_names=='depvar':
            model=f"{depvar}"
        elif model_names=='both':
            model=f"{row.label} ({depvar})"
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
    def plot_coef(self, model_index=1, sort=True, title=None, labels=None):
        '''
        Plot regression coefficients

        Input
           model_index the index of the model in the regression object
           sort boolean, if true the values are sorted in the plot
           title string, the title of the plot
           labels dict, the labels of the variables. Accepts regular expression.
        '''
        tab = self.reg.summ1[model_index-1].loc[lambda x:
                                              ~x['term'].str.contains(".tercept")]
        if sort:
            tab = tab.sort_values(['Coef.'], ascending=True)
        if labels:
            tab = tab.replace({'term':labels} , regex=True, inplace=False)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 6], tight_layout=True)
        #
        ax.errorbar(tab['Coef.'], tab['term'], xerr=tab['Std.Err.'],
                    fmt='.k', )
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


    def plot_pred(self, x, newdata=None, model_index=1,
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
            PCA object with fitted model and scores
        '''
        # arguments 
        ncomp=kws.get("ncomp", data.shape[1])
        niter=kws.get('niter', 10)
        inplace=kws.get('inplace', True)
        seed=kws.get('seed', 666)
        invert=kws.get('invert', None)
        # pca
        model = prince.PCA(
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
        self.fit = model.fit(data)
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
                

    # =====================================================
    # plots
    # =====================================================
    def plot(self, *args, **kws):
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
