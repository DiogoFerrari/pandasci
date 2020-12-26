import matplotlib.pyplot as plt
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
        self.reg = self.__run_regresion__(*args, **kws)
    

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
            return self.__run_binomial__(formulas, *args, **kws)
            

    def __run_gaussian__(self, *args, **kws):
        print("Gaussian regression not implemented yet")


    def __run_binomial__(self, formulas, *args, **kws):
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
                                'Obs':fit.nobs,
                                'aic':fit.aic,
                                'bic':fit.bic,
                                'r2':1-(fit.deviance/ fit.null_deviance),
                                'rmse':np.sqrt(np.mean((df['y']-fit.predict())**2))
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
        

    def __get_summary1__(self, fit):
        tab=fit.summary2().tables[1].reset_index( drop=False)
        tab.rename(columns={'index':'term'}, inplace=True)
        return tab

    def  __get_summary2__(self, fit):
        tab=summary_col(fit).tables[0]
        tab = pd.DataFrame(tab).reset_index( drop=False)
        tab.rename(columns={'index':'term'}, inplace=True)
        return tab
        
    def table(self, model_names='label',
              include_stats=['r2', 'bic', 'rmse', 'Obs']):
        '''
        Create table with all models
        
        Input
           model_names a string with 'label', 'depvar', 'both'
        '''
        tab_final = pd.DataFrame()
        for i, (idx, row) in enumerate(self.reg.iterrows()):
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
        return tab_final
            


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
