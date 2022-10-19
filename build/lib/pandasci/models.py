from .ds import *
from .rutils import *
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype
from pandas.api.types import is_bool_dtype, is_categorical_dtype
import numpy as np
import seaborn as sns
import prince # for PCA
from statsmodels.formula.api import glm as glm
from statsmodels.formula.api import logit as logit
from statsmodels.stats import proportion as pwr2prop
from statsmodels.api import families as family
import itertools
from itertools import product
from statsmodels.iolib.summary2 import summary_col
from scipy.stats import norm as qnorm
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pathlib
#
ru=rutils() ## my tools for R<->python

# * R packages

print("Loading R packages for module models of pandasci. It may take a while...")

# supress warnings
import warnings
warnings.filterwarnings("ignore")
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)   # will display errors, but not warnings
# 
# 
import rpy2.robjects as robj
import rpy2.rlike.container as rlc
import rpy2.robjects.lib.ggplot2 as gg
from rpy2.robjects import r, FloatVector, pandas2ri, StrVector
from rpy2.robjects.packages import importr
from rpy2.interactive import process_revents # to refresh graphical device
try:
    process_revents.start()                      # to refresh graphical device
except (OSError, IOError, BaseException) as e:
    pass
from rpy2.robjects.packages import data as datar
# import data from package: datar(pkg as loaded into python).fetch('data name')['data name']
# 
pandas2ri.activate()
stats = importr_or_install('stats')
base = importr_or_install('base')
utils = importr_or_install("utils")
ggtxt = importr_or_install("ggtext")
jtools=importr_or_install("jtools")
broom=importr_or_install("broom")
broomh=importr_or_install("broom.helpers")
metrics=importr_or_install("Metrics")
modelsummary=importr_or_install("modelsummary")
nnet=importr_or_install("nnet")
rpact = importr_or_install("rpact")
latex2exp=importr_or_install("latex2exp")
gridExtra = importr_or_install("gridExtra")
patchwork = importr_or_install("patchwork")
infer=importr_or_install("infer")
citools=importr_or_install("ciTools")
ggthemes=importr_or_install("ggthemes")
formula_tools=importr_or_install("formula.tools")

print("R packages loaded!")

# * functions
def ggtheme():
    g =gg.theme(
             ## ------
             ## legend
             ## ------ 
             legend_position = "right",
             # legend_position = [0.12, .96],
             legend_justification = FloatVector([0, 1]),
             legend_direction='vertical',
             # legend_direction='horizontal',
             legend_title = gg.element_text( size=11),
             # legend_text  = gg.element_text( size=10),
             # legend_text_legend=element_text(size=10),
             # legend_text_colorbar=None,
             # legend_box=None,
             # legend_box_margin=None,
             # legend_box_just=None,
             # legend_key_width=None,
             # legend_key_height=None,
             # legend_key_size=None,
             # legend_margin=None,
             # legend_box_spacing=None,
             # legend_spacing=None,
             # legend_title_align=None,
             # legend_entry_spacing_x=None,
             # legend_entry_spacing_y=None,
             # legend_entry_spacing=None,
             # legend_key=None,
             # legend_background=None,
             # legend_box_background=None,
             strip_background = gg.element_rect(colour="transparent",
                                                fill='transparent'),
             # strip_placement = "outside",
             strip_text_x        = gg.element_text(size=10, face='bold', hjust=0),
             strip_text_y        = gg.element_text(size=9, face="bold", vjust=0,
                                                   angle=-90),
             ##panel_grid_major  = element_blank(),
             # panel_grid_minor_x  = gg.element_blank(),
             # panel_grid_major_x  = gg.element_blank(),
             # panel_grid_minor_y  = gg.element_blank(),
             # panel_grid_major_y  = gg.element_blank(),
             panel_grid_minor_y  = gg.element_line(colour="grey", size=.3, linetype=3),
             panel_grid_major_y  = gg.element_line(colour="grey", size=.3, linetype=3),
             panel_grid_minor_x  = gg.element_line(colour="grey", size=.3, linetype=3),
             panel_grid_major_x  = gg.element_line(colour="grey", size=.3, linetype=3),
             # border 
             # ------
             panel_border      = gg.element_blank(),
             axis_line_x       = gg.element_line(colour="black", size=.2, linetype=1),
             axis_line_y       = gg.element_line(colour="black", size=.2, linetype=1),
             # axis_line_y       = gg.element_line(colour="black"),
             legend_background  = gg.element_rect(fill='transparent'),
             # legend_key_height = grid::unit(.1, "cm"),
             # legend_key_width  = grid::unit(.8, "cm")
             axis_ticks_x        = gg.element_blank(),
             axis_ticks_y        = gg.element_blank(),
             axis_text_y         = ggtxt.element_markdown(),
             plot_title	         = gg.element_text(hjust=0, size = 11,
                                                   colour='grey40', face='bold'),
             plot_subtitle	 = gg.element_text(hjust=0, size = 9,
                                                   colour='grey30'),
             axis_title_y        = gg.element_text(size=10, angle=90),
        )
    return g
def ggguides():
    g= gg.guides(colour = gg.guide_legend(title_position = "top",
                                          title_hjust=0),
                 fill = gg.guide_legend(title_position = "top",
                                        title_hjust=0),
                 shape = gg.guide_legend(title_position = "top",
                                         title_hjust=0),
                 linetype = gg.guide_legend(title_position = "top",
                                            title_hjust=0)
		 )
    return g        

# * Regression models
# *** class
class regression():
    def __init__(self, models=None, data=None,
                 na='omit', engine='r', *args, **kws):
        '''
        Run regression models
        
        Input
           models   : a dictionary of tuples. It can be either
                      - (formula str or dict, family str)
                      - (formula str or dict, family str, DataFrame)
        
                      formula str or dict
                      -------------------
                      The first element can be a string or a dictionary.
                      If a string is used, it must be a R-like
                      regression formula, i.e.,  <depvar> ~ <regressors>.

                      If a dictionary is used, it can have the following keys:values
                          - 'output'       : Required. The value must a string
                                             with the name of the output variable
                          - 'inputs'       : Required. For details,
                                             see models.models_utils().build_formula
                                             The basic structure are
                                             a. inputs = [...]
                                             b. inputs = {'oldnames': "newnames"}
                                             c. inputs = {'group': [...]}
                                             d. inputs = {'group': {'oldnames': "newnames"}}
                                         
                          - 'interactions' : a list of tuples of strings with
                                             the names of the interactive terms.
                                             Note that the strings must match
                                             the labels of the variables in
                                             case they are renamed in the
                                             'input.'
                                             The elements in each tuple will be
                                             an interaction
                                             a. formula
                                             b. interactions = [(...)]
                                             d. interactions = {'group': [(...)]}

                          - 'clusters'     : TBD

                      See example below
        
                      family
                      ------
                      The second string must be the family of the dependent
                      variable: gaussian, binomial, multinomial, poisson,
                      negative binomial, etc.

                      DataFrame
                      ---------
                      The third element, if used, must be the a DataFrame with
                      the data. In that case, the estimation will use the same
                      data for all models. A specific data for each model
                      can be provided using the tuple with the data included

           data     : a DataFrame. Only used if the data is not provided in the
                      model tuple. See argument 'models'


           na       : Not implemented. Only option for now is to exclude NAs

           engine   : 'python' or 'r.' Defines the software to use to run the 
                       regressions

        Example
        -------
        # from pandasci import ds
        # from pandasci import models as md
        from numpy.random import uniform as runif
        from numpy.random import choice as rsample
        n=100
        df = ds.eDataFrame({"y": runif(-3, 3, n),
                            "x1":runif(-3, 3, n),
                            "x2":runif(-3, 3, n),
                            "x3":runif(-3, 3, n),
                            "z": runif(-3, 3, n),
                            })
        df2 = df.mutate({'y2': rsample([0, 1], n)})

        mod = md.regression(
            models= {'model 1' : ("y  ~ x1"     , 'gaussian', df),
                     'model 2' : ("y2 ~ x1 + x2", 'binomial', df2)},
        )
        mod

        # Note: this model will run a gaussian linear model with the following
        # formula:
        #     y ~ x1 + x2 + x1*x2 + z*x3
        vars={
            'output': 'y',
            'inputs': ['x1', 'x2'],
            'interactions': [('x1', 'x2'), ('z', 'x3')]
        }
        mod = md.regression(models= {'model 1' : (vars, 'gaussian', df)})
        mod
        '''
        # 
        self.multinomial        = None
        self.labels             = []
        self.models             = []
        self.engine             = {}
        self.na                 = {}
        self.formula            = {}
        self.family             = {}
        self.data               = {}
        self.results            = eDataFrame()
        self.variables          = eDataFrame()
        self.variables_dict_raw = {}
        if models:
            self.add_models(models=models, data=data, 
                            na=na, engine=engine, *args, **kws)

# *** frontend
# **** core

    def add_models(self, models, data=None, na='omit',
                   engine='r', *args, **kws):
        '''
        Check docstring for pandasci.models.regression()
        '''
        assert isinstance(models, dict), "'formula' must be a dictionary of tuples"
        models = self.__build_formula_and_get_variables_dict__(models)
        self.__check_multinomial__(models)
        self.__set_engine__(models, engine)
        self.__set_na_action__(models, na)
        self.__set_model_info__(models, *args, **kws)
        self.__set_labels__(models)
        # must keep this order due to dependenci
        self.__build_coef_dict__()
        self.__set_data__(models, data, *args, **kws)
        self.__build_coef_dict_expand_categories__()
        self.__build_coef_df__()
        # 
        self.__run_regressions__(models, *args, **kws)
        # 
        # this must be after running the regression and keep this order
        # self.__build_variables_df__()
        # self.__get_var_group__()
        # self.__coefs_default__()


# **** summary

    def summary(self, models=None, *args, **kws):
        '''
        Summarize the estimation
        
        Input 
        -----
        models a string, list, or dictionary with the name of the models to
               print or save the summary. If a dictionary is used,
               the keys must the the model labels and the values the
               new labels to use in the summary.
               If None (default), print all models

        fn     optional. A string with the path to save the
               summary in a file.

               Note:
               -----
               If output_format is None (default), and fn is used, it saves 
               a latex (.tex), .csv, and an excel (.xlsx) file.
               All arguments of the functions that save those files can be used
               as arguments for the summary function.
               Example:
               latex    uses pandasci.ds.tolatex
               csv      uses pandasci.ds.to_csv
               excel    uses pandasci.ds.to_excel
               
               Hence, one can use arguments such as as align, caption, label, etc.
               to save the latex table.
               Ex: mod.summary(fn='table.tex', caption='A table')

        footnotes       a string or list of strings with the footnotes

        stars           boolean. If True (default), print a footnote
                        with the stars representing p-value levels

        output_format   string with the format of the output to save.
                        E.g.: 'latex', 'excel', 'csv'
                        See comments for the fn argument.

        replace a dictionary to replace content (e.g., labels of the covariates)
                using a regular expression. Passes as argument: pandas.replace(replace)
                Ex: replace={'term':{'educ':'Education'}} will replace
                all intances of 'educ' by 'education' in the column term
                (term is the column with the covariates name)

        vcov   a string or list with the variance-covariance matrix to use
               for the standard errors of the coefficients.
               It can be used to compute robust and clustered std. errors.

               For robust std errors, if a list is used,
               the models will use the covariance matrix in the order provided.
               It can be used to compute robust standard errors
               Uses R modelsummary in the background. This is an
               argument for that package.
               Ex: vcov="robust"
                   vcov=["robust", 'classica']

               For clustered std. errors, use a string with a right-hand
               formula with the variables to cluster. Note that
               the cluster variables must be included in the regression
               Ex: vcov = "~ time + state"
                   vcov = "~ time"
        
               
        
        cluster string with the name of the variables to compute
                clustered standard errors

        output_collect  boolean. If true, return a DataFrame with the summary

        '''
        res=None
        # 
        fn=kws.get("fn", None)
        if fn:
            kws.pop('fn')
            if isinstance(fn, pathlib.PurePath):
                fn = str(fn)
            fn = os.path.expanduser(fn)
        # 
        output_format=kws.get("output_format", None)
        if output_format:
            kws.pop('output_format')
        # 
        output_collect=kws.get("output_collect", None)
        replace=kws.get("replace", None)
        #
        # 
        models=self.models if not models else models
        models=[models] if isinstance(models, str) else models
        if not isinstance(models, dict ):
            models = {model:model for model in models}

        # basic table
        # save latex table
        # if output_format=='latex' or fn:
        #     tab_latex=(
        #         self.
        #         __get_summary_df__(output_collect=True, *args, **kws)
        #         # .select_cols(names={'term':'term'}|models )
        #     )
        #     print(tab_latex)
        # # 
        tab=(
            self.
            __get_summary_df__(output_collect=True, *args, **kws)
            .select_cols(names={'term':'term'}|models )
        )
        if replace:
            tab=tab.replace(replace, regex=True, inplace=False)
        tab=tab.rename_cols(columns={'term': ''}, tolower=False)
        # 
        if fn:
            if not output_format:
                output_format=['latex', 'csv', 'excel']
            elif isinstance(output_format, str):
                output_format=[output_format]
            # 
            fn_root=os.path.splitext(fn)[0]
            kws["stars"]=kws.get("stars", "+ p < 0.1, * p < 0.05, ** p < 0.01, *** p < 0.001")
            # 
            for format in output_format:
                if format=='latex':
                    kws.pop('fn', None)
                    self.__tolatex__(tab=tab, fn=f"{fn_root}.tex",
                                     *args, **kws)
                if format=='csv':
                    tabt=tab.bind_row(eDataFrame({'':[kws["stars"]]}))
                    tabt.to_csv(f"{fn_root}.csv", sep=';', index=False, decimal='.')
                if format=='excel':
                    tabt=tab.bind_row(eDataFrame({'':[kws["stars"]]}))
                    tabt.to_excel(f"{fn_root}.xlsx", sheet_name='Sheet1', index=False)
        # 
        if output_collect:
            res=tab
        else:
            tab.print()
        return res


    def get_odds(self, models=None, keep_raw=True, *args, **kws):
        '''
        Get odds ration for binomial model
        '''
        keep=['model', 'term']
        keep=keep+['estimate', 'conf.low', 'conf.high'] if keep_raw else keep
        keep+= ['odds', 'odds.low', 'odds.high',
                'perc_change', 'perc_change.low', 'perc_change.high'] 
        # 
        models=self.models if not models else models
        models=[models] if isinstance(models, str) else models
        est=(
            self
            .results
            .select_rows(query=f"model=={models}")
            .unnest(col='summ_tidy', id_vars=['model', 'family'])
            .select_rows(query=f"family=='binomial'")
            .mutate({
                'odds': lambda col: np.exp(col['estimate']),
                'odds.low': lambda col: np.exp(col['conf.low']),
                'odds.high': lambda col: np.exp(col['conf.high']),
                # 
                'perc_change': lambda col: 100*(np.exp(col['estimate'])-1),
                'perc_change.low': lambda col: 100*(np.exp(col['conf.low'])-1),
                'perc_change.high': lambda col: 100*(np.exp(col['conf.high'])-1),
            })
            
            .select_cols(names=keep)
        )
        return est

# **** predict

    def predict(self,
                models=None,
                predictor=None,
                predictor_values=robj.NULL,
                covars_at=robj.NULL,
                newdata=None
                ):
        '''
        See documentation for pandasci.models.plot_predict()
        '''
        mods=models
        predictor=predictor.replace('`', '')
        models_containing_predictor = self.__get_models_containing_variable__(predictor)
        assert models_containing_predictor, f'No model with predictor {predictor}'
        # 
        mods_list = {label:mod for label, mod in zip(self.results.label.values,
                                                     self.results['mod'].values)
                     if label in models_containing_predictor }
        rt=rtools(mods_list)
        pred=rt.predict(mods=mods,
                        predictor=predictor,
                        predictor_values=predictor_values,
                        covars_at=covars_at,
                        newdata=newdata)
        # recover the column names (dangerous zone)
        # ------------------------
        # print(pred.names())
        # columns={}
        # for varname in pred.names("."):
        #     match = process.extract(varname, self.get_coef_labels(),  limit=1)[0][0]
        #     columns[varname]=match
        # pred=pred.rename_cols(columns=columns, tolower=False)
        # print(pred.names())
        return pred



    def newdata(self,
                mods=None,
                predictor=None,
                predictor_values=robj.NULL,
                covars_at=robj.NULL,
                newdata=None
                ):
        mods_list = {label:mod for label, mod in zip(self.results.label.values,
                                                     self.results['mod'].values)}
        rt=rtools(mods_list)
        newdata=rt.newdata(mods=mods,
                           predictor=predictor,
                           predictor_values=predictor_values,
                           covars_at=covars_at
                           )
        return newdata
        
    
# **** plots

    def plot_coef(self,
                  x='estimate',
                  y='term',
                  text=False,
                  digits=2,
                  text_leg=True,
                  models=None,
                  regex=None,
                  # 
                  coefs=None,
                  coef_wrap=False,
                  # 
                  switch_axes=False,
                  # facet
                  facet=None,
                  facet_ncol=None,
                  facet_scales=None,
                  # color
                  color=None,
                  color_grey=False,
                  color_manual=False,
                  # shape
                  shape=None,
                  #
                  size=2.5,
                  errorbar_width=1,
                  alpha=1,
                  dodge=None,
                  # labels
                  xlab=None,
                  ylab=None,
                  # legend
                  leg_show       =True,
                  leg_ncol       =None,
                  leg_title      =None,
                  leg_title_pos  ='top',
                  leg_title_show =True,
                  leg_title_shape=None,
                  # title and caption
                  title    =None,
                  subtitle =None,
                  caption  =None,
                  # 
                  fn=None,
                  fn_format_fig=None,
                  fn_format_tab='xlsx',
                  fn_figsize = [8, 4]
                  ):
        '''
        Plot regression coefficients

        Input
        -----
        x             string with the x axis
        y             string with the y axis
        text          boolean or dictionary. If True, show the estimated values
                      If dictionary, it can contain the following:
                      key         value
                      ---         -----
                      size        number
                      face        "plain", "bold", "italic", "bold.italic"
                      angle       number between 0 and 1
                      vjust       number between 0 and 1       
                      hjust       number between 0 and 1

        digits        integer. Number of digits to plot if text is used. Default: 2

        text_leg      boolean. If True, add a footnote with p-value information
                      If text is used, it is also automatically used, unless
                      the user set it to False.
        
        models        string or list of strings with the labels indicating which 
                      models to plot
        regex         a regular expression to match with the labels of the 
                      models to print. Ignored if models is used.

        coefs         a list with the names of the coefficients to plot.
                      If None, it plots all the coefficients. If not,
                      the names in the list should match those that appear 
                      in the column 'term' in the <mod>.coefs_df, where
                      <mod> is an object returned by the models.regression()
                      function.
                      Note: for interactions, the coefficient labels must be
                            <varname>:<varname><category> or <varname>:<varname>

        coef_wrap    a integer to wrap the coefficient labels.

        switch_axes   boolean indicating if axes should be inverted or not

        facet         a dictionaty. The keys must be the labels to use for the facets.
                      The values must be a list of model labels. The models in the
                      list will go with the respective label

        color         same as for facet, but to group the models using a color
                      code instead 
        color_manual  a dictionary with {'category name':'color'}. Ignored if
                      color_grey is used or color not used
        color_grey    boolean indicating if a grey palette should be used for
                      the color code

        size           number with the size of the points
        errorbar_width width of the error bars
        alpha          a number between 0 and 1 indicating the transparency e for the
                       color codes
        dodge          number indicating the space between points when multiple models
                       are plotted

        xlab          string or a dictionary. IAccepts markdown. If a string,
                      it will be used as the title for the axis.
                      If a dictionary, it accepts following keys:values

                      key         value
                      ---         -----
                      title       a string with the label/title of the axis
                      size        size of the title of the axis
                      color       a string with the color of the title
                      lineheight  number with the height of the line for
                                  multiline title

        ylab           same as in xlab, but for y-axis. Accepts markdown.
        title          same as in xlab. Default is no title. Accepts markdown. 
        subtitle       same as in xlab. Default is no subtitle. Accepts markdown.
        caption        same as in xlab. Default is no caption. Accepts markdown.


        leg_show         boolean. If True, it shows the legend
        leg_ncol         integer with the number of columns for the legend
        leg_title        string with the title of the legend
        leg_title_pos    "top", 'bottom', 'left', or 'right' indicating the position
                         of the title of the legend
        leg_title_show   boolean indicating if the legend title should be displayed
        leg_title_shape  string with the title of the legend for the shapes of the points



        fn             a string with the full path for the file to save. If None
                       (default) no file is saved. The file extension will be
                       ignored. To set it, use fn_format_X.
        fn_format_fig  a string with the extension (format) to save the figure 
                       (pdf, png, etc.)
        fn_format_tab  a string with the extension (format) to save the underlying  
                       table used to generate the figure (xlsx, csv, etc)
        fn_figsize     a list with the size of the figure: [width, height]
        
        '''
        coef_labels = coefs # for compatibility
        models = self.__search_model_labels__(models, regex)

        remove_default_group_label=False
        xlab  = x if not xlab else xlab
        xlab  ={'title':xlab} if xlab and isinstance(xlab, str) else xlab
        xlab['title'] = xlab.get('title', x)
        # 
        ylab  = y if not ylab else ylab
        ylab  ={'title':ylab} if ylab and isinstance(ylab, str) else ylab
        ylab['title'] = ylab.get('title', y)
        # 
        title  = {} if not title else title
        title  ={'title':title} if title and isinstance(title, str) else title
        title['title'] = title.get('title', robj.NULL)
        # 
        subtitle  = {} if not subtitle else subtitle
        subtitle  ={'title':subtitle} if subtitle and isinstance(subtitle, str) else subtitle
        subtitle['title'] = subtitle.get('title', robj.NULL)
        # 
        caption  = {} if not caption else caption
        caption  ={'title':caption} if caption and isinstance(caption, str) else caption
        caption['title'] = caption.get('title', robj.NULL)
        # 
        sig_text  = {} if not isinstance(text, dict) else text

        if switch_axes:
            x, y=y, x
        leg_title       =robj.NULL if not leg_title  else leg_title 
        leg_title_shape =robj.NULL if not leg_title_shape  else leg_title_shape 
        title           =robj.NULL if not title  else title 
        subtitle        =robj.NULL if not subtitle  else subtitle 
        caption         =robj.NULL if not caption else caption               
        # 
        facet           =robj.NULL if not facet        else facet 
        facet_ncol      =robj.NULL if not facet_ncol   else facet_ncol 
        facet_scales    =robj.NULL if not facet_scales else facet_scales
        # 
        color           =robj.NULL if not color  else color 
        shape           =robj.NULL if not shape  else shape 
        xlab            =robj.NULL if not xlab  else xlab 
        ylab            =robj.NULL if not ylab  else ylab 

        # prepare dataframe 
        # -----------------
        models = self.labels if not models else models
        models = [models] if isinstance(models, str) else models
        facet  = facet if not facet else self.__plot_get_replace_dict__(facet) 
        color  = color if not color else self.__plot_get_replace_dict__(color) 
        shape  = shape if not shape else self.__plot_get_replace_dict__(shape) 
        # -----------------------------
        # get data 
        tmp=self.__plot_coef_prepare_data__(models, coef_labels,
                                            coef_wrap, text, color,
                                            facet, shape,
                                            digits=digits)
        if switch_axes:
            order = tmp[x].cat.categories[::-1]
            tmp[x].cat.set_categories(new_categories=order,
                                      ordered=False, inplace=True)
        # # replace * with x in interactions
        tmp=self.__plot_coef_set_interaction_str__(tmp,
                                                   interaction_str='&times;')
        # return tmp
        # -----------------------------
        # plot 
        # ----
        dodge=1/len(self.labels) if not dodge else dodge
        color='color' if color else robj.NULL
        color='label' if not color and len(models)>1 else color
        shape='shape' if shape else robj.NULL
        facet='facet' if facet else robj.NULL
        leg_ncol = len(models) if not leg_ncol else leg_ncol
        leg_title = color if leg_title_show and not leg_title else leg_title
        # 
        g = (
            gg.ggplot(tmp)
            + gg.geom_point(gg.aes_string(x=x, y=y, color=color, shape=shape),
                            size=size, alpha=alpha,
                            position=gg.position_dodge(dodge)) 
            # + gg.geom_segment(gg.aes_string(y=y, yend=y, x="-Inf", xend='Inf' ),
            #                   data=tmp.select_rows(keepna='estimate'))
            + gg.labs(
                x        = xlab.get("title", x),
                y        = ylab.get("title", y),
                color    = leg_title, 
                fill     = leg_title,
                linetype = leg_title,
                shape    = leg_title_shape,
                title    = title.get('title', robj.NULL),
                subtitle = subtitle.get('title', robj.NULL),
                caption  = caption.get('title', robj.NULL),
                )

            + gg.theme_bw()
            + self.__ggtheme__(xlab=xlab, ylab=ylab,
                               title=title, subtitle=subtitle)
            + self.__ggguides__(ncol =leg_ncol, leg_title_pos=leg_title_pos)
        )
        if color and not color_grey and not color_manual:
            g = g + ggthemes.scale_colour_tableau()
        elif color and not color_grey and color_manual:
            g = g + gg.scale_colour_manual(values=ru.dict2namedvector(color_manual)) 
        elif color and color_grey:
            g = g + gg.scale_colour_grey(start=0, end=.7, na_value="red") 

        if switch_axes:
            g = (g
                 + gg.geom_errorbar(gg.aes_string(x =x,
                                                  ymin ='conf.low',
                                                  ymax='conf.high',
                                                  color=color),
                                    size=errorbar_width,
                                    width=0, position=gg.position_dodge(dodge))
                 + gg.geom_hline(gg.aes_string(yintercept =0 ),
                                 linetype="dotted", col="red")
                 )
        else:
            g = (g
                 + gg.geom_errorbarh(gg.aes_string(y =y,
                                                   xmin ='conf.low',
                                                   xmax='conf.high',
                                                   color=color),
                                     size=errorbar_width,
                                     height =0, position=gg.position_dodge(dodge)) 
                 + gg.geom_vline(gg.aes_string(xintercept =0 ),
                                 linetype ="dotted", col="red", alpha=1)
                 )
        if facet:
            g  = (g
                  + gg.facet_wrap(f"~ {facet}" , ncol=facet_ncol,
                                  scales=facet_scales, labeller="label_value",
                                  dir ="h", as_table=True) 
                 )
        if not leg_show:
            g = g+gg.theme(legend_position="none")

        if text:
            g = g+ gg.geom_text( gg.aes_string(x=x, y=y, label='sig_text',
                                               color=color),
                                 show_legend = False,
                                 position    = gg.position_dodge(dodge),
                                 # 
                                 fontface    = sig_text.get('face', 'plain'),
                                 vjust       = sig_text.get('vjust', -1),
                                 hjust       = sig_text.get('hjust', .5),
                                 angle       = sig_text.get('angle', 0),
                                 size        = sig_text.get('size', 3)) 
            if text_leg:
                caption='* p < 0.1; ** p<0.05; *** p<0.01'
                g = (
                    g
                    + gg.labs(caption = caption) 
                    + gg.theme(plot_caption = gg.element_text(size = 8, hjust = 1,
                                                              ## family = "arial",
                                                              ## color = "black",
                                                              face = "italic")) 
                )
        # saving 
        # ------
        # saving 
        # ------
        self.__ggfigure_save__(
            tab           = tmp,
            g             = g,
            fn            = fn,
            fn_format_fig = fn_format_fig,
            fn_format_tab = fn_format_tab,
            fn_figsize    = fn_figsize,
            save_tab      = True)

        g.plot()
        return g


    def plot_predict(self,
                     models=None,
                     regex=None,
                     predictor=None,
                     predictor_values=None,
                     covars_at=None,
                     newdata=None,
                     # # Predicted data dictionary
                     # pred_dict=None,
                     # facet
                     facet=None,
                     facet_scales=None,
                     # color
                     color=None,
                     color_grey=False,
                     color_manual=False,
                     # line
                     linewidth=.9,
                     linetype=None,
                     # labels
                     xlab=None,
                     ylab=None,
                     # legend
                     leg_ncol       =None,
                     leg_show       =True,
                     leg_title      =None,
                     leg_title_pos  ='top',
                     leg_title_show =True,
                     leg_title_linetype=None,
                     # title and caption
                     title    =None,
                     subtitle =None,
                     caption  =None,
                     # saving
                     fn=None,
                     fn_format_fig=None,
                     fn_format_tab='xlsx',
                     fn_figsize = [8, 4]
                     ):
        '''
        Plot fitted values

        Input
        -----
        models        string or list of strings with the labels indicating whith models
                      to plot.
        regex         a regular expression to match with the labels of the 
                      models to print. Ignored if models is used.

        predictor     a string with the predictor
        
        predictor : a string with a variable name to get predicted
                    values (only used if newdata is not provided)

        predictor_values : (optional) a list of values for the predictors.
                           If None, they are generated automatically.

        covars_at  : a dict with name of the variable (key of the dict) 
                    and the values (values of the dict) to set for
                    prediction  (only used if newdata is not provided)

        newdata   : a DataFrame with the values of the covariates to
                    predict the output. If used, the other parameters
                    (predictor and covars_at) are ignored


        facet           a dictionary of string.
                        If dictionary, the keys must be the labels to use for the facets.
                        The values must be a list of model labels. The models in the
                        list will go with the respective label.
                        A string should be used whenever covars_at is used.
        
        facet_scales   'free', 'free_x', 'free_y'. Default is to use the same scale in
                        all facets

        color         same as for facet, but to group the models using a color
                      code instead 
        color_manual  a dictionary with {'category name':'color'}. Ignored if
                      color_grey is used or color not used
        color_grey    boolean indicating if a grey palette should be used for
                      the color code

        linetype     same as for facet, but group the models by linetype instead
        linewidth    number with the width of the fitted line
        
        xlab          string or a dictionary. IAccepts markdown. If a string,
                      it will be used as the title for the axis.
                      If a dictionary, it accepts following keys:values

                      key         value
                      ---         -----
                      title       a string with the label/title of the axis
                      size        size of the title of the axis
                      color       a string with the color of the title
                      lineheight  number with the height of the line for
                                  multiline title

        ylab           same as in xlab, but for y-axis. Accepts markdown.
        title          same as in xlab. Default is no title. Accepts markdown. 
        subtitle       same as in xlab. Default is no subtitle. Accepts markdown.
        caption        same as in xlab. Default is no caption. Accepts markdown.

        leg_show         boolean. If True, it shows the legend
        leg_ncol         integer with the number of columns for the legend
        leg_title        string with the title of the legend
        leg_title_pos    "top", 'bottom', 'left', or 'right' indicating the position
                         of the title of the legend
        leg_title_show   boolean indicating if the legend title should be displayed
        leg_title_linetype  string with the title of the legend for the linetypes


        fn             a string with the full path for the file to save. If None
                       (default) no file is saved. The file extension will be
                       ignored. To set it, use fn_format_X.
        fn_format_fig  a string with the extension (format) to save the figure 
                       (pdf, png, etc.)
        fn_format_tab  a string with the extension (format) to save the underlying  
                       table used to generate the figure (xlsx, csv, etc)
        fn_figsize     a list with the size of the figure: [width, height]
        
        '''

        xlab               =robj.NULL  if not xlab               else  xlab
        ylab               =robj.NULL  if not ylab               else  ylab
        leg_title          =robj.NULL  if not leg_title          else  leg_title
        predictor_values   =robj.NULL  if not predictor_values   else  predictor_values
        covars_at          =robj.NULL  if not covars_at          else  covars_at
        leg_title_linetype =robj.NULL  if not leg_title_linetype else  leg_title_linetype
        facet              =robj.NULL  if not facet              else  facet
        facet_scales       =robj.NULL  if not facet_scales       else  facet_scales
        color              =robj.NULL  if not color              else  color
        title              =robj.NULL  if not title              else  title
        subtitle           =robj.NULL  if not subtitle           else  subtitle
        caption            =robj.NULL  if not caption            else  caption
        # 
        # 
        models = self.__search_model_labels__(models, regex)
        # 
        xlab  = predictor.replace('`', '') if not xlab else xlab
        xlab  ={'title':xlab} if xlab and isinstance(xlab, str) else {'title': predictor}
        xlab['title'] = xlab.get('title', predictor)
        # 
        ylab  = 'Fitted values' if not ylab else ylab
        ylab  ={'title':ylab} if ylab and isinstance(ylab, str) else ylab
        ylab['title'] = ylab.get('title', 'Fitted values')
        # 
        title  = {} if not title else title
        title  ={'title':title} if title and isinstance(title, str) else title
        title['title'] = title.get('title', robj.NULL)
        # 
        subtitle  = {} if not subtitle else subtitle
        subtitle  ={'title':subtitle} if subtitle and isinstance(subtitle, str) else subtitle
        subtitle['title'] = subtitle.get('title', robj.NULL)
        # 
        # 
        caption  = {} if not caption else caption
        caption  ={'title':caption} if caption and isinstance(caption, str) else caption
        caption['title'] = caption.get('title', robj.NULL)
        # 
        pred=self.predict(
            models=models,
            predictor=predictor,
            predictor_values=predictor_values,
            covars_at=covars_at,
            newdata=newdata
        )
        if not covars_at:
            models   = self.labels if not models else models
            models   = [models] if isinstance(models, str) else models
            facet    = facet if not facet else\
                self.__plot_get_replace_dict__(facet) 
            color    = color if not color else\
                self.__plot_get_replace_dict__(color) 
            linetype = linetype if not linetype else\
                self.__plot_get_replace_dict__(linetype) 
            pred=(
                pred
                .mutate({'facet': lambda col: col['model_id']})
                .mutate({'color': lambda col: col['model_id']})
                .mutate({'linetype': lambda col: col['model_id']})
                .mutate({'group': lambda col: col['color']})
                .replace({'facet': facet} , regex=False)
                .replace({'color': color} , regex=False)
                .replace({'linetype': linetype} , regex=False)
                # .rename_cols(columns={'model_id':"Labels"}, tolower=False)
                .mutate({'Labels': lambda col: col['model_id']})
                # .rename_cols(columns={'model_id':"Labels"}, tolower=False)
                .mutate_type(from_to={"object": "str"})
            )
        else:
            models   = [models] if isinstance(models, str) else models
            assert len(models)==1 or len(covars_at.keys())==1, (
                "When using covars_at, use either one model or one covariate "+\
                "in covars_at. More than one for both is currently not supported.")
            assert not facet or isinstance(facet, str), "Facet must be a string or None when covars_at is used"
            assert not color or isinstance(color, str), "Color must be a string or None when covars_at is used"
            assert not linetype or isinstance(linetype, str), "Linetype must be a string or None when covars_at is used"
            if facet:
                pred=pred.mutate({'facet': lambda col: col[facet]})
            if color:
                pred=pred.mutate({'color': lambda col: col[color]})
                # pred=pred.rename_cols(columns={color:'color'}, tolower=False)
            if linetype:
                pred=pred.rename_cols(columns={linetype:'linetype'}, tolower=False)
            pred=(
                pred
                .mutate({'group': lambda col: col['model_id']})
                # .rename_cols(columns={'model_id':"Labels"}, tolower=False)
                .mutate({'Labels': lambda col: col['model_id']})
                .mutate_type(from_to={"object": "str"})
            )
        pred=pred.rename_cols(regex={"`":''}, tolower=False)

        # plot 
        # ----
        color              ='color' if color else robj.NULL
        color              ='Labels' if not color and len(models)>1 else color
        linetype           ='linetype' if linetype else robj.NULL
        linetype           ='Labels' if not linetype and len(models)>1 else linetype
        facet              ='facet' if facet else robj.NULL
        leg_ncol           = len(models) if not leg_ncol else leg_ncol
        leg_title_linetype = 'Labels' if leg_title_show and not leg_title_linetype else leg_title_linetype
        leg_title          = color if leg_title_show and not leg_title else leg_title
        # 
        
        x = f"`{predictor}`" if "`" not in predictor else predictor
        y = "pred"
        ymin='lower'
        ymax='upper'
        g = (
            gg.ggplot(pred)
            + gg.geom_line(gg.aes_string(x=x, y=y, group=robj.NULL, colour=color,
                                         linetype=linetype),
                           size=linewidth) 
            + gg.labs(
                x        = xlab.get('title'),
                y        = ylab.get('title'),
                color    = leg_title, 
                fill     = leg_title,
                linetype = leg_title_linetype,
                shape    = robj.NULL,
                title    = title.get('title', robj.NULL),
                subtitle = subtitle.get('title', robj.NULL),
                caption  = caption.get('title', robj.NULL),
            )
            + gg.theme_bw()
            + self.__ggtheme__(xlab=xlab, ylab=ylab,
                               title=title, subtitle=subtitle)
            + self.__ggguides__(ncol =leg_ncol, leg_title_pos=leg_title_pos)
        )
        if covars_at:
            g = g + gg.geom_ribbon(gg.aes_string(x=x, ymin = ymin, ymax = ymax,
                                                 # linetype=linetype,
                                                 fill=color,
                                                 # group="group"
                                                 ), alpha=.2)
        else:
            g = g + gg.geom_ribbon(gg.aes_string(x=x, ymin = ymin, ymax = ymax,
                                                 linetype=linetype,
                                                 fill=color,
                                                 group="group"
                                                 ), alpha=.2)
            
        # 
        if isinstance(pred[y].values[0], float):
            g = g + gg.scale_y_continuous(expand = FloatVector([0, 0]))
        else:
            g = g + gg.scale_y_discrete(expand = FloatVector([0, 0]))
        if isinstance(pred[x.replace('`', '')].values[0], float):
            g = g + gg.scale_x_continuous(expand = FloatVector([0, 0]))
        else:
            g = g + gg.scale_x_discrete(expand = FloatVector([0, 0]))
        # 
        if color and not color_grey and not color_manual:
            g = g + ggthemes.scale_colour_tableau()
            g = g + ggthemes.scale_fill_tableau()
        elif color and not color_grey and color_manual:
            g = g + gg.scale_colour_manual(values=ru.dict2namedvector(color_manual)) 
            g = g + gg.scale_fill_manual(values=ru.dict2namedvector(color_manual)) 
        elif color and color_grey:
            g = g + gg.scale_colour_grey(start=0, end=.7, na_value="red") 
            g = g + gg.scale_fill_grey(start=0, end=.7, na_value="red") 
        if facet:
            g  = (g
                  + gg.facet_wrap(f"~ {facet}" , ncol =robj.NULL,
                                  scales =facet_scales, labeller="label_value",
                                  dir ="h", as_table=True) 
                  )
        if not leg_show:
            g = g+gg.theme(legend_position="none")

        # saving 
        # ------
        self.__ggfigure_save__(
            tab           = pred,
            g             = g,
            fn            = fn,
            fn_format_fig = fn_format_fig,
            fn_format_tab = fn_format_tab,
            fn_figsize    = fn_figsize,
            save_tab      = True)
        
        g.plot()
        return g

    
# **** print

    def print(self, models=None, regex=None, *args, **kws):
        '''
        Print estimation results

        Input 
        -----
        models  a string or list of strings with the labels of the models to 
                print. If 'None', it prints all models 

        regex   a regular expression to match with the labels of the models to 
                print. Ignored if models is used.

        output_format  see pandasci.models.regression.summary

        fn             see pandasci.models.regression.summary
                
        '''
        models = self.__search_model_labels__(models, regex)
        # 
        res=self.__get_summaryr__(labels=models, *args, **kws)
        output_format= kws.get("output_format", 'data.frame')
        if kws.get("fn", False):
            output_format=kws.get("fn", output_format)

        print(f"", flush=True)
        self.__print_header_info__(labels=models)
        print(f"Estimation summary:\n", flush=True)
        if (output_format=='data.frame'):
            print(res.to_string(index=False), flush=True)
        else:
            print(res, flush=True)
        return None
        

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


# **** get
    def get_info(self, model=None):
        '''
        Input
        -----
        model   a string with the label of the model. If None,
                information for all models are printed
        '''
        if model:
            self.__get_info__(model)
        else:
            for model in self.results.label.values:
                self.__get_info__(model)


    def get_coef_labels(self):
        coeflabels=(
            self
            .results
            .unnest(col='summ_tidy', id_vars='label')
            .term.unique()
        )
        return list(coeflabels)
        

# **** others

    def build_formula(self, output, inputs, interactions=None, clusters=None):
        mu=models_utils()
        return mu.build_formula(output=output,
                                inputs=inputs,
                                interactions=interactions,
                                clusters=clusters)
# *** backend
# **** core/dispatcher

    def __run_regressions__(self, models, *args, **kws):
        for label, model in models.items():
            print(f"Running model {label}...", flush=True)
            self.results = self.results.bind_row(
                self.__run_dispatcher__(idx     = self.results.nrow+1,
                                        label   = label,
                                        formula = self.formula[label],
                                        family  = self.family[label],
                                        data    = self.data[label],
                                        engine  = self.engine[label],
                                        *args, **kws)
            )
            # for idx, (formula, family, data) in enumerate(zip(self.formula[label],
            #                                                   self.family[label],
            #                                                   self.data[label])):
            #     tmp = self.__run_dispatcher__(idx, label, formula, family, data,
            #                               *args, **kws)
            #     tmp['depvar']
            #     tab = pd.concat([tab, tmp], axis=0, ignore_index=True)

    def __run_dispatcher__(self, idx, label, formula, family, data, engine,
                           *args, **kws):
        if engine=='python':
            if family=='gaussian':
                tab = self.__run_gaussian__(idx, formula, label, data, *args, **kws)
            if family=='binomial':
                tab = self.__run_binomial__(idx, formula, label, data, *args, **kws)
            if family=='multinomial':
                tab = self.__run_multinomial__(idx, formula, label, data, *args, **kws)
        if engine=='r':
            if family=='gaussian':
                tab = self.__run_gaussianr__(idx, formula, label, data, *args, **kws)
            elif family=='binomial':
                tab = self.__run_binomialr__(idx, formula, label, data, *args, **kws)
            elif family=='multinomial':
                tab = self.__run_multinomialr__(idx, formula, label, data, *args, **kws)
            elif True:
                assert False, f"family {family}  not implemented"
        return tab


    def __reprr__(self, *args, **kws):
        self.print(models=None, *args, **kws)


    def __repr__(self):
        # if self.engine == 'python':
        #     print("to be implemented")
        # if self.engine == 'r':
        self.__reprr__()
        return ""


    def __build_formula_and_get_variables_dict__(self, models):
        for label, model_info in models.items():
            model_info = list(model_info)
            formula = model_info[0] 
            formula_arg  = None
            output       = None
            inputs       = None
            interactions = None
            clusters     = None
            if isinstance(formula, str):
                formula_arg  = formula
            elif not isinstance(formula, str):
                output       = formula['output']
                inputs       = formula['inputs']
                interactions = formula.get("interactions", None)
                clusters     = formula.get("clusters", None)
                formula      = self.build_formula(output, inputs, interactions, clusters)
            model_info[0] = formula
            model_info=tuple(model_info)
            models[label]=model_info
            # 
            self.variables_dict_raw[label]={
                'formula_arg':formula_arg,
                'output':output,
                "inputs":inputs,
                "interactions":interactions,
                "clusters":clusters
            }
        self.__build_coef_dict__()
        return models


    def __get_coef_groups__(self):
        groups=(
            self
            .results
            .unnest(col='summ_tidy', id_vars='label')
            .select_cols(names=['var_group', 'term', 'value_label'])
            .drop_rows(regex={'term':'Intercept'})
            # .drop_rows(dropna=True)
            # .values
            # .flatten()
        )
        coefs={}
        for group in groups.var_group.unique():
             unique_vars = set(
                groups
                .select_rows(query=f"var_group=='{group}'")
                # .value_label
                .term
                .values.tolist()
            )
             coefs[group]=list(unique_vars)
        return coefs

    def __get_var_group__(self):
        res_final=eDataFrame()
        for  model in self.labels:
            idx_model=self.results.select_rows(query=f"label=='{model}'").index[0]
            # Inputs 
            # ------
            input_vars=(
                self
                .variables
                .select_rows(query=f"model=='{model}'")
                .select_rows(query=f"part=='inputs'")
                .drop_rows(dropna='var_groups')
                .var_groups
                .values
                .flatten()[0]
            )
            dinputs={}
            for k, v in input_vars.items():
                for terms in v:
                    dinputs[terms]=k
            # interactions 
            # ------------
            ninterterms=[]
            dinter={}
            inter_vars=(
                self
                .variables
                .select_rows(query=f"model=='{model}'")
                .select_rows(query=f"part=='interactions'")
                .drop_rows(dropna='var_groups')
                .var_groups
                .values
                .flatten()[0]
            )
            if inter_vars:
                for k, v in inter_vars.items():
                    for terms in v:
                        if isinstance(terms, tuple):
                            nterms=len(terms)
                            terms=[re.sub(pattern='\)', repl='\)', string=s) for s in terms]
                            terms=[re.sub(pattern='\(', repl='\(', string=s) for s in terms]
                            regex="".join([f"(?=.*{s})" for s in terms])
                            dinter[regex]=k
                            ninterterms+=[nterms]
                        elif isinstance(terms, str):
                            dinter[terms]=k
                            ninterterms+=[1]
            # inputs 
            # ------
            res=(
                self
                .results
                # .select_rows(query=f"label=='{model}'")
                .summ_tidy[idx_model]
                .mutate({'var_group': lambda col: col['term']})
                .replace({'var_group':dinputs} , regex=False, inplace=False)
            )
            # interactions 
            # ------------
            if dinter:
                for i, (regex, group) in enumerate(dinter.items()):
                    idx=(
                        res
                        .select_rows(query=f"ninterterms=={ninterterms[i]}")
                        .select_rows(regex=regex)
                        .index.values
                    )
                    if idx:
                        res.loc[idx[0], 'var_group'] = group
            res_final=res_final.bind_row(res.mutate({'model_label': model}))
        res_final=(
            res_final
            .nest('model_label')
            .rename_cols(columns={'data':"summ_tidy"}, tolower=False)
            .mutate_rowwise({'summ_tidy': lambda col: col['summ_tidy'] .drop_cols(names='model_label')})
        )
        self.results=(
            self
            .results
            .drop_cols(names='summ_tidy')
            .join(res_final, how='left', left_on='label', right_on='model_label')
        )


    def __build_variables_df_anc__(self):
        for label in self.variables_dict_raw.keys():
            formula      = self.variables_dict_raw[label]['formula_arg']
            output       = self.variables_dict_raw[label]['output']
            inputs       = self.variables_dict_raw[label]['inputs']
            interactions = self.variables_dict_raw[label]['interactions']
            clusters     = self.variables_dict_raw[label]['clusters']
            #
            # 
            output_dic={}
            inputs_dic={}
            interactions_dic=None
            # 
            inputs_varlabels={}
            if not formula:
                formula = self.build_formula(output, inputs,
                                             interactions, clusters)
            # output 
            # ------
            if isinstance(output, str):
                output_dic[output]=output
            elif not output:
                vars=(
                    models_utils()
                    .formula2df(formula)
                    .select_rows(query=f"part=='output'")
                    .select_cols(names='var_list')
                    .values
                    .flatten()[0]
                )
                for v in vars:
                    output_dic[v]=v
            # inputs 
            # ------
            # Three cases based on dict values
            # no groups,    labels: a. inputs = {'oldnames': "newnames"} ()
            #    groups, no labels: b. inputs = {'group': [...]}
            #    groups,    labels: c. inputs = {'group': {'oldnames': "newnames"}}
            if isinstance(inputs, dict):
                inputs_dic=inputs
                for k, v in inputs.items():
                    # no groups for covar ({'oldnames': "newnames"})
                    if isinstance(v, str):
                        inputs_varlabels[k]=v
                    # k is a covar group
                    elif isinstance(v, list) or isinstance(v, dict):
                        if isinstance(v, list):
                            for vi in v:
                                inputs_varlabels[vi]=vi
                        elif isinstance(v, dict):
                            for ki, vi in v.items():
                                inputs_varlabels[ki]=vi
            # inputs are a list:
            elif isinstance(inputs, list):
                for v in inputs:
                    inputs_dic[v]=v   
                    inputs_varlabels[v]=v
            elif not inputs:
                vars=(
                    models_utils()
                    .formula2df(formula)
                    .select_rows(query=f"part=='inputs'")
                    .select_cols(names='var_list')
                    .values
                    .flatten()[0]
                )
                for v in vars:
                    v=re.sub(pattern='`', repl='', string=v)
                    inputs_dic[v]=v   
                    inputs_varlabels[v]=v
            # Interactions 
            # ------------
            if isinstance(interactions, dict) or isinstance(interactions, list):
                interactions_dic=interactions_dic

            # build data frame 
            # ----------------
            res = (
                models_utils()
                .formula2df(formula)
                .mutate({
                    'var_labels':None,
                    'var_groups':None,
                    'model': label,
                })
            )
            res.loc[res.part=='output', 'var_dict']=[output_dic]
            res.loc[res.part=='inputs', 'var_dict']=[inputs_dic]
            res.loc[res.part=='interactions', 'var_dict']=[interactions_dic]
            # 
            res.loc[res.part=='output', 'var_labels']=[output_dic]
            res.loc[res.part=='inputs', 'var_labels']=[inputs_varlabels]
            # 
            self.variables=self.variables.bind_row(res)


    def __build_variables_df__(self):
        self.variables=eDataFrame()
        for label in self.labels:
            formula      = self.variables_dict_raw[label]['formula_arg']
            output       = self.variables_dict_raw[label]['output']
            inputs       = self.variables_dict_raw[label]['inputs']
            interactions = self.variables_dict_raw[label]['interactions']
            clusters     = self.variables_dict_raw[label]['clusters']
            # 
            output_dic={}
            inputs_dic={}
            interactions_dic=None
            interactions_groups={}
            # 
            inputs_varlabels={}
            inputs_groups={}
            inputs_groups_final={}
            inputs_default_group=' '

            if not formula:
                formula = self.build_formula(output, inputs,
                                             interactions, clusters)
            # output 
            # ------
            if isinstance(output, str):
                output_dic={output: output}
            elif not output:
                vars=(
                    models_utils()
                    .formula2df(formula)
                    .select_rows(query=f"part=='output'")
                    .select_cols(names='var_list')
                    .values
                    .flatten()[0]
                )
                for v in vars:
                    output_dic[v]=v
            # inputs 
            # ------
            # Three cases based on dict values
            # no groups,    labels: a. inputs = {'oldnames': "newnames"} ()
            #    groups, no labels: b. inputs = {'group': [...]}
            #    groups,    labels: c. inputs = {'group': {'oldnames': "newnames"}}

            if isinstance(inputs, dict):
                inputs_dic=inputs
                for k, v in inputs.items():
                    # no groups for covar ({'oldnames': "newnames"})
                    if isinstance(v, str):
                        inputs_groups[inputs_default_group]=inputs_groups.get(" ", [])+[v]
                        inputs_varlabels[k]=v
                    # k is a covar group
                    elif isinstance(v, list) or isinstance(v, dict):
                        inputs_groups[k]=[]
                        if isinstance(v, list):
                            for vi in v:
                                inputs_varlabels[vi]=vi
                                inputs_groups[k]+=[vi]
                        elif isinstance(v, dict):
                            for ki, vi in v.items():
                                inputs_varlabels[ki]=vi
                                inputs_groups[k]+=[vi]
            # inputs are a list:
            elif isinstance(inputs, list):
                for v in inputs:
                    inputs_dic[v]=inputs_dic.get(" ", [])+[v]   
                    inputs_varlabels[v]=inputs_varlabels.get(" ", [])+[v]
                    inputs_groups[inputs_default_group]=inputs_groups.get(" ", [])+[v]
            elif not inputs:
                vars=(
                    models_utils()
                    .formula2df(formula)
                    .select_rows(query=f"part=='inputs'")
                    .select_cols(names='var_list')
                    .values
                    .flatten()[0]
                )
                for v in vars:
                    v=re.sub(pattern='`', repl='', string=v)
                    inputs_dic[v]=v   
            # 
            # collect categorical terms for inputs
            if not inputs_groups:
                inputs_groups={inputs_default_group: list(inputs_dic.values())}
            for group, terms in inputs_groups.items():
                inputs_groups_final[group]=[]
                for term in terms:
                    cat_term=self.__get_categorical_term_string__(model=label,
                                                                  term=term)
                    inputs_groups_final[group]+=cat_term if cat_term else [term]
            # Interactions 
            # ------------
            interactions_groups={}
            # case 1: interactions={'group': [(v1, v2), (v1, v3)]}
            if isinstance(interactions, dict):
                # create new interaction terms for categorical covariates
                interactions_dic=interactions_dic
                for group_label, interactions_set in interactions.items():
                    interactions_groups[group_label]=[]
                    for terms in interactions_set:
                        inter_list=[]
                        for term in terms:
                            cat_term=self.__get_categorical_term_string__(model=label,
                                                                          term=term)
                            inter_list+= [cat_term] if cat_term else [[term]]
                        # 
                        inter_list_final=list(itertools.product(*inter_list))
                        # inter_list_interim=list(itertools.product(*inter_list))
                        # inter_list_final=[":".join(termi) for termi in inter_list_interim]
                        # 
                        interactions_groups[group_label]+=inter_list_final
            # case 2: interactions=[(v1, v2), (v1, v3)]
            elif isinstance(interactions, list):
                inter_list=[]
                interactions_groups['Interactions']=interactions_groups.get('Interactions', [])+[]
                for interaction_set in interactions:
                    for terms in interaction_set:
                        cat_term=self.__get_categorical_term_string__(model=label,
                                                                      term=term)
                        inter_list+=[cat_term] if cat_term else [[term]]
                    # 
                    inter_list_final=list(itertools.product(*inter_list))
                    # inter_list_interim=list(itertools.product(*inter_list))
                    # inter_list_final=[":".join(termi) for termi in inter_list_interim]
                    # 
                    interactions_groups['Interactions']+=inter_list_final
            # when a formula was provided
            elif not interactions:
                inter_list=[]
                interactions=(
                    models_utils()
                    .formula2df(formula)
                    .select_rows(query=f"part=='interactions'")
                    .select_cols(names='var_list')
                    .values
                    .flatten()[0]
                )
                interactions_groups['Interactions']=interactions_groups.get('Interactions', [])+[]
                if interactions:
                    for interactioni in interactions:
                        interaction_set = interactioni.split(':')
                        for terms in interaction_set:
                            cat_term=self.__get_categorical_term_string__(model=label,
                                                                          term=term)
                            inter_list+=[cat_term] if cat_term else [[term]]
                        # 
                        inter_list_final=list(itertools.product(*inter_list))
                        # inter_list_interim=list(itertools.product(*inter_list))
                        # inter_list_final=[":".join(termi) for termi in inter_list_interim]
                        # 
                        interactions_groups['Interactions']+=inter_list_final

            # build the data frame 
            # --------------------
            res = (
                models_utils()
                .formula2df(formula)
                .mutate({
                    'var_labels':None,
                    'var_groups':None,
                    'model': label,
                })
            )
            # 
            res.loc[res.part=='output', 'var_labels']=[output_dic]
            res.loc[res.part=='output', 'var_dict']=[output_dic]
            # 
            res.loc[res.part=='inputs', 'var_dict']=[inputs_dic]
            res.loc[res.part=='inputs', 'var_labels']=[inputs_varlabels]
            res.loc[res.part=='inputs', 'var_groups']=[inputs_groups_final]
            # 
            res.loc[res.part=='interactions', 'var_dict']=[interactions_dic]
            res.loc[res.part=='interactions', 'var_groups']=[interactions_groups]
            # 
            self.variables=self.variables.bind_row(res)

    def __get_categorical_term_string__(self, model, term):
        vartype=(
            self
            .results
            .select_rows(query=f"label=='{model}'")
            .select_cols(names='summ_tidy')
            .values[0][0]
            .select_rows(query=f"var_label=='{term}'")
            .select_cols(names='var_type')
            .values
            .flatten()[0]
        )
        s=None
        if vartype=='categorical':
            s=[]
            for cat in self.data[model][term].unique():
                s+=[f"{term}{cat}"]
        return s

        
    def __search_model_labels__(self, models, regex):
        '''
        see documentation of models.regression.print()
        '''
        res = models
        if not models and regex:
            res = [mod for mod in self.labels if
                   bool(re.search(pattern=regex, string=mod))]
        return res


# **** estimation
# ***** gaussian

    def __run_gaussianr__(self, idx, formula, label, data, *args, **kws):
        fit = stats.lm(formula, data=data, family='gaussian')
        mod = pd.DataFrame({'idx'          :idx,
                            'label'        :label,
                            'model'        :label,
                            'depvar'       :formula.split("~")[0].strip(),
                            "formula"      :formula,
                            'family'       :'gaussian',
                            "mod"          :[fit],
                            "fit"          :[fit],
                            "summ_tidy"    :[self.__get_r_lm_summary_tidy__(fit)],
                            "summ_default" :[self.__get_r_lm_summary_default__(fit)],
                            "summ3"        :np.nan ,
                            'obs'          :self.__get_r_lm_nobs__(fit),
                            'aic'          :self.__get_r_lm_aic__(fit),
                            'bic'          :self.__get_r_lm_bic__(fit),
                            'r2'           :self.__get_r_lm_r2__(fit),
                            'r2adj'        :self.__get_r_lm_r2adj__(fit),
                            'rmse'         :self.__get_r_lm_rmse__(fit)
                            })
        return mod

    
# ***** binomial

    def __run_binomialr__(self, idx, formula, label, data, *args, **kws):
        fit = stats.glm(formula, data=data, family='binomial')
        mod = pd.DataFrame({'idx'          :idx,
                            'label'        :label,
                            'model'        :label,
                            'depvar'       :formula.split("~")[0].strip(),
                            "formula"      :formula,
                            'family'       :'binomial',
                            "mod"          :[fit],
                            "fit"          :[fit],
                            "summ_tidy"    :[self.__get_r_binomial_summary_tidy__(fit)],
                            "summ_default" :[self.__get_r_binomial_summary_default__(fit)],
                            "summ3"        :np.nan ,
                            'obs'          :self.__get_r_binomial_nobs__(fit),
                            'aic'          :self.__get_r_binomial_aic__(fit),
                            'bic'          :self.__get_r_binomial_bic__(fit),
                            'r2'           :self.__get_r_binomial_r2__(fit),
                            'r2adj'        :self.__get_r_binomial_r2adj__(fit),
                            'rmse'         :self.__get_r_binomial_rmse__(fit)
                            })
        return mod


# ***** multinomial

    def __run_multinomialr__(self, idx, formula, label, data, *args, **kws):
        fit = nnet.multinom(formula, data=data)
        mod = pd.DataFrame({'idx'          :idx,
                            'label'        :label,
                            'model'        :label,
                            'depvar'       :formula.split("~")[0].strip(),
                            "formula"      :formula,
                            'family'       :'multinomial',
                            "mod"          :[fit],
                            "fit"          :[fit],
                            "summ_tidy"    :[self.__get_r_multinomial_summary_tidy__(fit)],
                            "summ_default" :[self.__get_r_multinomial_summary_default__(fit)],
                            "summ3"        :np.nan ,
                            'obs'          :self.__get_r_multinomial_nobs__(fit),
                            'aic'          :self.__get_r_multinomial_aic__(fit),
                            'bic'          :self.__get_r_multinomial_bic__(fit),
                            'r2'           :self.__get_r_multinomial_r2__(fit),
                            'r2adj'        :self.__get_r_multinomial_r2adj__(fit),
                            'rmse'         :self.__get_r_multinomial_rmse__(fit)
                            })
        return mod


# **** summary
# ***** core

    def __tolatex__(self, tab, fn, *args, **kws):
        stars=kws.get("stars", None)
        footnotes=kws.get("footnotes", None)
        # remove keys that conflict with pandas to_latex
        kws.pop('stars', None)
        kws.pop('output_format', None)
        kws.pop('replace', None)
        kws.pop('footnotes', None)
        # if 'output_format' in kws.keys():
        #     del kws['output_format']
        # if 'replace' in kws.keys():
        #     del kws['replace']
        # 
        tab_latex=tab.tolatex(*args, **kws)
        if stars or footnotes:
            if stars:
                footnote=(
                    "\\\\begin{tablenotes}\\n"+\
                    f"\\\\item \\\\footnotesize ${stars}$\\n"+\
                    "\\\\end{tablenotes}"
                )
            else:
                footnote=(
                    "\\\\begin{tablenotes}\\n"+\
                    "\\\\end{tablenotes}"
                )
            # 
            tab_latex=re.sub(pattern='\\\\begin{tabular}',
                             repl='\\\\begin{threeparttable}\\n\\\\begin{tabular}',
                             string=tab_latex)
            tab_latex=re.sub(pattern='\\\\end{tabular}',
                             repl=f"\\\\end{{tabular}}\\n{footnote}\\n\\\\end{{threeparttable}}",
                             string=tab_latex)
            if footnotes:
                footnotes=[footnotes] if isinstance(footnotes, str) else footnotes
                for footnote in footnotes:
                    tab_latex=re.sub(
                        pattern='\\\\end{tablenotes}',
                        repl=f"\\\\item {footnote}\\n\\\\end{{tablenotes}}",
                        string=tab_latex)
        with open(fn, "w") as f:
            f.write(tab_latex)


    def __get_summary_df__(self, *args, **kws):
        res=self.__summaryr__(*args, **kws)
        return res
       


    def __summaryr__(self, *args, **kws):
        # self.__reprr__(*args, **kws)
        if kws.get("output_collect", False):
            return self.__get_summaryr__(*args, **kws)
        return None


    def __get_summaryr__(self, *args, **kws):
        output_format=kws.get("output_format", 'data.frame')
        if kws.get("fn", False):
            print(f"\nNote: 'fn' provided. Format will use fn extension.\n", flush=True)
            # 
            output_format=kws.get("fn", output_format)
            if isinstance(output_format, pathlib.PurePath):
                output_format=str(output_format)
        # 
        vcov=kws.get("vcov", 'classical')
        vcov=robj.Formula(vcov) if "~" in vcov else vcov
        footnotes=kws.get("footnotes", robj.NULL)
        # 
        models_to_print=kws.get("labels", None)
        models_to_print= self.labels if not models_to_print else models_to_print
        mods = ru.dict2namedlist({name:mod for name, mod in
                                  zip(self.labels, self.results.fit.values)
                                  if name in models_to_print
                                  })
        if self.multinomial:
            print("Note: Only classical Std.Errors are available for multinomial.")
            res= modelsummary.modelsummary_wide(mods,
                                                statistic='({conf.low}, {conf.high})',
                                                # + p < 0.1, * p < 0.05, ** p < 0.01, *** p < 0.001
                                                stars=True, ## c('*' = .1, '**' = .05, "***"=0.01),
                                                # vcov = vcov, 
                                                ## 
                                                coef_omit = "Intercept",
                                                ## coef_rename=c('vs'='first'),
                                                ## coef_map=c(), ## to reorder and rename/will omit those not included
                                                ## 
                                                ## align=c("l", "c"),
                                                ## gof_omit = 'Errors|F', # regexp to exclude stats summary (AIC, etc)
                                                notes = footnotes,
                                                ## output=fn # latex, work. excel, 'huxtable' etc
                                                output=output_format
                                                )
        else:
            res= modelsummary.modelsummary(mods,
                                           statistic='({conf.low}, {conf.high})',
                                           # + p < 0.1, * p < 0.05, ** p < 0.01, *** p < 0.001
                                           stars=True, ## 
                                           vcov = vcov,
                                           # cluster = cluster,
                                           ## 
                                           coef_omit = "Intercept",
                                           ## coef_rename=c('vs'='first'),
                                           ## coef_map=c(), ## to reorder and rename/will omit those not included
                                           ## 
                                           ## align=c("l", "c"),
                                           ## gof_omit = 'Errors|F', # regexp to exclude stats summary (AIC, etc)
                                           notes = footnotes,
                                           ## output=fn # latex, work. excel, 'huxtable' etc
                                           output=output_format
                                           )
        if output_format=='data.frame':
            res=(
                eDataFrame(
                    robj
                    .conversion
                    .rpy2py(res)
                )
                .drop_cols(names=['statistic', 'part'])
            )
            res.loc[res.duplicated(['term']), 'term'] = ""
        return res


# ***** Gaussian

    def __get_r_lm_summary_tidy__(self, fit):
        # res = ru.df2pandas(broom.tidy_lm(fit, conf_int=True))
        res = ru.df2pandas((broomh.tidy_add_term_labels(broomh.tidy_and_attach(fit))))
        res = res.rename_cols(columns={"label":"value_label"}, tolower=False)
        res=(
            res
            .replace({"term":{"`":''}} , regex=True, inplace=False)
            .mutate_rowwise({'ninterterms': lambda col: len(col['var_label'].split("*"))})
        )
        return res

    def __get_r_lm_summary_default__(self, fit):
        res = base.summary(fit)
        return res

    def __get_r_lm_nobs__(self, fit):
        res = broom.glance_lm(fit).rx2['nobs'][0]
        return res

    def __get_r_lm_aic__(self, fit):
        res = broom.glance_lm(fit).rx2['AIC'][0]
        return res

    def __get_r_lm_bic__(self, fit):
        res = broom.glance_lm(fit).rx2['BIC'][0]
        return res

    def __get_r_lm_r2adj__(self, fit):
        res = broom.glance_lm(fit).rx2['adj.r.squared'][0]
        return res

    def __get_r_lm_r2__(self, fit):
        res = broom.glance_lm(fit).rx2['r.squared'][0]
        return res

    def __get_r_lm_rmse__(self, fit):
        e=stats.residuals(fit)
        res = np.sqrt(np.mean(e**2))
        return res

# ***** Binomial

    def __get_r_binomial_summary_tidy__(self, fit):
        # res = ru.df2pandas(broom.tidy_glm(fit, conf_int=True))
        res = ru.df2pandas((broomh.tidy_add_term_labels(broomh.tidy_and_attach(fit))))
        res = res.rename_cols(columns={"label":"value_label"}, tolower=False)
        res=(
            res
            .replace({"term":{"`":''}} , regex=True, inplace=False)
            .mutate_rowwise({'ninterterms': lambda col: len(col['var_label'].split("*"))})
        )
        return res

    def __get_r_binomial_summary_default__(self, fit):
        res = base.summary(fit)
        return res

    def __get_r_binomial_nobs__(self, fit):
        res = broom.glance_glm(fit).rx2['nobs'][0]
        return res

    def __get_r_binomial_aic__(self, fit):
        res = broom.glance_glm(fit).rx2['AIC'][0]
        return res

    def __get_r_binomial_bic__(self, fit):
        res = broom.glance_glm(fit).rx2['BIC'][0]
        return res

    def __get_r_binomial_r2adj__(self, fit):
        # res = broom.glance_glm(fit).rx2['adj.r.squared'][0]
        res = np.nan 
        return res

    def __get_r_binomial_r2__(self, fit):
        # res = broom.glance_glm(fit).rx2['r.squared'][0]
        res = np.nan 
        return res

    def __get_r_binomial_rmse__(self, fit):
        e=stats.residuals(fit)
        res = np.sqrt(np.mean(e**2))
        return res

# ***** Multinomial

    def __get_r_multinomial_summary_tidy__(self, fit):
        res = ru.df2pandas(broom.tidy_multinom(fit, conf_int=True))
        res = res.replace({"term":{"`":''}} , regex=True, inplace=False)
        res=(
            res
            .replace({"term":{"`":''}} , regex=True, inplace=False)
            .mutate_rowwise({'ninterterms': lambda col: len(col['var_label'].split("*"))})
        )
        return res

    def __get_r_multinomial_summary_default__(self, fit):
        res = base.summary(fit)
        return res

    def __get_r_multinomial_nobs__(self, fit):
        res = broom.glance_multinom(fit).rx2['nobs'][0]
        return res

    def __get_r_multinomial_aic__(self, fit):
        res = broom.glance_multinom(fit).rx2['AIC'][0]
        return res

    def __get_r_multinomial_bic__(self, fit):
        # res = broom.glance_multinom(fit).rx2['BIC'][0]
        res = np.nan 
        return res

    def __get_r_multinomial_r2adj__(self, fit):
        # res = broom.glance_multinom(fit).rx2['adj.r.squared'][0]
        res = np.nan 
        return res

    def __get_r_multinomial_r2__(self, fit):
        # res = broom.glance_multinom(fit).rx2['r.squared'][0]
        res = np.nan 
        return res

    def __get_r_multinomial_rmse__(self, fit):
        e=stats.residuals(fit)
        res = np.sqrt(np.mean(e**2))
        return res

# **** predict


    def __get_models_containing_variable__(self, predictor):
        models = []
        for idx, row in self.results.iterrows():
            covars = ru.formula2varlist(row['formula'])
            if predictor in covars:
                models += [row['label']]
        return models

# **** plots


    def __plot_coef_prepare_data__(self, models,
                                   coef_labels, coef_wrap,
                                   text, color, facet, shape,
                                   digits=2):
        tmp=(
            self
            .__get_summary_tidy_and_merge_coef_r_coef_dict__()  
            .select_rows(query=f"model=={models}")
            .mutate({'label': lambda col: col['model']})
        )
        # term will be ordered
        tmp = self.__add_ordered_coefs__(tmp, coef_labels)
        tmp = tmp .mutate({'term': lambda col: col['value_label_final']})
        tmp = self.__plot_coef_get_data__(tmp, models, color, facet, shape)
        tmp = tmp.mutate_type(col2type=None, from_to={"object":'str'})
        tmp = tmp.drop_rows(regex={'term':'(^<b> *$)|(^<b>.*:.*$)'})
        # if not coef_labels:
        #     coef_labels=self.__get_coef_groups__()
        #     remove_default_group_label=True
        # if coef_labels:
        #     tmp=self.__plot_coef_order_terms__(coef_labels, coef_wrap,
        #                                        wrap_char="<br>",
        #                                        switch_axes=switch_axes)
        
        # wrap labels 
        # -----------
        if coef_wrap:
            tmp = (tmp
                   .mutate(var_to_wrap='term', wrap=coef_wrap, wrap_char='<br>')
                   .mutate(var_to_wrap='value_label', wrap=coef_wrap, wrap_char='<br>')
                   )

        # tmp = self.__plot_coef_clean_term__(tmp, coef_labels)
        # if tmp.drop_duplicates('term').nrow==tmp.drop_duplicates('value_label').nrow:
        #     tmp=tmp.mutate({'term': lambda col: col['value_label']})
        # if remove_default_group_label:
        # 
        # Add text
        # --------
        if text:
            tmp=(
                tmp
                .rename(columns={'p.value':'pvalue'}, inplace=False)
                .case_when({
                    'sig': {
                        f"(pvalue<0.01)": f"'***'",
                        f"(pvalue<0.05)": f"'**'",
                        f"(pvalue<0.1)": f"'*'",
                        True:"''",
                    }
                })
                .mutate_rowwise({'sig_text': lambda x: f"{round(x['estimate'], digits)} {x['sig']}"})
            )
        return tmp


    def __plot_coef_set_interaction_str__(self, tab,
                                          interaction_str='x'):
        cat_old = list(tab.term.cat.categories)
        cat_new = [re.sub(pattern='\\*', repl=interaction_str,
                          string=co) for
                   co in cat_old]
        for co, cn in zip(cat_old, cat_new):
            if cn!=co:
                tab.term=tab.term.cat.rename_categories({co: cn})
        return tab


    def __plot_get_replace_dict__(self, d):
        newdict={}
        for k, vs in d.items():
            for v in vs:
                newdict[v] = k
        return newdict


    def __ggtheme__(self, *args, **kws):
        g =gg.theme(
                 ## ------
                 ## legend
                 ## ------ 
                 legend_position = "top",
                 # legend_position = [0.12, .96],
                 legend_justification = FloatVector([0, .9]),
                 legend_direction='horizontal',
                 # legend_direction='horizontal',
                 legend_title = gg.element_text( size=11, face='bold'),
                 # legend_text  = gg.element_text( size=10),
                 # legend_text_legend=element_text(size=10),
                 # legend_text_colorbar=None,
                 # legend_box=None,
                 # legend_box_margin=None,
                 # legend_box_just=None,
                 # legend_key_width=None,
                 # legend_key_height=None,
                 # legend_key_size=None,
                 # legend_margin=None,
                 # legend_box_spacing=None,
                 # legend_spacing=None,
                 # legend_title_align=None,
                 # legend_entry_spacing_x=None,
                 # legend_entry_spacing_y=None,
                 # legend_entry_spacing=None,
                 # legend_key=None,
                 # legend_background=None,
                 # legend_box_background=None,
                 strip_background = gg.element_rect(colour="transparent",
                                                    fill='transparent'),
                 # strip_placement = "outside",
                 strip_text_x        = gg.element_text(size=10, face='bold', hjust=0),
                 strip_text_y        = gg.element_text(size=9, face="bold", vjust=0,
                                                       angle=-90),
                 ##panel_grid_major  = element_blank(),
                 # panel_grid_minor_x  = gg.element_blank(),
                 # panel_grid_major_x  = gg.element_blank(),
                 # panel_grid_minor_y  = gg.element_blank(),
                 # panel_grid_major_y  = gg.element_blank(),
                 panel_grid_minor_y  = gg.element_line(colour="grey", size=.3, linetype=3),
                 panel_grid_major_y  = gg.element_line(colour="grey", size=.3, linetype=3),
                 panel_grid_minor_x  = gg.element_line(colour="grey", size=.3, linetype=3),
                 panel_grid_major_x  = gg.element_line(colour="grey", size=.3, linetype=3),
                 # border 
                 # ------
                 panel_border      = gg.element_blank(),
                 axis_line_x       = gg.element_line(colour="black", size=.2, linetype=1),
                 axis_line_y       = gg.element_line(colour="black", size=.2, linetype=1),
                 # axis_line_y       = gg.element_line(colour="black"),
                 legend_background  = gg.element_rect(fill='transparent'),
                 # legend_key_height = grid::unit(.1, "cm"),
                 # legend_key_width  = grid::unit(.8, "cm")
                 axis_ticks_x        = gg.element_blank(),
                 axis_ticks_y        = gg.element_blank(),
                 axis_text           = ggtxt.element_markdown(),
                 axis_text_y         = ggtxt.element_markdown(),
                 axis_text_x         = ggtxt.element_markdown(),
                 axis_title_x        = ggtxt.element_markdown(
                     size  = kws.get("xlab", {}).get('size',11),
                     color = kws.get("xlab", {}).get('color','black'),
                 ),
                 axis_title_y        = ggtxt.element_markdown(
                     size  = kws.get("ylab", {}).get('size',11),
                     color = kws.get("ylab", {}).get('color','black'),
                 ),
                 plot_title	     = ggtxt.element_markdown(
                     hjust  = kws.get("title", {}).get('hjust',0),
                     size   = kws.get("title", {}).get('size',13),
                     colour = kws.get("title", {}).get('color', 'grey40'),
                     face   = kws.get("title", {}).get('face', 'bold')
                 ),
                 plot_subtitle       = ggtxt.element_markdown(
                     hjust  = kws.get("subtitle", {}).get('hjust',0),
                     size   = kws.get("subtitle", {}).get('size',11),
                     colour = kws.get("subtitle", {}).get('color', 'grey60'),
                     face   = kws.get("subtitle", {}).get('face', robj.NULL)
                 ),
            )
        return g


    def __ggguides__(self, ncol=1, leg_title_pos='top'):
        keywidth=2
        keyheight=.9
        g= gg.guides(colour = gg.guide_legend(title_position = leg_title_pos,
                                              ncol=ncol,
                                              size=8,
                                              keywidth=keywidth,
                                              keyheight=keyheight,
                                              title_hjust=0),
                     fill = gg.guide_legend(title_position = leg_title_pos,
                                            ncol=ncol,
                                            size=8,
                                            keywidth=keywidth,
                                            keyheight=keyheight,
                                            title_hjust=0),
                     shape = gg.guide_legend(title_position = leg_title_pos,
                                             ncol=ncol,
                                             size=8,
                                             keywidth=keywidth,
                                             keyheight=keyheight,
                                             title_hjust=0),
                     linetype = gg.guide_legend(title_position = leg_title_pos,
                                                ncol=ncol,
                                                size=8,
                                                keywidth=keywidth,
                                                keyheight=keyheight,
                                                title_hjust=0),
                     )
        return g        


    def __ggfigure_save__(self, tab, g,
                          fn, fn_format_fig, fn_format_tab, fn_figsize,
                          save_tab=True):
        '''
        See documentation in models.regression.plot_coef
        '''
        if fn:
            root = os.path.splitext(fn)[0]
            # table
            if save_tab:
                if fn_format_tab=='xlsx' or not fn_format_tab:
                    fn = f"{root}.xlsx"
                    tab.to_excel(fn, sheet_name='Sheet1', index=False)
                elif fn_format_tab=='csv':
                    fn = f"{root}.{fn_format_tab}"
                    tab.to_csv(fn, sep=';', index=False, decimal='.')
            # figure
            if not fn_format_fig:
                fns = [f"{root}.png", f"{root}.pdf"]
                [g.save(filename=str(fn), width=fn_figsize[0],
                        height=fn_figsize[1]) for fn in fns]
            else:
                fn = f"{root}.{fn_format_fig}"
                g.save(filename=str(fn), width=fn_figsize[0],
                       height=fn_figsize[1])
        

    def __plot_coef_get_data__(self, tab, models, color, facet, shape):
        tmp=(
            # self
            # .results
            tab
            .mutate({'facet': lambda col: col['label']})
            .mutate({'color': lambda col: col['label']})
            .mutate({'shape': lambda col: col['label']})
            .replace({'facet': facet} , regex=False)
            .replace({'color': color} , regex=False)
            .replace({'shape': shape} , regex=False)
        )
        # if coef_labels:
        #     coef_to_select = coef_labels if isinstance(coef_labels, list) else\
        #         [item for sublist in [*coef_labels.values()] for item in sublist]
        # else:
        #     coef_to_select = []
        # if '(Intercept)' not in coef_to_select:
        #     tmp=tmp.drop_rows(regex={'term':"Intercept"})
        return tmp


    def __plot_coef_clean_term__(self, tab, coef_labels):
        if not coef_labels:
            tab=(
                tab
                .mutate_rowwise({'term': lambda col: re.sub(pattern='\`',
                                                            repl=' ',
                                                            string=col['term'])})
                .mutate_rowwise({'term': lambda col: re.sub(pattern=':',
                                                            repl=' x ',
                                                            string=col['term'])})
            ) 
        else:
            cats=tab.term.cat.categories.tolist()
            cats=[re.sub(pattern='\`', repl=' ', string=c) for c in cats]
            cats=[re.sub(pattern=':', repl=' x ', string=c) for c in cats]
            tab=(
                tab
                .mutate_type(col2type={'term':'char'}  )
                .mutate_rowwise({'term': lambda col: re.sub(pattern='\`',
                                                            repl=' ',
                                                            string=col['term'])})
                .mutate_rowwise({'term': lambda col: re.sub(pattern=':',
                                                            repl=' x ',
                                                            string=col['term'])})
                .mutate_categorical(var='term', cats=cats, ordered=True)
            ) 
        return tab
        

    def __plot_coef_order_terms__(self, 
                                  coef_labels,
                                  coef_wrap,
                                  wrap_char,
                                  switch_axes):
        # 
        dfgroups =eDataFrame()
        coef_order=[]
        groups=[]
        # 
        if isinstance(coef_labels, dict):
            dfgroups = eDataFrame({'value_label': list(coef_labels)})
            for group, coefs in coef_labels.items():
                group=f"<b>{group}"
                groups    +=[group]
                # 
                coef_order+=[group]
                for coef in coefs:
                     coef_order+=[coef]
            dfgroups=(
                eDataFrame(list(product(groups, self.results.label)))
                .rename_cols(columns={0:'term', 1:'label'}, tolower=False)
            )
        else:
            coef_order=list(np.array(list(coef_labels)).flatten())
        # 
        if not switch_axes:
            coef_order=coef_order[::-1]
        res=(
            self
            .results
            .unnest(col='summ_tidy', id_vars='label')
            .bind_row(dfgroups)
            .select_rows(query=f"term=={coef_order}")
            .mutate_categorical(var='term', cats=coef_order, ordered=True,
                                wrap=coef_wrap, wrap_char=wrap_char)
            .mutate_case({
                 'value_label': {
                    f"(pd.isna(value_label))": "copy(term)" ,
                    True:f"copy(value_label)"
             	   }
             })
            # .mutate({'value_label': lambda col: col['term']})
        )
        # If caterogies across models and categorical variables are unique
        # use the value label of the cat var instead
        # Ex: Men (var value) instead of genderMen (term) 
        nterms         = len(res.term.unique())
        ncatvar_values = len(res.value_label.unique())
        if nterms == ncatvar_values:
            coef_order_value_labels = coef_order.copy()
            for idx, coefi in enumerate(coef_order):
                value=(
                    res
                    .select_rows(query=f"term=='{coefi}'")
                    .value_label.values[0]
                )
                if value!=coefi:
                    coef_order_value_labels[idx] = value
            res=(
                res
                .mutate_case({
                    'value_label': {
                        f"(pd.isna(value_label))": "copy(term)",
                        f"value_label!=term": "copy(value_label)",
                        True:f"copy(value_label)"
             	    }
                })
            )
            coef_order=coef_order_value_labels
            res=(
                res
                .mutate_categorical(var='value_label', cats=coef_order, ordered=True,
                                    wrap=coef_wrap, wrap_char=wrap_char)
            )
        else:
            res= res.mutate({'value_label': lambda col: col['term']})
        return res


# **** print

    def __print_header_info__(self, labels=None):
        self.__print_line__()
        res=(
            self
            .results
            .select_cols(names=['label','family', 'depvar'])
        )
        if labels:
            labels=[labels] if isinstance(labels, str) else labels
            res=res.select_rows(query=f"label =={labels}")
        print(res.to_string(index=False), flush=True)
        self.__print_line__()

        
    def __print_line__(self):
        print(f"====================================", flush=True)

        
# **** get

    def __get_variables__(self, formula):
        formula=re.sub(pattern=' ', repl='', string=formula)
        vars = re.split("~|\+|\||\*", formula)
        vars = [v for v in vars if v!='']
        return vars

    # def __get_data__(self, formula, na):
    #     vars = self.__get_variables__(formula)
    #     if na=='omit':
    #         res = self.data[vars].dropna(subset=None, axis=0)
    #         omitted = [index for index, row in
    #                    self.data[vars].iterrows() if row.isnull().any()]
    #     return res, omitted


    def __get_info__(self, model):
        info = self.results.select_rows(query=f"label=='{model}'")
        print(f"\nRegression label: {model}")
        print(f"Formula: {info['formula'].values[0]}")
        print(f"Family: {info['family'].values[0]}")
        # print(f"Function: {info['function']}")

    def __get_regression_tidy__(self, mod, fit, idx, formula, family, label):
        if self.engine=="python":
            tmp = pd.DataFrame({'idx':idx,
                                'label':label,
                                'depvar':formula.split("~")[0].strip(),
                                "formula":formula,
                                'family':family,
                                "mod":[mod],
                                "fit":[fit],
                                "summ_tidy":[self.__get_summary1__(fit)],
                                "summ_default":[self.__get_summary2__(fit)],
                                "summ3":[self.__get_summary3__(mod, fit)],
                                'Obs':fit.nobs,
                                'aic':fit.aic,
                                'bic':fit.bic,
                                'r2':1-(fit.deviance/ fit.null_deviance),
                                'rmse':np.nan ,
                                # 'rmse':np.sqrt(np.mean((self['y']-fit.predict())**2))
                                })
        return tmp

# **** set

    def __set_order_coefs__(self, models=None, coefs=None):
        if not models:
            models=self.models
        if not coefs:
            coefs=self.coefs_df.term.values.tolist()
        tab=(
            self
            .coefs_df
            .select_rows(query=f"model=={models}")
        )
        order=[]
        for model in self.models:
            for var in tab.term.values.tolist():
                if var not in order:
                    order+=[var]
        res=(
            self
            .coefs_df
            .select_rows(query=f"model=={models}")
            .select_rows(query=f"term=={coefs}")
            .mutate_categorical(var='term', cats=order, ordered=True, wrap=False)
        )
        return res


    def __set_engine__(self, models, engine):
        for label, dct in models.items():
            self.engine[label]=engine
            

    def __set_na_action__(self, models, na):
        for label, dct in models.items():
            self.na[label]=na


    def __set_model_info__(self, models, *args, **kws):
        for label, model in models.items():
            self.formula[label] = model[0]
            self.family[label]  = model[1]


    def __set_data__(self, models, data, *args, **kws):
        self.__build_variables_df_anc__()
        for label, model in models.items():
            cols=(
                self
                .variables
                .select_rows(query=f"model=='{label}'")
                .select_cols(names='var_labels')
                .drop_rows(dropna=True)
                .values
                .flatten()
                .tolist()
            )
            cols = {k:v for d in cols for k, v in d.items()}
            # vars = ru.formula2varlist(model[0])
            if len(model)==2:
                assert isinstance(data, eDataFrame), "'data' must be provided"
                self.data[label] = data.select_cols(names=cols)
            if len(model)==3:
                assert isinstance(model[2], eDataFrame), ("Third element of "+\
                                                          "'models' must be a DataFrame"
                                                          )
                self.data[label] = model[2].select_cols(names=cols)
            
        
                
    def __set_labels__(self, models):
        self.labels += list(models.keys())
        self.models += list(models.keys())

# **** build
# ***** build coef dict 

    def __build_coef_dict__(self):
        self.coefs_dict = {}
        for model in self.labels:
            # 
            self.coefs_dict[model] = {}
            self.coefs_dict[model]['inputs'] = {}
            self.coefs_dict[model]['interactions'] = {}
            # 
            formula      = self.variables_dict_raw[model]['formula_arg']
            inputs       = self.variables_dict_raw[model]['inputs']
            interactions = self.variables_dict_raw[model]['interactions']
            #
            inputs_default_group = ' '
            interactions_default_group = 'Interactions'
            # inputs 
            # ------
            # a. formula
            if not inputs:
                self.__build_coef_order_from_formula__(model,
                                                       inputs_default_group,
                                                       'inputs')
                # b. inputs = [...]
            elif isinstance(inputs, list):
                self.__build_coef_order_from_list_inputs__(model, inputs_default_group)
            elif isinstance(inputs, dict):
                for k, v in inputs.items():
                    # c. inputs = {'oldnames': "newnames"}
                    if isinstance(v, str):
                        self.__build_coef_order_from_dict_renaming_str_inputs__(
                            model,
                            group=inputs_default_group,
                            oldname=k,
                            newname=v)
                        # d. inputs = {'group': [...]}
                    elif isinstance(v, list):
                        self.__build_coef_order_from_dict_list_str_inputs__(model,
                                                                            group=k,
                                                                            variable_list=v)
                        # e. inputs = {'group': {'oldnames': "newnames"}}
                    elif isinstance(v, dict):
                        for oldname, newname in v.items():
                            self.__build_coef_order_from_dict_renaming_str_inputs__(
                                model=model,
                                group=k,
                                oldname=oldname,
                                newname=newname)
            # interactions 
            # ------------
            # a. formula
            if not interactions and formula:
                self.__build_coef_order_from_formula__(model,
                                                       interactions_default_group,
                                                       'interactions')
            # b. interactions = [(...)]
            elif isinstance(interactions, list):
                group=interactions_default_group
                self.__build_coef_order_from_list_interactions__(
                    model=model, interactions=interactions, group=group)
            #  interactions = {'group': [(...)], 'g2':[(...)]}
            elif isinstance(interactions, dict):
                self.__build_coef_order_from_dict_interactions__(
                    model, interactions_group=interactions)
            else:
                group=interactions_default_group
                self.coefs_dict[model]['interactions'][group]=[]
            

    def __build_coef_order_from_formula__(self, model, group, var_type):
        '''
        var_type is 'input' or 'interaction'
        '''
        formula=self.variables_dict_raw[model]['formula_arg']
        vars=(
            models_utils()
            .formula2df(formula)
            .select_rows(query=f"part=='{var_type}'")
            .select_cols(names='var_list')
            .values
            .flatten()[0]
        )
        if vars:
            vars = {v:v for v in vars} if var_type=='inputs' else vars
        else:
            vars={} if var_type=='inputs' else []
        self.coefs_dict[model][var_type][group]=vars


    def __build_coef_order_from_list_inputs__(self, model, inputs_default_group):
        # b. inputs = [...]
        vars =  self.variables_dict_raw[model]['inputs']
        vars = {v:v for v in vars}
        self.coefs_dict[model]['inputs'][inputs_default_group]=vars


    def __build_coef_order_from_dict_renaming_str_inputs__(self,
                                                           model,
                                                           group,
                                                           oldname,
                                                           newname):
        current_vars=self.coefs_dict[model]['inputs']
        # initialize in case it is empyt
        current_vars = current_vars.get(group, {})
        updated_vars = current_vars | {oldname:newname}
        self.coefs_dict[model]['inputs'][group]=updated_vars


    def __build_coef_order_from_dict_list_str_inputs__(self, model, group,
                                                       variable_list):
        vars = {v:v for v in variable_list}
        current_vars=self.coefs_dict[model]['inputs']
        # initialize in case it is empyt
        current_vars = current_vars.get(group, {})
        updated_vars = current_vars | vars
        self.coefs_dict[model]['inputs'][group]=updated_vars
    
    def __build_coef_order_from_dict_interactions__(self, model,
                                                    interactions_group):
        #  interactions_group = {'group': [(...)]}
        for group, interactions in interactions_group.items():
            terms_list=[":".join(terms) for terms in interactions]
            self.coefs_dict[model]['interactions'][group]=terms_list


            
    def __build_coef_order_from_list_interactions__(self,
                                                    model, 
                                                    interactions, group):
        #  interactions_group =  [(...)]
        self.coefs_dict[model]['interactions'][group]=[]
        for terms in interactions:
            nterms=len(terms)
            inter_combinations=[]
            for n in range(2, nterms+1):
                inter_combinations+=list(itertools.combinations(terms, n))
            terms_list=[":".join(terms) for terms in inter_combinations]
            self.coefs_dict[model]['interactions'][group]+=terms_list


# ***** build coef dict expand categories

    def __build_coef_dict_expand_categories__(self):
        coefs_dict_exp = {}
        for model, input_and_interactions in self.coefs_dict.items():
            coefs_dict_exp[model] = {}
            coefs_dict_exp[model]['inputs'] = {}
            coefs_dict_exp[model]['interactions'] = {}
            # inputs 
            # ------
            for group, variables in input_and_interactions['inputs'].items():
                coefs_dict_exp[model]['inputs'][group]={}
                for oldvar, newvar in variables.items():
                    cats=self.__build_coef_categorical_variable__(model,
                                                                  oldvar=oldvar,
                                                                  newvar=newvar)
                    if cats:
                        coefs_dict_exp[model]['inputs'][group]|=cats
                    else:
                        coefs_dict_exp[model]['inputs'][group][oldvar]=newvar
            # interactions 
            # ------------
            for group, variables in input_and_interactions['interactions'].items():
                coefs_dict_exp[model]['interactions'][group] = []
                for inter_term in variables:
                    terms_list=[]
                    terms=inter_term.split(":")
                    for term in terms:
                        cats=self.__build_coef_categorical_variable__(
                            model, oldvar=term, newvar=term)
                        if cats:
                            terms_list += [list(cats.values())]
                        else:
                            terms_list += [[term]]
                    terms_list=list(itertools.product(*terms_list))
                    terms_list=[":".join(term) for term in terms_list]
                    coefs_dict_exp[model]['interactions'][group] += terms_list
        self.coefs_dict_exp = coefs_dict_exp

    
    def __build_coef_categorical_variable_get_categories__(self, model, variable):
        vars = None
        if is_categorical_dtype(self.data[model][variable]):
            vars = self.data[model][variable].cat.categories.to_list()
        elif is_object_dtype(self.data[model][variable]):
            vars = self.data[model][variable].unique().tolist()
        return vars


    def __build_coef_categorical_variable__(self, model, oldvar, newvar):
        vars=None
        cats = self.__build_coef_categorical_variable_get_categories__(model=model,
                                                                       variable=newvar)
        if cats:
            oldvars = [f"{oldvar}{cat}" for cat in cats]
            newvars = [f"{newvar}{cat}" for cat in cats]
            vars = {o:n for o, n in zip(oldvars, newvars)}
        return vars



# ***** others

    def __build_coef_df__(self):
        d=self.coefs_dict_exp
        inputs = eDataFrame(
            pd.DataFrame.from_records(
                [
                    (model, group, varnew)
                for model, inputs_and_interactions in d.items()
                for group, variables in inputs_and_interactions['inputs'].items()
                for varnew in variables.values()
                    # for oldvarname, varname in group.items()
                ], columns=['model', 'value_group', 'term']
            )
        )
        interactions = eDataFrame(
            pd.DataFrame.from_records(
                [
                    (model, group, var)
                for model, inputs_and_interactions in d.items()
                for group, variables in inputs_and_interactions['interactions'].items()
                for var in variables
                    # for oldvarname, varname in group.items()
            ], columns=['model', 'value_group', 'term']
            )
        )
        res=(
            inputs
            .bind_row(interactions)
        )
        self.coefs_df=res


    def __build_coef_value_label_variable__(self, tab):
        if 'model' not in tab.names():
            tab=tab.mutate({'model': lambda col: col['label']})
        tab=(
            tab
            .groupby(['value_label', 'model'])
            .mutate({'value_label_occurrencies': lambda col: len(col['value_label'])})
            # 
            .groupby(['var_label', 'model'])
            .mutate({'value_label_occurrencies': lambda col: col['value_label_occurrencies'].max()})
            # 
            .mutate_rowwise({'value_label_detail': lambda col:
                             f"{col['value_label']} ({col['var_label']})"})
            .mutate_rowwise({'value_label_detail': lambda col:
                             self.__build_coef_value_label_variable_anc__(col)})
            # 
            .mutate_case({
                 'value_label_final': {
                    f"(value_label_occurrencies==1)": "copy(value_label)",
                    f"(value_label_occurrencies!=1)": "copy(value_label_detail)",
                   }
             })
        )
        return tab


    def __build_coef_value_label_variable_anc__(self, col):
        if col['var_type']=='interaction':
            interaction=col['variable']
            variables=interaction.split(":")
            variables=[var for var in variables if var not in col['value_label']]
            variables = ", ".join(variables)
            res=f"{col['value_label']} ({variables})"
        else:
            res = col['value_label_detail']
        return res


# **** others
# ** Old
# *** plot
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
        if   which=='coef':
            g= self.__plot_coefr__(args)
        elif which=='pred':
            g= self.__plot_predr__(args)
        # 
        # elif self.engine=='python':
        #     g= print("To be implemented")
        return g


    def dev_off(self):
        grd.dev_off()

    def __regression__(self, formula, family, na,  *args, **kws):
        self.engine=engine
        if self.engine=='python':
            res = self.__dispatcher__(formula, family, na, *args, **kws)
        else:
            res = self.__dispatcherr__(formula, family, na, *args, **kws)
        return res



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

# *** Ancillary

    def __check_multinomial__(self, models):
        '''
        Check if multinomial family is combined with other family distributions
        '''

        # check existing models
        notmultinomial = False
        if self.results.nrow==0:
            multinomial=False
        else:
            multinomial=self.multinomial
        for label, model in models.items():
            multinomial    = True if model[1] == 'multinomial' else multinomial
            notmultinomial = True if model[1] != 'multinomial' else notmultinomial

        if multinomial and notmultinomial:
            raise ValueError("Currently, combination of multinomial with other family "\
                             "distributions are not allowed")
        if multinomial and not notmultinomial:
            self.multinomial=True
                


# ** Python Engine (old/to remove)
# *** Regression


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
            tmp = self.__get_est_df__(mod, fit, idx, label, formula)
            tab=pd.concat([tab, tmp], axis=0, ignore_index=True)
        return tab
        

# *** Summary

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

    def predict_python(self, newdata, model_index=1):
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
        
# *** Plots
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
        pred = self.predict_python(newdata, model_index=model_index)
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


    


# ** draft

    def __get_summary_tidy_and_merge_coef_r_coef_dict__(self):
        tmp=(
            self
            .results
            .unnest(col='summ_tidy', id_vars='model')
        )
        taba=(
            self
            .__build_coef_value_label_variable__(tmp) 
            .drop_rows(regex='Intercept')
            .mutate_rowwise({'term_list': lambda col: col['term'].split(':')})
            .mutate_rowwise({'term_list': lambda col: sorted(col['term_list'])})
            .mutate_rowwise({'term_interac_reordered': lambda col: ":".join(col['term_list'])})
        )
        tabb=(
            self
            .coefs_df
            .mutate_rowwise({'term_list': lambda col: col['term'].split(':')})
            .mutate_rowwise({'term_list': lambda col: sorted(col['term_list'])})
            .mutate_rowwise({'term_interac_reordered': lambda col: ":".join(col['term_list'])})
            .rename_cols(columns={'term':'terms_dict'}, tolower=False)
            .drop_cols(names='term_list')
        )
        res=(
            taba
            .join(tabb, how='left', on=['term_interac_reordered', 'model'])
        )
        return res


    def __add_ordered_coefs__(self, tab, coef_labels=None,
                              coef_wrap=None, wrap_char=None):

        coef_labels_to_select=coef_labels
        coef_labels={}
        groups=[]
        for idx, row in self.coefs_df.iterrows():
            group=f"<b>{row['value_group']}"
            groups+=[group] if group not in groups else []
            term = row['term']
            if term not in coef_labels.get(group, []):
                coef_labels[group] = coef_labels.get(group, []) + [term]
            
        # dict->list
        order=self.__plot_coef_order_terms__(coef_labels=
                                             self.coefs_dict_exp)
        if coef_labels_to_select:
            order=[v for v in order if
                   v in coef_labels_to_select or
                   bool(re.search(pattern="\<b\>", string=v))
                   ]

        groups_df=(
            # ds.
            eDataFrame(list(product(self.models, groups)))
            .rename_cols(columns={0:'model', 1:'term'}, tolower=False)
            .mutate({
                "label" : lambda col: col['model'],
                "value_label_final": lambda col: col['term'],
                "terms_dict": lambda col: col['term'],
                     })
        )
        res=(
            tab
            .bind_row(groups_df)
            .mutate_categorical(var='terms_dict', cats=order[::-1],
                                ordered=True, wrap=False)
            .sort_values(['terms_dict'], ascending=True)
            .select_rows(query=f"term == {order}")
        )
        value_label_final_cats=(
            res
            .drop_duplicates(['value_label_final'])
            .select_cols(names=['value_label_final'])
            .values
            .flatten()
        )
        res=(
            res
            .mutate_categorical(var='value_label_final', cats=value_label_final_cats,
                                ordered=True, wrap=False)
        )
        return res
 
    # def __add_ordered_coefs__(self, tab, coef_labels=None,
    #                           coef_wrap=None, wrap_char=None):
    #     if not coef_labels:
    #         coef_labels=self.coefs_dict_exp
    #     order= self.__plot_coef_order_terms__(
    #         coef_labels=coef_labels,
    #         coef_wrap=coef_wrap,
    #         wrap_char=wrap_char
    #     )
    #     res=(
    #         eDataFrame({'coef_labels': order, "order":range(1, len(order)+1)})
    #         .join(tab, how='left',
    #               left_on=["coef_labels"],
    #               right_on=["terms_dict"])
    #         .mutate_categorical(var='coef_labels', cats=order[::-1],
    #                             ordered=True, wrap=False)
    #         .mutate_case({
    #             'value_label_final': {
    #                 f"pd.isna(value_label_final)": "copy(coef_labels)",
    #                 True:f"copy(value_label_final)"
    #             }
    #          })
    #     )
    #     cats=(
    #         res
    #         . sort_values(['order'], ascending=True)
    #         .select_cols(names=['value_label_final', 'order'])
    #         .drop_duplicates()
    #         .value_label_final
    #         .values
    #         .tolist()
    #     )
    #     res=res.mutate_categorical(var='value_label_final', cats=cats[::-1],
    #                                ordered=True, wrap=False)
    #     return res


    def __plot_coef_order_terms__(self, 
                                  coef_labels=None,
                                  coef_wrap=None,
                                  wrap_char=None):
        d={}
        for model, inp_and_interac in coef_labels.items():
            for group, vars in inp_and_interac['inputs'].items():
                d[group]=d.get(group, []) + list(vars.values())
        for model, inp_and_interac in coef_labels.items():
            for group, vars in inp_and_interac['interactions'].items():
                d[group]=d.get(group, []) + vars
        order=[]
        for group, vars in d.items():
            order+=[f"<b>{group}"]
            for var in vars:
                if var not in order:
                    order+=[var]
        return order

    
# * Causal Analysis and Inference
# ** DiD

class did():
    def __init__(self):
        # self.multinomial        = None
        # self.labels             = []
        # self.models             = []
        # self.engine             = {}
        # self.na                 = {}
        # self.formula            = {}
        # self.family             = {}
        self.data               = {}
        self.results            = eDataFrame()
        # self.variables          = eDataFrame()
        # self.variables_dict_raw = {}

# * Machine Learning
# ** Dimension Reduction
# *** PCA

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



# * Power and Sample size
# ** class

class power():
    def __init__(self,
                 diff,
                 alpha=0.05,
                 power=0.8,
                 two_tail=True,
                 ratio=1,
                 seq_design={
                     'typeOfDesign': 'OF',
                     'stops_at'    :[.33, .66, 1]
                 },
                 type=None,
                 type2prop={"prop1":None},
                 ):
        '''
        Compute the sample size

        Input 
        -----
        diff     : difference between average value of the outcome in the
                   two groups (e.g., different in proportions, mean, etc.)
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

        seq_design a dictionary with parameters for the R package 'rpact' for
                   group sequential design. The core parameters are:

                   typeOfDesign  string with function to compute the critical
                                 values at each peek. Default is 'OF'
                                 O’Brien-Fleming (Proschan et al., 2006).
                                 See rpact for other options
        
                   stops_at      Ration of the total sample to peek [.33, .66, 1]

        type*   : a dictionary with type-specific arguments. The * can be:
          
                  2prop : prop1  a list or float with the proportion(s)
                          presumed for group 1
        

        '''
        assert type, "'type' must be provided"
        assert diff, "The effect size 'diff' must be provided"
        
        if isinstance(diff, float):
            diff = [diff]
        self.diff=diff
        self.alpha=alpha
        self.power=power
        self.ratio=ratio
        self.two_tail=two_tail
        self.type=type
        self.data=None
        self.seq_design=seq_design
        self.type2prop=type2prop
        if type=='2prop':
            self.data = self.__2prop__()



# ** methods
# *** default

    def __str__(self):
        print(self.data, flush=True)
        return None

    def __repr__(self):
        print(self.data, flush=True)
        return ''


    
# *** plot

    def plot(self, cost_per_observation=None, design='both'):
        '''
        Plot the results of the power analysis

        # Input 
        # -----
        cost_per_observation  number indicating the cost per
                              observation. If provided,
                              plot the total cost.

        design    string. It can be,
                  - 'fixed'      plot only the fixed design
                  - 'sequential' plot only the sequential design
                  - 'both'       Default. Plot the fixed and the
                                 sequential design
        '''
        if self.type=='2prop':
            ax = self.__plot_2prop__(cost_per_observation, design)
        return ax


    def __plot_2prop__(self, cost_per_observation, design):

        vars=['prop1', 'prop2', 'sample_size_group1', 'sample_size_group2',
              'design', 'sample_size_group1_H1expected', 
              'diff', 'peek']
        # collecting data to plot 
        # ----------------------- 
        tab=self.__plot_select_design__(design)
        tab=(
            tab
            .select_cols(names=vars)
            .pivot_longer(id_vars=None, value_vars=['sample_size_group1',
                                                    'sample_size_group2'],
                          var_name='group', value_name='sample_size',
                          ignore_index=True)
            .mutate_type(col2type={'diff': str} )
            .mutate_case({
                 'Treatment group': {
                     f"(group=='sample_size_group1' )": "0 (e.g., control)", 
                     f"(group=='sample_size_group2' )": "1 (e.g., treatment)",
                   }
             })
            .combine(cols=['peek', 'diff'], colname='group_seq_design',
                     sep='_', remove=False)
        )
        # 
        # 
        tab_maximum = self.get_sample_size_max(design='Fixed design')
        # 
        extra_columns={}
        if cost_per_observation:
            tab_maximum=(
                tab_maximum   
                .mutate({
                    'Cost': lambda col: round(col['ntotal']*
                                              cost_per_observation, 2),
                })
                .mutate_rowwise({'Cost': lambda col: f"$ {col['Cost']}"})
            )
            extra_columns={"Cost":"Cost"}
        tab_maximum=tab_maximum.drop_cols(names='ntotal')
        tab_maximum=self.__get_main_columns__(tab_maximum,
                                              extra_columns=extra_columns)

        x = "prop2"
        y = "sample_size"
        color='diff'
        shape='design'
        size='`Treatment group`'
        group_seq_design='group_seq_design'
        twotailed = 'Yes' if self.two_tail else "No"
        labx= latex2exp.TeX("Proportion of 'positive' outcome cases (Y=1) "+\
                            'in the group of interest ($\pi_t$)')
        laby='Sample size required to detect the difference'
        labcolor= latex2exp.TeX('Group difference ($\\pi_t - \\pi_c$)')
        labshape='Design'
        title = "Sample size calculation"
        subtitle = latex2exp.TeX("Info:  "+
                    f"$\\alpha$: {self.alpha}; "
                    f"Power ($\\beta$): {self.power}; "+
                    f"Test : {self.type}; "+
                    f"Two-sided : {self.two_tail}; "+
                    f"Ratio between sample sizes in each group : {self.ratio}"
                    )
        # Plot 
        # ----
        g = gg.ggplot(tab)
        if design in ['fixed', 'both']:
            g = self.__plot_2prop_fixed_design__(g, x, y, color, shape, size, tab)
        if design in ['sequence', 'both']:
            g = self.__plot_2prop_sequence_design__(g, x, y,
                                                    color, shape, size, 
                                                    group_seq_design,
                                                    tab,
                                                    design)
            # 
        g = (
            g
            + gg.labs(
                x        = labx,
                y        = laby,
                color    = labcolor, 
                shape    = labshape,
                fill     = robj.NULL,
                linetype = robj.NULL,
                title    = title,
                subtitle = subtitle,
                caption  = robj.NULL
                )
            # + gg.scale_size_manual(cols)
            + gg.scale_colour_brewer(palette="Set1") 
            + gg.theme_bw()
            + ggtheme()
            + ggguides()
        )

        # table 
        # -----
        tab_maximum=self.__plot_select_design__(tab=tab_maximum, design=design)
        gtab = gridExtra.tableGrob(tab_maximum, rows = robj.NULL)
        layout = '''
        A
        A
        B'''

        g=patchwork.wrap_plots(A=g ,B=gtab, design=layout)
        print(g, flush=True)
        return g


    def __plot_2prop_fixed_design__(self, g, x, y, color, shape, size, tab):
        g = (
            g
            + gg.geom_point(gg.aes_string(x=x, y=y,
                                          colour=color,
                                          shape=shape,
                                          group=size
                                          ),
                             size=3.5, alpha=.4, position="identity",
                            data=tab.query(f"design=='Fixed design'")) 
            + gg.geom_line(gg.aes_string(x=x, y=y, group=robj.NULL, colour=color),
                           size=.6,
                            data=tab.query(f"design=='Fixed design'")) 
        )
        return g

    def __plot_2prop_sequence_design__(self, g, x, y,
                                       color, shape, size, 
                                       group_seq_design,
                                       tab,
                                       design):
        tab_expected=eDataFrame()
        if design in ['sequence', 'both']:
            tab_expected=(
                tab
                .query(f"design=='Group sequence'")
                .select_cols(names=['prop2', 'sample_size_group1_H1expected',
                                    'design', 'diff'])
                .mutate_rowwise({'design': lambda col: f"{col['design']} (expected)"})
                .mutate({'group_seq_design': lambda col: col['diff']})
                .drop_duplicates()
            )
        # 
        alpha=.4 if design=='sequence' else .1
        g = (
            g
            + gg.geom_point(gg.aes_string(x=x, y=y,
                                          colour=color,
                                          shape=shape,
                                          group=size
                                          ),
                             size=2.5, alpha=alpha, position="identity",
                            data=tab.query(f"design!='Fixed design'")) 
            + gg.geom_line(gg.aes_string(x=x, y=y, group=group_seq_design, colour=color),
                           size=.6, alpha=alpha, linetype=2,
                           data=tab.query(f"design!='Fixed design'")) 
            # 
            + gg.geom_point(gg.aes_string(x=x, y='sample_size_group1_H1expected',
                                          shape=shape,
                                          colour=color),
                           size=3, alpha=alpha+.1, linetype=1,
                           data=tab_expected) 
            + gg.geom_line(gg.aes_string(x=x, y='sample_size_group1_H1expected',
                                         group=group_seq_design, colour=color),
                           size=.6, alpha=alpha+.1, linetype=1,
                           data=tab_expected) 
        )
        return g


    def __plot_select_design__(self, design=None, tab=None):
        if design=='fixed':
            design=['Fixed design']
            # 
        elif design=='sequence':
            design=['Group sequence']
            # 
        elif design=='both':
            design=['Group sequence', 'Fixed design']
            # 
        if tab is None:
            res=self.data.select_rows(query=f"design=={design}")
        else:
            res=tab.select_rows(query=f"Design=={design}")
        return res
        
    # def __plot_2prop__(self):
    #     group_diff_label='Group difference ($\pi_t - \pi_c$)'
    #     taba = (
    #         self
    #         .data
    #         .pivot_longer(id_vars=None, value_vars=['sample_size_group1',
    #                                                 'sample_size_group2'],
    #                       var_name='Group', value_name='Sample size', ignore_index=True)
    #         .replace({'Group':{'sample_size_group1':"Reference (e.g. control group)",
    #                            'sample_size_group2':'Interest (e.g. treatment group)'}} ,
    #                  regex=False, inplace=False)
    #         .rename(columns={'diff':group_diff_label}, inplace=False)
    #         .mutate({group_diff_label: lambda col: round(col[group_diff_label], 4)})
    #     )
    #     tab=taba.query(f"design=='Single stop'")
    #     # 
    #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 6], tight_layout=True)
    #     sns.scatterplot(x='prop2', y='Sample size', data=tab,
    #                     style='Group', hue=group_diff_label, palette='RdBu',s=70,
    #                     ax=ax)
    #     sns.lineplot(x='prop2', y='Sample size', data=tab,
    #                     style='Group', hue=group_diff_label, palette='RdBu',
    #                     ax=ax, alpha=.3, legend=False)
    #     # 
    #     ax.set_xlabel("Proportion of 'positive' outcome cases (Y=1) "+\
    #                   'in the group of interest ($\pi_t$)')
    #     ax.set_ylabel('Sample size required to detect the difference')
    #     # 
    #     # grid
    #     ax.grid(b=None, which='major', axis='both', linestyle='-', alpha=.3)
    #     ax.set_axisbelow(True) # to put the grid below the plot
    #     # legend
    #     handles, labels = ax.get_legend_handles_labels()
    #     leg = ax.legend(loc='upper right')
    #     leg._legend_box.align = "left"
    #     # -------
    #     # Splines (axes lines)
    #     # -------
    #     ax.spines['bottom'].set_visible(True)
    #     ax.spines['left'].set_visible(True)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     title = "Sample size calculation"
    #     subtitle = ("Info:  "+
    #                 f"$\\alpha$: {self.alpha}; "+
    #                 f"Power ($\\beta$): {self.power}; "+
    #                 f"Test : {self.type}; "+
    #                 f"Two-sided : {self.two_tail}; "+
    #                 f"Ratio between sample sizes in each group : {self.ratio}"
    #                 )


    #     # tabgd=taba.query(f"design=='Group sequence'")
    #     # print(tabgd)
    #     # sns.scatterplot(x='prop2', y='Sample size', data=tabgd,
    #     #                 hue=group_diff_label, palette='RdBu',s=20,
    #     #                 legend=False,
    #     #                 alpha=.1,
    #     #                 ax=ax)
    #     # -----
    #     # Title
    #     # -----
    #     plt.subplots_adjust(top=.78)
    #     xcoord=-.07
    #     ycoord=1.13
    #     yoffset=.07
    #     ax.annotate(title, xy=(xcoord, ycoord),
    #                xytext=(xcoord, ycoord), xycoords='axes fraction',
    #                        size=15, alpha=.6)
    #     ax.annotate(subtitle, xy=(xcoord, ycoord-yoffset),
    #                xytext=(xcoord, ycoord-yoffset), xycoords='axes fraction',
    #                        size=11, alpha=.6)
    #     # maximum value
    #     max_sample_sizes = (
    #         tab
    #         .groupby([group_diff_label, 'Group'])
    #         .apply(lambda x: x.nlargest(1, columns=['Sample size']))
    #     )
    #     for idx, row in max_sample_sizes.iterrows():
    #         x = row['prop2']
    #         y = row['Sample size']
    #         group = row['Group']
    #         # 
    #         va = 'bottom' if 'Reference' in group else 'top'
    #         txt = f"Max: {int(row['Sample size']):,}\n({group})"
    #         txt = f'\n{txt}' if va == 'top' else f"{txt}\n"
    #         color='red' if va == 'top' else 'green'
    #         # 
    #         ax.scatter(x, y, s=60, color=color)
    #         ax.text(x, y, s=txt,
    #                 ha='center', va=va, ma='center',
    #                 fontdict=dict(weight='normal', style='italic',
    #                               color=color, fontsize=10, alpha=1))
    #     return ax
        

# *** misc

    def get_design(self, value_group1, diff, raw_fit=False, summary=True):
        '''
        Print output for a specific design obtained from value of the output
        for group 1 and the difference in the output value from this group
        and the other one

        Input
        -----

        value_group1   value of the output for the group 1 (e.g., output proportion
                       in the control group in a 2 proportion test).

        diff           difference in the value of the output between groups

        raw_fit        boolean print the raw R output instead of the tidy version

        summary        boolean, used only when 'raw_fit=True'. Print summary only
        
        '''
        tab=eDataFrame()
        if self.type=='2prop':
            tab=(
                self
                .data
                .select_rows(query=f"prop1=={value_group1}")
                .select_rows(query=f"diff=={diff}")
            )
        if tab.nrow==0:
            tab = power(diff=diff, 
                        alpha=self.alpha,
                        power=self.power,
                        ratio=self.ratio,
                        seq_design=self.seq_design,
                        type2prop=self.type2prop,
                        type=self.type
                        )
            tab = tab.data
        if not raw_fit:
            tab = self.__get_main_columns__(tab)
            print(tab.to_string(index=False, max_colwidth=13,
                                formatters={
                                    'N (group c)' : lambda x: '%.1f' % x,
                                    'N (group t)' : lambda x: '%.1f' % x,
                                    'N (total)'   : lambda x: '%.1f' % x,
                                    'Sig. level'  : lambda x: '%.4f' % x,
                                    'Power'       : lambda x: '%.4f' % x
                                })
                  )
        else:
            res=tab.fit.reset_index(drop=True)[0]
            res_design=tab.fit_design.reset_index(drop=True)[0]
            print(res_design)
            if not summary:
                print(res, flush=True)
            else:
                print(base.summary(res))


    def get_sample_size_max(self, design='Fixed design'):
        maxvalue=self.data.query(f"design=='Fixed design'").select_cols(names=['sample_size_total']).max()[0]
        if self.type=='2prop':
            group1='prop1'
            group2='prop2'
        tab_maximum = (
            self
            .data
            .query(f"sample_size_total=={maxvalue}")
            .select_cols(names=[group1, group2, 'diff'])
            .join(self.data, how='left', conflict="keep_all", suffixes=["_x", "_y"] )
            .mutate({
                'sample_size_group1': lambda col: round(col['sample_size_group1'], 0),
                'sample_size_group2': lambda col: round(col['sample_size_group2'], 0),
                'sample_size_total' : lambda col: round(col['sample_size_total'], 0),
                'ntotal'            : lambda col: round(col['sample_size_total'], 0), # for costs
                'critical_zvalue'   : lambda col: round(col['critical_zvalue'], 4),
                'siglevels'         : lambda col: round(col['siglevels'], 6),
                'stopProb'          : lambda col: round(col['stopProb'], 6),
            })
        )
        return tab_maximum

        
# ** methods (hidden)
# *** 2prop

    def __2prop__(self):
        prop1s=self.__2prop_get_prop1__()
        res=eDataFrame()
        for diffi in self.diff:
            prop2s = prop1s + diffi
            for prop1, prop2 in zip(prop1s, prop2s):
                if 0 < prop2 <1:
                    tmp = self.__2prop_seq_design__(prop1=prop1,
                                                    prop2=prop2,
                                                    diffi=diffi)

                    res = res.bind_row(tmp)
        return res

    def __2prop_seq_design__(self,
                             prop1,
                             prop2,
                             diffi):
        # 
        tail = 2 if self.two_tail else 1
        # 
        gdesign_info=rpact.getDesignGroupSequential(
            sided = tail,
            alpha = self.alpha,
            beta = 1-self.power,
            typeOfDesign = self.seq_design['typeOfDesign']
            # 
            # informationRates = self.seq_design["stops_at"]
            ## futilityBounds = c(0, 0.05)
        )
        gdesign=rpact.getSampleSizeRates(gdesign_info,
                                         pi1 = prop1,
                                         pi2 = prop2)
        es_cohen_hi = pwr2prop.proportion_effectsize(
            prop1=prop2,
            prop2=prop1
        ) # = 2*asin(sqrt(p1))-2*asin(sqrt(p2)) (Cohen, 1988)
        # 
        # get R slots 
        # -----------
        slot1 = gdesign.slots['.xData']
        slot2 = gdesign.slots['.xData']
        slot2 = slot2.find('.design')
        slot2 = slot2.slots['.xData']
        # 
        fixed_design_group1        = slot1.find('nFixed1')
        fixed_design_group2        = slot1.find('nFixed2')
        peeks                      = slot2.find('stages')
        peeks                      = slot2.find('stages')
        critical_zvalues           = slot2.find('criticalValues')
        sigLevels                  = slot1.find('criticalValuesPValueScale').flatten()
        alphaSpent                 = slot2.find('alphaSpent')
        numberOfSubjects           = slot1.find('numberOfSubjects').flatten()
        numberOfSubjects1          = slot1.find('numberOfSubjects1').flatten()
        numberOfSubjects2          = slot1.find('numberOfSubjects2').flatten()
        expectedNumberOfSubjectsH1 = slot1.find('expectedNumberOfSubjectsH1').flatten()[0]
        infoRates                  = slot1.find('informationRates').flatten()
        stopProb                   = gdesign.slots['.xData'].find('rejectPerStage').flatten()
        # stopProb: probability of rejecting H0 on that stage (add up to power level)
        npeeks=len(peeks)
        # 
        base = eDataFrame({
            "prop1"      : [prop1],
            "prop2"      : [prop2],
            'diff'       : [diffi],
            'es_cohen_h' : [es_cohen_hi],
            'pwr'        : [self.power],
            'alpha'      : [self.alpha],
            'two-sided'  : [self.two_tail],
            'fit'        : [gdesign],
            'fit_design' : [gdesign_info]
        })
        seq_design = eDataFrame(
            {
                "sample_size_group1"            : numberOfSubjects1,
                "sample_size_group2"            : numberOfSubjects2,
                "sample_size_total"             : numberOfSubjects,
                "sample_size_group1_H1expected" : [expectedNumberOfSubjectsH1/2]*npeeks,
                "sample_size_total_H1expected"  : [expectedNumberOfSubjectsH1]*npeeks,
                "sample_size_peek_perc"         : infoRates,
                "peek"                          : peeks,
                "critical_zvalue"               : critical_zvalues,
                "siglevels"                     : sigLevels,
                "siglevelsCum"                  : alphaSpent,
                "stopProb"                      : stopProb,
                "design"                        : 'Group sequence'
                # 
            }
        )
        fixed_design = eDataFrame(
            {
                "sample_size_group1"            : fixed_design_group1 ,
                "sample_size_group2"            : fixed_design_group2 ,
                "sample_size_total"             : fixed_design_group1+fixed_design_group2,
                "sample_size_group1_H1expected" : fixed_design_group1,
                "sample_size_total_H1expected"  : fixed_design_group1+fixed_design_group2,
                "sample_size_peek_perc"         : 1,
                "peek"                          : np.max(peeks),
                "critical_zvalue"               : np.abs(qnorm.ppf(self.alpha/tail)),
                "siglevels"                     : self.alpha,
                "siglevelsCum"                  : self.alpha,
                "stopProb"                      : self.power,
                "design"                        : 'Fixed design'
                # 
            }
        )
        tab= (base
              .bind_col((seq_design .bind_row(fixed_design)), ignore_index=False)
              .fillna(inplace=False, axis=0, method="ffill")
              )
        return tab

    def __2prop_get_prop1__(self):
        if not self.type2prop['prop1']:
            prop1s = np.array([0.01] +
                              [round(x, 2) for x in list(np.linspace(0.0, 1, 21))][1:-1] +
                              [0.99])
        else:
            prop1s=self.type2prop['prop1']
            if isinstance(prop1s, float):
                prop1s = np.array([prop1s])
            if isinstance(prop1s, list):
                prop1s = np.array(prop1s)
        return prop1s



# *** misc

    def __get_main_columns__(self, tab, extra_columns={}):
        tab=(
            tab
            .select_cols(names={'design'            : 'Design',
                                'peek'              : 'Peek',
                                'prop1'             : "pi_c",
                                'prop2'             : "pi_t",
                                'sample_size_group1': "N (group c)",
                                'sample_size_group2': "N (group t)",
                                'sample_size_total' : "N (total)",
                                'critical_zvalue'   : "z-value",
                                'siglevels'         : "Sig. level",
                                'stopProb'          : 'Power',
                                } | extra_columns)
        )
        return eDataFrame(tab)



# * Other Tests
# ** chisquared

class chisq_test():

    def __init__(self, v1, v2, data, groups=None):
        '''
        Compute Pearson's Chi-squared test of indepenence

        v1   string with the name of the first variable

        v2   string with the name of the second variable

        data a DataFrame with the variables

        group  string with the name of the grouping variable.
               If provided, tests will be performed within each group.
        '''
        self.v1=v1
        self.v2=v2
        self.groups=groups if not groups or isinstance(groups, list) else [groups]
        self.data=data
        self.results=self.__chisq_test__()

    def __chisq_test__(self):
        if self.groups:
            res=(
                self
                .data
                .nest(self.groups)
                .mutate_rowwise({"res": lambda col: self.__chisq_test_main__(
                    self.v1,
                    self.v2,
                    col['data'])})
                .unnest(col='res', id_vars=self.groups)
            )
        else:
            res=self.__chisq_test_main__(self.v1,
                                         self.v2,
                                         self.data)
        return res

    def __chisq_test_main__(self, v1, v2, data):
        freq_table = data.tab(row=v1, col=v2) 
        contingency_table = data.tabn(v1, v2, False, False)
        n = data.tabn(v1, v2).max()['Total']
        res_raw = infer.chisq_test(data, robj.Formula(f'{v1} ~ {v2}'))
        hyp =("Null hypothesis (H0): "+
              "The row and column variables of the "+
              "contingency table are independent"+
              "\nAlt. hypothesis (H1): "+
              "The row and column variables are dependent")
        res = ru.df2pandas(res_raw)
        # 
        mod = pd.DataFrame({'v1'           :v1,
                            'v2'           :v2,
                            'test'         :'chi-squared',
                            'hypotheses'   :hyp,
                            "cont_tab"     :[contingency_table],
                            "freq_tab"     :[freq_table],
                            "mod"          :[res_raw],
                            "fit"          :[res_raw],
                            'n (total)'    :n
                            }).reset_index(drop=True)
        mod=eDataFrame(mod).bind_col(res.reset_index(drop=True),
                                     ignore_index=False)
        return mod

    def __repr__(self):
        print(f"Chi-squared test of independence\n", flush=True)
        hyp=self.results.hypotheses.values[0]
        print(f"{hyp}\n", flush=True)
        cols = ['v1', 'v2', 'statistic', 'chisq_df', 'p_value', 'n (total)']
        if self.groups:
            tab=(
                self
                .results
                .select_cols(names=self.groups+cols)
            )
        else:
            tab=(
                self
                .results
                .select_cols(names=cols)
            )
        print(tab.to_string())
        return ''

    # def contingency_table(self):
    #     if self.groups:
    #         cols=self.groups+['freq_tab']
    #         tab=(
    #             self
    #             .results
    #             .select_cols(names=cols)
    #             .unnest(col='freq_tab', id_vars=self.groups)
    #         )
    #     else:
    #         cols='freq_tab'
    #         tab=(
    #             self
    #             .results
    #             .select_cols(names=cols)
    #             .values[0][0]
    #         )
    #     print(tab.to_string(index=False))

# * R tools
# ** rtools 

class rtools():
    def __init__(self, mod={}):
        '''
        Use
        Input 
        -----
        mod  : a dictionary of regression models estimated in R via rpy2
               The key must be a label for the model, the value
               the output of the R function, returned into python
        '''
        if mod:
            assert isinstance(mod, dict), "mod must be a dictionary"
        self.mod=mod


    def add_model(self, mod):
        '''
        Add new model to the object

        Input
        -----

        mod  : an model estimated in R
        '''
        for mod_label, mod_fit in mod.items():
            self.mod[mod_label]=mod_fit
            

    def predict(self,
                mods=None,
                predictor=None,
                predictor_values=robj.NULL,
                covars_at=robj.NULL,
                newdata=None):
        '''
        Return predicted values (see jtools make_predictions)

        Input
    	-----
           mods      : a string with the label of the model. If None
                       will run the precition for all models
           predictor : a string with a variable name to get predicted
                       values (only used if newdata is not provided)
           predictor_values : (optional) a list of values for the predictors.
                              If None, they are generated automatically.
           covars_at    : a dict with name of the variable (key of the dict) 
                       and the values (values of the dict) to set for
                       prediction  (only used if newdata is not provided)
           newdata   : a DataFrame with the values of the covariates to
                       predict the output. If used, the other parameters
                       (predictor and covars_at) are ignored

        Output
    	------
           predicted values
        '''
        # convert covars_at to list of vectors
        assert self.mod, 'No models in the rtools object!'
        assert predictor or isinstance(newdata, eDataFrame), (
            "Either the predictor a "+\
            "'newdata' must be provided")
            
        pred=eDataFrame()
        if not mods:
            mods=list(self.mod.keys())
        if not isinstance(mods, list):
            mods=[mods]
        # check if predictor is in the models
        mods_label_list=mods
        for idx, mod in enumerate(mods_label_list):
            f=base.as_character(stats.formula(self.mod[mod]))[0]
            if predictor not in ru.formula2varlist(f):
                mods.pop(idx)

        if not isinstance(newdata, eDataFrame):
            newdata_provided=False
        for mod in mods:
            assert mod in list(self.mod.keys()), (
                "Model name not found! Check 'mods'"  
            )
            if not newdata_provided:
                newdata = self.newdata(mod,
                                       predictor=predictor,
                                       predictor_values=predictor_values,
                                       covars_at=covars_at)
            predtmp = self.__predict__(mod, newdata)
            pred=pred.bind_row(predtmp.mutate({'model_id': mod}))
        try:
            pred=pred.drop_cols(names='.rownames')
        except (OSError, IOError, BaseException, KeyError) as e:
            pass
        return eDataFrame(pred)


    def newdata(self, mods=None, predictor=robj.NULL,
                predictor_values=robj.NULL,
                covars_at=None):
        '''
        Return predicted values (see jtools make_predictions)

        Input
    	-----
           mods      : a string with the label of the model. If None
                       will run the precition for all models

           predictor : a string with a variable name to get predicted
                       values

           predictor_values : (optional) a list of values for the predictors.
                              If None, they are generated automatically.

           covars_at     : a dict with name of the variable (key of the dict) 
                           and the values (values of the dict) to set for
                           prediction

        Output
    	------
           predicted values
        '''
        assert self.mod, 'No models in the rtools object!'
        assert predictor, "Predictor must be provided"
        # 
        if covars_at:
            assert isinstance(covars_at, dict), "covars_at must be a dictionary!"
            covars_at=self.__dict2lov__(covars_at)
        else:
            covars_at=robj.NULL
        # 
        if isinstance(predictor_values, list):
            predictor_values=np.array(predictor_values)
        # 
        if not mods:
            mods=list(self.mod.keys())
        if not isinstance(mods, list):
            mods=[mods]
        # 
        pred=eDataFrame()
        for mod in mods:
            assert mod in list(self.mod.keys()), (
                "Model name not found! Check 'mods'"  
            )
            tmp = jtools.make_new_data(model=self.mod[mod],
                                       pred=predictor,
                                       pred_values=predictor_values,
                                       at=covars_at
                                       )
            tmp=ru.df2pandas(tmp)
            pred=pred.bind_row(tmp.mutate({'model_id': mod}))
        return eDataFrame(pred)

    # =====================================================
    # utilities
    # =====================================================
    def __dict2lov__(self, dict):
        '''
        Convert python dictionary to an R named list of vectors
        '''
        dict_final={}
        for k,v in dict.items():
            if isinstance(v[0], str):
                if isinstance(v, str):
                    v=[v]
                dict_final[k] = robj.StrVector(v)
            else:
                dict_final[k] = robj.FloatVector(v)
        return robj.vectors.ListVector(dict_final)


    def __predict__(self, mod, newdata):
        rmod = self.mod[mod]
        rmod_class=base.attr(rmod, "class")[0]

        # rmod_class=mod.results.family
        # Linear model 
        # ------------
        if rmod_class=='lm':
            # print(f"TBD", flush=True)
            pred_data=(
                ru
                .df2pandas(citools.add_ci_lm(fit=rmod, df=newdata))
                .rename_cols(columns={'LCB0.025':'lower',
                                      'UCB0.975':'upper',
                                      }, tolower=False)
            )
        # Gneeralized Linear model 
        # ------------------------
        if rmod_class=='glm':
            pred_data=(
                ru
                .df2pandas(citools.add_ci_glm(fit=rmod, df=newdata))
                .rename_cols(columns={'LCB0.025':'lower',
                                      'UCB0.975':'upper',
                                      }, tolower=False)
            )
            
        # Multinomial model 
        # -----------------
        if rmod_class=='multinom':
            pred=stats.predict(rmod,
                               type='probs',
                               newdata=newdata,
                               conf_int=.95,
                               se_fit=True)
            cols=rmod.rx2['lab']
            pred=eDataFrame(pred, columns=cols).reset_index(drop=True)
            pred_data=broom.augment_nls(rmod,
                                        newdata=newdata,
                                        conf_int=True,
                                        type_predict="probs")
            pred_data=(
                ru.df2pandas(pred_data)
                # eDataFrame(pred_data)
                .reset_index(drop=True)
                .rename_cols(columns={'.fitted':'fitted'}, tolower=False)
                .bind_col(pred, ignore_index=False)
            )
        # dangerous zone
        # --------------
        # print(pred_data.names())
        colnames_correct    = newdata.names()
        colnames_to_replace = pred_data.names()[0:len(colnames_correct)]
        cols={old:new for old, new in zip(colnames_to_replace, colnames_correct)}
        pred_data = pred_data.rename_cols(columns=cols, tolower=False)
        # print(pred_data.names())
        return pred_data

# * Utils

class models_utils():
    def __init__(self):
        pass
    
    def build_formula(self, output, inputs, interactions=None, clusters=None):
        '''
        Inputs
        ------

        output   a string with the name of the output

        inputs   a list or dictionary with the additive covariates.
                 If a dictionary is used, there are three possibilities:
                    a. inputs = [...]
                    b. inputs = {'oldnames': "newnames"}
                    c. inputs = {'group': [...]}
                    d. inputs = {'group': {'oldnames': "newnames"}}

        interactions a list with tuples. Each tuple must contain
                     strings with the name of the variables
                     for the respective interactive term 

        clusters     TBD

        '''
        # output 
        # ------
        if isinstance(output, dict):
            output=list(output.keys())[0]
        
        # inputs 
        # ------
        # three possibilities:
        # list:
        #   inputs = [...]
        # dict:
        #   a. inputs = {'oldnames': "newnames"}
        #   b. inputs = {'group': [...]}
        #   c. inputs = {'group': {'oldnames': "newnames"}}
        input_list_final = []
        rename_vars      = {}
        if isinstance(inputs, list):
            input_list_final = [f"`{x}`" for x in inputs] # add ` character 
        # 
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, str):    # a. inputs = {'oldnames': "newnames"}
                    input_list_final += [f"`{v}`"]
                    rename_vars[k] = v
                elif isinstance(v, list): # b. inputs = {'group': [...]}
                    input_list_final += [f"`{x}`" for x in v]
                elif isinstance(v, dict): # c. inputs = {'group': {'oldnames': "newnames",...}}
                    for vkey, vvalue in v.items():
                        input_list_final += [f"`{vvalue}`"]
                        rename_vars[vkey] = vvalue
        inputs=input_list_final 
        # inputs = [f"`{x}`" for x in [*inputs.values()]] if isinstance(inputs, dict)\
        #     else [f"`{x}`" for x in inputs]
        inputs = " + ".join(inputs)

        # interactions 
        # ------------
        if isinstance(interactions, dict):
            interactions= list(itertools.chain(*[v for k, v in interactions.items()]))
        if interactions:
            interactions_formula=''
            for interaction_terms in interactions:
                interaction_terms = [f"`{x}`" for x in interaction_terms] 
                interaction = "*".join(interaction_terms)
                interactions_formula += f" + {interaction}"
            interactions = interactions_formula 

        # Formlua 
        # -------
        f = f"{output} ~ {inputs}" if not interactions else \
            f"{output} ~ {inputs} + {interactions}"
        return f


    def formula2df(self, formula):
        inputs = list(formula_tools.rhs_vars(robj.Formula(formula)))
        # inputs = [f"`{i}`" for i in inputs]
        output = list(formula_tools.lhs_vars(robj.Formula(formula)))
        # output = [f"`{i}`" for i in output]
        interactions=[term.strip() for term in formula.split('~')[1].split('+')\
                      if "*" in term]
        interactions=[i.replace('*', ":") for i in interactions]
        # interactions=[tuple(i.split('*')) for i in interactions]
        interactions=interactions if interactions else None
        res=eDataFrame(
            {"part"     : ['output', 'inputs', 'interactions', 'clusters'],
             'var_list' : [output,
                           inputs,
                           interactions,
                           None
                           ],
             'var_dict' : [None,
                           None,
                           None,
                           None
                           ],
             # 'grouped': ['No', "No", "No", "No"],
             # 'renamed': ["No", "No", "No", "No"]
             }
        )
        return res


    
