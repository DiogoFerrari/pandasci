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
import itertools
from statsmodels.iolib.summary2 import summary_col
from scipy.stats import norm as qnorm

#  R packages
# supress warnings
import warnings
warnings.filterwarnings("ignore")
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)   # will display errors, but not warnings


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
stats = importr('stats')
base = importr('base')
utils = importr("utils")
ggtxt = importr("ggtext")
jtools=importr("jtools")
broom=importr("broom")
metrics=importr("Metrics")
modelsummary=importr("modelsummary")
nnet=importr("nnet")
rpact = importr("rpact")
latex2exp=importr("latex2exp")
gridExtra = importr("gridExtra")
patchwork = importr("patchwork")


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
                     'stops_at':[.33, .66, 1]
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

    def plot(self, cost_per_observation=None):
        if self.type=='2prop':
            ax = self.__plot_2prop__(cost_per_observation)
        return ax

    def __plot_2prop__(self, cost_per_observation):

        vars=['prop1', 'prop2', 'sample_size_group1', 'sample_size_group2',
              'design', 'sample_size_group1_H1expected', 
              'diff', 'peek']
        tab=(
            self
            .data
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
        tab_expected=(
            tab
            .query(f"design=='Group sequence'")
            .select_cols(names=['prop2', 'sample_size_group1_H1expected',
                                'design', 'diff'])
            .mutate_rowwise({'design': lambda col: f"{col['design']} (expected)"})
            .mutate({'group_seq_design': lambda col: col['diff']})
            .drop_duplicates()
        )
        maxvalue=self.data.query(f"design=='Fixed design'").select_cols(names=['sample_size_total']).max()[0]
        tab_maximum = (
            self
            .data
            .query(f"sample_size_total=={maxvalue}")
            .select_cols(names=['prop1', 'prop2', 'diff'])
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
                                              extra_columns=extra_columns
                                              )

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
        g = (
            gg.ggplot(tab)
            + gg.geom_point(gg.aes_string(x=x, y=y,
                                          colour=color,
                                          shape=shape,
                                          group=size
                                          ),
                             size=3.5, alpha=.4, position="identity",
                            data=tab.query(f"design=='Fixed design'")) 
            + gg.geom_point(gg.aes_string(x=x, y=y,
                                          colour=color,
                                          shape=shape,
                                          group=size
                                          ),
                             size=2.5, alpha=.1, position="identity",
                            data=tab.query(f"design!='Fixed design'")) 
            + gg.geom_line(gg.aes_string(x=x, y=y, group=robj.NULL, colour=color),
                           size=.6,
                            data=tab.query(f"design=='Fixed design'")) 
            + gg.geom_line(gg.aes_string(x=x, y=y, group=group_seq_design, colour=color),
                           size=.6, alpha=.1, linetype=2,
                           data=tab.query(f"design!='Fixed design'")) 
            # 
            + gg.geom_point(gg.aes_string(x=x, y='sample_size_group1_H1expected',
                                          shape=shape,
                                          colour=color),
                           size=3, alpha=.2, linetype=1,
                           data=tab_expected) 
            + gg.geom_line(gg.aes_string(x=x, y='sample_size_group1_H1expected',
                                         group=group_seq_design, colour=color),
                           size=.6, alpha=.2, linetype=1,
                           data=tab_expected) 
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
        gtab = gridExtra.tableGrob(tab_maximum, rows = robj.NULL)
        layout = '''
        A
        A
        B'''

        g=patchwork.wrap_plots(A=g ,B=gtab, design=layout)
        print(g, flush=True)
        return g


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

    def get_design(self, value_group1, diff):
        tab=eDataFrame()
        if self.type=='2prop':
            tab=(
                self
                .data
                .select_rows(query=f"prop1=={value_group1}")
                .select_rows(query=f"diff=={diff}")
            )
        if tab.nrow==0:
            tab = power(diff=diff, type2prop={"prop1":value_group1}, type='2prop')
            tab = tab.data
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
        gdesign=rpact.getDesignGroupSequential(
            sided = tail,
            alpha = self.alpha,
            beta = 1-self.power,
            # informationRates = self.seq_design["stops_at"]
            ## futilityBounds = c(0, 0.05)
        )
        gdesign=rpact.getSampleSizeRates(gdesign,
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
        return tab



# * Regression models
# ** Class
class regression():
    def __init__(self, formula, data, family='gaussian', na='omit',
                 engine='python', *args, **kws):
        '''
        Run regression models
        
        Input
           formula  : a tuple (str, str) or a dictionary of tuples (str, str).
                      The first string of the tuple must be a R-like
                      regression formula, i.e.,  <depvar> ~ <regressors>.
                      The second string must be the family of the dependent
                      variable: gaussian, binomial, multinomial, poisson,
                      negative binomial
                      If a dictionary is used, the key must be the label
                      of the respective model.
           engine   : 'python' or 'r.' Defines the software to use to run the 
                       regressions
        '''
        assert isinstance(formula, tuple) or isinstance(formula, dict),\
            "'formula' must be a tuple or dictionary"
        self.multinomial = False
        self.__check_multinomial__(formula)
        formula = {"Model 1": formula} if isinstance(formula, str) else formula
        # 
        self.data        = eDataFrame(data)
        self.engine      = engine
        self.formulas    = formula
        self.na          = na
        self.data        = data
        self.results     = self.__run_regressions__(*args, **kws)

# ** Utils
# *** TODO Core

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


    def __run_regressions__(self, *args, **kws):
        assert (isinstance(self.formulas, tuple) or isinstance(self.formulas, dict)) ,\
            "'formulas' must be a typle or dictionary"
        if isinstance(self.formulas, tuple):
            self.formulas = {'Model 1':self.formulas}
        tab=eDataFrame()
        for idx, (label, (formula, family)) in enumerate(self.formulas.items()):
            tmp = self.__run_dispatcher__(idx, label, formula, family,
                                          *args, **kws)
            tmp['depvar']
            tab = pd.concat([tab, tmp], axis=0, ignore_index=True)
        return eDataFrame(tab)


    def __run_dispatcher__(self, idx, label, formula, family, *args, **kws):
        if self.engine=='python':
            if family=='gaussian':
                tab = self.__run_gaussian__(idx, formula, label, *args, **kws)
            if family=='binomial':
                tab = self.__run_binomial__(idx, formula, label, *args, **kws)
            if family=='multinomial':
                tab = self.__run_multinomial__(idx, formula, label, *args, **kws)
        if self.engine=='r':
            if family=='gaussian':
                tab = self.__run_gaussianr__(idx, formula, label, *args, **kws)
            if family=='binomial':
                tab = self.__run_binomialr__(idx, formula, label, *args, **kws)
            if family=='multinomial':
                tab = self.__run_multinomialr__(idx, formula, label, *args, **kws)
        return tab


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

# *** Summary (computing)

    # def summary_default(self, model=None):
    #     if model:
    #         print(self.regression[model]['sumamry_default'])
    #     else:
    #         for model in self.regression.keys():
    #             print(self.regression[model]['summary_default'])
            
    # def summary_tidy(self, model=None, join=True, get=True):
    #     res = eDataFrame() if join or model else {}
    #     if model:
    #         res = self.regression[model]['sumamry_tidy']
    #         res['model']=model
    #     else:
    #         for model in self.regression.keys():
    #             resi=self.regression[model]['summary_tidy']
    #             resi['model']=model
    #             if join:
    #                 res = res.bind_row(resi)
    #             else:
    #                 res[model]=resi
    #     return res if get else print(res)


# *** Summary (printing)

    def summary(self, *args, **kws):
        '''

        Input 
        -----
        vcov   a string of list with the variance-covariance matrix to use
               for the standard errors of the coefficients. It a list is used,
               the models will use the covariance matrix in the order provided.

        fn     a string with a path and file name to save the output.

        output_format   string with the format of the output. E.g.: 'latex',
                        'huxtable', 'DataFrame' (default).
        output_collect  boolean. If true, return a DataFrame with the summary

        '''
        if self.engine == 'python':
            print("to be implemented")
        if self.engine == 'r':
            res=self.__summaryr__(*args, **kws)
        return res
       

    def __repr__(self):
        if self.engine == 'python':
            print("to be implemented")
        if self.engine == 'r':
            self.__reprr__()
        return ""

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

    def __check_multinomial__(self, formula):
        '''
        Check if multinomial family is combined with other family distributions
        '''
        multinomial=False
        notmultinomial = False
        for label, (formula, family) in formula.items():
            multinomial    = True if family == 'multinomial' else multinomial
            notmultinomial = True if family != 'multinomial' else notmultinomial
            
        if multinomial and notmultinomial:
            raise ValueError("Currently, combination of multinomial with other family "\
                             "distributions are not allowed")
        if multinomial and not notmultinomial:
            self.multinomial=True

# ** Python Engine
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


    
# ** R engine
# *** Regression
# **** gaussian

    def __run_gaussianr__(self, idx, formula, label, *args, **kws):
        data=self.data
        fit = stats.lm(formula, data=self.data, family='gaussian')
        mod = pd.DataFrame({'idx'          :idx,
                            'label'        :label,
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

    
# **** binomial

    def __run_binomialr__(self, idx, formula, label, *args, **kws):
        data=self.data
        fit = stats.glm(formula, data=self.data, family='binomial')
        mod = pd.DataFrame({'idx'          :idx,
                            'label'        :label,
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


# **** multinomial

    def __run_multinomialr__(self, idx, formula, label, *args, **kws):
        data=self.data
        fit = nnet.multinom(formula, data=self.data)
        mod = pd.DataFrame({'idx'          :idx,
                            'label'        :label,
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


# *** Summary (computing)
# **** Gaussian

    def __get_r_lm_summary_tidy__(self, fit):
        res = broom.tidy_lm(fit, conf_int=True)
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

# **** Binomial

    def __get_r_binomial_summary_tidy__(self, fit):
        res = broom.tidy_glm(fit, conf_int=True)
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

# **** Multinomial

    def __get_r_multinomial_summary_tidy__(self, fit):
        res = broom.tidy_multinom(fit, conf_int=True)
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

# *** Summary (printing)

    def __summaryr__(self, *args, **kws):
        self.__reprr__(*args, **kws)
        if kws.get("output_collect", False):
            return self.__get_summaryr__(*args, **kws)
        return None

    def __reprr__(self, *args, **kws):

        res=self.__get_summaryr__(*args, **kws)
        output_format= kws.get("output_format", 'data.frame')
        if kws.get("fn", False):
            output_format=kws.get("fn", output_format)

        print(f"", flush=True)
        self.__print_header_info__()
        print(f"Estimation summary:\n", flush=True)
        if (output_format=='data.frame'):
            print(res.to_string(index=False), flush=True)
        else:
            print(res, flush=True)
        return None


# *** Plots
        
# *** Ancillary

    def __get_summaryr__(self, *args, **kws):
        output_format=kws.get("output_format", 'data.frame')
        if kws.get("fn", False):
            print(f"\nNote: 'fn' provided. Format will use fn extension.\n", flush=True)
            output_format=kws.get("fn", output_format)
        vcov=kws.get("vcov", 'classical')
        footnotes=kws.get("footnotes", robj.NULL)
        if self.multinomial:
            res= modelsummary.modelsummary_wide(self.results.fit.values,
                                                statistic='({conf.low}, {conf.high})',
                                                stars=True, ## c('*' = .1, '**' = .05, "***"=0.01),
                                                vcov = vcov, #"classical", "robust", "stata", "HC4", "HC0",
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
            res= modelsummary.modelsummary(self.results.fit.values,
                                           statistic='({conf.low}, {conf.high})',
                                           stars=True, ## c('*' = .1, '**' = .05, "***"=0.01),
                                           vcov = vcov, #"classical", "robust", "stata", "HC4", "HC0",
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


    def __print_header_info__(self):
        self.__print_line__()
        res=(
            self
            .results
            .select_cols(names=['label','family', 'depvar'])
        )
        print(res.to_string(index=False), flush=True)
        self.__print_line__()

        
    def __print_line__(self):
        print(f"====================================", flush=True)
        
        
# * PCA

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



# * Regression tools
# ** jtools 

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
            

    def predict(self, mods=None,
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
        assert predictor, "Predictor must be provided"
            
        pred=eDataFrame()
        if not mods:
            mods=list(self.mod.keys())
        if not isinstance(mods, list):
            mods=[mods]
        for mod in mods:
            assert mod in list(self.mod.keys()), (
                "Model name not found! Check 'mods'"  
            )
            newdata = self.newdata(mod, predictor=predictor,
                                   predictor_values=predictor_values,
                                   covars_at=covars_at)
            predtmp = self.__predict__(mod, newdata)
            pred=pred.bind_row(predtmp.mutate({'model_id': mod}))
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
            tmp=eDataFrame(tmp)
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
        # Multinomial model 
        # -----------------
        if rmod_class=='multinom':
            pred=stats.predict(rmod,
                               type='probs',
                               newdata=newdata,
                               conf_int=.95,
                               se_fit=True)
            cols=rmod.rx2['lab']
            pred=eDataFrame(pred, columns=cols)
            pred_data=broom.augment_nls(rmod,
                                        newdata=newdata,
                                        conf_int=True,
                                        type_predict="probs")
            pred_data=(
                eDataFrame(pred_data)
                .reset_index(drop=True)
                .rename_cols(columns={'.fitted':'fitted'}, tolower=False)
                .bind_col(pred.reset_index(drop=True), ignore_index=False)
            )
        return pred_data
