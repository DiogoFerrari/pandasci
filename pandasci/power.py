from .ds import *
import numpy as np
import pandas as pd
import itertools as it
from statsmodels.stats import proportion as pwr2prop
from scipy.stats import norm as qnorm

# * R Modules
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
#
def importr_or_install(packname, contriburl="http://cran.r-project.org"):
    '''To install R packages if needed'''
    try:
        rpack = importr(packname)
    except (RRuntimeError, PackageNotInstalledError):
        print(f"Installing R package {packname}...")
        utils.install_packages(packname)
        rpack = importr(packname)
    return rpack

# 
pwrss   = importr_or_install("pwrss")
pwrssint = importr_or_install("InteractionPoweR")
rpact = importr_or_install("rpact")
# 
latex2exp=importr_or_install("latex2exp")
gridExtra = importr_or_install("gridExtra")
ggtxt = importr_or_install("ggtext")
patchwork = importr_or_install("patchwork")




# * functions

def ggtheme(legend_position='right'):
    g =gg.theme(
             ## ------
             ## legend
             ## ------ 
             legend_position = legend_position,
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
             panel_grid_minor_y  = gg.element_line(colour="grey",
                                                   size=.3, linetype=3),
             panel_grid_major_y  = gg.element_line(colour="grey",
                                                   size=.3, linetype=3),
             panel_grid_minor_x  = gg.element_line(colour="grey",
                                                   size=.3, linetype=3),
             panel_grid_major_x  = gg.element_line(colour="grey",
                                                   size=.3, linetype=3),
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


def ggguides(ncol=1):
    keywidth=2
    keyheight=.9
    leg_title_pos="top"
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


# * info

# Source:
# One time analysis:
# - Shinny: https://pwrss.shinyapps.io/index/
# - Site  : https://cran.r-project.org/web/packages/pwrss/vignettes/examples.html
# Interaction:
# - https://dbaranger.github.io/InteractionPoweR/articles/InteractionPoweRvignette.html
# - https://dbaranger.github.io/InteractionPoweR/reference/power_interaction.html
# Sequential design:

# * classes

class power_seq_design():
    
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
                 ngroups=2,
                 *args, **kws
                 ):
        '''
        Compute the sample size

        Input 
        -----
        diff     : difference between average value of the outcome in the
                   two groups (e.g., different in proportions, mean, etc.)
        ngroups  : int, number of groups, including all treatment and control
                   groups, to compute the total sample size
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
        self.ngroups=ngroups
        if type=='2prop':
            self.data = self.__2prop__()
        self.__sample_size_all_groups__()


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
                                    'N (t+c)'     : lambda x: '%.1f' % x,
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


    # 2 prop 
    # ------
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


    # plot 
    # ----
    def plot(self, cost_per_observation=None, design='both',
             ngroups=None, subtitle_info=None, *args, **kws):
        if self.type=='2prop':
            ax = self.__plot_2prop__(
                cost_per_observation = cost_per_observation,
                design               = design,
                ngroups              = ngroups if not None else self.ngroups,
                subtitle_info        = subtitle_info,
                *args, **kws)
        return ax


    def __plot_2prop__(self, cost_per_observation, design, ngroups,
                       subtitle_info, *args, **kws):

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
            # total sample size 
            # -----------------
            .assign(ntotal = lambda col: col['sample_size']*ngroups)
        )
        # 
        # Data for table 
        # --------------
        extra_columns={}
        tab_maximum = self\
            .get_sample_size_max(design='Fixed design')\
            .assign(ntotal=lambda col: col['sample_size_group1']*ngroups)
        if cost_per_observation:
            tab_maximum=(
                tab_maximum   
                .mutate({'Cost':
                         lambda col: round(col['ntotal']*
                                           cost_per_observation, 2),
                         })
                .mutate_rowwise({'Cost': lambda col: f"$ {col['Cost']}"})
            )
            extra_columns={"Cost":"Cost"}
        tab_maximum=self.__get_main_columns__(tab_maximum,
                                              extra_columns=extra_columns)
        tab=tab.assign(sample_size=lambda col: col['ntotal'])
        x = "prop2"
        y = "sample_size"
        color='diff'
        shape='design'
        fill = 'diff'
        size='`Treatment group`'
        group_seq_design='group_seq_design'
        twotailed = 'Yes' if self.two_tail else "No"
        labx= latex2exp.TeX("Proportion of 'positive' outcome cases (Y=1) "+\
                            'in the group of interest ($\pi_t$)')
        laby='Total sample size required to detect the effect'
        labcolor= latex2exp.TeX('Group difference ($\\pi_t - \\pi_c$)')
        labfill=latex2exp.TeX('Group difference ($\\pi_t - \\pi_c$)')
        labshape='Design'
        title = "Sample size calculation"
        subtitle = (
            "Info:  "+
            f"$\\alpha$: {self.alpha}; "
            f"Power ($\\beta$): {self.power}; "+
            f"Test : {self.type}; "+
            f"Two-sided : {self.two_tail}; "+
            f"Ratio between sample sizes in each group : {self.ratio}; "
        )
        # 
        if subtitle_info is not None:
            subtitle += subtitle_info
        else:
            subtitle += f"Treatment arms: {ngroups}"
        subtitle = latex2exp.TeX(subtitle)
        # 
        # Plot 
        # ----
        g = gg.ggplot(tab)
        if design in ['sequence', 'both']:
            g = self.__plot_2prop_sequence_design__(g, x, y,
                                                    color, shape, size, 
                                                    fill,
                                                    group_seq_design,
                                                    tab,
                                                    design, ngroups)
        if design in ['fixed', 'both']:
            g = self.__plot_2prop_fixed_design__(g, x, y, color, shape, size, fill, tab)

        g = (
            g
            + gg.scale_shape_manual(values=[22,24,21])
            + gg.labs(
                x        = labx,
                y        = laby,
                color    = labcolor, 
                shape    = labshape,
                fill     = labfill,
                linetype = 'd',
                title    = title,
                subtitle = subtitle,
                caption  = robj.NULL
                )
            # + gg.scale_size_manual(cols)
            # + gg.scale_colour_brewer(palette="Set1") 
            # + gg.scale_shape_discrete() 
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


    def __plot_2prop_fixed_design__(self, g, x, y, color, shape, size, fill, tab):
        g = (
            g
            + gg.geom_line(gg.aes_string(x=x, y=y, group=robj.NULL, color=fill),
                           # color='black',
                           size=.6,
                           data=tab.query(f"design=='Fixed design'")) 
            + gg.geom_point(gg.aes_string(x=x, y=y,
                                          shape=shape,
                                          fill=fill,
                                          group=size
                                          ),
                            # fill='black',
                            color='white',
                             size=3.5, position="identity",
                            data=tab.query(f"design=='Fixed design'")) 
        )
        return g


    def __plot_2prop_sequence_design__(self, g, x, y,
                                       color, shape, size, 
                                       fill,
                                       group_seq_design,
                                       tab,
                                       design,
                                       ngroups):
        tab_expected=eDataFrame()
        # Expected values 
        # ---------------
        if design in ['sequence', 'both']:
            tab_expected=(
                tab
                .query(f"design=='Group sequence'")
                .select_cols(names=['prop2', 'sample_size_group1_H1expected',
                                    'design', 'diff'])
                .mutate_rowwise({
                    'design': lambda col: f"{col['design']} (expected)"
                })
                .mutate({'group_seq_design': lambda col: col['diff']})
                .drop_duplicates()
                .assign(ntotal = lambda col: col['sample_size_group1_H1expected']*ngroups)
            )
        # 
        alpha=.4 if design=='sequence' else .1
        g = (
            g
            + gg.geom_line(gg.aes_string(x=x, y=y,
                                         group=group_seq_design,
                                         color=fill,
                                         ),
                           # colour='gray',
                           size=.6, linetype=2,
                           data=tab.query(f"design!='Fixed design'")) 
            + gg.geom_point(gg.aes_string(x=x, y=y,
                                          shape=shape,
                                          fill=fill,
                                          group=size
                                          ),
                            color='white',
                            # fill='gray',
                            size=2.5, position="identity",
                            data=tab.query(f"design!='Fixed design'")) 
            # 
            # + gg.geom_line(gg.aes_string(x=x, y='sample_size_group1_H1expected',
            + gg.geom_line(gg.aes_string(x=x, y='ntotal',
                                         group=group_seq_design,
                                         color=fill
                                         ),
                           # colour='gray',
                           size=.6, linetype=1,
                           data=tab_expected) 
            # + gg.geom_point(gg.aes_string(x=x, y='sample_size_group1_H1expected',
            + gg.geom_point(gg.aes_string(x=x, y='ntotal',
                                          fill=fill,
                                          shape=shape,
                                          ),
                            color='white',
                            # fill='gray',
                            size=3, linetype=1,
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
        

    # hidden methods 
    # --------------
    def __str__(self):
        print(self.data, flush=True)
        return None


    def __repr__(self):
        print(self.data, flush=True)
        return ''


    def __get_main_columns__(self, tab, extra_columns={}):
        tab=(
            tab
            .select_cols(names={'design'            : 'Design',
                                'peek'              : 'Peek',
                                'prop1'             : "pi_c",
                                'prop2'             : "pi_t",
                                'sample_size_group1': "N (c)",
                                'sample_size_group2': "N (t)",
                                'sample_size_total' : "N (c+t)",
                                'ntotal'            : "N (total)",
                                'critical_zvalue'   : "z-value",
                                'siglevels'         : "Sig. level",
                                'stopProb'          : 'Power',
                                } | extra_columns)
        )
        return eDataFrame(tab)


    def __sample_size_all_groups__(self):
        self.data["sample_size_final"] = \
            self.data["sample_size_group1"]* self.ngroups


class power():
        
    def __init__(self, *args, **kws):
        '''
        Compute the sample size

        Input 
        -----
        alpha    : float
                   Type I error level. Defaul 0.05

        power    : float
                   power of the test. Defualt 0.8
        
        alternative : str
                     "not equal", "greater", "less", "equivalent",
                     "non-inferior", or "superior".

        tau      : float
                   Value used for different power calculations
                   difference in proportions between the groups;
                   difference in means between the groups;
                 
        ncontrol : int
                   number of subjects in the control group

        ngroups  : int
                   total number of treatment groups, including the
                   control group(s). Default 2 (one treatment and one
                   control group)

        seq_design : dict
                     Values defined by R pkg rpact
                     key            value
                    'typeOfDesign': 'OF'          (Default)
                    'stops_at'    : [.33, .66, 1] (Default)

        interaction_effects : dict
                              See parameters in interaction_efffects()

        '''
        self.alpha=kws.get("alpha", .05)
        self.power=kws.get("power", .8)
        self.alternative=kws.get("alternative", 'not equal')
        self.ncontrol=kws.get("ncontrol", False)
        self.ngroups=kws.get("ngroups", 2)
        self.seq_design=kws.get("seq_design", {"typeOfDesign": 'OF',
                                               "stops_at":[[.33, .66, 1]]})
        interaction_effects = kws.get("interaction_effects", False)
        # 
        self.compute='power' if self.ncontrol else 'sample size'
        self.tau=kws.get("tau", .1)
        self.diff_prop=kws.get("tau", .1)
        self.diff_means=kws.get("tau", .1)

        # compute
        # -------
        self.__difference_in_proportions__()
        self.__difference_in_proportions_seq_design__()
        self.__difference_in_means__()
        self.__collect__()
        self.est_interaction_count = 0 
        if interaction_effects:
            self.interaction_effects(**interaction_effects)
        
        
    # Fixed design
    # ------------
    def __difference_in_means__(self):
        tau = np.linspace(.1, 1, 10)
        mu2 = 0
        mu1 = tau+mu2
        if self.compute=='sample size':
            est = pwrss.pwrss_z_2means(mu1 = mu1,
                                       mu2 = mu2,
                                       alpha = self.alpha,
                                       power = self.power,
                                       verbose=False,
                                       alternative = self.alternative)
        else:
            est = pwrss.pwrss_z_2means(mu1 = mu1,
                                       mu2 = mu2,
                                       alpha = self.alpha,
                                       n = self.ncontrol,
                                       verbose=False,
                                       alternative = self.alternative)
            

        # results
        self.est_diff_means = {
            'est' : est,
            'tidy': self.__tidy__(obj=est,
                                  test='Difference in means',
                                  par1='mu1',
                                  par2='mu2',
                                  par1_label='Effect size',
                                  par2_label='Mean of the control group',
                                  outcome='Continuous',
                                  )
        }


    def __difference_in_proportions__(self):
        tau = self.diff_prop
        p1   = np.linspace(.05, .95, 15)
        p2  = p1+tau

        p1  = FloatVector(p1[(0<p2) & (p2<1)])
        p2  = FloatVector(p2[(0<p2) & (p2<1)])
        if self.compute=='sample size':
            est = pwrss.pwrss_z_2props(p1 = p1,
                                       p2 = p2,
                                       alpha = self.alpha,
                                       power = self.power,
                                       arcsin_trans = False,
                                       verbose=False,
                                       alternative = self.alternative)
        else:
            est = pwrss.pwrss_z_2props(p1 = p1,
                                       p2 = p2,
                                       alpha = self.alpha,
                                       arcsin_trans = False,
                                       verbose=False,
                                       n = self.ncontrol,
                                       alternative = self.alternative)
            

        # results
        self.est_diff_prop = {
            'est' : est,
            'tidy': self.__tidy__(obj=est,
                                  test='Difference in proportions',
                                  par1='p1',
                                  par2='p2',
                                  par1_label='Proportion of positive cases in the control group',
                                  par2_label='Proportion of positive cases in the treatment group',
                                  outcome='Binary',
                                  )
        }


    def __linear_regression__(self, binary_predictor=True):
        # tau = np.linspace(.1, 1, 10)
        # mu2 = 0
        # mu1 = tau+mu2
        # if self.compute=='power':
        #     est = pwrss.pwrss_z_2means(mu1 = mu1,
        #                                mu2 = mu2,
        #                                alpha = self.alpha,
        #                                power = self.power,
        #                                alternative = self.alternative)
        # else:
        #     est = pwrss.pwrss_z_2means(mu1 = mu1,
        #                                mu2 = mu2,
        #                                alpha = self.alpha,
        #                                n = self.ncontrol,
        #                                alternative = self.alternative)
            

        # # results
        # self.est_diff_means = {
        #     'est' : est,
        #     'tidy': self.__tidy__(obj=est,
        #                           test='Difference in means',
        #                           par1='mu1',
        #                           par2='mu2',
        #                           par1_label='Effect size',
        #                           par2_label='Mean of the control group',
        #                           )
        # }
        pass


    def interaction_effects(self, *args, **kws):
        '''
        Compute power for a two-way interaction. Function from
        R package InteractionPoweR. It run simulations to compute the
        power

        n : int or list of integers
            sample sizd for the simulation to compute power. Default 1000

        interaction_effect : float or list of floats
                             expected interaction effect. Default .1.

        k_y  : int or list of integers
               Number of categories of the outcome. Zero (0) indicates a
               continuous outcome (default)

        k_x1 : int or list of integers
               Number of categories of x1. Zero (0) indicates a
               continuous covariate. Default 2.

        k_x2 : int or list of integers
               Number of categories of x2. Zero (0) indicates a
               continuous covariate  (default)

        r_x1_x2 : float or list of floats
                  correlation between the interaction terms x1 nd x2
                  Default 0

        r_x1_y  : float or list of floats
                  correlation between the interaction term x1 and
                  the outcome. Default .1.

        r_x2_y  : float or list of floats
                  correlation between the interaction term x2 and
                  the outcome. Default .1.

        ncores  : int
                  number of cores to run in paralell. Default
                  uses half the number of available cores

        niter   : int
                  number of iterations for the simulation. Default 1,000

        '''
        self.est_interaction_count += 1

        ncores   = kws.get("ncores", len(os.sched_getaffinity(0))/2)
        niter    = kws.get("niter", 1000)

        n        = kws.get("n", [1000])
        r_x1x2_y = kws.get("interaction_effect", [.1])
        r_x1_y   = kws.get("r_x1_y", [.1])
        r_x2_y   = kws.get("r_x2_y", [.1])
        r_x1_x2  = kws.get("r_x1_x2", [0])
        k_y      = kws.get("k_y", [0])
        k_x1     = kws.get("k_x1", [2])
        k_x2     = kws.get("k_x2", [0])

        n        = n if isinstance(n, list) else [n]
        r_x1x2_y = r_x1x2_y if isinstance(r_x1x2_y, list) else [r_x1x2_y]
        r_x1_y   = r_x1_y if isinstance(r_x1_y, list) else [r_x1_y]
        r_x2_y   = r_x2_y if isinstance(r_x2_y, list) else [r_x2_y]
        r_x1_x2  = r_x1_x2 if isinstance(r_x1_x2, list) else [r_x1_x2]
        k_y      = k_y if isinstance(k_y, list) else [k_y]
        k_x1     = k_x1 if isinstance(k_x1, list) else [k_x1]
        k_x2     = k_x2 if isinstance(k_x2, list) else [k_x2]

        res = pd.DataFrame()
        comb = list(it.product(n, r_x1x2_y, r_x1_y, r_x2_y, r_x1_x2, k_y, k_x1, k_x2))
        for i, (n, r_x1x2_y, r_x1_y, r_x2_y, r_x1_x2, k_y, k_x1, k_x2) in enumerate(comb):
            print(f'Computing iteractions for combination {i+1} of {len(comb)}...')
            tmp = self.__interaction_effects__(
                n        = n,
                ncores   = ncores,
                niter    = niter,
                r_x1x2_y = r_x1x2_y,
                r_x1_x2  = r_x1_x2,
                r_x1_y   = r_x1_y,
                r_x2_y   = r_x2_y,
                k_y      = k_y,
                k_x1     = k_x1,
                k_x2     = k_x2
            )
            res = pd.concat([res, tmp])
        if self.est_interaction_count==1:
            self.est_interaction = res.assign(ran=self.est_interaction_count)
        else:
            self.est_interaction = pd.concat([
                self.est_interaction, 
                res.assign(ran=self.est_interaction_count)])


    def __interaction_effects__(self, n, ncores, niter, r_x1x2_y,
                                r_x1_x2, r_x1_y, r_x2_y, k_y,
                                k_x1, k_x2, *args, **kws):
        res = pwrssint.power_interaction(
            alpha    = self.alpha,             # alpha, for the power analysis
            N        = FloatVector([n]),
            r_x1x2_y = r_x1x2_y,
            r_x1_x2  = r_x1_x2,
            r_x1_y   = r_x1_y,
            r_x2_y   = r_x2_y,
            k_y      = k_y,
            k_x1     = k_x1,
            k_x2     = k_x2,
            cl       = ncores,
            n_iter   = niter
        )
        match k_y:
            case 0:
                y = 'Continuous'
            case 2:
                y = 'Binary'
            case a if a > 2:
                y = f'Categorical ({k_y} cats)'
        match k_x1:
            case 0:
                x1 = 'Continuous'
            case 2:
                x1 = 'Binary'
            case a if a > 2:
                x1 = f'Categorical ({k_x1} cats)'
        match k_x2:
            case 0:
                x2 = 'Continuous'
            case 2:
                x2 = 'Binary'
            case a if a > 2:
                x2 = f'Categorical ({k_x2} cats)'

        res = pd.DataFrame({'ntot':res.rx2['N'],
                            'power': res.rx2['pwr'],
                            'interaction_effect':r_x1x2_y,
                            'y': y,
                            'x1': x1,
                            'x2': x2,
                            "cor_x1_x2"  : r_x1_x2,
                            "cor_x1_y"   : r_x1_y,
                            "cor_x2_y"   : r_x2_y,
                            'niter'      : niter,
                            })
        return res


    # Sequential design 
    # -----------------
    def __difference_in_proportions_seq_design__(self):
        '''This function using the class power_seq_design'''
        two_tail = True if self.alternative=='not equal' else False
        res = power_seq_design(
            diff=self.diff_prop,
            alpha=self.alpha,
            power=self.power,
            two_tail=two_tail,
            ratio=1,
            seq_design=self.seq_design,
            type='2prop',
            type2prop={"prop1":None}
        )
        tidy = (
            res.data
            .rename(columns={
                'prop1'     :'par1',
                'prop2'     :'par2',
                'diff'      : "tau",
                'pwr'       :'power',
                'two-sided' :'two-tail'
            })
            .assign(**{
                'test'       : 'Difference in proportions',
                'stat'       : 'z',
                "par1_label" : 'Proportion of positive cases in the control group',
                "par2_label" : 'Proportion of positive cases in the treatment group',
                'ntot'       : lambda col: col["sample_size_total"],
                'ncontrol'   : lambda col: col['sample_size_group1'],
                'outcome'    : 'binary'
            })
        )
        self.est_diff_prop_seq_design = {
            'est': res,
            'tidy':tidy
        }

        
    # plot 
    # ----
    def plot(self, design='sequential', cost_per_observation=None,
             ngroups=2, subtitle_info=None, *args, **kws):
        '''
        Plot power analysis

        Input 
        -----
        cost_per_observation  number indicating the cost per
                              observation. If provided,
                              plot the total cost.

        design    string. It can be,
                  - 'fixed'      plot only the fixed design
                  - 'sequential' plot only the sequential design
                  - 'both'       Default. Plot the fixed and the
                                 sequential design
        ngroups   integer with the total number of experimental
                  groups, including control and treatment

        subtitle_info str
                            customize the info in the plot subtitle
                            about the treatment arms

        For interaction plot 
        --------------------
        ran int
            Number of the ran (see self.est_interaction)
            Default None (plot all ran)

        power_threshold float
                        horizontal line with power threshold

        
        '''
        g=None
        if design=='sequential':
            g = (
                self
                .est_diff_prop_seq_design['est']
                .plot(design='both',
                      cost_per_observation=cost_per_observation,
                      ngroups=ngroups, *args, **kws)
            )
        elif design=='fixed':
            g = self.plot_fixed_design(
                cost_per_observation=cost_per_observation,
                ngroups=ngroups, subtitle_info=subtitle_info)
        elif design=='interaction':
            if self.est_interaction_count>0:
                ran=kws.get("ran", None)
                power_threshold=kws.get("power_threshold", .8)
                g = self.plot_interaction(
                    ran=ran,
                    power_threshold=power_threshold)
            else:
                print("\nPower analysis for interaction not yet "+
                      'conducted. Run interaction_effects() first.')
                
        return g


    def plot_fixed_design(self, cost_per_observation, ngroups,
                          *args, **kws):
        tab = (
            self
            .table
            .query(f"design=='fixed'")
            .assign(ntot = lambda col: col['ntot']*ngroups,
                    label = lambda col: [f"{np.abs(tau)}\n{n}"
                                         if not cost_per_observation else
                                         f"{np.abs(tau)}\n{n}\n$ {int(n*cost_per_observation):,.0f}"
                                         for tau, n in
                                         zip(col['tau'].round(2), col['ntot'].astype(int))]
                    )
        )
        # 
        x = "par1"
        y = "ntot" if self.compute=='sample size' else 'power'
        fill='test'
        label='label'
        facet1='outcome'
        facet2=robj.NULL
        xlab= latex2exp.TeX("\overset{\\textbf{Binary}: Proportion of 'positive' outcome cases (Y=1) "+\
                            'in the group of interest ($\pi_t$)}'+\
                            "{\\textbf{Continuous}: Effect size (e.g., difference in averages)}")
        ylab='Total sample size required to detect the effect'
        title = "Sample size calculation"
        subtitle = latex2exp.TeX(
            "Info:  "+
            f"$\\alpha$: {self.alpha}; "
            f"Power ($\\beta$): {self.power}; "+
            f"Alternative : {self.alternative}; "+
            f"Treatment arms: {ngroups}"
        )
        g = (
            gg.ggplot(tab)
            + gg.geom_line(gg.aes_string(x=x, y=y, color=fill), alpha=.3) 
            + gg.geom_point(gg.aes_string(x=x, y=y, fill=fill), size=2) 
            + gg.geom_text(gg.aes_string(x=x, y=y, label=label, color=fill),
                           show_legend=False, parse=False, vjust=-.2, hjust=.5,
                           check_overlap = True, 
                           fontface='bold', position=gg.position_dodge(0),
                           angle=0, size=2.5) 
            + gg.scale_y_continuous(expand = FloatVector([0.15, 0]))
            + gg.scale_x_continuous(expand = FloatVector([0.07, 0]))
            + gg.facet_wrap(f"~{facet1}", scales='free')
            + gg.theme_bw()
            + ggtheme(legend_position='top')
            + ggguides(ncol=len(tab.test.unique()))
            + gg.labs(
                x        = xlab,
                y        = ylab,
                color    = robj.NULL, 
                fill     = robj.NULL,
                linetype = robj.NULL,
                shape    = robj.NULL,
                title    = title,
                subtitle = subtitle,
                caption  = 'Point labels indicate effect size (top) and sample size (bottom)'
            )

        )
        g.plot()
        return g


    def plot_interaction(self, ran, power_threshold, *args, **kws):
        tab = (
            self
            .est_interaction
            .assign(facet1 = lambda col: [f"{v} (Interaction effect size)" for v
                                          in col['interaction_effect']],
                    facet2 = lambda col: [f"X2: {x2}" for x2 in col['x2']])
        )
        if ran is not None:
            tab = tab.query(f"ran=={range()}")

        x = "ntot"
        y = "power"
        fill='y'
        facet1='facet1'
        facet2='facet2'
        #
        xlab = 'Sample size'
        ylab = 'Power to detect the interaction effect'
        title = "Sample size calculation"
        subtitle = latex2exp.TeX(
            "Info:  "+
            f"$\\alpha$: {self.alpha}; "
            f"Power threshold ($\\beta$): {power_threshold}; "+
            f"Alternative : {self.alternative}; "+
            f"Treatment: {tab.x1.unique()}"
        )
        leg_title='Outcome'
        leg_title_lt='Treatment'
        g = (
            gg.ggplot(tab)
            + gg.geom_hline(gg.aes_string(yintercept=.8 ),linetype="dashed", col="red")
            + gg.geom_line(gg.aes_string(x=x, y=y, color=fill)) 
            + gg.geom_point(gg.aes_string(x=x, y=y, fill=fill)) 
            + gg.facet_grid(f"{facet2} ~ {facet1}") 
            + gg.theme_bw()
            + ggtheme(legend_position='top')
            + ggguides(ncol=4)
            + gg.labs(
                x        = xlab,
                y        = ylab,
                color    = leg_title,
                fill     = leg_title,
                linetype = leg_title_lt,
                shape    = robj.NULL,
                title    = title,
                subtitle = subtitle,
            )
        )
        g.plot()
        return g

        

    # Ancillary functions 
    # -------------------
    def __repr__(self):
        print("\n")
        print("Maximum sample size per type of test:")
        print(
            self.table
            .groupby(['test'])
            .apply(lambda col: col.query('ntot==ntot.max()'))
            .filter(['ntot', 'par1_value', 'par2_value'])
            .reset_index(drop=False)
            .sort_values(['ntot'], ascending=True)
            .to_string()
        )
        return ""


    def __tidy__(self, obj, test, par1, par2, par1_label, par2_label,
                 outcome):
        par1 = obj.rx2['parms'].rx2[par1]
        par2 = obj.rx2['parms'].rx2[par2]
        if len(par2)<len(par1):
            par2 = list(par2)*len(par1)
        tau  = par2-par1
        if not self.ncontrol:
            n1   = obj.rx2['n'][:len(par1)]
            n2   = obj.rx2['n'][len(par1):]
            power=self.power
        else:
            n1   = obj.rx2['n'][0]
            n2   = obj.rx2['n'][0]
            power= obj.rx2['power']
        d = {
            'test'       : test,
            'stat'       : obj.rx2['test'][0],
            'par1'       : par1,
            'par2'       : par2,
            'par1_label' : par1_label,
            'par2_label' : par2_label,
            'tau'        : tau,
            'ntot'       : n1*self.ngroups,
            'ncontrol'   : n1,
            'power'      : power,
            'alpha'      : self.alpha,
            'two-tail'   : self.alternative,
            'design'     : 'fixed',
            'outcome'    : outcome,
        }
        res = pd.DataFrame(d)
        return res


    def __collect__(self):
        self.table = pd.concat([self.est_diff_prop['tidy'],
                                self.est_diff_means['tidy'],
                                self.est_diff_prop_seq_design['tidy'],
                                ])


