
# * R modules
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
from rpy2.robjects.packages import data as datar
# import data from package: datar(pkg as loaded into python).fetch('data name')['data name']
# 
pandas2ri.activate()
stats = importr('stats')
base = importr('base')
utils = importr("utils")

# * rutils class

class rutils():
    def __init__(self):
        pass

    def strr(self, obj):
        print(utils.str(obj))
    
    def str(self, obj):
        if isinstance(obj, robj.methods.RS4):
            print(f"\nSlots level 1\n", flush=True)
            print(tuple(obj.slots))
            print(f"\nNames level 1\n", flush=True)
            print(tuple(obj.names))
            for slot in tuple(obj.slots):
                print(f"=========================", flush=True)
                print(f"Slot '{slot}'")
                try:
                    print(f"{tuple(obj.slots[slot].names)}")
                except (OSError, IOError, SyntaxError, NameError, TypeError, ZeroDivisionError) as e:
                    print("empty ... ")
        # print("\nNote: Run nested function to go down the levels. "+
        #       "E.g.: str(extract_info(obj, <slot>, <field>))\n")


    def extract_info(self, obj, slot, field):
        res = obj.slots[slot].find(field)
        # print("\nNote: Run nested function to go down the levels. "+
        #       "E.g.: extract_info(extract_info(obj, <slot>, <field>))\n")
        return res


    def dict2namedlist(self, dict):
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

    def dict2namedvector(self, dict):
        values=list(dict.values())
        string = [True if isinstance(v, str) else False for v in values ]
        if any(string):
            values = [str(v) for v in values]
            v = StrVector(values)
        else:
            v = FloatVector(values)
        v.names = list(dict.keys())
        return v
        
