import pkg_resources
import pandas as pd

def load_survey():
    """Return a dataframe with European Social Survey sample data.

    Contains the following fields:
        index   
        name     
        name.full

    """
    stream = pkg_resources.resource_stream(__name__, './survey.csv')
    return read_data(stream, encoding='utf-8', sep=';')

