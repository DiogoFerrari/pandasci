from pandasci import ds
import pandas as pd
import bs4 as bs
import requests as rq
import lxml
import webbrowser


# * functions (beautful soap)

class scraping:

    def __init__(self, url):
        '''
        Return beautifulsoap object

        Input
    	-----
           url  : URL address

        Output
    	------
           BS object
        '''
        print(f"\n\nReading URL {url}...\n", flush=True)
        self.url=url
        self.source=self.getHTML(url)
        self.tables=self.__get_tables__()

    def getHTML(self, url):
        # handle "Page not found", which urlopen returns some HTML error by default
        try:
            html = rq.get(url, auth=('user', 'pass')).content
        except:
            print("Page not found")
            return None
        # handle server not found, which Beautiful soup returns None by default
        try:
            html = bs.BeautifulSoup(html, 'html.parser')
        except AttributeError as e:
            print("Server not found")
            return None
        return html

    # function returns none if tag is not found
    def msgTagNotFound(self, tag, attrs):
            print("")
            print("Tag", """+tag+""","and/or attributes:")
            for k, v in attrs.items():
                print("""+k+""",":", """+v+""")
                print("not found")


    def getAllTags(self, tag, attrs, soup):
        results = soup.findAll(tag, attrs = attrs)
        if results == []:
            msgTagNotFound(tag, attrs)
            return None
        else:
            return results


    def getTags(self, tag, attrs, soup):
        results = soup.find(tag, attrs = attrs)
        if results == []:
            takeNotFoundMsg(tag, attrs)
            return None
        else:
            return results

    def __get_tables__(self):
        '''
        Extract tables in the HTML page

        Input
    	-----
    	   self  : arg1

        Output
    	------
           return tables as data.frame
        '''
        tablist = self.source.find_all('table')
        tablist = pd.read_html(str(tablist))

        tabs = ds.eDataFrame()
        for i, tab in enumerate(tablist):
            label=f"Table {i}"
            tabs=tabs.bind_row(ds.eDataFrame(
                {'id':label,
                 'tab':[tab]}
            ))
        return tabs

    def get_table(self, idx):
        return  self.tables.select_rows(index=[idx]).unnest('tab', 'id')
        

    def tables_glimpse(self):
        '''
        See headings of all tables

        Input
    	-----

        Output
    	------
           Print tables
        '''
        for idx, row in self.tables.iterrows():
            print(f"===================================", flush=True)
            print(f"Table Id.  : {row.id}", flush=True)
            print(f"Table index: {idx}", flush=True)
            print(f"", flush=True)
            print(f"{row.tab}", flush=True)
        print(f"===================================", flush=True)

    def open_url(self):
        webbrowser.open(self.url)
