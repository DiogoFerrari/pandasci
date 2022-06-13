import bs4 as bs
import requests as rq


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
        self.url=url
        self.source=self.getHTML(url)

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
        return soup

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

