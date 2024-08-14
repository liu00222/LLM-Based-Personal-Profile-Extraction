from html.parser import HTMLParser


class HTMLParser(HTMLParser):

    def __init__(self, tag_list=('p', 'h1', 'h2', 'li'), include_link=True):
        super().__init__()
        self.data = []
        self.capture = False
        self.include_link = include_link
        self.tag_list = tag_list

    def handle_starttag(self, tag, attrs):
        if tag in self.tag_list:
            self.capture = True
        
        if tag in ('a') and self.include_link:
            if len(attrs) > 0:
                for i in range(len(attrs)):
                    self.data.append(attrs[i][0])
                    self.data.append(attrs[i][1])
        
        if tag in ('img') and self.include_link:
            for i in range(len(attrs)):
                if attrs[i][0] == 'src':
                    self.data.append(attrs[i][1])

    def handle_endtag(self, tag):
        if tag in self.tag_list:
            self.capture = False

    def handle_data(self, data):
        if self.capture:
            self.data.append(data)

def parsed_data_to_string(dataset, parsed_data, model_name=''):
    res = ''
    for i in range(len(parsed_data)):
        curr = parsed_data[i].strip().replace('\n', '').replace('\t', '').replace('href', '')
        if curr not in ('#', ''):
            res = f'{res}\n{curr}'
    return res