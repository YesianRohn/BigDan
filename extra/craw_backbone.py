import requests, re, json, numpy as np
from bs4 import BeautifulSoup

url = 'https://paperswithcode.com/sota/image-classification-on-imagenet?tag_filter=171'
output_dir = 'extra/output/'
raw_page_output = output_dir + 'raw_page.txt'
page_output = output_dir + 'page.txt'
raw_metrics_output = output_dir + 'raw_metrics.txt'
metrics_output = output_dir + 'backbone.txt'

def RM(patt, sr):
	mat = re.search(patt, sr, re.DOTALL | re.MULTILINE)
	return mat.group(1) if mat else ''

def getPage(url, cookie='', proxy='', timeout=5):
	try:
		headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
		if cookie != '': headers['cookie'] = cookie
		if proxy != '': 
			proxies = {'http': proxy, 'https': proxy}
			resp = requests.get(url, headers=headers, proxies=proxies, timeout=timeout)
		else: resp = requests.get(url, headers=headers, timeout=timeout)
		content = resp.content
		headc = content[:min([3000,len(content)])].decode(errors='ignore')
		charset = RM('charset="?([-a-zA-Z0-9]+)', headc)
		if charset == '': charset = 'utf-8'
		content = content.decode(charset, errors='replace')
	except Exception as e:
		print(e)
		content = ''
	return content

def saveRawPage(raw_page): # debug only
    with open(raw_page_output, 'w') as fout:
        fout.write(raw_page)

def savePage(url):
    page = getPage(url)
    saveRawPage(page) # debug only
    page = re.sub('[\r\t\n]', '', page)
    with open(page_output, 'w') as fout:
        fout.write(page)

def loadPage():
    with open(page_output, 'r') as fin:
        content = fin.readlines()
    return str(content[0])

def saveRawMetrics(raw_metrics): # debug only
    with open(raw_metrics_output, 'w', encoding='utf-8') as fout: # for debug
        fout.write(str(raw_metrics))

def saveMetrics(page):
    soup = BeautifulSoup(page, features='lxml')
    raw_metrics = json.loads(soup.find('script', {'id': 'evaluation-table-data'}).get_text())
    saveRawMetrics(raw_metrics) # debug only
    metrics = []
    for metric in raw_metrics:
        method = str(metric['method'])
        acc = str(metric['metrics']['Top 1 Accuracy'])
        param = str(metric['metrics']['Number of params'])
        if acc == 'None' or param == 'None': continue
        if acc.find('%') == -1: continue # acc += '%'
        if param.find('M') == -1: continue # param += 'M'
        score = str(float(acc[:acc.find('%')]) / 100 * np.exp(-np.log10(float(param[:param.find('M')]) * 1e6 / 1e8 + 1)))
        metrics.append((method, acc, param, score))
    metrics.sort(key=lambda t: -float(t[3]))
    with open(metrics_output, 'w', encoding='utf-8') as fout:
        for metric in metrics:
            fout.write('\t'.join([str(x) for x in metric]) + '\n')

if __name__ == '__main__':
    savePage(url) # need to run it at first time!
    page = loadPage()
    saveMetrics(page)
