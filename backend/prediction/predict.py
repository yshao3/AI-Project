# # Save Model Using Pickle
import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
import urllib2
from bs4 import BeautifulSoup as bs

def get_historical_data(name, number_of_days):
	data = []
	url = "https://finance.yahoo.com/quote/" + name + "/history/"
	rows = bs(urllib2.urlopen(url).read()).findAll('table')[0].tbody.findAll('tr')

	for each_row in rows:
		divs = each_row.findAll('td')
		if divs[1].span.text  != 'Dividend': #Ignore this row in the table
			#I'm only interested in 'Close' price; For other values, play with divs[1 - 5]
			data.append({'Date': divs[0].span.text, 'Close': float(divs[4].span.text.replace(',',''))})

	return data[:number_of_days]

def main():
	data = get_historical_data(path, 30)
	past = []
	for d in data:
		past.append(d['Close'])
	loaded_model = pickle.load(open(path+".sav", 'rb'))
	result = loaded_model.predict(np.array([past]))
	print "Predict ["+path+"]'s close price:", result[0]
	x = np.arange(0,30)
	plt.title('Past 30 days close price: ['+path+']')
	plt.plot(x,past)
	plt.show()

if __name__=='__main__':
	if len(sys.argv) < 1:
		print "error: parameters are not enough:"
	path = sys.argv[1]
	main()
#Test
