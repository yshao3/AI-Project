import csv
from datetime import datetime
dic = {}
def getSPPE(path):
	with open(path,'rb') as p:
		reader = csv.reader(p)
		for row in reader:
			if (len(row) >= 2):
				try:
					key = row[0].split("-")
					if len(key) < 2: 
						continue
					k = key[0][-2:]+'-'+key[1]
					date = datetime.strptime(k,'%y-%m')
					k = str(date.strftime('%y-%m'))
					if k not in dic:
						dic[k] = {}
					dic[k]['PE'] = float (row[1])
				except ValueError:
					continue
				
def getSP(path):
	with open(path,'rb') as p:
		reader = csv.reader(p)
		for row in reader:
			if (len(row) >= 4):
				try:
					date = datetime.strptime(row[1],'%m/%d/%y')
					k = str(date.strftime('%y-%m'))
					# print k
					if k not in dic:
						dic[k] = {}
					dic[k]['SP'] = float (row[2])
					dic[k]['SPn'] = float (row[3])
				except ValueError:
					continue

# def getSPYTD(path):
# 	with open(path,'rb') as p:
# 		reader = csv.reader(p)
# 		for row in reader:
# 			if (len(row) >= 2):
# 				try:
# 					key = row[0].split("-")
# 					if len(key) < 2: 
# 						continue
# 					k = key[0][2:]+'-'+key[1]
# 					# print(k)
# 					if k not in dic:
# 						dic[k] = {}
# 					if 'YTD' not in dic[k]:
# 						dic[k]['YTD'] =0
# 					dic[k]['YTD'] += float (row[1])
# 				except ValueError:
# 					continue
				
def getGoldM(path):
	with open(path,'rb') as p:
		reader = csv.reader(p)
		for row in reader:
			try:
				if (len(row) >= 4):
					date = datetime.strptime(row[1],'%m/%d/%y')
					k = str(date.strftime('%y-%m'))
					# print k
					if k not in dic:
						dic[k] = {}
					dic[k]["Gold"] = float (row[2])
					dic[k]["Goldn"] = float (row[3])
			except ValueError:
					continue

def getLibor(path):
	with open(path,'rb') as p:
		reader = csv.reader(p)
		for row in reader:
			if (len(row) >= 2):
				try:
					key = row[0].split("-")
					if len(key) < 2: 
						continue
					k = key[0][-2:]+'-'+key[1]
					date = datetime.strptime(k,'%y-%m')
					k = str(date.strftime('%y-%m'))
					print k
					if k not in dic:
						dic[k] = {}
					if 'Libor' not in dic[k]:
						dic[k]['Libor'] =0
					dic[k]['Libor'] += float (row[1])
				except ValueError:
					continue
	
def getOilM(path):
	with open(path,'rb') as p:
		reader = csv.reader(p)
		for row in reader:
			if (len(row) >= 4):
				try:
					key = row[1].split("-")
					if len(key) < 3: 
						continue
					k = key[0][2:]+'-'+key[1]
					date = datetime.strptime(k,'%y-%m')
					k = str(date.strftime('%y-%m'))
					# print k
					if k not in dic:
						dic[k] = {}
					dic[k]['Oil'] = float (row[2])
					dic[k]['Oiln'] = float (row[3])
				except ValueError:
					continue

def getDJ(path):
	with open(path,'rb') as p:
		reader = csv.reader(p)
		for row in reader:
			if (len(row) >= 4):
				try:
					date = datetime.strptime(row[1],'%m/%d/%y')
					k = str(date.strftime('%y-%m'))
					# print k
					if k not in dic:
						dic[k] = {}
					dic[k]['DJ'] = float (row[2])
					dic[k]['DJn'] =float (row[3])
				except ValueError:
					continue
getSP("SP500.csv")
getSPPE("SP500PE.csv")
# getSPYTD("SP500YTD.csv")
getDJ("DJM.csv")
getOilM("OilM.csv")
getGoldM("GoldM.csv")
getLibor("LibD.csv")
# print(dic['17-01'])
writer = csv.writer(open("features.csv","w"))
writer.writerow(["Key", "SP", "SPn", "PE","DJ","DJn","Gold", "Goldn","Oil","Oiln","Libor"])
for k in dic.keys():
	key = str(k)
	if 'SP' not in dic[k]:
		sp = 0
	else:
		sp =dic[k]['SP']
	if 'DJ' not in dic[k]:
		dj = 0
	else:
		dj =dic[k]['DJ']
	if 'DJn' not in dic[k]:
		djn = 0
	else:
		djn =dic[k]['DJn']
	if 'Oil' not in dic[k]:
		oil = 0
	else:
		oil =dic[k]['Oil']
	if 'Oiln' not in dic[k]:
		oiln = 0
	else:
		oiln =dic[k]['Oiln']
	# if 'YTD' not in dic[k]:
	# 	ytd = 0
	# else:
	# 	# print dic[k]['YTD']
	# 	ytd =dic[k]['YTD']
	if 'PE' not in dic[k]:
		pe = 0
	else:
		pe =dic[k]['PE']
	if 'Libor' not in dic[k]:
		lib = 0
	else:
		lib =dic[k]['Libor']
	if 'SPn' not in dic[k]:
		spn = 0
	else:
		spn =dic[k]['SPn']
	if 'Gold' not in dic[k]:
		gold = 0
	else:
		gold =dic[k]['Gold']
	if 'Goldn' not in dic[k]:
		goldn = 0
	else:
		goldn =dic[k]['Goldn']
	writer.writerow([key, sp, spn, pe,dj,djn, gold,goldn,oil,oiln,lib])