import numpy as np
import pymysql, random, json

def connect_db(database):
	try:
		conn = pymysql.connect(host='localhost', port=3306, user='root', password='pkuoslab', db=database, charset='utf8')
		return conn
	except:
		print ('Exception: MySQL Connection')
		return None

def fetch_champion_dict(jsonfile):
	with open(jsonfile) as jsonfile:
		champion_dict = json.load(jsonfile)
	return champion_dict
		
def fetch_both_sides_tgp(gamemode, champion_dict):
	X_train_lst = []
	y_train_lst = []
	X_test_lst = []
	y_test_lst = []
	X_train = np.array(X_train_lst)
	y_train = np.array(y_train_lst)
	X_test = np.array(X_test_lst)
	y_test = np.array(y_test_lst)
	conn = connect_db('lol_prediction_tgp')
	if conn == None: return (X_train, y_train, X_test, y_test)
	cursor = conn.cursor()
	try:
		ifexists = cursor.execute("select * from game_"+gamemode)
		if ifexists == 0:
			cursor.close()
			conn.close()
			return (X_train, y_train, X_test, y_test)
		else:
			all_set = cursor.fetchall()
			cursor.close()
			conn.close()
	except:
		print ("Exception: Selecting")
		return (X_train, y_train, X_test, y_test)	
	cnt = 0
	for line in all_set:
		X = np.zeros(len(champion_dict)*2)
		y = line[3]
		valid = True
		for i in range(10):
			curchampionindex = int(champion_dict[str(line[6*i+5])][0])
			kill = line[6*i+6]
			death = line[6*i+7]
			assist = line[6*i+8]
			dmg = line[6*i+9]
			if dmg < 500 or (kill+assist)/(death+1) < 0.1:
				valid = False
				break
			if i < 5:
				X[curchampionindex] = 1
			else :
				X[curchampionindex+len(champion_dict)] = 1
		if valid:
			cnt += 1
			if (cnt % 4 != 0) :
				X_train_lst.append(X)
				y_train_lst.append(y)
			else :
				X_test_lst.append(X)
				y_test_lst.append(y)
	X_train = np.array(X_train_lst)
	y_train = np.array(y_train_lst)
	X_test = np.array(X_test_lst)
	y_test = np.array(y_test_lst)
	print ('Success: Fetch Both Sides '+str(cnt))
	return (X_train, y_train, X_test, y_test)

def fetch_one_side_tgp(gamemode, champion_dict):
	X_train_lst = []
	y_train_lst = []
	X_test_lst = []
	y_test_lst = []
	X_train = np.array(X_train_lst)
	y_train = np.array(y_train_lst)
	X_test = np.array(X_test_lst)
	y_test = np.array(y_test_lst)
	conn = connect_db('lol_prediction_tgp')
	if conn == None: return (X_train, y_train, X_test, y_test)
	cursor = conn.cursor()
	try:
		ifexists = cursor.execute("select * from game_"+gamemode)
		if ifexists == 0:
			cursor.close()
			conn.close()
			return (X_train, y_train, X_test, y_test)
		else:
			all_set = cursor.fetchall()
			cursor.close()
			conn.close()
	except:
		print ("Exception: Selecting")
		return (X_train, y_train, X_test, y_test)
	cnt = 0
	for line in all_set:
		X = np.zeros(len(champion_dict))
		y = line[3]
		valid = True
		for i in range(0, 5):
			curchampionindex = int(champion_dict[str(line[6*i+5])][0])
			kill = line[6*i+6]
			death = line[6*i+7]
			assist = line[6*i+8]
			dmg = line[6*i+9]
			if dmg < 500 or (kill+assist)/(death+1) < 0.1:
				valid = False
				break
			X[curchampionindex] = 1
		if valid:
			cnt += 1
			if (cnt % 8 != 0 and cnt % 8 != 3) :
				X_train_lst.append(X)
				y_train_lst.append(y)
			else :
				X_test_lst.append(X)
				y_test_lst.append(y)
		X = np.zeros(len(champion_dict))
		y = 1-line[3]
		valid = True
		for i in range(5, 10):
			curchampionindex = int(champion_dict[str(line[6*i+5])][0])
			kill = line[6*i+6]
			death = line[6*i+7]
			assist = line[6*i+8]
			dmg = line[6*i+9]
			if dmg < 500 or (kill+assist)/(death+1) < 0.1:
				valid = False
				break
			X[curchampionindex] = 1
		if valid:
			cnt += 1
			if (cnt % 8 != 0 and cnt % 8 != 3) :
				X_train_lst.append(X)
				y_train_lst.append(y)
			else :
				X_test_lst.append(X)
				y_test_lst.append(y)
	X_train = np.array(X_train_lst)
	y_train = np.array(y_train_lst)
	X_test = np.array(X_test_lst)
	y_test = np.array(y_test_lst)
	print ('Success: Fetch One Side '+str(cnt))
	return (X_train, y_train, X_test, y_test)
	
def fetch_both_sides_riot(mapid, gametype, subtype, gamemode, createdate, champion_dict):
	X_train_lst = []
	y_train_lst = []
	X_test_lst = []
	y_test_lst = []
	X_train = np.array(X_train_lst)
	y_train = np.array(y_train_lst)
	X_test = np.array(X_test_lst)
	y_test = np.array(y_test_lst)
	conn = connect_db('lol_prediction_riot')
	if conn == None: return (X_train, y_train, X_test, y_test)
	cursor = conn.cursor()
	try:
		ifexists = cursor.execute("select bluesidewin, bluesidechampion, redsidechampion from game_riot where mapid="+mapid+" and gametype='"+gametype+"' and subtype='"+subtype+"' and gamemode='"+gamemode+"' and createdate>="+createdate[0]+" and createdate<="+createdate[1])
		if ifexists == 0:
			cursor.close()
			conn.close()
			return (X_train, y_train, X_test, y_test)
		else:
			all_set = cursor.fetchall()
			cursor.close()
			conn.close()
	except:
		print ("Exception: Selecting")
		return (X_train, y_train, X_test, y_test)
	cnt = 0
	for line in all_set:
		X = np.zeros(len(champion_dict)*2)
		y = line[0]
		for championidstr in line[1].split(';'):
			championid = int(champion_dict[championidstr][0])
			X[championid] = 1
		for championidstr in line[2].split(';'):
			championid = int(champion_dict[championidstr][0])
			X[championid+len(champion_dict)] = 1
		cnt += 1
		if (cnt % 4 != 0) :
			X_train_lst.append(X)
			y_train_lst.append(y)
		else :
			X_test_lst.append(X)
			y_test_lst.append(y)
	X_train = np.array(X_train_lst)
	y_train = np.array(y_train_lst)
	X_test = np.array(X_test_lst)
	y_test = np.array(y_test_lst)
	print ('Success: Fetch Both Sides '+str(cnt))
	return (X_train, y_train, X_test, y_test)
	
def fetch_one_side_riot(mapid, gametype, subtype, gamemode, createdate, champion_dict):
	X_train_lst = []
	y_train_lst = []
	X_test_lst = []
	y_test_lst = []
	X_train = np.array(X_train_lst)
	y_train = np.array(y_train_lst)
	X_test = np.array(X_test_lst)
	y_test = np.array(y_test_lst)
	conn = connect_db('lol_prediction_riot')
	if conn == None: return (X_train, y_train, X_test, y_test)
	cursor = conn.cursor()
	try:
		ifexists = cursor.execute("select bluesidewin, bluesidechampion, redsidechampion from game_riot where mapid="+mapid+" and gametype='"+gametype+"' and subtype='"+subtype+"' and gamemode='"+gamemode+"' and createdate>="+createdate[0]+" and createdate<="+createdate[1])
		if ifexists == 0:
			cursor.close()
			conn.close()
			return (X_train, y_train, X_test, y_test)
		else:
			all_set = cursor.fetchall()
			cursor.close()
			conn.close()
	except:
		print ("Exception: Selecting")
		return (X_train, y_train, X_test, y_test)
	cnt = 0
	for line in all_set:
		X = np.zeros(len(champion_dict))
		y = line[0]
		for championidstr in line[1].split(';'):
			championid = int(champion_dict[championidstr][0])
			X[championid] = 1
		cnt += 1
		if (cnt % 8 != 0 and cnt % 8 != 3):
			X_train_lst.append(X)
			y_train_lst.append(y)
		else:
			X_test_lst.append(X)
			y_test_lst.append(y)
		X = np.zeros(len(champion_dict))
		for championidstr in line[2].split(';'):
			championid = int(champion_dict[championidstr][0])
			X[championid] = 1
		cnt += 1
		if (cnt % 8 != 0 and cnt % 8 != 3):
			X_train_lst.append(X)
			y_train_lst.append(1-y)
		else:
			X_test_lst.append(X)
			y_test_lst.append(1-y)
	X_train = np.array(X_train_lst)
	y_train = np.array(y_train_lst)
	X_test = np.array(X_test_lst)
	y_test = np.array(y_test_lst)
	print ('Success: Fetch One Side '+str(cnt))
	return (X_train, y_train, X_test, y_test)
	
def read_file_both_sides(data_set, champion_match, champion_num):		
	fin = open (champion_match, "r")		
	dict = {}		
	#stat = {}		
	for line in fin:		
		championindex = int(line.split(' ')[0])		
		championid = int(line.split(' ')[1])		
		dict[championid] = championindex		
	#	stat[championid] = {'k':0, 'd':0, 'a':0, 'appear':0, 'win':0, 'dmg':0}		
		
	fin.close()		
	X_train_lst = []		
	y_train_lst = []		
	X_test_lst = []		
	y_test_lst = []		
	fin = open (data_set, "r")
	cnt = 0		
	for line in fin:		
		type = int(line.split(' ')[0])		
		X = np.zeros(champion_num*2)		
		y = int(line.split(' ')[71])		
		valid = True		
		for i in range(10):		
			curchampionindex = dict[int(line.split(' ')[7*i+3])]		
			kill = int(line.split(' ')[7*i+4])		
			death = int(line.split(' ')[7*i+5])		
			assist = int(line.split(' ')[7*i+6])		
			dmg = int(line.split(' ')[7*i+7])		
			if dmg < 1000 or (kill+assist)/(death+1) < 0.1:		
				valid = False		
				break		
			if i < 5:		
				X[curchampionindex] = 1		
			else :		
				X[curchampionindex+champion_num] = 1		
		if valid:	
			cnt += 1	
			if (type < 45) :		
				X_train_lst.append(X)		
				y_train_lst.append(y)		
			else :		
				X_test_lst.append(X)		
				y_test_lst.append(y)		
		
	fin.close()		
	X_train = np.array(X_train_lst)		
	y_train = np.array(y_train_lst)		
	X_test = np.array(X_test_lst)		
	y_test = np.array(y_test_lst)
	print ('Success: Fetch Both Sides '+str(cnt))	
	return (X_train, y_train, X_test, y_test)		
		
def read_file_one_side(data_set, champion_match, champion_num):		
	fin = open (champion_match, "r")		
	dict = {}		
	#stat = {}		
	for line in fin:		
		championindex = int(line.split(' ')[0])		
		championid = int(line.split(' ')[1])		
		dict[championid] = championindex		
	#	stat[championid] = {'k':0, 'd':0, 'a':0, 'appear':0, 'win':0, 'dmg':0}		
		
	fin.close()		
	X_train_lst = []		
	y_train_lst = []		
	X_test_lst = []		
	y_test_lst = []		
	fin = open (data_set, "r")
	cnt = 0		
	for line in fin:		
		type = int(line.split(' ')[0])		
		X = np.zeros(champion_num)		
		y = int(line.split(' ')[71])		
		for i in range(5):		
			curchampionindex = dict[int(line.split(' ')[7*i+3])]		
			X[curchampionindex] = 1		
		if (type < 45) :		
			X_train_lst.append(X)		
			y_train_lst.append(y)		
		else :		
			X_test_lst.append(X)		
			y_test_lst.append(y)		
		X = np.zeros(champion_num)		
		y = 1-y		
		for i in range(5, 10):		
			curchampionindex = dict[int(line.split(' ')[7*i+3])]		
			X[curchampionindex] = 1		
		if (type < 45) :		
			X_train_lst.append(X)		
			y_train_lst.append(y)		
		else :		
			X_test_lst.append(X)		
			y_test_lst.append(y)		
		cnt += 2
	fin.close()		
	X_train = np.array(X_train_lst)		
	y_train = np.array(y_train_lst)		
	X_test = np.array(X_test_lst)		
	y_test = np.array(y_test_lst)	
	print ('Success: Fetch One Side '+str(cnt))	
	return (X_train, y_train, X_test, y_test)
	
def read_file_both_sides_old(data_set, champion_match, champion_num):		
	fin = open (champion_match, "r")		
	dict = {}		
	#stat = {}		
	for line in fin:		
		championindex = int(line.split(' ')[0])		
		championid = int(line.split(' ')[1])		
		dict[championid] = championindex		
	#	stat[championid] = {'k':0, 'd':0, 'a':0, 'appear':0, 'win':0, 'dmg':0}		
		
	fin.close()		
	X_train_lst = []		
	y_train_lst = []		
	X_test_lst = []		
	y_test_lst = []		
	fin = open (data_set, "r")
	cnt = 0		
	for line in fin:		
		type = int(line.split(' ')[1])		
		X = np.zeros(champion_num*2)		
		y = int(line.split(' ')[32])		
		valid = True		
		for i in range(10):		
			curchampionindex = dict[int(line.split(' ')[3*i+4])]				
			if i < 5:		
				X[curchampionindex] = 1		
			else :		
				X[curchampionindex+champion_num] = 1		
		if valid:	
			cnt += 1	
			if (type < 45) :		
				X_train_lst.append(X)		
				y_train_lst.append(y)		
			else :		
				X_test_lst.append(X)		
				y_test_lst.append(y)		
		
	fin.close()		
	X_train = np.array(X_train_lst)		
	y_train = np.array(y_train_lst)		
	X_test = np.array(X_test_lst)		
	y_test = np.array(y_test_lst)		
	print ('Success: Fetch Both Sides '+str(cnt))	
	return (X_train, y_train, X_test, y_test)		
		
def read_file_one_side_old(data_set, champion_match, champion_num):		
	fin = open (champion_match, "r")		
	dict = {}		
	#stat = {}		
	for line in fin:		
		championindex = int(line.split(' ')[0])		
		championid = int(line.split(' ')[1])		
		dict[championid] = championindex		
	#	stat[championid] = {'k':0, 'd':0, 'a':0, 'appear':0, 'win':0, 'dmg':0}		
		
	fin.close()		
	X_train_lst = []		
	y_train_lst = []		
	X_test_lst = []		
	y_test_lst = []		
	fin = open (data_set, "r")	
	cnt = 0	
	for line in fin:		
		type = int(line.split(' ')[1])		
		X = np.zeros(champion_num)		
		y = int(line.split(' ')[32])		
		for i in range(5):		
			curchampionindex = dict[int(line.split(' ')[3*i+4])]		
			X[curchampionindex] = 1		
		if (type < 45) :		
			X_train_lst.append(X)		
			y_train_lst.append(y)		
		else :		
			X_test_lst.append(X)		
			y_test_lst.append(y)		
		X = np.zeros(champion_num)		
		y = 1-y		
		for i in range(5, 10):		
			curchampionindex = dict[int(line.split(' ')[3*i+4])]		
			X[curchampionindex] = 1		
		if (type < 45) :		
			X_train_lst.append(X)		
			y_train_lst.append(y)		
		else :		
			X_test_lst.append(X)		
			y_test_lst.append(y)		
		cnt += 2
	fin.close()		
	X_train = np.array(X_train_lst)		
	y_train = np.array(y_train_lst)		
	X_test = np.array(X_test_lst)		
	y_test = np.array(y_test_lst)		
	print ('Success: Fetch One Side '+str(cnt))	
	return (X_train, y_train, X_test, y_test) 
