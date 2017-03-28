import numpy as np
import pymysql, random, json

def connect_db():
	try:
		conn = pymysql.connect(host='localhost', port=3306, user='root', password='pkuoslab', db='lol_prediction', charset='utf8')
		return conn
	except:
		print ('Exception: MySQL Connection')
		return None

def fetch_champion_dict(jsonfile):
	with open(jsonfile) as jsonfile:
		champion_dict = json.load(jsonfile)
	return champion_dict
		
def fetch_both_sides(gamemode, champion_dict, train_set_rate=0.75):
	X_train_lst = []
	y_train_lst = []
	X_test_lst = []
	y_test_lst = []
	conn = connect_db()
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
			if (random.random() < train_set_rate) :
				X_train_lst.append(X)
				y_train_lst.append(y)
			else :
				X_test_lst.append(X)
				y_test_lst.append(y)
	X_train = np.array(X_train_lst)
	y_train = np.array(y_train_lst)
	X_test = np.array(X_test_lst)
	y_test = np.array(y_test_lst)
	print ('Success: Fetch Both Sides')
	return (X_train, y_train, X_test, y_test)

def fetch_one_side(gamemode, champion_dict, train_set_rate=0.75):
	X_train_lst = []
	y_train_lst = []
	X_test_lst = []
	y_test_lst = []
	conn = connect_db()
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
			if (random.random() < train_set_rate) :
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
			if (random.random() < train_set_rate) :
				X_train_lst.append(X)
				y_train_lst.append(y)
			else :
				X_test_lst.append(X)
				y_test_lst.append(y)
	X_train = np.array(X_train_lst)
	y_train = np.array(y_train_lst)
	X_test = np.array(X_test_lst)
	y_test = np.array(y_test_lst)
	print ('Success: Fetch One Side')
	return (X_train, y_train, X_test, y_test)