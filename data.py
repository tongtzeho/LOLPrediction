import numpy as np

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

	fin.close()
	X_train = np.array(X_train_lst)
	y_train = np.array(y_train_lst)
	X_test = np.array(X_test_lst)
	y_test = np.array(y_test_lst)
	return (X_train, y_train, X_test, y_test)