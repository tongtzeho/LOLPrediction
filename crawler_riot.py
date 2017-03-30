import pymysql, random, requests, json, time, re, multiprocessing, os, datetime

start_time = "2017-03-01 00:00:00"
end_time = "2017-03-16 00:00:00"

class RiotAPI(object):
	key = ""
	region = ""
	proxy = {'http': 'http://127.0.0.1:1080'}
	
	def __init__(self, region, key):
		self.region = region
		self.key = key
		
	def request_api_url(self, api_url):
		err_time = 0
		while err_time < 100:
			try:
				if err_time % 2 == 0:
					r = requests.get(api_url+'?api_key='+self.key, timeout=20)
				else:
					r = requests.get(api_url+'?api_key='+self.key, proxies=self.proxy, timeout=20)
				status_code = r.status_code
				if status_code in (404, 400, 500, 429, 503, 401, 403):
					err_time += 1
					print ("Exception: Status Code = "+str(status_code))
					time.sleep(2)
					continue
				else:
					time.sleep(0.75)
					return r.json()
			except:
				err_time += 1
				print ("Exception: Internet Error")
				time.sleep(2)
				continue
	
	def get_champion(self):
		return self.request_api_url('https://'+self.region+'.api.pvp.net/api/lol/static-data/'+self.region+'/v1.2/champion')
		
	def get_summoner(self, summonernames):
		return self.request_api_url('https://'+self.region+'.api.pvp.net/api/lol/'+self.region+'/v1.4/summoner/by-name/'+summonernames)
		
	def get_recentgame(self, summonerid):
		return self.request_api_url('https://'+self.region+'.api.pvp.net/api/lol/'+self.region+'/v1.3/game/by-summoner/'+summonerid+'/recent')

class MySQL(object):
	
	host = 'localhost'
	port = 3306
	user = ''
	password = ''
	database = ''
	charset = 'utf8'
	
	def __init__(self, user, password, database):
		self.user = user
		self.password = password
		self.database = database
		
	def connect(self):
		try:
			conn = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password, db=self.database, charset=self.charset)
			return conn
		except:
			print ('Exception: MySQL Connection')
			time.sleep(1)
			return None
	
	def get_summonerid(self, region):
		conn = self.connect()
		if (conn == None): return []
		cursor = conn.cursor()
		try:
			ifexists = cursor.execute("select summonerid from player_riot where region = '"+region+"'")
			if ifexists == 0:
				cursor.close()
				conn.close()
				return []
			else:
				fetchdata = cursor.fetchall()
				cursor.close()
				conn.close()
				result = []
				for item in fetchdata:
					result.append(item[0])
				random.shuffle(result)
				return result
		except:
			print ('Exception: Get SummonerID From MySQL')
			cursor.close()
			conn.close()
			return []
			
	def update_summonerid(self, region, summonerid):
		conn = self.connect()
		if (conn == None): return False
		cursor = conn.cursor()
		try:
			cursor.execute("insert into player_riot values ('"+region+"', "+summonerid+")")
			conn.commit()
			cursor.close()
			conn.close()
			return True
		except:
			cursor.close()
			conn.close()
			return False
			
	def search_game(self, region, gameid):
		conn = self.connect()
		if (conn == None): return -1
		cursor = conn.cursor()
		try:
			ifexists = cursor.execute("select region,gameid from game_riot where region = '"+region+"' and gameid = "+gameid)
			cursor.close()
			conn.close()
			return ifexists
		except:
			print ('Exception: Search Game - '+region+' : '+gameid)
			cursor.close()
			conn.close()
			return -1
			
	def update_game(self, region, gameid, gameinfo):
		conn = self.connect()
		if (conn == None): return False
		cursor = conn.cursor()
		try:
			if gameinfo == None:
				print ('Failed: Insert Game - '+region+' : '+gameid)
				cursor.close()
				conn.close()
				return False
			if gameinfo[4] == None or gameinfo[5] == None or gameinfo[6] == None:
				print ('Failed: Insert Game - '+region+' : '+gameid)
				ret = False
				pass
			else:
				try:
					cmd = "insert into game_riot values ('"+region+"', "+gameid+", "+gameinfo[0]+", '"+gameinfo[1]+"', '"+gameinfo[2]+"', '"+gameinfo[3]+"', '"+gameinfo[4]+"', '"+gameinfo[5]+"', "+gameinfo[6]+", "+gameinfo[7]+")"
					cursor.execute(cmd)
					conn.commit()
					print ('Insert New Game - '+region+' : '+gameid)
					ret = True
				except:
					ret = False
			cursor.close()
			conn.close()
			return ret
		except:
			print ('Exception: Insert Game - '+region+' : '+gameid)
			cursor.close()
			conn.close()
			return False
		
def parse_champion(champion):
	result = {}
	i = 0
	try:
		for c in champion['data'].values():
			result[c['id']] = (i, c['name'])
			i += 1
	except:
		print ('Exception: Parse Champion')
	return result
	
def parse_recentgame(game):
	result = {}
	try:
		for newgame in game['games']:
			try:
				if newgame['invalid']: continue
				newitem = [str(newgame['mapId']), newgame['gameType'], newgame['subType'], newgame['gameMode'], None, None, None, str(newgame['createDate']), []]
				bluesidechampionidstr = ""
				redsidechampionidstr = ""
				if newgame['teamId'] == 100: # blue side
					if newgame['stats']['win']: newitem[6] = '1' # blue side win?
					else: newitem[6] = '0'
					bluesidechampionidstr += ";"+str(newgame['championId'])
				elif newgame['teamId'] == 200: # red side
					if newgame['stats']['win']: newitem[6] = '0'
					else: newitem[6] = '1'
					redsidechampionidstr += ";"+str(newgame['championId'])
				else: continue
				for fellow in newgame['fellowPlayers']:
					newitem[8].append(str(fellow['summonerId']))
					if fellow['teamId'] == 100:
						bluesidechampionidstr += ";"+str(fellow['championId'])
					elif fellow['teamId'] == 200:
						redsidechampionidstr += ";"+str(fellow['championId'])
					else:
						break
				if len(bluesidechampionidstr.split(';')) != len(redsidechampionidstr.split(';')): continue
				newitem[4] = bluesidechampionidstr[1:]
				newitem[5] = redsidechampionidstr[1:]
				result[str(newgame['gameId'])] = newitem
			except:
				continue
		return result
	except:
		print ('Exception: Parse Game Detail')
		return result

def main_loop(region, key, rate):
	mysql = MySQL('root', 'pkuoslab', 'lol_prediction_riot')
	summoneridlst = mysql.get_summonerid(region)
	crawler = RiotAPI(region, key)
	while (len(summoneridlst)):
		for summonerid in summoneridlst:
			if (os.path.isfile('exit0')):
				print ('Process "'+region+'" Exit')
				return
			try:
				if random.random() < rate:
					gamedict = parse_recentgame(crawler.get_recentgame(str(summonerid)))
					if len(gamedict):
						for gameid, gameinfo in gamedict.items():
							if mysql.search_game(region, gameid) == 0:
								mysql.update_game(region, gameid, gameinfo)
							if gameinfo != None and gameinfo[8] != None and len(gameinfo[8]):
								for summonerid in gameinfo[8]:
									mysql.update_summonerid(region, summonerid)
			except:
				print ('Exception: Unknown Error')
				continue
		summoneridlst = mysql.get_summonerid(region)
	print ('No Summoner in Region '+region)

if __name__ == '__main__':
	if os.path.isfile('exit0'): os.remove('exit0')
	if not os.path.isfile('Key'):
		print ('Key Not Found')
		exit()
	fin = open('Key', 'r')
	key = fin.read()
	fin.close()
	param_list = [['NA', key, 0.7], ['EUW', key, 0.7], ['KR', key, 0.7]]
	processes = []
	for param in param_list:
		if param == param_list[0]: continue
		processes.append(multiprocessing.Process(target = main_loop, args = (param[0], param[1], param[2])))
	for p in processes:
		p.start()
	main_loop(param_list[0][0], param_list[0][1], param_list[0][2])
	for p in processes:
		p.join()
	print ("All Processes Exit")
	exit()