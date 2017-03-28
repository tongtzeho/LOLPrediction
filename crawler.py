from selenium import webdriver
import pymysql, random, requests, json, time, re, multiprocessing, os, datetime

start_time = "2017-03-01 00:00:00"
end_time = "2017-03-16 00:00:00"

class Daiwan(object):

	BASE_URL = 'http://lolapi.games-cube.com/'

	token = ''
	phantomjs_path = 'phantomjs/bin/phantomjs.exe'
	
	account = []
	account_id = 0
	
	proxy = {'http': 'http://127.0.0.1:1080'}

	def __init__(self, account_file):
		fin = open(account_file, 'r')
		for line in fin:
			self.account.append((line.split(' ')[0], line.split(' ')[1].replace('\r', "").replace('\n', "")))

	def update_token(self, username, password):
		while True:
			try:
				driver = webdriver.PhantomJS(executable_path=self.phantomjs_path)
				driver.set_page_load_timeout(30)
				driver.get('http://user.games-cube.com/login.aspx')
				driver.find_element_by_id("txt_username").send_keys(username)
				driver.find_element_by_id("txt_password").send_keys(password)
				driver.find_element_by_id("btn_ok").click()
				time.sleep(3)
				driver.get('http://admin.games-cube.com/api/LoLToken.aspx')
				driver.find_element_by_id("lnk_request").click()
				time.sleep(3)
				data = driver.page_source
				driver.quit()
				matcher = re.findall('<h4>[\r\n ]*[A-Z0-9\-]+[\n\r ]*</h4>', data, re.S)
				if len(matcher):
					self.token = matcher[0].replace('<h4>', "").replace('</h4>', "").replace('\n', "").replace('\r', "").replace(' ', "")
					print (username+" Update Token："+self.token)
					return
				else:
					self.token = ""
					print (username+" Cannot Update Token")
			except:
				print ("Exception: "+username+" Update Token")
		
	def request_api_url(self, api_url):
		err_time = 0
		while True:
			try:
				headers = {'DAIWAN-API-TOKEN': self.token}
				ret = requests.get(self.BASE_URL+api_url, headers = headers).json()
				while 'msg' in ret and ret['msg'].startswith('令牌信息已经无效或已经被销毁'):
					print ("Invalid Token")
					self.account_id = (self.account_id+1) % len(self.account)
					self.update_token(self.account[self.account_id][0], self.account[self.account_id][1])
					headers = {'DAIWAN-API-TOKEN': self.token}
					ret = requests.get(self.BASE_URL+api_url, headers = headers).json()
				if (str(ret).startswith('API calls quota exceeded!')):
					ret = requests.get(self.BASE_URL+api_url, headers = headers, proxies=self.proxy).json()
					while 'msg' in ret and ret['msg'].startswith('令牌信息已经无效或已经被销毁'):
						print ("Invalid Token")
						self.account_id = (self.account_id+1) % len(self.account)
						self.update_token(self.account[self.account_id][0], self.account[self.account_id][1])
						headers = {'DAIWAN-API-TOKEN': self.token}
						ret = requests.get(self.BASE_URL+api_url, headers = headers, proxies=self.proxy).json()
				if str(ret).startswith('API calls quota exceeded!') or ('msg' in ret and ret['msg'].startswith('令牌信息已经无效或已经被销毁')):
					continue
				return ret
			except:
				err_time += 1
				print ("Exception: Internet")
				time.sleep(1)
				if err_time > 500: return {}
		
	def get_area(self):
		return self.request_api_url('Area')
		
	def get_champion(self):
		return self.request_api_url('champion')
		
	def get_userarea(self, keyword):
		return self.request_api_url('UserArea?keyword='+keyword)
		
	def get_battlesummaryinfo(self, qquin, area):
		return self.request_api_url('BattleSummaryInfo?qquin='+qquin+'&vaid='+area)
	
	def get_combatlist(self, qquin, area, page):
		return self.request_api_url('CombatList?qquin='+qquin+'+&vaid='+area+'&p='+page) # 0..49
	
	def get_gamedetail(self, qquin, area, gameid):
		return self.request_api_url('GameDetail?qquin='+qquin+'&vaid='+area+'&gameid='+gameid)

class MySQL(object):
	
	host = 'localhost'
	port = 3306
	user = ''
	password = ''
	database = 'lol_prediction'
	charset = 'utf8'
	
	def __init__(self, user, password):
		self.user = user
		self.password = password
		
	def connect(self):
		try:
			conn = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password, db=self.database, charset=self.charset)
			return conn
		except:
			print ('Exception: MySQL Connection')
			time.sleep(1)
			return None
	
	def get_qquin(self, area):
		conn = self.connect()
		if (conn == None): return set()
		cursor = conn.cursor()
		try:
			ifexists = cursor.execute("select qquin from player where area = "+area)		
			if ifexists == 0:
				cursor.close()
				conn.close()
				return set()
			else:
				result_set = cursor.fetchall()
				cursor.close()
				conn.close()
				qquin_set = set()
				for s in result_set:
					qquin_set.add(s[0])
				return qquin_set
		except:
			print ('Exception: Get QQUIN From MySQL')
			cursor.close()
			conn.close()
			return set()
	
	def update_playerinfo(self, area, qquin, playerinfo):
		conn = self.connect()
		if (conn == None): return False
		cursor = conn.cursor()
		try:
			if playerinfo[0] == None:
				print ('Failed: Update '+area+'-'+qquin+' Player Info')
				return False
			ifexists = cursor.execute("select area,qquin from player where area="+area+" and qquin='"+qquin+"'")
			if ifexists == 0:
				cursor.execute("insert into player (area,qquin,aram_game,aram_win,aram_lose,aram_leave) values ("+area+",'"+qquin+"',"+playerinfo[0]+","+playerinfo[1]+","+playerinfo[2]+","+playerinfo[3]+")")
				conn.commit()
				cursor.close()
				conn.close()
				print (area+'-'+qquin+' : New Player')
				return True
			else:
				cursor.execute("update player set aram_game="+playerinfo[0]+", aram_win="+playerinfo[1]+", aram_lose="+playerinfo[2]+", aram_leave="+playerinfo[3]+" where area="+area+" and qquin='"+qquin+"'")
				conn.commit()
				cursor.close()
				conn.close()
				print (area+'-'+qquin+' : Update Player Info')
				return True
		except:
			print ('Exception: Update '+area+'-'+qquin+' Player Info')
			cursor.close()
			conn.close()
			return False
			
	def update_playername(self, area, qquin, name, nametime):
		conn = self.connect()
		if (conn == None): return False
		cursor = conn.cursor()
		try:
			ifexists = cursor.execute("select area,qquin,name,nametime from player where area="+area+" and qquin='"+qquin+"'")
			if ifexists == 0:
				cursor.execute("insert into player (area,qquin,name,nametime) values ("+area+",'"+qquin+"','"+name+"','"+nametime+"')")
				conn.commit()
				cursor.close()
				conn.close()
				print (area+'-'+qquin+' : New Player '+name)
				return True
			else:
				select_result = cursor.fetchall()[0]
				if select_result[3] == None or (str(select_result[2]) != name and str(select_result[3]) < nametime):
					cursor.execute("update player set name='"+name+"', nametime='"+nametime+"' where area="+area+" and qquin='"+qquin+"'")
					conn.commit()
				cursor.close()
				conn.close()
				print (area+'-'+qquin+' : Update Player Name '+name)				
				return True
		except:
			print ('Exception: Update '+area+'-'+qquin+' Player Name')
			cursor.close()
			conn.close()
			return False
			
	def search_game(self, mode, area, gameid):
		conn = self.connect()
		if (conn == None): return -1
		cursor = conn.cursor()
		try:
			ifexists = cursor.execute("select area,gameid from game_"+mode+" where area="+area+" and gameid="+gameid)
			cursor.close()
			conn.close()
			return ifexists
		except:
			print ('Exception: Search '+area+'-'+gameid)
			cursor.close()
			conn.close()
			return -1
			
	def insert_game(self, mode, area, gameid, detail):
		conn = self.connect()
		if (conn == None): return False
		cursor = conn.cursor()
		try:
			if detail == None or detail[2] == None:
				print ('Failed: Insert '+area+'-'+gameid+' Game Detail')
				return False
			cmd = "insert into game_"+mode+" values ("+area+","+gameid+",'"+detail[3]+"',"+str(detail[2])
			for i in range(0, 5):
				cmd += ",'"+detail[0][i][0]+"',"+detail[0][i][2]+","+detail[0][i][4]+","+detail[0][i][5]+","+detail[0][i][6]+","+detail[0][i][7]
			for i in range(0, 5):
				cmd += ",'"+detail[1][i][0]+"',"+detail[1][i][2]+","+detail[1][i][4]+","+detail[1][i][5]+","+detail[1][i][6]+","+detail[1][i][7]
			cmd += ")"
			cursor.execute(cmd)
			conn.commit()
			cursor.close()
			conn.close()
			return True			
		except:
			print ('Exception: Insert '+area+'-'+gameid)
			cursor.close()
			conn.close()
			return False
		
def parse_champion(champion):
	result = {}
	i = 0
	try:
		for c in champion['data']:
			result[c['id']] = (i, c['ename'])
			i += 1
	except:
		print ('Exception: Parse Champion')
	return result

def parse_playerinfo(info):
	try:
		record = [None, None, None, None]
		# game_type 1 for normal
		# game_type 4 for S7 solo-rank
		# game_type 6 for ARAM
		for battleinfo in info['data'][0]['batt_sum_info']:
			if battleinfo['battle_type'] == 6:
				record = [str(battleinfo['total_num']), str(battleinfo['win_num']), str(battleinfo['lose_num']), str(battleinfo['leave_num'])]
				break
		return record
	except:
		print ('Exception: Parse Player Info')	
		return [None]
	
def parse_gamedetail(game_detail):
	try:
		result = [[], [], None, game_detail['data'][0]['battle']['start_time']]
		for r in game_detail['data'][0]['battle']['gamer_records']:
			if r['team'] == 100:
				result[0].append((r['qquin'], r['name'], str(r['champion_id']), r['win'], str(r['champions_killed']), str(r['num_deaths']), str(r['assists']), str(r['total_damage_dealt_to_champions'])))
			elif r['team'] == 200:
				result[1].append((r['qquin'], r['name'], str(r['champion_id']), r['win'], str(r['champions_killed']), str(r['num_deaths']), str(r['assists']), str(r['total_damage_dealt_to_champions'])))
			# r['team'] = 100 for blue side, r['team'] = 200 for red side
			# r['win'] = 2 for lose, r['win'] = 1 for win
		result[2] = 2-result[0][0][3] # 1 for blue side win, 0 for red side win
		return result
	except:
		print ('Exception: Parse Game Detail')
		return None
		
def main_loop(area):
	mysql = MySQL('root', 'pkuoslab')
	qquin_set = mysql.get_qquin(area)
	crawler = Daiwan('Account.txt')
	while len(qquin_set):
		qquin_tuple = list(qquin_set)
		random.shuffle(qquin_tuple)
		qquin_tuple = tuple(qquin_tuple)
		for qquin in qquin_tuple:
			if (os.path.isfile('exit')):
				print ('Process '+area+' Exit')
				return
			mysql.update_playerinfo(area, qquin, parse_playerinfo(crawler.get_battlesummaryinfo(qquin, area)))
			normal_gameid = [] # map = 11, type = 3, mode = 1
			rank_gameid = [] # map = 11, type = 4, mode = 4
			aram_gameid = [] # map = 12, type = 6, mode = 6
			arurf_gameid = [] # map = 11, type = 3, mode = 24
			page = 0
			while True:
				try:
					combat_list = crawler.get_combatlist(qquin, area, str(page))
					if combat_list['data'][0]['list_num'] == 0: break
					expire = False
					for battle in combat_list['data'][0]['battle_list']:
						if battle['battle_time'] < start_time:
							expire = True
							break
						if battle['battle_time'] <= end_time:
							if battle['battle_map'] == 11 and battle['game_type'] == 3 and battle['game_mode'] == 1: normal_gameid.append(str(battle['game_id']))
							elif battle['battle_map'] == 11 and battle['game_type'] == 4 and battle['game_mode'] == 4: rank_gameid.append(str(battle['game_id']))
							elif battle['battle_map'] == 12 and battle['game_type'] == 6 and battle['game_mode'] == 6: aram_gameid.append(str(battle['game_id']))
							elif battle['battle_map'] == 11 and battle['game_type'] == 3 and battle['game_mode'] == 24: arurf_gameid.append(str(battle['game_id']))
					if expire: break
					page += 1
				except:
					print ('Exception: '+area+'-'+qquin+' Get Combat List')
					break
			mode_list = {'normal': normal_gameid, 'rank': rank_gameid, 'aram': aram_gameid, 'arurf': arurf_gameid}
			for gamemode, gameidlist in mode_list.items():
				for gameid in gameidlist:
					if (os.path.isfile('exit')):
						print ('Process '+area+' Exit')
						return
					if mysql.search_game(gamemode, area, gameid) == 0:
						detail = parse_gamedetail(crawler.get_gamedetail(qquin, area, gameid))
						mysql.insert_game(gamemode, area, gameid, detail)
						try:
							for i in range(0, 2):
								for j in range(0, 5):
									qquin_set.add(detail[i][j][0])
									mysql.update_playername(area, detail[i][j][0], detail[i][j][1], detail[3])
						except:
							print ('Exception: '+area+'-'+gameid+' Update Other Player')
							continue
	print ('Process '+area+' Has No qquin(s)')

if __name__ == '__main__':	
	#crawler = Daiwan('Account.txt')
	#champion_dict = parse_champion(crawler.get_champion())
	#with open('champion.json', 'w') as outfile:  
	#	json.dump(champion_dict, outfile, ensure_ascii=False)
	#	outfile.write('\n')
	if os.path.isfile('exit'): os.remove('exit')
	if not os.path.isfile('settings.txt'):
		print ('"settings.txt" File Not Found')
		exit()
	fin = open('settings.txt', 'r')
	data = fin.read()
	param_list = data.split(' ')
	processes = []
	for param in param_list:
		if param == param_list[0]: continue
		processes.append(multiprocessing.Process(target = main_loop, args = (param.replace('\r', '').replace('\n', ''),)))
	for p in processes:
		p.start()
	main_loop(param_list[0].replace('\r', '').replace('\n', ''))
	for p in processes:
		p.join()
	print ("All Processes Exit")
