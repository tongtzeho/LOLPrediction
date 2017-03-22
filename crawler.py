from selenium import webdriver
import random, requests, json, time, re, multiprocessing, os, datetime

start_time = int(time.mktime(time.strptime("2017-03-07 00:00:00", "%Y-%m-%d %H:%M:%S")))
end_time = int(time.mktime(time.strptime("2017-03-09 00:00:00", "%Y-%m-%d %H:%M:%S")))

class Daiwan(object):
	"""docstring for Daiwan."""

	BASE_URL = 'http://lolapi.games-cube.com/'

	token = ''
	phantomjs_path = 'phantomjs/bin/phantomjs.exe'
	
	account = []
	account_id = 0

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
				return ret
			except:
				print ("Exception: Internet")
				time.sleep(1)
		
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

def parse_gamedetail(game_detail, champion_dict = None):
	try:
		result = [[], [], None, game_detail['data'][0]['battle']['start_time']]
		for r in game_detail['data'][0]['battle']['gamer_records']:
			if r['team'] == 100:
				result[0].append((r['qquin'], r['name'], r['champion_id'], r['win'], r['team'], r['champions_killed'], r['num_deaths'], r['assists'], r['total_damage_dealt_to_champions']))
			elif r['team'] == 200:
				result[1].append((r['qquin'], r['name'], r['champion_id'], r['win'], r['team'], r['champions_killed'], r['num_deaths'], r['assists'], r['total_damage_dealt_to_champions']))
			if champion_dict != None:
				print (str(r['team'])+" "+str(champion_dict[r['champion_id']]))
	except:
		print ('Exception: Parse Game Detail')
		return None
	
def write_playerinfo(area, qquin, info):
	try:
		record = [None, ['0', '0', '0', '0'], None, None, ['0', '0', '0', '0'], None, ['0', '0', '0', '0']]
		# game_type 1 for normal
		# game_type 4 for S7 solo-rank
		# game_type 6 for ARAM
		for battleinfo in info['data'][0]['batt_sum_info']:
			if battleinfo['battle_type'] == 1 or battleinfo['battle_type'] == 4 or battleinfo['battle_type'] == 6:
				record[battleinfo['battle_type']] = [str(battleinfo['total_num']), str(battleinfo['win_num']), str(battleinfo['lose_num']), str(battleinfo['leave_num'])]
		fout = open(area+"/PlayerInfo/"+qquin+".txt", "w")
		fout.write(record[1][0]+' '+record[1][1]+' '+record[1][2]+' '+record[1][3]+"\n")
		fout.write(record[4][0]+' '+record[4][1]+' '+record[4][2]+' '+record[4][3]+"\n")
		fout.write(record[6][0]+' '+record[6][1]+' '+record[6][2]+' '+record[6][3]+"\n")
		fout.close()
		print (area+' '+qquin+' Update Info')
	except:
		print ('Exception: '+area+' '+qquin+' Update Info')
		
def main_loop(area):
	if not os.path.isfile(area+'/qquin.txt'):
		print (area+' QQUIN File Not Found')
		return
	fin = open(area+'/qquin.txt')
	qquin_set = set()
	for line in fin:
		qquin_set.add(line.replace('\r', "").replace('\n', ""))
	crawler = Daiwan('Account.txt')
	while True:
		qquin_tuple = list(qquin_set)
		random.shuffle(qquin_tuple)
		qquin_tuple = tuple(qquin_tuple)
		for qquin in qquin_tuple:
			battle_summary_info = crawler.get_battlesummaryinfo(qquin, area)
			write_playerinfo(area, qquin, battle_summary_info)
			page = 0
			normal_gameid = [] # map = 11, type = 3, mode = 1
			rank_gameid = [] # map = 11, type = 4, mode = 4
			aram_gameid = [] # map = 12, type = 6, mode = 6
			arurf_gameid = [] # map = 11, type = 3, mode = 24
			while True:
				try:
					combat_list = crawler.get_combatlist(qquin, area, str(page))
					if combat_list['data'][0]['list_num'] == 0: break
					page += 1
					expire = False
					for battle in combat_list['data'][0]['battle_list']:
						if int(time.mktime(time.strptime(battle['battle_time'], "%Y-%m-%d %H:%M:%S"))) < start_time:
							expire = True
							break
						if int(time.mktime(time.strptime(battle['battle_time'], "%Y-%m-%d %H:%M:%S"))) <= end_time:
							if battle['battle_map'] == 11 and battle['game_type'] == 3 and battle['game_mode'] == 1: normal_gameid.append(str(battle['game_id']))
							elif battle['battle_map'] == 11 and battle['game_type'] == 4 and battle['game_mode'] == 4: rank_gameid.append(str(battle['game_id']))
							elif battle['battle_map'] == 12 and battle['game_type'] == 6 and battle['game_mode'] == 6: aram_gameid.append(str(battle['game_id']))
							elif battle['battle_map'] == 11 and battle['game_type'] == 3 and battle['game_mode'] == 24: arurf_gameid.append(str(battle['game_id']))
					if expire: break
				except:
					print ('Exception: '+area+' '+qquin+' Get Combat List')
					break
			print (normal_gameid)
			print (rank_gameid)
			print (aram_gameid)
			print (arurf_gameid)
			
			# 查询每页战绩
				# 对每场比赛，如果模式符合，看是否已存在，如果不存在，查询比赛具体
					# 把同场比赛的不在set里的玩家加进set且查询战绩
			# 更新本地qquin.txt
		return

if True:
	crawler = Daiwan('Account.txt')
	area = '20'
	qquin55k = 'U15681153143084694807'
	qquintzh = 'U9426748735989830127'
	qquinllg = 'U6894554802621839420'
	gameidllga20 = '1282678561'
	#battle_summary_info = crawler.get_battlesummaryinfo(qquintzh, area)
	#write_playerinfo(area, qquintzh, battle_summary_info)
	#print (crawler.get_combatlist(qquinllg, area, '0'))
	#print (crawler.get_combatlist(qquinllg, area, '1'))
	#print (crawler.get_gamedetail(qquinllg, area, gameidllga20))
	#champion_dict = parse_champion(crawler.get_champion())
	#with open('champion.json', 'w') as outfile:  
	#	json.dump(champion_dict, outfile, ensure_ascii=False)  
	#	outfile.write('\n')
	parse_gamedetail(crawler.get_gamedetail(qquinllg, area, gameidllga20), parse_champion(crawler.get_champion()))
	exit()

if __name__ == '__main__':	
	main_loop('20')
