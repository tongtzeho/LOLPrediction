CREATE TABLE Player (
	area TINYINT NOT NULL,
	qquin VARCHAR(28) NOT NULL,
	name VARCHAR(40),
	nametime VARCHAR(21),
	aram_game INTEGER,
	aram_win INTEGER,
	aram_lose INTEGER,
	aram_leave INTEGER,
	PRIMARY KEY (area, qquin)
);

CREATE TABLE Game_Rank (
	area TINYINT NOT NULL,
	gameid BIGINT NOT NULL,
	starttime VARCHAR(21) NOT NULL,
	bluesidewin TINYINT NOT NULL,
	qquin_b0 VARCHAR(28) NOT NULL,
	champion_b0 SMALLINT NOT NULL,
	kill_b0 SMALLINT NOT NULL,
	death_b0 SMALLINT NOT NULL,
	assist_b0 SMALLINT NOT NULL,
	damage_b0 INT NOT NULL,
	qquin_b1 VARCHAR(28) NOT NULL,
	champion_b1 SMALLINT NOT NULL,
	kill_b1 SMALLINT NOT NULL,
	death_b1 SMALLINT NOT NULL,
	assist_b1 SMALLINT NOT NULL,
	damage_b1 INT NOT NULL,
	qquin_b2 VARCHAR(28) NOT NULL,
	champion_b2 SMALLINT NOT NULL,
	kill_b2 SMALLINT NOT NULL,
	death_b2 SMALLINT NOT NULL,
	assist_b2 SMALLINT NOT NULL,
	damage_b2 INT NOT NULL,
	qquin_b3 VARCHAR(28) NOT NULL,
	champion_b3 SMALLINT NOT NULL,
	kill_b3 SMALLINT NOT NULL,
	death_b3 SMALLINT NOT NULL,
	assist_b3 SMALLINT NOT NULL,
	damage_b3 INT NOT NULL,
	qquin_b4 VARCHAR(28) NOT NULL,
	champion_b4 SMALLINT NOT NULL,
	kill_b4 SMALLINT NOT NULL,
	death_b4 SMALLINT NOT NULL,
	assist_b4 SMALLINT NOT NULL,
	damage_b4 INT NOT NULL,
	qquin_r0 VARCHAR(28) NOT NULL,
	champion_r0 SMALLINT NOT NULL,
	kill_r0 SMALLINT NOT NULL,
	death_r0 SMALLINT NOT NULL,
	assist_r0 SMALLINT NOT NULL,
	damage_r0 INT NOT NULL,
	qquin_r1 VARCHAR(28) NOT NULL,
	champion_r1 SMALLINT NOT NULL,
	kill_r1 SMALLINT NOT NULL,
	death_r1 SMALLINT NOT NULL,
	assist_r1 SMALLINT NOT NULL,
	damage_r1 INT NOT NULL,
	qquin_r2 VARCHAR(28) NOT NULL,
	champion_r2 SMALLINT NOT NULL,
	kill_r2 SMALLINT NOT NULL,
	death_r2 SMALLINT NOT NULL,
	assist_r2 SMALLINT NOT NULL,
	damage_r2 INT NOT NULL,
	qquin_r3 VARCHAR(28) NOT NULL,
	champion_r3 SMALLINT NOT NULL,
	kill_r3 SMALLINT NOT NULL,
	death_r3 SMALLINT NOT NULL,
	assist_r3 SMALLINT NOT NULL,
	damage_r3 INT NOT NULL,
	qquin_r4 VARCHAR(28) NOT NULL,
	champion_r4 SMALLINT NOT NULL,
	kill_r4 SMALLINT NOT NULL,
	death_r4 SMALLINT NOT NULL,
	assist_r4 SMALLINT NOT NULL,
	damage_r4 INT NOT NULL,
	PRIMARY KEY (area, gameid)
);

CREATE TABLE Game_Normal (
	area TINYINT NOT NULL,
	gameid BIGINT NOT NULL,
	starttime VARCHAR(21) NOT NULL,
	bluesidewin TINYINT NOT NULL,
	qquin_b0 VARCHAR(28) NOT NULL,
	champion_b0 SMALLINT NOT NULL,
	kill_b0 SMALLINT NOT NULL,
	death_b0 SMALLINT NOT NULL,
	assist_b0 SMALLINT NOT NULL,
	damage_b0 INT NOT NULL,
	qquin_b1 VARCHAR(28) NOT NULL,
	champion_b1 SMALLINT NOT NULL,
	kill_b1 SMALLINT NOT NULL,
	death_b1 SMALLINT NOT NULL,
	assist_b1 SMALLINT NOT NULL,
	damage_b1 INT NOT NULL,
	qquin_b2 VARCHAR(28) NOT NULL,
	champion_b2 SMALLINT NOT NULL,
	kill_b2 SMALLINT NOT NULL,
	death_b2 SMALLINT NOT NULL,
	assist_b2 SMALLINT NOT NULL,
	damage_b2 INT NOT NULL,
	qquin_b3 VARCHAR(28) NOT NULL,
	champion_b3 SMALLINT NOT NULL,
	kill_b3 SMALLINT NOT NULL,
	death_b3 SMALLINT NOT NULL,
	assist_b3 SMALLINT NOT NULL,
	damage_b3 INT NOT NULL,
	qquin_b4 VARCHAR(28) NOT NULL,
	champion_b4 SMALLINT NOT NULL,
	kill_b4 SMALLINT NOT NULL,
	death_b4 SMALLINT NOT NULL,
	assist_b4 SMALLINT NOT NULL,
	damage_b4 INT NOT NULL,
	qquin_r0 VARCHAR(28) NOT NULL,
	champion_r0 SMALLINT NOT NULL,
	kill_r0 SMALLINT NOT NULL,
	death_r0 SMALLINT NOT NULL,
	assist_r0 SMALLINT NOT NULL,
	damage_r0 INT NOT NULL,
	qquin_r1 VARCHAR(28) NOT NULL,
	champion_r1 SMALLINT NOT NULL,
	kill_r1 SMALLINT NOT NULL,
	death_r1 SMALLINT NOT NULL,
	assist_r1 SMALLINT NOT NULL,
	damage_r1 INT NOT NULL,
	qquin_r2 VARCHAR(28) NOT NULL,
	champion_r2 SMALLINT NOT NULL,
	kill_r2 SMALLINT NOT NULL,
	death_r2 SMALLINT NOT NULL,
	assist_r2 SMALLINT NOT NULL,
	damage_r2 INT NOT NULL,
	qquin_r3 VARCHAR(28) NOT NULL,
	champion_r3 SMALLINT NOT NULL,
	kill_r3 SMALLINT NOT NULL,
	death_r3 SMALLINT NOT NULL,
	assist_r3 SMALLINT NOT NULL,
	damage_r3 INT NOT NULL,
	qquin_r4 VARCHAR(28) NOT NULL,
	champion_r4 SMALLINT NOT NULL,
	kill_r4 SMALLINT NOT NULL,
	death_r4 SMALLINT NOT NULL,
	assist_r4 SMALLINT NOT NULL,
	damage_r4 INT NOT NULL,
	PRIMARY KEY (area, gameid)
);

CREATE TABLE Game_ARAM (
	area TINYINT NOT NULL,
	gameid BIGINT NOT NULL,
	starttime VARCHAR(21) NOT NULL,
	bluesidewin TINYINT NOT NULL,
	qquin_b0 VARCHAR(28) NOT NULL,
	champion_b0 SMALLINT NOT NULL,
	kill_b0 SMALLINT NOT NULL,
	death_b0 SMALLINT NOT NULL,
	assist_b0 SMALLINT NOT NULL,
	damage_b0 INT NOT NULL,
	qquin_b1 VARCHAR(28) NOT NULL,
	champion_b1 SMALLINT NOT NULL,
	kill_b1 SMALLINT NOT NULL,
	death_b1 SMALLINT NOT NULL,
	assist_b1 SMALLINT NOT NULL,
	damage_b1 INT NOT NULL,
	qquin_b2 VARCHAR(28) NOT NULL,
	champion_b2 SMALLINT NOT NULL,
	kill_b2 SMALLINT NOT NULL,
	death_b2 SMALLINT NOT NULL,
	assist_b2 SMALLINT NOT NULL,
	damage_b2 INT NOT NULL,
	qquin_b3 VARCHAR(28) NOT NULL,
	champion_b3 SMALLINT NOT NULL,
	kill_b3 SMALLINT NOT NULL,
	death_b3 SMALLINT NOT NULL,
	assist_b3 SMALLINT NOT NULL,
	damage_b3 INT NOT NULL,
	qquin_b4 VARCHAR(28) NOT NULL,
	champion_b4 SMALLINT NOT NULL,
	kill_b4 SMALLINT NOT NULL,
	death_b4 SMALLINT NOT NULL,
	assist_b4 SMALLINT NOT NULL,
	damage_b4 INT NOT NULL,
	qquin_r0 VARCHAR(28) NOT NULL,
	champion_r0 SMALLINT NOT NULL,
	kill_r0 SMALLINT NOT NULL,
	death_r0 SMALLINT NOT NULL,
	assist_r0 SMALLINT NOT NULL,
	damage_r0 INT NOT NULL,
	qquin_r1 VARCHAR(28) NOT NULL,
	champion_r1 SMALLINT NOT NULL,
	kill_r1 SMALLINT NOT NULL,
	death_r1 SMALLINT NOT NULL,
	assist_r1 SMALLINT NOT NULL,
	damage_r1 INT NOT NULL,
	qquin_r2 VARCHAR(28) NOT NULL,
	champion_r2 SMALLINT NOT NULL,
	kill_r2 SMALLINT NOT NULL,
	death_r2 SMALLINT NOT NULL,
	assist_r2 SMALLINT NOT NULL,
	damage_r2 INT NOT NULL,
	qquin_r3 VARCHAR(28) NOT NULL,
	champion_r3 SMALLINT NOT NULL,
	kill_r3 SMALLINT NOT NULL,
	death_r3 SMALLINT NOT NULL,
	assist_r3 SMALLINT NOT NULL,
	damage_r3 INT NOT NULL,
	qquin_r4 VARCHAR(28) NOT NULL,
	champion_r4 SMALLINT NOT NULL,
	kill_r4 SMALLINT NOT NULL,
	death_r4 SMALLINT NOT NULL,
	assist_r4 SMALLINT NOT NULL,
	damage_r4 INT NOT NULL,
	PRIMARY KEY (area, gameid)
);

CREATE TABLE Game_ARURF (
	area TINYINT NOT NULL,
	gameid BIGINT NOT NULL,
	starttime VARCHAR(21) NOT NULL,
	bluesidewin TINYINT NOT NULL,
	qquin_b0 VARCHAR(28) NOT NULL,
	champion_b0 SMALLINT NOT NULL,
	kill_b0 SMALLINT NOT NULL,
	death_b0 SMALLINT NOT NULL,
	assist_b0 SMALLINT NOT NULL,
	damage_b0 INT NOT NULL,
	qquin_b1 VARCHAR(28) NOT NULL,
	champion_b1 SMALLINT NOT NULL,
	kill_b1 SMALLINT NOT NULL,
	death_b1 SMALLINT NOT NULL,
	assist_b1 SMALLINT NOT NULL,
	damage_b1 INT NOT NULL,
	qquin_b2 VARCHAR(28) NOT NULL,
	champion_b2 SMALLINT NOT NULL,
	kill_b2 SMALLINT NOT NULL,
	death_b2 SMALLINT NOT NULL,
	assist_b2 SMALLINT NOT NULL,
	damage_b2 INT NOT NULL,
	qquin_b3 VARCHAR(28) NOT NULL,
	champion_b3 SMALLINT NOT NULL,
	kill_b3 SMALLINT NOT NULL,
	death_b3 SMALLINT NOT NULL,
	assist_b3 SMALLINT NOT NULL,
	damage_b3 INT NOT NULL,
	qquin_b4 VARCHAR(28) NOT NULL,
	champion_b4 SMALLINT NOT NULL,
	kill_b4 SMALLINT NOT NULL,
	death_b4 SMALLINT NOT NULL,
	assist_b4 SMALLINT NOT NULL,
	damage_b4 INT NOT NULL,
	qquin_r0 VARCHAR(28) NOT NULL,
	champion_r0 SMALLINT NOT NULL,
	kill_r0 SMALLINT NOT NULL,
	death_r0 SMALLINT NOT NULL,
	assist_r0 SMALLINT NOT NULL,
	damage_r0 INT NOT NULL,
	qquin_r1 VARCHAR(28) NOT NULL,
	champion_r1 SMALLINT NOT NULL,
	kill_r1 SMALLINT NOT NULL,
	death_r1 SMALLINT NOT NULL,
	assist_r1 SMALLINT NOT NULL,
	damage_r1 INT NOT NULL,
	qquin_r2 VARCHAR(28) NOT NULL,
	champion_r2 SMALLINT NOT NULL,
	kill_r2 SMALLINT NOT NULL,
	death_r2 SMALLINT NOT NULL,
	assist_r2 SMALLINT NOT NULL,
	damage_r2 INT NOT NULL,
	qquin_r3 VARCHAR(28) NOT NULL,
	champion_r3 SMALLINT NOT NULL,
	kill_r3 SMALLINT NOT NULL,
	death_r3 SMALLINT NOT NULL,
	assist_r3 SMALLINT NOT NULL,
	damage_r3 INT NOT NULL,
	qquin_r4 VARCHAR(28) NOT NULL,
	champion_r4 SMALLINT NOT NULL,
	kill_r4 SMALLINT NOT NULL,
	death_r4 SMALLINT NOT NULL,
	assist_r4 SMALLINT NOT NULL,
	damage_r4 INT NOT NULL,
	PRIMARY KEY (area, gameid)
);