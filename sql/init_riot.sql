CREATE TABLE player_riot (
	region CHAR(6) NOT NULL,
	summonerid BIGINT NOT NULL,
	PRIMARY KEY (region, summonerid)
);

CREATE TABLE game_riot (
	region CHAR(6) NOT NULL,
	gameid BIGINT NOT NULL,
	mapid TINYINT NOT NULL,
	gametype VARCHAR(40) NOT NULL,
	subtype VARCHAR(40) NOT NULL,
	gamemode VARCHAR(40) NOT NULL,
	bluesidechampion VARCHAR(40) NOT NULL,
	redsidechampion VARCHAR(40) NOT NULL,
	bluesidewin TINYINT NOT NULL,
	createdate BIGINT NOT NULL,
	PRIMARY KEY (region, gameid)
);

CREATE INDEX index_type ON game_riot(mapid, gametype, subtype, gamemode, createdate);

INSERT INTO player_riot VALUES ('NA', 58597238);
INSERT INTO player_riot VALUES ('NA', 26564459);

INSERT INTO player_riot VALUES ('EUW', 34463698);
INSERT INTO player_riot VALUES ('EUW', 33787513);

INSERT INTO player_riot VALUES ('KR', 54970691);
INSERT INTO player_riot VALUES ('KR', 3801955);
