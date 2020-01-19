------------------------------------------------------------------------------------------------------------------------
-- Drop a database if it already exists
DROP DATABASE IF EXISTS `football_players`;
------------------------------------------------------------------------------------------------------------------------
-- Create a database
CREATE DATABASE `football_players`;
------------------------------------------------------------------------------------------------------------------------
-- Use a database
USE `football_players`;
------------------------------------------------------------------------------------------------------------------------
-- Create a table
-- Use data types: tiny int, varchar,
-- Use NOT NULL
-- Use AUTO_INCREMENT
-- Use PRIMARY KEY
-- Use DEFAULT value
CREATE TABLE `players` (
  `player_id` int(11) NOT NULL AUTO_INCREMENT,
  `player_name` varchar(50) NOT NULL,
  `ucl` tinyint(4) NOT NULL,
  `team_prefix` char(3) NOT NULL,
  `fav_team` varchar(50) DEFAULT NULL,
  `Ballon dor` smallint(6) NOT NULL DEFAULT '0',
  `date_of_born` date NOT NULL,
  `market_value` decimal(9,2) NOT NULL,
  PRIMARY KEY (`player_id`)
);
------------------------------------------------------------------------------------------------------------------------
-- Insert into table
INSERT INTO `players` VALUES (1,'Cristiano Ronaldo', 4,'JUV','Manchester United', 5, '1985-02-25', 150.50);
INSERT INTO `players` VALUES (2,'Marcus Rashford', 0, 'MUN','Manchester United', 0, '1985-05-12', 120.50);
------------------------------------------------------------------------------------------------------------------------
-- Creating a foreign key
  KEY `fk_client_id_idx` (`client_id`),
  KEY `fk_invoice_id_idx` (`invoice_id`),
  CONSTRAINT `fk_payment_client` FOREIGN KEY (`client_id`) REFERENCES `clients` (`client_id`) ON UPDATE CASCADE,
  CONSTRAINT `fk_payment_invoice` FOREIGN KEY (`invoice_id`) REFERENCES `invoices` (`invoice_id`) ON UPDATE CASCADE
------------------------------------------------------------------------------------------------------------------------
-- SELECT statement
SELECT player_id, player_name
FROM players
WHERE player_id = 1
ORDER BY player_name ASC
SELECT
    first,
    last,
    pay * 10 + 9
SELECT first, last, pay, pay / 10 + 9 AS 'bonus' FROM employees WHERE first = 'Dragos'
SELECT DISTINCT first FROM employees
SELECT * FROM employees WHERE dob = '1990-01-01'
SELECT * FROM employees WHERE first = 'Alex' AND NOT  pay > 8000
AND before OR
SELECT * FROM employees WHERE first = 'Alex' OR first = 'Dragos' OR first = 'Elena'
SELECT * FROM employees WHERE first IN ('Raluca', 'Dragos', 'Elena')
SELECT * FROM employees WHERE age BETWEEN 30 AND 35
SELECT * FROM employees WHERE first LIKE '%s' # Dragos, Marius, Narcis
SELECT * FROM employees WHERE first NOT LIKE '%a%' # Ion
SELECT * FROM employees WHERE first LIKE 'D____s' # Dragos
% - any number of characters
_ - a single character
def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None

conn.create_function("REGEXP", 2, regexp)
c.execute("""SELECT * FROM employees WHERE first REGEXP ?""", [r'tei'])
REGEXP:  ^ beginning: r'^Dra' $ end: r'gos$' | logical or: r'(Dragos|Marius)' r'[aty]z' r'[a-z]l'
SELECT * FROM employees WHERE first IS NULL
SELECT * FROM employees ORDER BY pay DESC, age ASC
SELECT first, last, 10 * id AS points FROM employees ORDER BY first
SELECT *, pay / age AS rate FROM employees ORDER BY rate DESC
SELECT * FROM employees LIMIT 5
SELECT * FROM employees LIMIT 5, 5
SELECT *      FROM employees e     JOIN team_supported t ON e.id = t.emp_id
SELECT p.name AS angajat, pd.name AS manager     FROM people p     JOIN people pd ON p.id = pd.reports_to
UPDATE people SET first_name = 'Dragos' WHERE id IN (SELECT ppl_id              FROM teams              WHERE team = 'MANCHESTER UNITED')
CREATE TABLE teams_copy AS SELECT * FROM teams
INSERT INTO teams_copy SELECT * FROM teams WHERE team = 'MANCHESTER UNITED'
INSERT INTO teams (name) VALUES  ('STEAUA'), ('RAPID'), ('DINAMO')
1 a 1 b 1 c 2 a 2 b 2 c 3 a 3 b 3 c
SELECT first_name, last_name, team, 'Smecher' as status FROM people JOIN teams ON ppl_id = id WHERE team REGEXP 'MANCHESTER UNITED' UNION SELECT first_name, last_name, team, 'Fraier' as status FROM people JOIN teams ON ppl_id = id WHERE team REGEXP 'LIVERPOOL'
SELECT first_name, last_name, team, duration FROM people P JOIN teams t     ON ppl_id = id JOIN years_of_supporting     USING (ppl_id) WHERE first_name = 'Dragos' AND team REGEXP 'MANCHESTER UNITED'
SELECT first_name, last_name, team, duration FROM teams NATURAL JOIN years_of_supporting JOIN people ON ppl_id = id
SELECT * FROM people p JOIN teams t      ON p.id = t.id sau USING (id)
SELECT * FROM people p LEFT JOIN teams t ON p.id = t.ppl_id
SELECT * FROM people p JOIN teams t ON p.id = t.ppl_id JOIN trophees tr ON tr.team = t.team
