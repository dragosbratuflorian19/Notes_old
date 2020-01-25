------------------------------------------------------------------------------------------------------------------------
-- General information
C.R.U.D. : Create.Read.Update.Delete
Query: A google search is a Query
-- Data Types
INT - Whole Number
DECIMAL(M,N) - Decimal number : M - total number of digits, N - number of digits after the decimal point
VARCHAR(l) - String of text of length l
BLOB - binary large object; it stores large data (images, files)
DATE - date of format: YYYY-MM-DD
TIMESTAMP - of format YYYY-MM-DD HH:MM:SS - used for recordings
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
  `player_id` int(11) AUTO_INCREMENT,
  `player_name` varchar(50) NOT NULL,
  `ucl` tinyint(4) NOT NULL,
  `team_prefix` char(3) NOT NULL,
  `fav_team` varchar(50) DEFAULT NULL,
  `Ballon dor` smallint(6) NOT NULL DEFAULT '0',
  `date_of_born` date NOT NULL,
  `market_value` decimal(9,2) UNIQUE ,
  PRIMARY KEY (`player_id`)
);
------------------------------------------------------------------------------------------------------------------------
-- Delete a table
DROP TABLE players;
------------------------------------------------------------------------------------------------------------------------
-- Add/Delete a column
ALTER TABLE players ADD form INT(5);
ALTER TABLE players DROP COLUMN form;
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
-- Update some values
UPDATE players
SET team = 'Manchester United', fav_team = 'ManU'
WHERE team = 'Manchester Utd' OR team = 'Man Utd';
------------------------------------------------------------------------------------------------------------------------
-- DELETE a column
DELETE FROM players
WHERE player_id = 2
------------------------------------------------------------------------------------------------------------------------
-- SELECT statement
-- WHERE statement
-- ORDER BY statement
SELECT player_id, player_name
FROM players
WHERE player_id = 1
ORDER BY player_name ASC
------------------------------------------------------------------------------------------------------------------------
-- NEW column as ALIAS
SELECT first, last, pay, pay / 10 + 9 AS 'bonus'
FROM employees
WHERE first = 'Dragos'
------------------------------------------------------------------------------------------------------------------------
-- DISTINCT
SELECT DISTINCT first
FROM employees
------------------------------------------------------------------------------------------------------------------------
-- Multiple filtering (AND before OR)
SELECT *
FROM employees
WHERE first = 'Alex' AND NOT  pay > 8000
------------------------------------------------------------------------------------------------------------------------
-- IN
SELECT *
FROM employees
WHERE first IN ('Raluca', 'Dragos', 'Elena')
------------------------------------------------------------------------------------------------------------------------
-- BETWEEN
SELECT *
FROM employees
WHERE age BETWEEN 30 AND 35
------------------------------------------------------------------------------------------------------------------------
-- LIKE
SELECT *
FROM employees
WHERE first LIKE '%s' -- Dragos, Marius, Narcis
WHERE first NOT LIKE '%a%' -- Ion
WHERE first LIKE 'D____s' -- Dragos
------------------------------------------------------------------------------------------------------------------------
-- REGEX
-- % - any number of characters
-- _ - a single character
------------------------------------------------------------------------------------------------------------------------
-- MULTIPLE ORDERING
SELECT *
FROM employees
ORDER BY pay DESC, age ASC
------------------------------------------------------------------------------------------------------------------------
-- FILTERING
SELECT *, pay / age AS rate
FROM employees
ORDER BY rate DESC
------------------------------------------------------------------------------------------------------------------------
-- LIMIT
SELECT * FROM employees LIMIT 5
------------------------------------------------------------------------------------------------------------------------
-- Skip first n rows
SELECT *
FROM employees LIMIT 5, 5
------------------------------------------------------------------------------------------------------------------------
-- JOIN
SELECT *
FROM employees e
JOIN team_supported t ON e.id = t.emp_id
------------------------------------------------------------------------------------------------------------------------
-- SELF JOIN
SELECT p.name AS angajat, pd.name AS manager
FROM people p
JOIN people pd ON p.id = pd.reports_to
------------------------------------------------------------------------------------------------------------------------
-- SUBQUERIES
UPDATE people
SET first_name = 'Dragos'
WHERE id IN
    (   SELECT ppl_id
        FROM teams
        WHERE team = 'MANCHESTER UNITED'
    )
------------------------------------------------------------------------------------------------------------------------
-- TABLE COPY
CREATE TABLE teams_copy AS
SELECT * FROM teams
------------------------------------------------------------------------------------------------------------------------
-- TABLE COPY
INSERT INTO teams_copy
SELECT *
FROM teams
WHERE team = 'MANCHESTER UNITED'
------------------------------------------------------------------------------------------------------------------------
-- MULTIPLE INSERT
INSERT INTO teams (name)
VALUES
('STEAUA'),
('RAPID'),
('DINAMO')
------------------------------------------------------------------------------------------------------------------------
-- UNION
SELECT first_name, last_name, team, 'Smecher' as status
FROM people
JOIN teams ON ppl_id = id
WHERE team LIKE 'MANCHESTER UNITED'
UNION
SELECT first_name, last_name, team, 'Fraier' as status
FROM people
JOIN teams ON ppl_id = id
WHERE team LIKE 'LIVERPOOL'
------------------------------------------------------------------------------------------------------------------------
-- MULTIPLE JOINS
SELECT first_name, last_name, team, duration
FROM people P
JOIN teams t ON ppl_id = id
JOIN years_of_supporting USING (ppl_id)
WHERE first_name = 'Dragos' AND team LIKE 'MANCHESTER UNITED'
------------------------------------------------------------------------------------------------------------------------
-- NATURAL JOIN
SELECT first_name, last_name, team, duration
FROM teams
NATURAL JOIN years_of_supporting
JOIN people ON ppl_id = id
------------------------------------------------------------------------------------------------------------------------
-- USING
SELECT *
FROM people p
JOIN teams t USING (id)
------------------------------------------------------------------------------------------------------------------------
-- OUTER JOIN
SELECT * FROM people p
LEFT JOIN teams t ON p.id = t.ppl_id
------------------------------------------------------------------------------------------------------------------------
-- JOIN E TABLES
SELECT *
FROM people p
JOIN teams t ON p.id = t.ppl_id
JOIN trophees tr ON tr.team = t.team
