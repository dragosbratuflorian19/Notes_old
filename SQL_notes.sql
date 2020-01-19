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
  `kids` tinyint(4) NOT NULL,
  `team_prefix` char(3) NOT NULL,
  `fav_team` varchar(50) DEFAULT NULL,
  `Ballon dor` smallint(6) NOT NULL DEFAULT '0',
  `date_of_born` date NOT NULL,
  `market_value` decimal(9,2) NOT NULL,
  PRIMARY KEY (`player_id`)
);
------------------------------------------------------------------------------------------------------------------------
-- Insert into table
INSERT INTO `players` VALUES (1,'Cristiano Ronaldo', 4,'JUV','Manchester United');
INSERT INTO `players` VALUES (2,'Marcus Rashford', 0, 'MUN','Manchester United');
------------------------------------------------------------------------------------------------------------------------
-- Creating a foreign key
  KEY `fk_client_id_idx` (`client_id`),
  KEY `fk_invoice_id_idx` (`invoice_id`),
  CONSTRAINT `fk_payment_client` FOREIGN KEY (`client_id`) REFERENCES `clients` (`client_id`) ON UPDATE CASCADE,
  CONSTRAINT `fk_payment_invoice` FOREIGN KEY (`invoice_id`) REFERENCES `invoices` (`invoice_id`) ON UPDATE CASCADE
------------------------------------------------------------------------------------------------------------------------