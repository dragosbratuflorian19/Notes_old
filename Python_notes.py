##################################################################################
# Variable arguments
from sys import argv
script, first, second, third = argv
print('{first} & {second} & {third}')
# When called: python hello.py Wayne Fucking Rooney
##################################################################################
# Dictionary .get method
teams = {'UTD': 'United', 'CHE': 'Chelsea',
         'ARS': 'Arsenal', 'LIV': 'Liverpool'}
team = teams.get('BAR')
print(team)  # None
team = teams.get('BAR', 'Does Not Exist!!!')
print(team)  # Does Not Exist!!!
##################################################################################
# Classes Inheritance
class Patruped():
  def mers(self):
    print("merge in 4 membre")

class Caine(Patruped):
  def __init__(self, name):
    self.name = name
  def latra(self):
    print("{} face ham ham".format(self.name))

class Pisica(Patruped):
  def __init__(self, name):
    self.name = name
  def miaun(self):
    print("{} face miaun miaun".format(self.name))

oscar = Caine(name='Oscar')
oscar.latra()  # Oscar face ham ham
codita = Pisica(name='Codita')
codita.miaun()  # Codita face miaun miaun
##################################################################################
# Path file handling
from pathlib import Path

path = Path("Jucatori")
path.mkdir()  # creeaza folderul jucatori
path.rmdir()  # Sterge folderul jucatori
from pathlib import Path

path = Path()
for file in path.glob('*.*'):
  path.glob('*')
  path.glob('*.xls')
  print(file)
##################################################################################
# Classes super method
class Parent(object):
  def override(self):
    print("PARENT override()")
  def implicit(self):
    print("PARENT implicit()")
  def altered(self):
    print("PARENT altered()")

class Child(Parent):
  def override(self):
    print("CHILD override()")
  def altered(self):
    print("CHILD, BEFORE PARENT altered()")
    super(Child, self).altered()
    print("CHILD, AFTER PARENT altered()")

dad = Parent()
son = Child()
dad.implicit()  # PARENT implicit()
son.implicit()  # PARENT implicit()
dad.override()  # PARENT override()
son.override()  # CHILD override()
dad.altered()  	# PARENT altered()
son.altered()   # CHILD, BEFORE PARENT altered()
				# PARENT altered()
				# CHILD, AFTER PARENT altered()
##################################################################################
# Formatted print
print('{0} {1} {0}'.format(True, False))  # True False True
print('{0:04} {1:.5f}'.format(5, 5))  # 0005 5.00000
print('{0:>4} {1:>32}'.format('*', '#'))  # *                                #
for i in range(16):
	print('{0:3} {1:16}'.format(i, 10**i))
# Print a size x size multiplication table
##################################################################################
# Table trick
size = int(input("Please enter the table size: "))
for row in range(1, size + 1):
  for column in range(1, size + 1):
    product = row * column  # Compute product
    print('{0:4}'.format(product), end='')  # Display product
  print()
##################################################################################
# Classical print format
print("I will inject %s here and %d here" %
      ('text', 34))  # I will inject text here and 34 here
##################################################################################
# Args and Kwargs
def myfunc(*args, **kwargs):
  print(f"I would like {args[1]} {kwargs['car']}.")


myfunc(10, 20, 30, food='Pizzas', car='Mercedeses') # I would like 20 Mercedeses.
##################################################################################
# Map, filter and lambda
def printare(nume):
  return f'{nume} e BO$$'
def patrat(numar):
  return numar ** 2
def check_nume(numele):
  return 'elena' in numele.lower()

nume_fete = ['Elena Popescu', 'Ivan Elena', 'Simona Urs', 'Raluca Elena']
numere = [1, 2, 3, 4, 5]
catalog = ['Costel', 'Nicu', 'Gigel']
lista = list(map(printare, catalog))
print(lista)  # ['Costel e BO$$', 'Nicu e BO$$', 'Gigel e BO$$']
for item in map(patrat, numere):
  print(item, end=" ")  # 1 4 9 16 25
lista3 = list(filter(check_nume, nume_fete))
print('\n', lista3)  # ['Elena Popescu', 'Ivan Elena', 'Raluca Elena']
numere = [2, 5, 7, 8, 4]
lista = list(map(lambda num: num ** 2, numere))
print(lista)  # [4, 25, 49, 64, 16]
titulaturi = ['Raluca', 'Elena', 'Bianca', 'Casandra']
lista2 = list(filter(lambda nume: 'ca' in nume.lower(), titulaturi))
print(lista2)  # ['Raluca', 'Bianca', 'Casandra']
##################################################################################
# Polymorphism
class Dog():
  def __init__(self, name):
    self.name = name
  def speak(self):
    return self.name + ' says wof'

class Cat():
  def __init__(self, name):
    self.name = name
  def speak(self):
    return self.name + ' says meow'

niko = Dog('niko')
felix = Cat('felix')
print(niko.speak())
print(felix.speak())

for pet in [niko, felix]:
  print(pet.speak())
def speaking_function(pet):
  print(pet.speak())

print(speaking_function(niko))
print(speaking_function(felix))
##################################################################################
# Magic functions 
class Book:
  def __init__(self, titlu, autor, pagini):
    self.titlu = titlu
    self.autor = autor
    self.pagini = pagini
  def __str__(self):
    return f'{self.titlu}, scrisa de {self.autor}'
  def __len__(self):
    return self.pagini
  def __del__(self):
    print('O carte a fost stearsa')

carte = Book("Cool", 'BRATU', 100)
print(carte)
print(len(carte))
del carte
##################################################################################
# __name__ , '__main__'
def funct():
  print('FUNC() IN ONE.PY')


print('TOP LEVEL IN ONE.PY')

if __name__ == '__main__':
  print('ONE.PY is being run directly')
else:
  print('ONE.PY has been imported')

# two.py
import one

print('TOP LEVEL IN TWO.PY')

one.funct()

if __name__ == '__main__':
  print('TWO.PY is run directly')
else:
  print('Two.py has been imported')
##################################################################################
# Unittest

# def cap_text(text):
#     return text.title()
import unittest
import cap

class TestCap(unittest.TestCase):
  def test_one_word(self):
    text = 'python'
    result = cap.cap_text(text)
    self.assertEqual(result, 'Python')

  def test_multiple_words(self):
    text = 'molty python'
    result = cap.cap_text(text)
    self.assertEqual(result, 'Molty Python')

if __name__ == '__main__':
  unittest.main()
# def cap_adu(*args):
#    return sum(args)

import unittest
import cap


class TestAdunare(unittest.TestCase):
  def test_two_nums(self):
    a = 4
    b = 6
    result = cap.cap_adu(a, b)
    self.assertEqual(result, 10)

  def test_three_nums(self):
    a = 5
    b = 60
    c = 4
    result = cap.cap_adu(a, b, c)
    self.assertEqual(result, 69)


if __name__ == '__main__':
  unittest.main()

##################################################################################
# Funcsaption
def hello(name='Dragos'):
  print('The hello() func has been executed')

  def greet():
    return '\t This is a greet funct inside hello!'
  def welcome():
    return '\t this is inside hello'
  print('I am going to return a funct')
  if name == 'Dragos':
    return greet
  else:
    return welcome


my_new_funct = hello('Dragos')
print(my_new_funct())


def hello():
  return 'Hi Dragos!'


def other(some_def_func):
  print('Other code runs here!')
  print(some_def_func())


other(hello)
##################################################################################
# Decorators
def new_decorator(original_func):
  def wrap_func():
    print('some extra code, before the original function')
    original_func()
    print('Some extra code, after the original func')
  return wrap_func

def func_needs_decorator():
  print('I want to be decorated!')

@new_decorator
def func_needs_decorator():
  print('I want to be decorated!')

func_needs_decorator()

def inmultire_impartire(original_func):
  def wrap_func(a, b):
    print(f'produsul a 2 numere este{a * b}')
    original_func(a, b)
    print(f'catul a 2 numere este{a / b}')
  return wrap_func

@inmultire_impartire
def operatii_matematice(a, b):
  print(f'suma a 2 numere este{a + b}')

operatii_matematice(3, 5)
##################################################################################
# Generators
def create_cubes(n):
  for x in range(n):
    yield x**3

generatorul = create_cubes(9)
copie_gen = create_cubes(9)
lista = list(generatorul)
print(type(generatorul))  # <class 'generator'>
print(lista)  # [0, 1, 8, 27, 64, 125, 216, 343, 512]
print(next(copie_gen))  # 0
print(next(copie_gen))  # 1
print(next(copie_gen))  # 8
##################################################################################
# Iterators
s = 'Hello'
s_iter = iter(s)
print(next(s_iter))  # H
print(next(s_iter))  # e
##################################################################################
# Collections: Counter
from collections import Counter

lista = [1, 2, 3, 1, 2, 2, 2, 3, 1, 2, 8]
sir = 'asdadfafasdasdafsdasfdvn'
text = 'Ana are mere si pere si mere da mere nu pere poate mere'
cuvinte = text.split()
print(Counter(lista))  # Counter({2: 5, 1: 3, 3: 2, 8: 1})
# Counter({'a': 7, 'd': 6, 's': 5, 'f': 4, 'v': 1, 'n': 1})
print(Counter(sir))
print(Counter(cuvinte))  # Counter({'mere': 4, 'si': 2, 'pere': 2, 'Ana': 1
# , 'are': 1, 'da': 1, 'nu': 1, 'poate': 1})
# c.clear() - Resets all counts
# list(c) - list unique elements
# set(c) - convert to  set
# dict(c) - convert tot a regular dictionary
# c.items() - comvert to a list of (elem, cnt) pairts
# Counter(dict(list_of_pairs)) - convert from a list of (elem, cnt) pairs
# c.most_common() [:-n-1:-1] - n Least common elements
# c += Counter() - Remove zero and negative counts
##################################################################################
# Collections: Default dict
from collections import defaultdict

d = defaultdict(lambda: 0)
d['one']
d['two'] = 2
# defaultdict(<function <lambda> at 0x0000000002201828>, {'a': 0, 'b': 2})
print(d)
##################################################################################
# Datetime
import datetime
t = datetime.time(5, 25, 1)
print(t)  # 05:25:01
print(t.hour)  # 5
print(datetime.time.min)  # 00:00:00
print(datetime.time.max)  # 23:59:59.999999

import datetime
today = datetime.date.today()
print(today)  # 2019-10-14
print(today.year)  # 2019
print(datetime.date.min)  # 0001-01-01
print(datetime.date.max)  # 9999-12-31
d1 = datetime.date(2015, 3, 11)
print(d1)  # 2015-03-11
d2 = d1.replace(year=1990)
print(d2)  # 1990-03-11
print(d1 - d2)  # 9131 days, 0:00:00
##################################################################################
# PDB Module
import pdb

x = [1, 3, 4]
y = 2
z = 3
result = y + z
print(result)
pdb.set_trace()
result2 = y + x
print(result2)
##################################################################################
# Cronometer
import timeit

print(timeit.timeit('"-".join(str(n) for n in range(100))', number=10000))
# 0.402033814
print(timeit.timeit('"-".join([str(n) for n in range(100)])', number=10000))
# 0.345903018
print(timeit.timeit('"-".join(map(str,range(100)))', number=10000))
# 0.222068923
##################################################################################
# Regular expressions
import re

. - Any character except new line
\d - digit(0 - 9)
\D - Not a digit(0 - 9)
\w - Word character(a - z, A - Z, 0 - 9, _)
\W - Not a word character(a - z, A - Z, 0 - 9, _)
\s - Whitespaces(space, tab, newline)
\S - Not whitespaces(space, tab, newline)
\b - Word boundary
\B - Not word boundary
^ - Beginning of a string
$ - End of a string

Quantifiers:
* - 0 or More
+ - 1 or More
? - 0 or One
{3} - Exact number
{3, 4} - Range of numbers(min, max)

Groups:
| - Either or
() - Group

text_phrase = 'sdsd..sssddd...sdddsddd...dsds...dsssss...sdddd'
text_patterns = ['sd*',  # s followed by zero or more d's
                 'sd+',  # s followed by one or more d's
                 'sd?',  # s followed by zero or one d's
                 'sd{3}',  # s followed by three d's
                 'sd{2,3}',  # s followed by two to three d's
                 ]
def multi_re_find(patterns, phrase):
  for pattern in patterns:
    print(f'searching the phrase using the re check: {pattern}')
    print(re.findall(pattern, phrase))
    print('\n')

multi_re_find(text_patterns, text_phrase)
##################################################################################
# Tricks
dividend = int(input('Enter dividend: '))
divisor = int(input('Enter divisor: '))
msg = dividend / divisor if divisor != 0 else 'Error, cannot divide by zero'
print(msg)
n = int(input("Enter a number: "))
print('|', n, '| = ', (-n if n < 0 else n), sep='')
nume = 'dragos' if 'e' in 'Daniel' else 'Bratu'
print(nume)
##################################################################################
# Time module
from time import perf_counter, sleep
print("Enter your name: ", end="")
start_time = perf_counter()
name = input()
elapsed = perf_counter() - start_time
print(name, "it took you", elapsed, "seconds to respond")
for count in range(10, -1, -1):  # Range 10, 9, 8, ..., 0
  print(count)  # Display the count
  sleep(1)  # Suspend execution for 1 second
##################################################################################
# RJust, LJust
print(word.rjust(10, "*"))  # ******ABCD
print(word.rjust(3, "*"))  # ABCD
print(word.rjust(15, ">"))  # >>>>>>>>>>>ABCD
print(word.rjust(10))  # ABCD
##################################################################################
# One line prime numbers
print([p for p in range(2, 80) if not [x for x in range(2, p) if p % x == 0]])
##################################################################################
# Dictionary comprehension
lista = ['a', 'b', 'c']
numere = [1, 2, 3]
dictionar = dict(zip(lista, numere))
print(dictionar)
##################################################################################
# Unpacking
_, _, *y, _, _ = x
s = y
print(s)  # [3, 4]
##################################################################################
# Linked lists
class Node:
  def __init__(self, informatie=None):
    self.informatie = informatie
    self.adresa_urmatoare = None


class LinkedList:
  def __init__(self):
    self.informatie_inceput = None

  def printing(self):
    printvalue = self.informatie_inceput
    while printvalue is not None:
      print(printvalue.informatie)
      printvalue = printvalue.adresa_urmatoare

  def ins_beg(self, o_informatie):
    new = Node(o_informatie)
    new.adresa_urmatoare = self.informatie_inceput
    self.informatie_inceput = new

  def ins_end(self, o_informatie):
    newest = Node(o_informatie)
    if self.informatie_inceput == None:
      self.informatie_inceput = newest
      return
    lastNode = self.informatie_inceput
    while(lastNode.adresa_urmatoare):
      lastNode = lastNode.adresa_urmatoare
    lastNode.adresa_urmatoare = newest

  def ins_bet(self, node, o_informatie):
    newofall = Node(o_informatie)
    newofall.adresa_urmatoare = node.adresa_urmatoare
    node.adresa_urmatoare = newofall

  def deletion(self, informatia_stearsa):
    new = self.informatie_inceput
    if new is not None:
      if new.informatie == informatia_stearsa:
        self.informatie_inceput = new.adresa_urmatoare
        new = None
        return

    while new is not None:
      if new.informatie == informatia_stearsa:
        break
      prev = new
      new = new.adresa_urmatoare

    if new == None:
      return

    prev.adresa_urmatoare = new.adresa_urmatoare
    new = None


x = LinkedList()
x.informatie_inceput = Node('Manchester City')
data2 = Node('Liverpool')
data3 = Node('Tottenham')

x.informatie_inceput.adresa_urmatoare = data2
data2.adresa_urmatoare = data3
x.ins_beg('Manchester United')
x.ins_end('Everton')
x.ins_bet(x.informatie_inceput.adresa_urmatoare.adresa_urmatoare, 'Chelsea')
x.ins_bet(x.informatie_inceput.adresa_urmatoare.adresa_urmatoare, 'Barcelona')
x.deletion('Barcelona')
x.printing()
##################################################################################
# Objects sum
class Operations():
  def __init__(self, *args):
    if len(args) == 0:
      self.numbers = (0, 0)
    else:
      self.numbers = args

  def __add__(self, other):
    sum = tuple(x + y for x, y in zip(self.numbers, other.numbers))
    return Operations(*sum)

  def __mul__(self, other):
    mul = tuple(x * y for x, y in zip(self.numbers, other.numbers))
    return Operations(*mul)


# also add/sub/mul/div/floor/mod/power/lshift/rshift/and/or/xor
x = Operations(4, 5)
y = Operations(1, 4)
t = Operations(3, 2)
z = x + y + t
w = x * y * t
print(z.numbers)
print(w.numbers)
##################################################################################
# Objects comparison
class Comparison():
  def __init__(self, x):
    self.x = x

  def __lt__(self, other):
    return self.x < other.x

  def __gt__(self, other):
    return self.x > other.x

  def __eq__(self, other):
    return self.x == other.x


if __name__ == '__main__':
  obj1 = Comparison(2)
  obj2 = Comparison(3)
  print(obj1 < obj2)  # True
  print(obj1 > obj2)  # False
  print(obj1 == obj2)  # False
# also iadd/isub/imul/idiv/ifloordivi/mod/ipower/ilshift/irshift/iand/ior/ixor
##################################################################################
# Dictionaries sum
class dictionary(dict):
  def __add__(self, other):
    self.update(other)
    return dictionary(self)


dict1 = dictionary({'firstname': 'Dragos'})
dict2 = dictionary({'lastname': 'BRATU'})
print(dict1 + dict2)
##################################################################################
# Virtual environment
pip install virtualenv
virtualenv project1_env
source project1_env / bin / activate
pip freeze - -local > requirements.txt
deactivate
pip install - r requirements.txt
##################################################################################
# Generator comprehension
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
my_gen = (n * n for n in nums)
##################################################################################
# Object sorting
class Angajat():
  def __init__(self, nume, varsta, salariu):
    self.nume = nume
    self.varsta = varsta
    self.salariu = salariu

  def __repr__(self):
    return f'({self.nume}, {self.varsta}, {self.salariu})'


a1 = Angajat('Dragos', 25, 60000)
a2 = Angajat('Vlad', 28, 80000)
a3 = Angajat('Iulian', 24, 30000)

angajati = [a1, a2, a3]

ang_sort_sal = sorted(angajati, key=lambda e: e.salariu, reverse=True)
print(ang_sort_sal)
##################################################################################
# OS module
import os

print(os.getcwd())
os.chdir('D:\\LocalData\\py06366\\OneDrive - Alliance')
print(os.removedirs('D:\\LocalData\\py06366\\OneDrive - Alliance\\MaaanU'))

import os

for dir_path, dir_names, file_names in os.walk('C:\\Users\\py06366\\OneDrive - Alliance\\Pyhon\\a\\PyCharm'):
  print('Current Path:', dir_path)
  print('Directories:', dir_names)
  print('Files:', file_names)

##################################################################################
# File handling
with open('client1.py', 'r+') as f:
  size_to_read = 10
  f_contents = f.read(size_to_read)
  while len(f_contents) > 0:
    print(f_contents, end='#')
    f_contents = f.read(size_to_read)
with open('client1.py', 'r+') as f:
  size_to_read = 10
  f_contents = f.read(size_to_read)
  print(f_contents)
  f.seek(0)
  f_contents = f.read(size_to_read)
  print(f_contents)
# renaming files
os.chdir('C:\\Users\\py06366\\OneDrive - Alliance\\Pyhon\\P1\\moby\\Fisiere')
for f in os.listdir():
  file_name, file_extention = os.path.splitext(f)
  echipa, categoria, numar = file_name.split('-')
  echipa = echipa.strip()
  categoria = categoria.strip()
  numar = numar.strip()
  nume_nou = f'{numar}-{echipa}{file_extention}'
  os.rename(f, nume_nou)
##################################################################################
# Random module
import random

value = random.uniform(1, 10)
print(value)  # 4.15449409933993

colors = ['Red', 'Black', 'Green']
roulette = random.choices(colors, weights=[18, 18, 2], k=5)
print(roulette)  # ['Red', 'Black', 'Black', 'Red', 'Black']
##################################################################################