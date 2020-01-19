# VARIABLE ARGUMENTS
from sys import argv
script, first, second, third = argv
print(f'{first} & {second} & {third}')
# When called: python hello.py Wayne Fucking Rooney
###################################################
# JOIN
things = ['United', 'Zaha', 'Wazza', 'Chalton', 'Pogba', 'Trafford']
print(' '.join(things))  # United Zaha Wazza Chalton Pogba Trafford
print('#'.join(things[1:4]))  # Zaha#Wazza#Chalton
###################################################
# DICTIONARY .GET METHOD
teams = {'UTD': 'United', 'CHE': 'Chelsea',
         'ARS': 'Arsenal', 'LIV': 'Liverpool'}
team = teams.get('BAR')
print(team)  # None
team = teams.get('BAR', 'Does Not Exist')
print(team)  # Does Not Exist
###################################################
# Inheritance:  INHERITANCE
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
oscar.mers()  # merge in 4 membre
codita = Pisica(name='Codita')
codita.miaun()  # Codita face miaun miaun
###################################################
# PATHLIB
from pathlib import Path

path = Path("Jucatori")
path.mkdir()  # create a folder
path.rmdir()  # delete a folder
path = Path()
for file in path.glob('*.*'):  # path.glob('*'), path.glob('*.xls')
  print(file)
###################################################
# INHERITANCE
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
dad.altered()   # PARENT altered()
son.altered()   # CHILD, BEFORE PARENT altered()
# PARENT altered()
# CHILD, AFTER PARENT altered()
###################################################
# ROUND TRUNCATE
print(int(28.71))  # 28
print(round(93.34836, 2))  # 93.35
print(round(19.47, 0))  # 19.0
print(round(28793.54836, 0))  # 28794.0
print(round(28793.54836, 1))  # 28793.5
print(round(28793.54836, -1))  # 28790.0
###################################################
# PRINTING
print('{0:04} {1:.5f}'.format(5, 5))  # 0005 5.00000
print('{0:>4} {1:>32}'.format('*', '#'))  # *                                %
###################################################
# FORMATED TABLE
size = 9
for row in range(1, size + 1):
  for column in range(1, size + 1):
    product = row * column
    print('{0:4}'.format(product), end='')
  print()
###################################################
# % FORMATED STRING
print("I will inject %s here and %d here" % ('text', 34))
###################################################
# ENUMERATE
o_lista = ['ala', 'bala', 'portocala']
for elem in enumerate(o_lista, start=1):
  print(elem, end="")  # (1, 'ala')(2, 'bala')(3, 'portocala')
###################################################
# ZIP
lista1 = [1, 2, 3, 4, 5, 6]
lista2 = ['a', 'b', 'c', 'd', 'e']
lista3 = [100, 200, 300, 400]
for item in zip(lista1, lista2, lista3):
  print(item)
#                   (1, 'a', 100)
#                   (2, 'b', 200)
#                   (3, 'c', 300)
#                   (4, 'd', 400)

###################################################
# ARGS KWARGS
def myfunc(*args, **kwargs):
  print(f"I would like {args[-1]} {kwargs['car']}es.")
  print(kwargs)
  print(args)

myfunc(20, 30, car='Mercedes', another='Something')
# I would like 30 Mercedeses.
# {'car': 'Mercedes', 'another': 'Something'}
# (20, 30)
###################################################
# MAP FILTER
def printare(nume):
  return f'{nume} e BO$$'
def patrat(numar):
  return numar ** 2
def check_nume(numele):
  return 'elena' in numele.lower()

nume_fete = ['Elena Popescu', 'Ivan Elena', 'Simona Urs', 'Raluca Elena']
numere = [1, 2, 3, 4, 5]
catalog = ['Costel', 'Nicu', 'Gigel'] 
print(list(map(printare, catalog)))  # ['Costel e BO$$', 'Nicu e BO$$', 'Gigel e BO$$']
print(list(map(patrat, numere)))  # 1 4 9 16 25
print(list(filter(check_nume, nume_fete)))  # ['Elena Popescu', 'Ivan Elena', 'Raluca Elena']
###################################################
# LAMBDA FUNCTION
numere = [2, 5, 7, 8, 4]
lista = list(map(lambda num: num ** 2, numere))
print(lista)  # [4, 25, 49, 64, 16]

titulaturi = ['Raluca', 'Elena', 'Bianca', 'Casandra']
lista2 = list(filter(lambda nume: 'ca' in nume.lower(), titulaturi))
print(lista2)  # ['Raluca', 'Bianca', 'Casandra']
###################################################
# CLASS
name = 'Henrik'
class Caine:
  species = 'mammal'
  def __init__(self, breed, name):
    self.breed = breed
    self.name = name
    self.specius = Caine.species + 'ius'
  def ce_face(self, numar):
    print(f'{self.name} face ham ham de {numar} ori')

caini = {'Igor': 'pitbull', 'Lord': 'lup', 'Oscar': 'husky'}
for nume, rasa in caini.items():
  nume = Caine(rasa, nume)
  nume.ce_face(4)
  print(nume.specius)
  print(f'Si este rasa: {nume.breed}, din regnul {nume.species}')
###################################################
# POLYMORPHISM
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
  print(type(pet))
  print(pet.speak())
def pet_speak(pet):
  print(pet.speak())

print(pet_speak(niko))
print(pet_speak(felix))
###################################################
  # __STR__ __LEN__ __DEL__
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
###################################################
# __NAME__ __MAIN__
def printare():
  print(__name__)
# if imported: <name_of_the_file>
# if not imported: __main__
###################################################
# UNITTEST
import unittest
def cap_adu(*args):
  return sum(args)

class TestAdunare(unittest.TestCase):
  def test_two_nums(self):
    a = 4
    b = 6
    result = cap_adu(a, b)
    self.assertEqual(result, 10)

  def test_three_nums(self):
    a = 5
    b = 60
    c = 4
    result = cap_adu(a, b, c)
    self.assertEqual(result, 69)

unittest.main()


def hello(name='Dragos'): FUNCSAPTION
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
###################################################
# FUNCSAPTION
def hello():  
  return 'Hi Dragos!'
def other(some_def_func):
  print('Other code runs here!')
  print(some_def_func())

other(hello)
###################################################
# DECORATORS
def inmultire_impartire(original_func):

  def wrap_func(*args):
    product = 1
    for number in args:
      product *= number
    print(f'produsul numerelor {args} este {product}')
    original_func(*args)
    print(f'catul primelor 2 numere {args[0]} si {args[1]} este {args[0] / args[1]}')
  return wrap_func

@inmultire_impartire
def adunare(*args):
  print(f'suma numerelor {args} este {sum(args)}')

adunare(3, 5, 6, 10)
# produsul numerelor (3, 5, 6, 10) este 900
# suma numerelor (3, 5, 6, 10) este 24
# catul primelor 2 numere 3 si 5 este 0.6
###################################################
# GENERATORS
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
###################################################
# ITER
s = 'Hello' 
s_iter = iter(s)
print(next(s_iter))  # H
print(next(s_iter))  # e
###################################################
# DEFAULT DICT
from collections import defaultdict 

d = defaultdict(lambda: 0)
d['one']
d['two'] = 2
print(d)  # defaultdict(<function <lambda> at 0x0000000002201828>, {'a': 0, 'b': 2})
###################################################
# DATETIME MODULE
import datetime 
t = datetime.time(5, 25, 1)
print(t)  # 05:25:01
print(t.hour)  # 5
today = datetime.date.today()
print(today)  # 2019-10-14
print(today.year)  # 2019
d1 = datetime.date(2015, 3, 11)
print(d1)  # 2015-03-11
d2 = d1.replace(year=1990)
print(d2)  # 1990-03-11
print(d1 - d2)  # 9131 days, 0:00:00
bday = datetime.datetime(1994, 4, 19)
print(f'My bday is on {bday:%B %d, %Y}')  # My bday is on April 19, 1994
###################################################
# TIMEIT
import timeit 

print(timeit.timeit('"-".join(str(n) for n in range(100))', number=10000))
# 0.402033814
print(timeit.timeit('"-".join([str(n) for n in range(100)])', number=10000))
# 0.345903018
print(timeit.timeit('"-".join(map(str,range(100)))', number=10000))
# 0.222068923
###################################################
# NUMBERS
print(hex(13))  # 0xd
print(bin(14))  # 0b1110
print(pow(2, 3, 3))  # (2^3)%3 = 2
print(abs(-2))  # 2
print(round(3.1))  # 3.0 (Always float)
print(round(1.2345, 2))  # 1.23
###################################################
# STRINGS
s = 'United'
print(s.isalnum())  # True
print(s.islower())  # False
print(s.isalpha())  # True
###################################################
# SETS
s = set() 
s.add(1)
s.add(2)
print(s)  # {1, 2}
s.clear()
print(s)  # set()
s = {1, 2, 3}
sc = s.copy()
print(sc)  # {1, 2, 3}
s.add(4)
print(s)  # {1, 2, 3, 4}
print(s.difference(sc))  # {4}
s1 = {1, 2, 3}
s2 = {1, 4, 5}
s1.difference_update(s2)
print(s1)  # {2, 3}
s.discard(2)  # {1, 2, 3, 4}
print(s)  # {1, 3, 4}
s1 = {1, 2, 3}
s2 = {1, 2, 4}
print(s1.intersection(s2))  # {1, 2}
s1.intersection_update(s2)
print(s1)  # {1, 2}
s1 = {1, 2}
s2 = {1, 2, 4}
s3 = {5}
print(s1.isdisjoint(s2))  # False
print(s1.isdisjoint(s3))  # True
print(s1.issubset(s2))  # True
print(s2.issuperset(s1))  # True
print(s1.symmetric_difference(s2))  # {4}
print(s1.union(s2))  # {1, 2, 4}
###################################################
# COMPREHENSIVE DICTIONARIES
z = {k: v**2 for k, v in zip(['a', 'b'], range(2))} 
print(z)  # {'a': 0, 'b': 1}
###################################################
# COMPREHENSIVE IF
dividend = int(input('Enter dividend: ')) 
divisor = int(input('Enter divisor: '))
msg = dividend / divisor if divisor != 0 else 'Error you stupid shit, cannot divide by zero'
print(msg)

n = int(input("Enter a number: "))
print('|', n, '| = ', (-n if n < 0 else n), sep='')

nume = 'dragos' if 'e' in 'Daniel' else 'Bratu'
print(nume)
###################################################
# TIME MODULE
from time import perf_counter, sleep
print("Enter your name: ", end="")
start_time = perf_counter()
name = input()
elapsed = perf_counter() - start_time
print(name, "it took you", elapsed, "seconds to respond")
for count in range(10, -1, -1):  # Range 10, 9, 8, ..., 0
  print(count)  # Display the count
  sleep(1)  # Suspend execution for 1 second
###################################################
# HIDDEN FUNCTIONS
len(sir) == sir.__len__()
sir[2] == sir.__getitem__(2)
###################################################
# ONE LINER PRIME NUMBERS
print([p for p in range(2, 80) if not [x for x in range(2, p) if p % x == 0]])
###################################################
# DICTIONARIES
lista = ['a', 'b', 'c']
numere = [1, 2, 3]
dictionar = dict(zip(lista, numere))
print(dictionar)
###################################################
# SETS OPERATIONS
a = {1, 2, 3, 4}  
b = {3, 4, 5, 6, 7}
print(a | b)  # {1, 2, 3, 4, 5, 6, 7}
print(a & b)  # {3, 4}
print(a - b)  # {1, 2}
print(a ^ b)  # {1, 2, 5, 6, 7}
###################################################
# UNPACKING
x = 1, 2, 3, 4, 5, 6
_, _, *y, _, _ = x
print(y)  # [3, 4]
###################################################
#  LINKED LISTS
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
###################################################
#  MAGIC METHOD
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
# also iadd/isub/imul/idiv/ifloordivi/mod/ipower/ilshift/irshift/iand/ior/ixor  MAGIC METHOD
###################################################
# MAGIC METHOD
class dictionary(dict):
  def __add__(self, other):
    self.update(other)
    return dictionary(self)

dict1 = dictionary({'firstname': 'Dragos'})
dict2 = dictionary({'lastname': 'BRATU'})
print(dict1 + dict2)
###################################################
# MAGIC METHOD
class LenthConversion:
  value = {'mm': 0.001, 'cm': 0.01, 'm': 1, 'km': 1000, 'in': 0.254, 'ft': 0.3048,
           'yd': 0.9144, 'mi': 1609.344}
  def __init__(self, x, value_unit='m'):
    self.x = x
    self.value_unit = value_unit
  def Convert_to_meters(self):
    return self.x * LenthConversion.value[self.value_unit]
  def __add__(self, other):
    ans = self.Convert_to_meters() + other.Convert_to_meters()
    return LenthConversion(ans / LenthConversion.value[self.value_unit], self.value_unit)
  def __str__(self):
    return str(self.Convert_to_meters)
  def __repr__(self):
    return "LengthConversion(" + str(self.x) + ' ' + self.value_unit + ")"


obj1 = LenthConversion(0, 'm') + LenthConversion(52, 'cm')
print(repr(obj1))
print(obj1)
###################################################
# VIRTUALENV
pip install virtualenv
virtualenv project1_env
source project1_env / bin / activate
pip freeze - -local > requirements.txt
deactivate
pip install - r requirements.txt
###################################################
# ONE LINE GENERATOR
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
my_gen = (n * n for n in nums)
###################################################
# OBJECTS SORTING
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
###################################################
# OS MODULE
import os

print(os.getcwd())
os.chdir('D:\\LocalData\\py06366\\OneDrive - Alliance')
print(os.removedirs('D:\\LocalData\\py06366\\OneDrive - Alliance\\MaaanU'))

# Date of modification
from datetime import datetime

mod_time = os.stat('Server.py').st_mtime
print(datetime.fromtimestamp(mod_time))

# walking
for dir_path, dir_names, file_names in os.walk(os.getcwd()):
  print('Current Path:', dir_path)
  print('Directories:', dir_names)
  print('Files:', file_names)
# RENAME FILES  
os.chdir('C:\\Users\\py06366\\OneDrive - Alliance\\Pyhon\\P1\\moby\\Fisiere')
for f in os.listdir():
  file_name, file_extention = os.path.splitext(f)
  echipa, categoria, numar = file_name.split('-')
  echipa = echipa.strip()
  categoria = categoria.strip()
  numar = numar.strip()
  nume_nou = f'{numar}-{echipa}{file_extention}'
  os.rename(f, nume_nou)
###################################################
# OPEN BIG FILES
with open('client1.py', 'r+') as f:
  
  size_to_read = 10
  f_contents = f.read(size_to_read)
  while len(f_contents) > 0:
    print(f_contents, end='#')
    f_contents = f.read(size_to_read)

###################################################
# RANDOM MODULE
import random 

value = random.uniform(1, 10)
print(value)  # 4.15449409933993

colors = ['Red', 'Black', 'Green']
roulette = random.choices(colors, weights=[18, 18, 2], k=5)
print(roulette)  # ['Red', 'Black', 'Black', 'Red', 'Black']
###################################################
# REGEX
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
###################################################
# LOGGER DECORATOR
from functools import wraps 

def my_logger(original_function):
  import logging
  logging.basicConfig(filename=f'{original_function.__name__}.log', level=logging.INFO)

  @wraps(original_function)
  def wrapper(*args, **kwargs):
    logging.info(f'Ran with args: {args}, and {kwargs}')
    return original_function(*args, **kwargs)
  return wrapper

def my_timer(original_function):
  import time

  @wraps(original_function)
  def wrapper(*args, **kwargs):
    t1 = time.time()
    result = original_function(*args, **kwargs)
    t2 = time.time() - t1
    print(f'{original_function.__name__} ran in : {t2} seconds.')
    return result
  return wrapper

import time

@my_logger
@my_timer
def display(echipa, trofee):
  time.sleep(1)
  print(f'Display ran with arguments {echipa}, {trofee}')


display('Manchester', 20)
###################################################
# DELIMITER
pi = 3.14159265 
print(f'pi este {pi:.3f}')
###################################################