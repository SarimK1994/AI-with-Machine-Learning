#Name of authors: Evan DeBiasse
#Date: 6/4/2018
#Purpose: Introduction to python programming

#Python is a dynamic typing langugage and is also a scripting language
#Data type: integer: 1 and floating point number: 1.2

res0 = 2 ** 4
print("res0: ", res0)
print("type of res0: ", type(res0))

# dynamic-typing is double curse: 1. easier to learn 2. sometimes difficult for big project

res1 = 4 % 2
print("res1: ", res1)

x = 2
y = 3
x = x + x
res2 = x + y
print("res2: ", res2)

# String
# indexing String
#
s = 'abcdefghijk'
res3 = s[4]
print("res3: ", res3)

# you can use slice notation to grab slices of the string
# all this knowledge we learn in the intro section is important to build AI applications
# slice is upto not include the limit the following is up to and not including 3
res4 = s[:3]
print("res4: ", res4)

#Starting index IS included, but 6 is not for example this output is "def"
res5 = s[3:6]
print("res5: ", res5)

#List
#Lists are a sequence of elements
#
my_list = ['a','b','c']
my_list.append('d')
print("my_list: ", my_list)

res6 = my_list[1]
print("res6: ", res6)

#list is mutable and tuple is immutable
#tuple is the pair of item
#
my_list[0] = 'NEW'
print("my_list: ", my_list)

#Dictionary
#Dictionay behaves like key-value pair just like hash table
#
d1 = {'key1':'value','key2':123}
res8 = d1['key1']
print("res8: ", res8)

#Dictionary can take any item as its value. It is quite flexible
#value for d2 is a list
d2 = {'k1':[1,2,3]}
res9 = d2['k1']
res9a = d2['k1'][0]
print("res9: ", res9)
print("res9a: ", res9a)

#since python is so flexible the responsibility falls on the shoulders of the programmer

#tuples
#tuples are immutable and do not support item assignment
#
t = (1,2,3)
res10 = t[0]
print("res10 ", res10)
# This would give you trouble t[0] = 'NEW' because tuples are immutable (gives error doesn't suppoer assignment

#Sets
#a set is a collection unique elements
# this is set function
res11 = set([1,1,2,2,2,5,5,5,6,6])
print("res11: ", res11)
#Notice repeated items are not printed, only unique elements are printed

#add an element to set
s = {1,2,3}
s.add(5)
print("set s: ", s)

# Comparison operators
# compare 2 strings
#Assigning a boolean
res12 = 'hi' == 'bye'
print("res12: ", res12)

#logic operator
res13 = (1 > 2) or (2 < 3)
print("res13: ", res13)

#if elif else statements
#python does not use bracket or curly brackets to separate the block of code execution
#it uses the whitespace or indentation instead
#
print()
if 1 == 2:
    print('first')
    a = 5
else:
    print('last')
#notice no curly braces, also the colons denote the if statement

# check multiple conditions using else if
#
if 1 == 2:
    print('first')
elif 4 == 4:
    print('second')
elif 3 == 3:
    print('middle')
else:
    print('Last')

print()

#while loop
#
i = 1
while i < 5:
    print('i is: {}'.format(i))
    i = i + 1

#list comprehension
#Go through each elemenet of x and for each element, calculate its square the result is a list
#Calculate the square of each item of a list
x = [1,2,3,4]
res15 = [num**2 for num in x]
print("res15: ",res15)

#function
#
def square(num):
    """

    :param num: num is integer
    :return: square of a number
    """

#above is a documentation string implemented by the 3 quotations

    return num**2

res16 = square(2)
print("res16:", res16)

#NumPy Arrays, which provide lots of mathematical equatioins for us
#NumPy arrays are the building block for our first neural network
#create NumPy array from Python list
import numpy as np
my_list = [1,2,3]
arr1 = np.array(my_list)

#we want to figure out the shape of this array
#shape information is critical for any AI codes
shape_arr1 = arr1.shape
print("shape_arr1: ",shape_arr1)
#basically this is rank-one array

# each list is one row
# matrix is important because all your data is matrix of number
my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
arr2 = np.array(my_matrix)
shape_arr2 = arr2.shape
print("shape_arr2: ", shape_arr2)
print()

#arrange: return evenly spaced values within a given inteval
arr3 = np.arange(0,10)
print("arr3: ", arr3)
print()

arr4 = np.zeros(3)
print("arr4: ", arr4)

arr5 = np.zeros((2,5))
print("arr5: ", arr5)
print()

arr6 = np.ones((2,5))
print("arr6", arr6)
print()

# Numpy has lots of ways to create random number array
# randn return a sample from the standard normal or Gaussian distribution
#
arr7 = np.random.randn(2,3)
print("arr7: ", arr7)
print()

arr8 = np.random.randn(2)
print("arr8: ", arr8)
print()

#generate 10 random integers
arr9 = np.random.randint(1,100,10)
print("arr9: ", arr9)
print()

arr10 = np.arange(25) # 1d array
arr11 = arr10.reshape(5,5) #reshape from 1d array to 5x5 2d array
print("arr11: ", arr11)
print()

#max find max values
#
arr11 = np.random.randint(0,50,10)
print("arr11: ", arr11)
res23 = arr11.max()
print("res23: ",res23)

#argmax find the index location of max values
res23a = arr11.argmax()
print("arr23a: ", res23a)

#numpy array indexing
arr12 = np.arange(0,11)
res25 = arr12[8]
print("res25: ", res25)

res26 = arr12[1:5] #Slice notation
print("res26: ", res26)
print()

res27 = arr12[:6] #Up to but not including 6
print("res27: ", res27)
print()

res28 = arr12[5:] #From index 5 all the way to the end
print("res28: ", res28)
print()

#indexing 23 matricies
arr13 = np.array([[5,10,15],[20,25,30],[35,40,45]])

#get row 0
res29 = arr13[0]
print("res29: ", res29)
print()

# row 2 index 1
res30 = arr13[2,1]
print("res30: ", res30)
print()

res31 = arr13[:2,1:] #gets top right corner
print("res31: ", res31)
print()

#get first 2 rows
res32 = arr13[:2]
print("res32: ", res32)
print()

arr14 = np.arange(1,11)
bool_arr1 = arr14 > 4
print("bool_arr1: ", bool_arr1)
print()

#we can get results when the boolean array happens to be true
#
arr15 = arr14[arr14 > 4]
print("arr15: ", arr15)
print()

#Numpy Operations
#Arithmetic for Numpy
#
arr16 = np.arange(0,11)
arr17 = arr16 * arr16 # element by element operation
for item in arr17:
    print(item)
print()

arr18 = arr17 + arr17 # element by element addition
for item in arr18:
    print(item)
print()

# you can do array with exponenets
#
arr20 = arr16 ** 2 #square each element
print("arr20: ", arr20)
print()

#Universal array operation
#Element by element square function
arr21 = np.sqrt(arr16)
print("arr21: ", arr21)
print()

#calculate exponential
arr22 = np.exp(arr16)
print("arr22: ", arr22)
print()

#Sometimes numpy will issue warning instead of error on certain math operations
arr23 = np.log(arr16)
print("arr23: ", arr23)
print()

#Numpy provides lots of useful mathematica operations for building AI apps
#that is why we do not use regular python

