#!/usr/bin/python

####################################################
# CS 5600/6600/7890: Assignment 1: Problems 1 & 2
# Zachary Hook
# A02231717
#####################################################

import numpy as np
import math

class and_perceptron:

    def __init__(self):
        self.a = .51
        self.b = .51
        self.bias = -1
        pass

    def output(self, x):
        right = x[0]
        left = x[1]
        val = math.ceil((right*self.a)+(left*self.b)+self.bias)
        if val > 0:
            val = 1
        else:
            val = 0
        return val

class or_perceptron:
    def __init__(self):
        self.a = 1
        self.b = 1
        self.bias = -.99
        pass

    def output(self, x):
        right = x[0]
        left = x[1]
        val = math.ceil((right*self.a)+(left*self.b)+self.bias)
        if val > 0:
            val = 1
        else:
            val = 0
        return val

class not_perceptron:
    def __init__(self):
        self.bias = .5
        self.b = -1
        pass

    def output(self, x):
        not_perc = x[0]
        val = math.ceil((not_perc*self.b)+self.bias)
        if val > 0:
            val = 1
        else:
            val = 0
        return val

class xor_perceptron:
    def __init__(self):
        pass

    def output(self, x):
        or_res = or_perceptron().output(x)
        and_res = and_perceptron().output(x)
        x[0] = and_res
        and_res = not_perceptron().output(np.array([x[0]]))
        x[0] = or_res
        x[1] = and_res
        and_res = and_perceptron().output(x)
        return and_res


class xor_perceptron2:
    def __init__(self):
        self.oneA = .51
        self.oneB = .51
        self.oneBias = -1
        self.twoA = 1.01
        self.twoB = 1.01
        self.twoBias = -1
        self.threeA = 1
        self.threeB = 1
        self.threeBias = -1.01
        pass

    def output(self, x):
        first = (x[0]*self.oneA)+(x[1]*self.oneB)+self.oneBias
        if first > 0:
            first = 1
        else:
            first = 0
        second = (x[0]*self.twoA)+(x[1]*self.twoB)+self.twoBias
        if second > 0:
            second = 1
        else:
            second = 0
        final = (first*self.threeA)+(second*self.threeB)+self.threeBias
        if final > 0:
            final = np.array([1])
        else:
            final = np.array([0])
        return final

### ================ Unit Tests ====================

# let's define a few binary input arrays.    
x00 = np.array([0, 0])
x01 = np.array([0, 1])
x10 = np.array([1, 0])
x11 = np.array([1, 1])

# let's test the and perceptron.
def unit_test_01():
    andp = and_perceptron()
    assert andp.output(x00) == 0
    assert andp.output(x01) == 0
    assert andp.output(x10) == 0
    assert andp.output(x11) == 1
    print('all andp assertions passed...')

# let's test the or perceptron.
def unit_test_02():
    orp = or_perceptron()
    assert orp.output(x00) == 0
    assert orp.output(x01) == 1
    assert orp.output(x10) == 1
    assert orp.output(x11) == 1
    print('all orp assertions passed...')

# let's test the not perceptron.
def unit_test_03():
    notp = not_perceptron()
    assert notp.output(np.array([0])) == 1
    assert notp.output(np.array([1])) == 0
    print('all notp assertions passed...')

# let's test the 1st xor perceptron.
def unit_test_04():
    xorp = xor_perceptron()
    assert xorp.output(x00) == 0
    assert xorp.output(x01) == 1
    assert xorp.output(x10) == 1
    assert xorp.output(x11) == 0
    print('all xorp assertions passed...')

# let's test the 2nd xor perceptron.
def unit_test_05():
    xorp2 = xor_perceptron2()
    assert xorp2.output(x00)[0] == 0
    assert xorp2.output(x01)[0] == 1
    assert xorp2.output(x10)[0] == 1
    assert xorp2.output(x11)[0] == 0
    print('all xorp2 assertions passed...')

unit_test_01()
unit_test_02()
unit_test_03()
unit_test_04()
unit_test_05()
        
        

    
        





