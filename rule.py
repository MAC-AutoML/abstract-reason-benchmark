import math
import numpy as np
import math


def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a / b if b != 0 else "error"  

def modulus(a, b):
    return a % b if b != 0 else "error"  


def power(a, b):
    return a ** b

def square_root(a):
    return math.sqrt(a) if a >= 0 else "error"  


def natural_log(a):
    return math.log(a) if a > 0 else "error"  

def common_log(a):
    return math.log10(a) if a > 0 else "error"  


def intersection(set_a, set_b):
    result = set_a & set_b
    return result if result else '{}'

def union(set_a, set_b):
    result = set_a | set_b
    return result if result else '{}'

def difference(set_a, set_b):
    result = set_a - set_b
    return result if result else '{}'



def absolute_value(a):
    return abs(a)


def factorial(a):
    return math.factorial(a) if a >= 0 else "error"  


def maximum(a, b):
    return max(a, b)

def minimum(a, b):
    return min(a, b)


def average(data):
    return np.mean(data)


def variance(data):
    return np.var(data)

def standard_deviation(data):
    return np.std(data)


def combination(n, k):
    return math.comb(n, k)


def permutation(n, k):
    return math.perm(n, k)


def logical_and(a, b):
    return 1 if a and b else 0

def logical_or(a, b):
    return 1 if a or b else 0

def logical_not(a):
    return 1 if not a else 0

def logical_xor(a, b):
    return 1 if (a and not b) or (not a and b) else 0

def binary_and(a, b, length_bit):
    result = int(a, 2) & int(b, 2)
    return bin(result)[2:].zfill(length_bit)  

def binary_or(a, b, length_bit):
    result = int(a, 2) | int(b, 2)
    return bin(result)[2:].zfill(length_bit)

def binary_not(a, length_bit):
    result = (1 << length_bit) - 1 - int(a, 2)  
    return bin(result)[2:].zfill(length_bit)

def binary_xor(a, b, length_bit):
    result = int(a, 2) ^ int(b, 2)
    return bin(result)[2:].zfill(length_bit)

def binary_logical_left_shift(a, n, length_bit):
    
    result = int(a, 2) << n
    return bin(result)[2:].zfill(length_bit)[-length_bit:]  

def binary_arithmetic_left_shift(a, n, length_bit):
    
    return binary_logical_left_shift(a, n, length_bit)

def binary_logical_right_shift(a, n, length_bit):
    
    result = int(a, 2) >> n
    return bin(result)[2:].zfill(length_bit)[-length_bit:]  

def binary_arithmetic_right_shift(a, n, length_bit):
    
    result = int(a, 2) >> n
    
    if a[0] == '1':  
        result |= (1 << (length_bit - n)) - 1  
    return bin(result)[2:].zfill(length_bit)[-length_bit:]  

def binary_circular_left_shift(a, n, length_bit):
    
    n = n % length_bit  
    return bin((int(a, 2) << n | int(a, 2) >> (length_bit - n)) & ((1 << length_bit) - 1))[2:].zfill(length_bit)

def binary_circular_right_shift(a, n, length_bit):
    
    n = n % length_bit  
    return bin((int(a, 2) >> n | int(a, 2) << (length_bit - n)) & ((1 << length_bit) - 1))[2:].zfill(length_bit)


def binary_check_bit(a, n, length_bit):
    
    result = int(a, 2) & (1 << n)
    return result != 0

def binary_set_bit(a, n, length_bit):
    
    result = int(a, 2) | (1 << n)
    return bin(result)[2:].zfill(length_bit)

def binary_clear_bit(a, n, length_bit):
    
    result = int(a, 2) & ~(1 << n)
    return bin(result)[2:].zfill(length_bit)

def binary_toggle_bit(a, n, length_bit):
    
    result = int(a, 2) ^ (1 << n)
    return bin(result)[2:].zfill(length_bit)


def count_substr(a, b):
    return str(a.count(b))

def to_base(n, num):
    
    if num == 0:
        return "0"
    digits = []
    while num:
        digits.append(int(num % n))
        num //= n
    return ''.join(str(x) for x in digits[::-1])

def add_base(n, a, b):
    
    num_a = int(a, n)
    num_b = int(b, n)
    
    return to_base(n, num_a + num_b)

def subtract_base(n, a, b):
    
    num_a = int(a, n)
    num_b = int(b, n)
    
    result = num_a - num_b
    return to_base(n, result) if result >= 0 else "-" + to_base(n, abs(result))

def multiply_base(n, a, b):
    
    num_a = int(a, n)
    num_b = int(b, n)
    
    return to_base(n, num_a * num_b)


def reverse_string(s):
    return s[::-1]


def concatenate_strings(s1, s2):
    return s1 + s2


def repeat_string(s, ):
    return s * 2


def get_length(s):
    return len(s)


def is_subsequence(A: str, B: str) -> bool:
    
    i, j = 0, 0
    
    while i < len(A) and j < len(B):
        
        if A[i] == B[j]:
            j += 1
        
        i += 1
    
    if j == len(B):
        return 'T'
    else:
        return 'F'

from typing import List, Callable

def create_polynomial(coefficients: List[float]) -> Callable[[float], float]:
    def polynomial(x: float) -> float:
        
        return sum(coef * (x ** i) for i, coef in enumerate(coefficients))
    return polynomial

def linear_fun(x, a, b):
    return a * x + b

def quadratic_fun(x, a, b, c):
    return a * x**2 + b * x + c

def exponential_fun(x, a, b):
    return a * math.exp(b * x)

def logarithmic_fun(x, a, b):
    if x > 0:  
        return a * math.log(b * x)
    else:
        raise ValueError("x must be greater than 0 for logarithmic function")

def sine_fun(x, a, b):
    return a * math.sin(b * x)

def cosine_fun(x, a, b):
    return a * math.cos(b * x)

def square_wave(x, frequency):
    frequency = frequency / 10
    
    return 1 if math.sin(2 * math.pi * frequency * x) >= 0 else -1

def triangle_wave(x, frequency):
    frequency = frequency / 10
    period = 1.0 / frequency
    x = x % period
    return 4 * x / period - 1 if x < period / 2 else 3 - 4 * x / period

def sawtooth_wave(x, frequency):
    frequency = frequency / 10
    period = 1.0 / frequency
    return 2 * (x % period) / period - 1

def sin_square_wave(x, frequency):
    frequency = frequency / 10
    return math.sin(2 * math.pi * frequency * x)**2



def sort_list(lst, descending=False):
    descending = bool(descending % 2)
    
    return sorted(lst, reverse=descending) if lst else '[]'

def filter_less_than(lst, threshold):
    
    result = [x for x in lst if x > threshold]
    return result if result else '[]'

def deduplicate(lst):
    
    result = list(set(lst))
    return result

def find_minimum(lst):
    
    return min(lst) if lst else '[]'

def find_maximum(lst):
    
    return max(lst) if lst else '[]'

def calculate_median(lst):
    
    if not lst:
        return '[]'
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    mid = n // 2
    return sorted_lst[mid]

from collections import Counter

def find_mode(lst):
    
    if not lst:
        return '[]'
    count = Counter(lst)
    max_count = max(count.values())
    modes = [k for k, v in count.items() if v == max_count]
    return modes

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

def days_between_dates(date1, date2):

    
    d1 = date(date1[0], date1[1], date1[2])
    d2 = date(date2[0], date2[1], date2[2])

    
    return abs((d2 - d1).days)


def add_days(date_input, days):
    d = date(date_input[0], date_input[1], date_input[2])
    new_date = d + timedelta(days=days)
    return [new_date.year, new_date.month, new_date.day]


def add_months(date_input, months):
    d = date(date_input[0], date_input[1], date_input[2])
    new_date = d + relativedelta(months=months)
    return [new_date.year, new_date.month, new_date.day]


def add_years(date_input, years):
    d = date(date_input[0], date_input[1], date_input[2])
    new_date = d + relativedelta(years=years)
    return [new_date.year, new_date.month, new_date.day]



def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def calculate_age(birth_date, current_date):
    birth = date(birth_date[0], birth_date[1], birth_date[2])
    current = date(current_date[0], current_date[1], current_date[2])
    age = current.year - birth.year
    
    
    if (current.month, current.day) < (birth.month, birth.day):
        age -= 1
    
    return age



bit_function_list = [
    {"fun": binary_and, "args": 2, "name": "binary_and"},
    {"fun": binary_or, "args": 2, "name": "binary_or"},
    {"fun": binary_not, "args": 1, "name": "binary_not"},
    {"fun": binary_xor, "args": 2, "name": "binary_xor"},
]

bit_shift_function_list = [
    {"fun": binary_logical_left_shift, "args": 1, "name": "logical_left_shift", "parameter":1},
    {"fun": binary_logical_right_shift, "args": 1, "name": "logical_right_shift", "parameter":1},
    {"fun": binary_arithmetic_left_shift, "args": 1, "name": "arithmetic_left_shift", "parameter":1},
    {"fun": binary_arithmetic_right_shift, "args": 1, "name": "arithmetic_right_shift", "parameter":1},
    {"fun": binary_circular_left_shift, "args": 1, "name": "circular_left_shift", "parameter":1},
    {"fun": binary_circular_right_shift, "args": 1, "name": "circular_right_shift", "parameter":1},
]

bit_op_function_list = [
    {"fun": binary_check_bit, "args": 1, "name": "check_bit", "parameter":1},
    {"fun": binary_set_bit, "args": 1, "name": "set_bit", "parameter":1},
    {"fun": binary_clear_bit, "args": 1, "name": "clear_bit", "parameter":1},
    {"fun": binary_toggle_bit, "args": 1, "name": "toggle_bit", "parameter":1},
]

list_function_list = [
    {"fun": sort_list, "args": 1, "name": "sort", "parameter":0},
    {"fun": filter_less_than, "args": 1, "name": "filter", "parameter":1},
    {"fun": deduplicate, "args": 1, "name": "deduplicate", "parameter":0},
]

list_cnt_function_list = [
    {"fun": find_minimum, "args": 1, "name": "min", "parameter":0},
    {"fun": find_maximum, "args": 1, "name": "max", "parameter":0},
    {"fun": calculate_median, "args": 1, "name": "median", "parameter":0},
    {"fun": find_mode, "args": 1, "name": "majority", "parameter":0},
]

set_function_list = [

    {"fun": intersection, "args": 2, "name": "intersection"},
    {"fun": union, "args": 2, "name": "union"},
    {"fun": difference, "args": 2, "name": "difference"},
]

str_function_list = [
    {"fun": count_substr, "args": 1, "name": "conut"},
]

mul_function_list = [
    {"fun": multiply, "args": 2, "name": "*"},
]

add_function_list = [
    {"fun": add, "args": 2, "name": "+"},

]

div_function_list = [
    {"fun": divide, "args": 2, "name": "/"},
]

sub_function_list = [
    {"fun": subtract, "args": 2, "name": "-"},
]

square_function_list = [
    {"fun": square_root, "args": 1, "name": "square root"},
]

mul_base_function_list = [
    {"fun": multiply_base, "args": 2, "name": "*"},
]

add_base_function_list = [
    {"fun": add_base, "args": 2, "name": "+"},

]

sub_base_function_list = [
    {"fun": subtract_base, "args": 2, "name": "-"},
]


str_op_function_list = [
    {"fun": reverse_string, "args": 1, "name": "reverse"},
    {"fun": concatenate_strings, "args": 2, "name": "concatenate"},
    {"fun": repeat_string, "args": 1, "name": "repeat"},
    {"fun": get_length, "args": 1, "name": "get_length"},
]

linear_function_list = [
    {"fun": linear_fun, "args": 1, "name": "fun", "parameter":2},
]

quadratic_function_list = [
    {"fun": quadratic_fun, "args": 1, "name": "fun", "parameter":3},
]

exponential_function_list = [
    {"fun": exponential_fun, "args": 1, "name": "fun", "parameter":2},
]

logarithmic_function_list = [
    {"fun": logarithmic_fun, "args": 1, "name": "fun", "parameter":2},
]

sine_function_list = [
    {"fun": sine_fun, "args": 1, "name": "fun", "parameter":2},
]

cosine_function_list = [
    {"fun": cosine_fun, "args": 1, "name": "fun", "parameter":2},
]

square_wave_function_list = [
    {"fun": square_wave, "args": 1, "name": "fun", "parameter":1},
]

triangle_wave_function_list = [
    {"fun": triangle_wave, "args": 1, "name": "fun", "parameter":1},
]

sawtooth_wave_function_list = [
    {"fun": sawtooth_wave, "args": 1, "name": "fun", "parameter":1},
]

sin_square_wave_function_list = [
    {"fun": sin_square_wave, "args": 1, "name": "fun", "parameter":1},
]

data_function_list = [
    {"fun": days_between_dates, "args": 2, "name": "days_between_dates", "parameter":0},
]

base_function_list = [
    {"fun": to_base, "args": 1, "name": "to base", "parameter":1},
]

substr_function_list = [
    {"fun": is_subsequence, "args": 2, "name": "contains(in order)", "parameter":0},
]


if __name__ == "__main__":
    a = 10
    b = 5
    data = [1, 2, 3, 4, 5]
    set_a = {1, 2, 3}
    set_b = {3, 4, 5}
    
    
    print("Addition:", add(a, b))
    print("Subtraction:", subtract(a, b))
    print("Multiplication:", multiply(a, b))
    print("Division:", divide(a, b))
    print("Modulus:", modulus(a, b))
    print("Power:", power(a, b))
    print("Square Root:", square_root(a))
    print("Natural Log:", natural_log(a))
    print("Common Log:", common_log(a))
    
    
    
    print("Absolute Value:", absolute_value(-a))
    print("Factorial:", factorial(5))
    
    
    
    
    
    print("Combination C(5, 2):", combination(5, 2))
    print("Permutation P(5, 2):", permutation(5, 2))
    
    
    print("Logical AND (True, False):", logical_and(True, False))
    print("Logical OR (True, False):", logical_or(True, False))
    print("Logical NOT (True):", logical_not(True))
    print("Logical XOR (True, False):", logical_xor(True, False))
    a = '0101'
    b = '0011'
    bit_length = len(a)
    print("AND:", binary_and(a, b, bit_length)) 
    print("OR:", binary_or(a, b, bit_length))  
    print("NOT:", binary_not(a, bit_length))  
    print("XOR:", binary_xor(a, b, bit_length)) 
    print('Conut: ', count_substr('strawberry', 'r'))
    print(add_base(3, "101", "201"))     
    print(subtract_base(3, "111", "2"))
    print(multiply_base(3, "121", "22"))
