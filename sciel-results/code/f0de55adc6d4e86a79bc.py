import random
import time
from math import gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    # Use 80s deadline to be safe within 90s limit
    deadline = start_time + 80.0
    
    max_n = 22
    numbers = list(range(1, max_n + 1))
    
    # Calculate LCM of 1..22
    # This is constant for the problem
    L_max = reduce(lcm, numbers, 1)
    
    # Precompute weights w_x = L_max / x
    weights = {x: L_max // x for x in numbers}
    
    best_set = []
    best_size = 0
    
    # To ensure we have a valid result even if random fails immediately
    # Try a few deterministic orders first
    deterministic_orders = [
        numbers[::-1], # Descending
        numbers,       # Ascending
        [21, 22, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] # Mixed
    ]
    
    for order in deterministic_orders:
        current_sums = {0}
        selected = []
        
        for x in order:
            w = weights[x]
            new_sums = {s + w for s in current_sums}
            
            if current_sums.isdisjoint(new_sums):
                selected.append(x)
                current_sums |= new_sums
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set = selected[:]
                    if best_size == max_n:
                        return best_set
                        
    # Randomized search
    # We can run many iterations
    while time.time() < deadline:
        order = numbers[:]
        random.shuffle(order)
        
        current_sums = {0}
        selected = []
        
        for x in order:
            w = weights[x]
            new_sums = {s + w for s in current_sums}
            
            if current_sums.isdisjoint(new_sums):
                selected.append(x)
                current_sums |= new_sums
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set = selected[:]
                    if best_size == max_n:
                        return best_set
                        
    # Return the best set found
    return best_set if best_set else [1]
