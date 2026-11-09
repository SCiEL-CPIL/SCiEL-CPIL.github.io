import time
import random

def _greedy_construction(order):
    selected = []
    reachable = 1  # sum 0
    mask = (1 << 201) - 1
    
    for x in order:
        # Check if 200-x is reachable
        if 200 - x >= 0 and ((reachable >> (200 - x)) & 1):
            continue
        
        selected.append(x)
        reachable = (reachable | (reachable << x)) & mask
        
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_set = []
    best_size = 0
    
    # Deterministic orders
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 49, -1)), # Middle out?
    ]
    
    for order in orders:
        current_set = _greedy_construction(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    # Random restarts
    while time.time() < deadline:
        order = list(range(1, 101))
        random.shuffle(order)
        current_set = _greedy_construction(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return best_set
