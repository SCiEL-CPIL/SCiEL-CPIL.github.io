import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    all_elements = set(range(2, 129))
    
    # 1. Greedy Initialization
    def greedy_construct():
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    initial_seq = greedy_construct()
    current_set = set(initial_seq)
    best_set = set(current_set)
    best_len = len(current_set)
    
    # Cache for conflict count? 
    # Since we change one element at a time, we could optimize, 
    # but full re-calc is simple and fast enough for N~40.
    
    def count_conflicts(seq):
        if not seq:
            return 0
        n = len(seq)
        prods = []
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.append(p)
        c = Counter(prods)
        conflicts = 0
        for v in c.values():
            if v > 1:
                conflicts += v * (v - 1) // 2
        return conflicts

    def get_score(seq):
        # Maximize length, minimize conflicts
        # If valid (conflicts=0), score = len
        # If invalid, score = len - 100 * conflicts
        # This encourages moving to valid states
        conf = count_conflicts(seq)
        if conf == 0:
            return len(seq)
        return len(seq) - 100 * conf

    current_seq = sorted(current_set)
    current_score = get_score(current_seq)
    
    T = 10.0
    cool_rate = 0.999
    
    while time.time() < deadline:
        # Generate neighbor
        r = random.random()
        if r < 0.5:
            # Add
            available = list(all_elements - current_set)
            if not available:
                continue
            x = random.choice(available)
            new_set = current_set | {x}
        else:
            # Remove
            if not current_set:
                continue
            x = random.choice(list(current_set))
            new_set = current_set - {x}
            
        new_seq = sorted(new_set)
        new_score = get_score(new_seq)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_set = new_set
            current_seq = new_seq
            current_score = new_score
            
            # Check if we found a new best valid solution
            if count_conflicts(current_seq) == 0:
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_set = set(current_set)
                    
        T *= cool_rate
        
        # If T gets too low, restart with higher T if we are stuck?
        # Or just let it run. 
        # Simple restart strategy: if no improvement for a while, reset T?
        # For simplicity, we just keep cooling. 
        # If we hit a valid state, we stay there usually.
        
        # Optional: If we are in a valid state, bias moves towards adding
        # This is handled by the score function implicitly (adding increases len, 
        # but might increase conflicts, penalized heavily).
        
    # Return the best valid sequence found
    return sorted(best_set)

# EVOLVE-BLOCK-END
