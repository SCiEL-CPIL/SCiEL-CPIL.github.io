import random
import time
from fractions import Fraction

def _check_distinct_sums(subset):
    """Check if all reciprocal subset sums are distinct."""
    if not subset:
        return True
    sums = {Fraction(0)}
    for x in subset:
        rec = Fraction(1, x)
        new_sums = set()
        for s in sums:
            new_sums.add(s + rec)
        if len(new_sums) != len(sums):
            return False
        sums = new_sums
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Seed solution
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Elements
    elements = list(range(1, 23))
    reciprocals = {x: Fraction(1, x) for x in elements}
    
    def _try_greedy(order):
        """Greedy build given a specific order of elements."""
        current_set = []
        current_sums = {Fraction(0)}
        for x in order:
            if x in current_set:
                continue
            rec = reciprocals[x]
            added_sums = {s + rec for s in current_sums}
            if not added_sums.intersection(current_sums):
                current_set.append(x)
                current_sums = current_sums.union(added_sums)
        return current_set

    # 1. Try deterministic orders
    # Descending index (smallest reciprocals first)
    order_desc = list(range(22, 0, -1))
    set_desc = _try_greedy(order_desc)
    if len(set_desc) > best_size:
        best_set = set_desc
        best_size = len(set_desc)
        
    # Ascending index (largest reciprocals first)
    order_asc = list(range(1, 23))
    set_asc = _try_greedy(order_asc)
    if len(set_asc) > best_size:
        best_set = set_asc
        best_size = len(set_asc)
        
    # 2. Randomized search
    # We know best_size is at least 9.
    # We want to find larger.
    # We don't need to iterate K down from 22 if we just greedily build and take the result.
    # The greedy build will naturally stop when it can't add more.
    # So we just run many randomized greedy builds and keep the longest.
    
    attempts = 0
    max_attempts = 100000
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        pool = elements[:]
        random.shuffle(pool)
        
        current_set = []
        current_sums = {Fraction(0)}
        
        for x in pool:
            rec = reciprocals[x]
            added_sums = {s + rec for s in current_sums}
            if not added_sums.intersection(current_sums):
                current_set.append(x)
                current_sums = current_sums.union(added_sums)
                
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
            # If we reach a very high size, we can stop or continue to see if we can go higher?
            # The max possible is 22.
            # If we find 16, it's good. If we find 17, even better.
            # We should continue to try to beat it.
            
    return best_set

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Seed solution
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Elements
    elements = list(range(1, 23))
    reciprocals = {x: Fraction(1, x) for x in elements}
    
    def _try_greedy(order):
        """Greedy build given a specific order of elements."""
        current_set = []
        current_sums = {Fraction(0)}
        for x in order:
            rec = reciprocals[x]
            added_sums = {s + rec for s in current_sums}
            if not added_sums.intersection(current_sums):
                current_set.append(x)
                current_sums = current_sums.union(added_sums)
        return current_set

    # 1. Try deterministic orders
    # Descending index (smallest reciprocals first)
    order_desc = list(range(22, 0, -1))
    set_desc = _try_greedy(order_desc)
    if len(set_desc) > best_size:
        best_set = set_desc
        best_size = len(set_desc)
        
    # Ascending index (largest reciprocals first)
    order_asc = list(range(1, 23))
    set_asc = _try_greedy(order_asc)
    if len(set_asc) > best_size:
        best_set = set_asc
        best_size = len(set_asc)
        
    # 2. Randomized search
    attempts = 0
    max_attempts = 100000
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        pool = elements[:]
        random.shuffle(pool)
        
        current_set = []
        current_sums = {Fraction(0)}
        
        for x in pool:
            rec = reciprocals[x]
            added_sums = {s + rec for s in current_sums}
            if not added_sums.intersection(current_sums):
                current_set.append(x)
                current_sums = current_sums.union(added_sums)
                
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
    return best_set
# EVOLVE-BLOCK-END
