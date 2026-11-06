"""Rewritten program: local search to maximize distinct 3-AP differences for 32 integers in 1..512."""
import time
import random
import numpy as np

def _score(S):
    """Calculate the number of distinct d > 0 such that a, a+d, a+2d in S."""
    S_set = set(S)
    valid_ds = set()
    elems = list(S)
    n = len(elems)
    for i in range(n):
        x = elems[i]
        for j in range(i + 1, n):
            y = elems[j]
            d = y - x
            # Check if x-d is in S (then x-d, x, y form a 3-AP with diff d)
            # or y+d is in S (then x, y, y+d form a 3-AP with diff d)
            if (x - d) in S_set or (y + d) in S_set:
                valid_ds.add(d)
    return len(valid_ds)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Start with a random set of 32 distinct integers in 1..512
    rng = random.Random(42)
    
    def random_set():
        return set(rng.sample(range(1, 513), 32))
    
    best_set = random_set()
    best_score = _score(best_set)
    
    # Local search: try moving one element to a new random position
    while time.time() < deadline:
        # Try several random moves
        for _ in range(50):
            if time.time() >= deadline:
                break
            new_set = set(best_set)
            # Remove a random element
            to_remove = rng.choice(list(new_set))
            new_set.remove(to_remove)
            # Add a new random element not in new_set
            available = [x for x in range(1, 513) if x not in new_set]
            new_elem = rng.choice(available)
            new_set.add(new_elem)
            
            new_score = _score(new_set)
            if new_score > best_score:
                best_set = new_set
                best_score = new_score
    
    return list(best_set)
