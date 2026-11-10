import time
import random
import math
from itertools import combinations

def _score(A):
    """Compute max(|A+A|, |A*A|) for set A (list of distinct ints)."""
    n = len(A)
    sumset = set()
    prodset = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sumset.add(ai + A[j])
            prodset.add(ai * A[j])
    return max(len(sumset), len(prodset))

def _valid(A, max_val=10000):
    """Check if A is valid: 20 distinct positive integers <= max_val."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 for x in A):
        return False
    if any(x > max_val for x in A):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin to be safe against 60s/90s limits
    
    # Initial set: 1 to 20
    best_A = list(range(1, 21))
    best_score = _score(best_A)
    
    # Try some structured initial sets
    # Geometric-like or arithmetic
    candidate_sets = [
        list(range(1, 21)),          
        list(range(2, 22)),          
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 3, 6, 12, 24, 48, 96], # Mix
    ]
    
    for cs in candidate_sets:
        if _valid(cs):
            s = _score(cs)
            if s < best_score:
                best_score = s
                best_A = list(cs)
    
    random.seed(42)
    
    # Fast local search: random perturbations with small steps
    # We will perform many small iterations instead of few large ones
    
    current_A = list(best_A)
    current_score = best_score
    
    # Optimization: Pre-compute some stats? Not really.
    # We'll just run a loop that checks time frequently.
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Randomly pick an index to change
        i = random.randint(0, 19)
        old_val = current_A[i]
        
        # Choose a strategy: small step or large jump
        if random.random() < 0.7:
            # Small step
            delta = random.randint(-10, 10)
            new_val = old_val + delta
        else:
            # Large jump
            new_val = random.randint(1, 10000)
            
        if new_val < 1 or new_val > 10000:
            continue
            
        # Check distinctness efficiently
        # Since N=20, checking `new_val in current_A` is O(N)
        if new_val in current_A:
            continue
            
        new_A = current_A.copy()
        new_A[i] = new_val
        
        s = _score(new_A)
        
        if s < current_score:
            current_A = new_A
            current_score = s
            if current_score < best_score:
                best_score = current_score
                best_A = list(current_A)
        else:
            # Accept with some probability (simulated annealing style) to escape local minima?
            # Given time constraints, simple hill climbing with random restarts is safer.
            # Let's just keep the current state.
            pass
            
        # Check time inside the loop to break out immediately
        if time.time() > deadline:
            break
            
    return best_A
