import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0
    
    # We want to find a set of size 12.
    # If we find one, we can stop early or use remaining time to verify.
    # Since 12 is the max, any valid 12 is optimal.
    
    best_set = []
    
    # Candidate pool: Large numbers to maximize sum range
    # Range 128-256
    candidates_base = list(range(128, 257))
    
    max_restarts = 5000
    
    for i in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        random.shuffle(candidates_base)
        
        # Try to build a set of size 12
        # Greedy selection with backtracking? Or just random subset of 12?
        # Random subset of 12 from 128-256:
        # There are C(129, 12) possibilities.
        # We can just pick 12 random distinct numbers.
        
        current_set = random.sample(candidates_base, 12)
        
        # Check validity
        if _is_valid(current_set):
            return [int(x) for x in current_set]
            
        # If we didn't find a valid set of 12, try size 11?
        # But we want to maximize.
        # We can keep a best_set found so far.
        
    # Fallback: return best set found (might be empty if none valid found in time)
    # But we should ensure we return *something* valid.
    # A single element is always valid.
    if not best_set:
        return [256]
        
    return [int(x) for x in best_set]

def _is_valid(S):
    # S is a list of distinct integers
    # Generate all subset sums
    sums = set()
    for i in range(len(S)):
        # Iterative update
        # To avoid O(2^n) list creation each time, we can just use a set
        # But we need to check divisibility.
        pass
