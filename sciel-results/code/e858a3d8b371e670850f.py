import random
import time
import sys
from itertools import combinations

def _is_b2_subset(subset):
    """Check if all unordered pair sums (including diagonals) are unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_b2_size(ambient):
    """Exhaustively compute the largest subset with all unordered pair sums unique."""
    n = len(ambient)
    # Precompute all pair sums for the ambient set
    # We want the largest subset where all pair sums (i,j) with i<=j are unique
    
    # Use backtracking with pruning
    best_size = 0
    used_sums = set()
    
    def backtrack(idx, current_size, current_sums):
        nonlocal best_size
        if current_size > best_size:
            best_size = current_size
        
        # Pruning: even if we take all remaining, can't beat best
        if current_size + (n - idx) <= best_size:
            return
        
        for i in range(idx, n):
            val = ambient[i]
            # Check if adding val creates any duplicate sum
            conflict = False
            new_sums_to_add = []
            for j in range(len(ambient[:i])):
                # Wait, we need to check against all elements already in the subset
                # Actually, we need to track which elements are in the subset
                pass
            
            # Let me redo this with explicit subset tracking
            pass
    
    # Simpler approach: iterate over all subsets in decreasing size order
    # For n=12, total subsets = 2^12 = 4096, very manageable
    for size in range(n, 0, -1):
        found = False
        for combo in combinations(range(n), size):
            subset = [ambient[i] for i in combo]
            if _is_b2_subset(subset):
                found = True
                break
        if found:
            return size
    return 1

def construct():
    """Return exactly 12 distinct integers in [0,128] that minimize the largest B2 subset size."""
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # The ambient set [0,11] has max B2 subset size... let's think about it.
    # For [0,1,2,...,11], pair sums range from 0 to 22.
    # The largest Sidon (B2) subset of [0,11]: 
    # A Sidon set has all pair sums unique.
    # Known: in [0,n], max Sidon set size is about sqrt(n).
    # For [0,11], we can check: {0,1,3,7,11}? Let's verify.
    # Actually, let's just compute it.
    
    # Strategy: search for ambient sets that minimize the max B2 subset size.
    # The minimum possible is 1 (if no two elements have same pair sum structure) 
    # but with 12 elements, we need at least some structure.
    # Actually, for any 12 distinct integers, the max B2 subset is at least... 
    # By Ramsey-type arguments, it should be at least 3 or 4.
    
    # Let's try random search with scoring
    best_score = 13  # worse than any real value
    best_set = None
    rng = random.Random(42)
    
    # First, evaluate the default [0,11]
    default_set = list(range(12))
    default_score = _max_b2_size(default_set)
    best_score = default_score
    best_set = default_set[:]
    
    # Try some structured sets
    # Sets with large gaps to break Sidon structure
    # A B2 set requires no four elements a,b,c,d with a+b=c+d (a,b,c,d not necessarily distinct, but pairs unordered with replacement)
    # So we want sets where many 4-tuples satisfy a+b=c+d
    
    # Try sets based on powers or exponential
    # Actually, let's try random sets and evaluate
    
    candidates = [
        list(range(12)),
        [0, 1, 2, 4, 8, 16, 32, 64, 96, 120, 126, 128],
        [0, 2, 4, 8, 16, 32, 64, 96, 112, 120, 126, 128],
        [0, 1, 3, 7, 15, 31, 63, 95, 119, 127, 128, 126],
    ]
    
    for cand in candidates:
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        score = _max_b2_size(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
        if best_score == 1:
            break
        if time.time() > deadline:
            break
    
    # Random search
    attempts = 0
    while time.time() < deadline:
        # Generate random 12-element set in [0,128]
        cand = sorted(rng.sample(range(129), 12))
        score = _max_b2_size(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            if best_score == 1:
                break
        attempts += 1
    
    return best_set

if __name__ == "__main__":
    result = construct()
    print(result)
    print(f"Max B2 size: {_max_b2_size(result)}")
