import time
import random
from itertools import combinations

def is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def has_sidon_k(A, k):
    # Check if any k-element subset is Sidon
    # Optimization: sort A? Not strictly necessary for small N.
    for comb in combinations(A, k):
        if is_sidon(list(comb)):
            return True
    return False

def max_sidon_size(A):
    # Check from high to low to find the maximum size
    # Max possible is 12.
    # We want to minimize this, so we check small k first to see if it's possible to be small?
    # No, max_sidon_size returns the MAXIMUM size.
    # If has_sidon_k(A, 4) is False, then max size <= 3.
    # If has_sidon_k(A, 4) is True, max size >= 4.
    
    # To minimize the return value, we want to return 3, 4, 5...
    # The evaluator wants the MINIMUM of the MAXIMUM Sidon subset size.
    # So we want a set where the largest Sidon subset is as small as possible.
    # If we can make it so that NO 4-set is Sidon, the answer is 3.
    # If we can't, but some 4-sets are Sidon and no 5-sets are, the answer is 4.
    
    # Strategy:
    # 1. Check if any 4-set is Sidon. If not, return 3.
    # 2. If yes, check if any 5-set is Sidon. If not, return 4.
    # 3. If yes, check if any 6-set is Sidon. If not, return 5.
    # 4. If yes, return 6 (or higher, but 6 is likely the floor for random sets in small ranges? 
    #    Actually, in [0, 25], it's hard to avoid 4-sets? 
    #    Let's check [0, 11]. 
    #    {0, 1, 2, 3} is not Sidon (0+2=1+1).
    #    {0, 1, 2, 4}? 0+4=4, 1+3(no 3), 2+2=4. Collision.
    #    It seems possible to have max Sidon size 3 or 4.
    
    if not has_sidon_k(A, 4):
        return 3
    
    if not has_sidon_k(A, 5):
        return 4
        
    if not has_sidon_k(A, 6):
        return 5
        
    if not has_sidon_k(A, 7):
        return 6
        
    # Fallback
    return 7 # Placeholder for >6

def construct():
    start_time = time.time()
    deadline = start_time + 50 # Safety margin
    
    best_A = None
    best_score = 12 # Max possible is 12, so any valid set is better
    
    # Focus on small range [0, 25] as per knowledge
    search_range = 26
    
    # Try a few random restarts in the small range
    # Also try the specific "dense" initial guess
    initial_candidates = [
        list(range(12)),
        list(range(0, 12, 1)), # same
    ]
    
    # Generate some random candidates in [0, 25]
    for _ in range(50):
        cand = random.sample(range(search_range), 12)
        initial_candidates.append(cand)
        
    # Evaluate initial candidates
    for A in initial_candidates:
        if len(set(A)) != 12:
            continue
        score = max_sidon_size(A)
        if score < best_score:
            best_score = score
            best_A = A[:]
            if best_score == 3:
                return best_A
                
    # Local Search
    # We want to minimize best_score.
    # If best_score is 4, we try to get 3.
    # If best_score is 5, we try to get 4.
    
    A = best_A[:]
    
    # To speed up local search, we can track the "bad" subsets.
    # But for simplicity, we just evaluate.
    
    # We will perform local search on the best candidate found so far.
    # If we find a 3, we return immediately.
    
    while time.time() < deadline:
        idx = random.randint(0, 11)
        old_val = A[idx]
        
        # Try moving to a value in the small range
        # To encourage collisions, pick a value close to existing ones?
        # Or just random in [0, 25].
        new_val = random.randint(0, search_range - 1)
        
        if new_val in A:
            continue
            
        A[idx] = new_val
        
        current_score = max_sidon_size(A)
        
        if current_score < best_score:
            best_score = current_score
            best_A = A[:]
            if best_score == 3:
                return best_A
        else:
            # Revert if worse or equal?
            # If equal, we might want to keep it to explore?
            # But reverting is safer to maintain a good baseline.
            A[idx] = old_val
            
    return best_A
