import time
import random
from itertools import combinations

def is_sidon(S):
    """Check if a list of integers is a Sidon set."""
    sums = set()
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sidon_5(S):
    """Count the number of 5-element subsets that are Sidon."""
    count = 0
    for subset in combinations(S, 5):
        if is_sidon(subset):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Start with dense set
    S = list(range(12))
    
    # Check if it already works
    if count_sidon_5(S) == 0:
        return S
        
    # Local search
    best_S = S[:]
    best_count = count_sidon_5(S)
    
    while time.time() < deadline:
        improved = False
        # Try to improve by swapping
        for _ in range(1000): # Inner loop
            if time.time() > deadline:
                break
            if best_count == 0:
                return best_S
                
            i = random.randint(0, 11)
            old_val = S[i]
            # Try a new value
            # Focus on small range to keep density
            new_val = random.randint(0, 25)
            
            if new_val == old_val:
                continue
            if new_val in S:
                continue
                
            # Apply swap
            S[i] = new_val
            new_count = count_sidon_5(S)
            
            if new_count < best_count:
                best_count = new_count
                best_S = S[:]
                improved = True
                if new_count == 0:
                    return best_S
            else:
                # Revert
                S[i] = old_val
                
        if not improved:
            # Restart with random dense set
            S = random.sample(range(0, 21), 12)
            S.sort()
            new_count = count_sidon_5(S)
            if new_count < best_count:
                best_count = new_count
                best_S = S[:]
                if new_count == 0:
                    return best_S

    return best_S
