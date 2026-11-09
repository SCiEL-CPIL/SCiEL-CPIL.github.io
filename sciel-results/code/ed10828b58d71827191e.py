import random
import time
from itertools import combinations

def largest_sidon_subset(S):
    """Find largest subset of S with all pair sums (i<=j) unique."""
    S = sorted(S)
    n = len(S)
    
    # Quick check: if n is small, brute force
    for k in range(n, 1, -1):
        for subset in combinations(S, k):
            sums = set()
            valid = True
            for i in range(len(subset)):
                for j in range(i, len(subset)):
                    s = subset[i] + subset[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                return k
    return 2

def construct():
    # We want to minimize the largest Sidon subset size.
    # A set where every 5-element subset has a repeated pair sum is ideal (max Sidon = 4).
    # Or even better, max Sidon = 3.
    
    # Strategy: Use a dense set of small numbers.
    # If we take {0, 1, 2, ..., 11}, what is the largest Sidon subset?
    # In {0..11}, a Sidon set of size 5?
    # 0,1,3,7,11 -> sums: 0,1,3,7,11, 2,4,8,12, 6,10,14, 10,14,18 -> 10 repeats (3+7=10, 1+9? no 9 not in set. 0+10? no. 7+3=10. 1+9? no. 2+8? no. 4+6? no. 5+5=10. 5 is not in set. Wait.
    # Let's check 0,1,3,7,11:
    # Pairs:
    # 0+0=0, 0+1=1, 0+3=3, 0+7=7, 0+11=11
    # 1+1=2, 1+3=4, 1+7=8, 1+11=12
    # 3+3=6, 3+7=10, 3+11=14
    # 7+7=14, 7+11=18
    # 11+11=22
    # Duplicate: 14 (3+11 and 7+7). So not Sidon.
    
    # Actually, it is known that the largest Sidon subset of {0, ..., n-1} is about sqrt(n).
    # For n=12, sqrt(12) ~ 3.46. So likely max Sidon size is 4 or 5.
    # Let's just compute it for {0..11}.
    
    # But we can do better. We want to force collisions.
    # Consider a set composed of two interleaved arithmetic progressions with small differences.
    # Or simply a set of consecutive integers.
    
    # Let's try to optimize locally from a dense start.
    # Start with {0, 1, ..., 11}.
    # Calculate current max Sidon size.
    
    best_set = list(range(12))
    best_size = largest_sidon_subset(best_set)
    
    # If best_size is already 3, we are done.
    # If it's 4, we try to find a set with size 3.
    # If it's 5, we try to find a set with size 4.
    
    # Let's try a few specific structured sets that are known to have many additive collisions.
    # Sets with small diameter are good candidates.
    
    candidates = []
    
    # 1. Consecutive integers
    for start in range(0, 117):
        S = [start + i for i in range(12)]
        candidates.append(S)
        
    # 2. Two interleaved APs
    # e.g., 0, 2, 4, ..., 10 and 1, 3, 5, ..., 11 (this is just 0..11)
    # Try: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    # Try: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12 (skip 11)
    
    # 3. Random sets with small values
    random.seed(123)
    for _ in range(2000):
        # Sample 12 distinct values from [0, 20] to force collisions
        if len(range(21)) >= 12:
            S = random.sample(range(21), 12)
            candidates.append(S)
            
    for _ in range(2000):
        # Sample 12 distinct values from [0, 15]
        if len(range(16)) >= 12:
            S = random.sample(range(16), 12)
            candidates.append(S)

    # Evaluate candidates
    for S in candidates:
        size = largest_sidon_subset(S)
        if size < best_size:
            best_size = size
            best_set = S
            if best_size == 3:
                # 3 is the theoretical minimum for 12 distinct integers?
                # Can we have max Sidon size 2?
                # That would mean every 3-element subset has a repeated pair sum.
                # For any 3 distinct ints a<b<c, a+b = a+c => b=c (impossible)
                # a+b = b+c => a=c (impossible)
                # a+c = b+c => a=b (impossible)
                # a+a = b+b => a=b
                # a+a = b+c?
                # If we require ALL 3-subsets to be non-Sidon, then max Sidon is 2.
                # Is it possible to have 12 integers where every triple is non-Sidon?
                # This is a strong condition.
                # Let's check if best_size can be 2.
                # If we find a set with max Sidon 2, we return it immediately.
                return best_set

    # If we didn't find 3, try local search to improve the best set found so far.
    # We want to minimize largest_sidon_subset.
    
    current_set = best_set[:]
    current_size = best_size
    
    start_time = time.time()
    max_time = 50.0
    
    while (time.time() - start_time) < max_time:
        improved = False
        # Try swapping one element with a random value
        for _ in range(100):
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            new_val = random.randint(0, 128)
            
            if new_val == old_val:
                continue
            if new_val in current_set:
                continue
                
            current_set[idx] = new_val
            new_size = largest_sidon_subset(current_set)
            
            if new_size < current_size:
                current_size = new_size
                if current_size < best_size:
                    best_size = current_size
                    best_set = current_set[:]
                    if best_size == 3:
                        return best_set
                improved = True
            else:
                current_set[idx] = old_val
                
        if not improved:
            # Try more aggressive moves: swap two elements
            for _ in range(50):
                i, j = random.sample(range(12), 2)
                vi, vj = current_set[i], current_set[j]
                
                # Try replacing vi with a random small value
                new_vi = random.randint(0, 20)
                if new_vi in current_set and new_vi != vi:
                    continue
                if new_vi == vj:
                    continue
                
                current_set[i] = new_vi
                new_size = largest_sidon_subset(current_set)
                
                if new_size < current_size:
                    current_size = new_size
                    if current_size < best_size:
                        best_size = current_size
                        best_set = current_set[:]
                        if best_size == 3:
                            return best_set
                else:
                    current_set[i] = vi
                    
    return best_set
