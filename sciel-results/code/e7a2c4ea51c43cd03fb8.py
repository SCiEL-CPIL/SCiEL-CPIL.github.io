"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools

def max_sidon_subset_brute(s):
    """
    Compute the largest Sidon subset of s.
    Since len(s) is small (12), we can check all subsets.
    A set is Sidon if all sums a+b (a<=b) are unique.
    """
    s = sorted(s)
    n = len(s)
    max_size = 0
    
    # Iterate through all subsets by size, descending
    # We can stop early if we find a Sidon subset of size k, 
    # but we need the MAXIMUM, so we check from n down to 1.
    # However, checking all subsets of size k is C(n, k).
    # Sum of C(12, k) is 4096.
    # For each subset, check if Sidon: O(k^2).
    
    # Optimization: If a subset of size k is Sidon, any subset of it is Sidon.
    # We want the largest k.
    
    # Simple approach: Check all subsets.
    # To be faster, we can use bitmasks or recursion.
    # Given N=12, simple iteration is fine.
    
    # We'll check subsets of size n, n-1, ..., 1
    # and return the first one found to be Sidon.
    
    for k in range(n, 0, -1):
        found = False
        for subset_indices in itertools.combinations(range(n), k):
            subset = [s[i] for i in subset_indices]
            # Check if subset is Sidon
            sums = set()
            is_sidon = True
            for i in range(k):
                for j in range(i, k):
                    val = subset[i] + subset[j]
                    if val in sums:
                        is_sidon = False
                        break
                    sums.add(val)
            if not is_sidon:
                continue
            
            # If we reach here, subset is Sidon
            max_size = k
            found = True
            break # No need to check other subsets of this size
        
        if found:
            return max_size
            
    return 0

def construct():
    random.seed(42)
    
    # Strategy: 
    # 1. Generate diverse candidate sets.
    #    - Random sets
    #    - Clustered sets (many small numbers)
    #    - Arithmetic progressions
    # 2. Evaluate each with max_sidon_subset_brute.
    # 3. Keep the best.
    
    best_set = None
    best_val = 12 # Max possible is 12
    
    start_time = time.time()
    timeout = 50
    
    # Initial candidates
    candidates = []
    
    # 1. Random sets
    for _ in range(50):
        c = random.sample(range(129), 12)
        candidates.append(c)
        
    # 2. Clustered sets (e.g., two clusters of 6)
    for _ in range(50):
        c1 = random.sample(range(0, 40), 6)
        c2 = random.sample(range(40, 129), 6)
        c = c1 + c2
        if len(set(c)) == 12:
            candidates.append(c)
            
    # 3. Arithmetic progressions
    for _ in range(20):
        a = random.randint(0, 50)
        d = random.randint(1, 10)
        c = [a + i*d for i in range(12)]
        if max(c) < 129 and len(set(c)) == 12:
            candidates.append(c)
            
    # 4. "Dense" low number sets
    for _ in range(50):
        c = random.sample(range(0, 30), 12)
        candidates.append(c)
        
    # 5. Specific structured sets known to have collisions
    # e.g. {0,1,2,3,4,5, 10,11,12,13,14,15}
    candidates.append([0,1,2,3,4,5, 10,11,12,13,14,15])
    candidates.append([0,1,2,3,4,5, 6,7,8,9,10,11])
    candidates.append([0, 2, 4, 6, 8, 10, 20, 22, 24, 26, 28, 30])

    # Evaluate initial candidates
    for c in candidates:
        if len(set(c)) != 12:
            continue
        val = max_sidon_subset_brute(c)
        if val < best_val:
            best_val = val
            best_set = c[:]
            
    # Local search around the best
    current = best_set[:]
    current_val = best_val
    
    while time.time() - start_time < timeout:
        candidate = current[:]
        # Random perturbation
        for _ in range(random.randint(1, 3)):
            i = random.randint(0, 11)
            # Try to move element i to a nearby value or random
            if random.random() < 0.5:
                delta = random.randint(-5, 5)
                new_val = candidate[i] + delta
            else:
                new_val = random.randint(0, 128)
            
            if 0 <= new_val <= 128 and new_val not in candidate:
                candidate[i] = new_val
        
        if len(set(candidate)) != 12:
            continue
            
        val = max_sidon_subset_brute(candidate)
        
        if val < current_val:
            current = candidate
            current_val = val
            if val < best_val:
                best_val = val
                best_set = candidate[:]
        elif val == current_val and random.random() < 0.1:
            current = candidate
            
    return best_set
# EVOLVE-BLOCK-END
