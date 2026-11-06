import itertools
import random

def is_sidon(subset):
    """
    Check if a subset is a Sidon set.
    A Sidon set requires all unordered pair sums (including diagonals a+a) to be unique.
    """
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def has_sidon_4(S):
    """
    Check if S contains any Sidon subset of size 4.
    """
    for combo in itertools.combinations(S, 4):
        if is_sidon(combo):
            return True
    return False

def count_sidon_4(S):
    """
    Count the number of Sidon subsets of size 4 in S.
    """
    count = 0
    for combo in itertools.combinations(S, 4):
        if is_sidon(combo):
            count += 1
    return count

def construct():
    # Goal: Find a set of 12 distinct integers in [0, 128]
    # such that the largest Sidon subset has size 3.
    # This is equivalent to finding a set with NO Sidon subset of size 4.
    
    rng = random.Random(42)
    
    # Generate a pool of candidates
    candidates = []
    
    # 1. Dense Arithmetic Progressions
    # APs have high additive energy, leading to many sum collisions.
    for start in range(0, 15):
        for step in range(1, 5):
            S = [start + i * step for i in range(12)]
            if max(S) <= 128:
                candidates.append(S)
                
    # 2. Random dense sets in small ranges
    # Small ranges force sum collisions.
    for _ in range(5000):
        # Sample 12 distinct integers from [0, 20]
        S = sorted(rng.sample(range(0, 21), 12))
        candidates.append(S)
        
    # 3. Unions of small clusters
    # e.g., {0,1,2} U {3,4,5} ... with perturbations
    for _ in range(5000):
        S = []
        base = 0
        for i in range(4):
            # Add 3 numbers in a cluster
            start = base + rng.randint(0, 2)
            S.extend([start + rng.randint(0, 1) for _ in range(3)])
            base += 4 + rng.randint(0, 2)
        
        # Ensure distinctness and range
        S = sorted(set(S))
        if len(S) < 12:
            # Fill with random small numbers
            while len(S) < 12:
                S.append(rng.randint(0, 20))
            S = sorted(set(S))
        if len(S) == 12 and max(S) <= 128:
            candidates.append(S)

    # Check candidates for absence of Sidon 4-subsets
    for S in candidates:
        if not has_sidon_4(S):
            return S

    # If no perfect candidate found, use local search to minimize Sidon 4-count
    # starting from the best candidate found so far.
    
    best_S = None
    min_count = float('inf')
    
    for S in candidates:
        c = count_sidon_4(S)
        if c < min_count:
            min_count = c
            best_S = S[:]
            
    if min_count == 0:
        return best_S
        
    # Local Search
    S = best_S[:]
    for _ in range(20000):
        idx = rng.randint(0, 11)
        old_val = S[idx]
        
        # Try to replace with a value in a dense region
        new_val = rng.randint(0, 25)
        
        if new_val in S and new_val != old_val:
            continue
            
        S[idx] = new_val
        S.sort()
        
        c = count_sidon_4(S)
        
        if c < min_count:
            min_count = c
            best_S = S[:]
            if min_count == 0:
                return best_S
        else:
            # Rollback
            S[idx] = old_val
            S.sort()
            
    return best_S
