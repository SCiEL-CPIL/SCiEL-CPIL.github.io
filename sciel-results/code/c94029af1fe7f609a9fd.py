"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random

# EVOLVE-BLOCK-START

def _is_sidon4(subset):
    """Check if a 4-element tuple has all pair sums (i<=j) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _has_sidon4(arr):
    """Check if any 4-subset of arr is Sidon. Returns True if exists."""
    for sub4 in itertools.combinations(arr, 4):
        if _is_sidon4(sub4):
            return True
    return False

def _max_sidon_size(candidates):
    """Find the largest Sidon subset size of the given 12-element set."""
    for size in range(12, 0, -1):
        for combo in itertools.combinations(candidates, size):
            n = len(combo)
            sums = set()
            ok = True
            for i in range(n):
                for j in range(i, n):
                    s = combo[i] + combo[j]
                    if s in sums:
                        ok = False
                        break
                    sums.add(s)
                if not ok:
                    break
            if ok:
                return size
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    best_set = list(range(12))
    best_size = _max_sidon_size(best_set)  # should be 4
    
    # If we can find a set with no Sidon 4-subset, max size <= 3
    # Fast check: just look for any Sidon 4-subset
    
    # Strategy 1: Structured sets - dense clusters
    # Consecutive integers in a small range
    for start in range(0, 118):
        if time.time() > deadline:
            break
        candidates = list(range(start, start + 12))
        if max(candidates) > 128:
            break
        if not _has_sidon4(candidates):
            # Verify full max Sidon size
            ms = _max_sidon_size(candidates)
            if ms < best_size:
                best_size = ms
                best_set = candidates
                if best_size <= 3:
                    return best_set
    
    # Strategy 2: Arithmetic progressions
    for d in range(1, 12):
        for a in range(0, 129 - 11*d):
            if time.time() > deadline:
                break
            candidates = [a + i*d for i in range(12)]
            if max(candidates) > 128:
                continue
            if not _has_sidon4(candidates):
                ms = _max_sidon_size(candidates)
                if ms < best_size:
                    best_size = ms
                    best_set = candidates
                    if best_size <= 3:
                        return best_set
    
    # Strategy 3: Random search with fast Sidon-4 check
    random.seed(42)
    for attempt in range(20000):
        if time.time() > deadline:
            break
        candidates = sorted(random.sample(range(129), 12))
        # Fast check: does it have any Sidon 4-subset?
        if not _has_sidon4(candidates):
            ms = _max_sidon_size(candidates)
            if ms < best_size:
                best_size = ms
                best_set = candidates
                if best_size <= 3:
                    return best_set
    
    # Strategy 4: Local search starting from best found
    if best_size > 3:
        current_set = best_set[:]
        current_has_s4 = _has_sidon4(current_set)
        
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Pick a random element to replace
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            
            # Try several replacements
            found_improvement = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                new_val = random.randint(0, 128)
                if new_val in current_set:
                    continue
                
                neighbor = current_set[:]
                neighbor[idx] = new_val
                neighbor.sort()
                
                # Fast check
                if not _has_sidon4(neighbor):
                    ms = _max_sidon_size(neighbor)
                    if ms < best_size:
                        best_size = ms
                        best_set = neighbor[:]
                        if best_size <= 3:
                            return best_set
                    current_set = neighbor
                    found_improvement = True
                    break
            
            # Occasionally try swapping two elements
            if not found_improvement and random.random() < 0.3:
                idx2 = random.randint(0, 11)
                if idx2 != idx:
                    neighbor = current_set[:]
                    neighbor[idx], neighbor[idx2] = neighbor[idx2], neighbor[idx]
                    neighbor.sort()
                    if not _has_sidon4(neighbor):
                        ms = _max_sidon_size(neighbor)
                        if ms < best_size:
                            best_size = ms
                            best_set = neighbor[:]
                            if best_size <= 3:
                                return best_set
                        current_set = neighbor
    
    # Strategy 5: Targeted construction - try to make all 4-subsets non-Sidon
    # Use sets with many elements in pairs (duplicates in sum space)
    # e.g., {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14}
    structured_candidates = [
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 2, 4, 5, 7, 9, 10, 12, 14, 15, 17, 18],
        [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13],
        [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17],
        [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21],
        [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27],
    ]
    
    for cand in structured_candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) == 12 and max(cand) <= 128:
            if not _has_sidon4(cand):
                ms = _max_sidon_size(cand)
                if ms < best_size:
                    best_size = ms
                    best_set = cand[:]
                    if best_size <= 3:
                        return best_set
    
    # Strategy 6: More aggressive random search with bias toward dense sets
    for attempt in range(50000):
        if time.time() > deadline:
            break
        # Bias toward small values
        max_val = random.choice([20, 30, 40, 50, 80, 128])
        candidates = sorted(random.sample(range(max_val + 1), 12))
        if not _has_sidon4(candidates):
            ms = _max_sidon_size(candidates)
            if ms < best_size:
                best_size = ms
                best_set = candidates
                if best_size <= 3:
                    return best_set
    
    return best_set

# EVOLVE-BLOCK-END
