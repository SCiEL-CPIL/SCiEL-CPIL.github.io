"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter
from itertools import combinations

def _count_duplicate_sums(subset):
    """Count how many unordered pair sums (including diagonals) are duplicated."""
    n = len(subset)
    sums = [subset[i] + subset[j] for i in range(n) for j in range(i, n)]
    c = Counter(sums)
    return sum(k * (k - 1) // 2 for k in c.values())

def _is_sidon_subset(vals):
    """Check if a subset has all unordered pair sums unique (including diagonals)."""
    s = len(vals)
    pair_sums = set()
    for i in range(s):
        for j in range(i, s):
            ps = vals[i] + vals[j]
            if ps in pair_sums:
                return False
            pair_sums.add(ps)
    return True

def _max_sidon_subset_size_fast(subset, deadline=None):
    """
    Find largest subset with all unordered pair sums unique (including diagonals).
    Uses greedy + backtracking with pruning for n=12.
    """
    n = len(subset)
    best = 0
    
    # Sort by value to help with pruning
    indices = list(range(n))
    
    # Try sizes from n down to 1
    for size in range(n, 0, -1):
        if deadline and time.time() > deadline:
            break
        
        found = False
        for combo in combinations(indices, size):
            if deadline and time.time() > deadline:
                break
            vals = [subset[i] for i in combo]
            if _is_sidon_subset(vals):
                best = size
                found = True
                break
        
        if found:
            break
    
    return best

def _local_search_exhaustive(subset, deadline, max_passes=10):
    """
    Exhaustive local search: for each element, try all possible values 0..128.
    Accept if duplicate sum count increases.
    """
    subset = list(subset)
    n = len(subset)
    current_dups = _count_duplicate_sums(subset)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            old_val = subset[i]
            best_new_val = old_val
            best_new_dups = current_dups
            
            # Try all possible new values
            for new_val in range(129):
                if new_val == old_val:
                    continue
                # Check uniqueness
                if any(subset[j] == new_val for j in range(n) if j != i):
                    continue
                
                subset[i] = new_val
                new_dups = _count_duplicate_sums(subset)
                
                if new_dups > best_new_dups:
                    best_new_dups = new_dups
                    best_new_val = new_val
            
            if best_new_val != old_val:
                subset[i] = best_new_val
                current_dups = best_new_dups
                improved = True
        
        if not improved:
            break
            
    return subset

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best_subset = list(range(12))
    best_score = _max_sidon_subset_size_fast(best_subset, deadline)
    
    random.seed(42)
    
    # List of initial candidates: random + structured
    candidates = []
    
    # 1. Random candidates with more collisions
    for _ in range(30):
        candidates.append(random.sample(range(129), 12))
        
    # 2. Structured candidates: Arithmetic Progressions with small differences
    for start in range(0, 120, 5):
        for d in [1, 2, 3, 4, 5, 6, 7, 8]:
            ap = [start + i * d for i in range(12)]
            if all(0 <= x <= 128 for x in ap):
                candidates.append(ap)
                
    # 3. Structured candidates: Symmetric sets around a center
    for c in range(60, 100, 5):
        # Dense symmetric set
        sym = [c-5, c-4, c-3, c-2, c-1, c, c+1, c+2, c+3, c+4, c+5, c+6]
        if all(0 <= x <= 128 for x in sym):
            candidates.append(sym)
        
        # More spread symmetric
        sym2 = [c-10, c-8, c-6, c-4, c-2, c, c+2, c+4, c+6, c+8, c+10, c+12]
        if all(0 <= x <= 128 for x in sym2):
            candidates.append(sym2)
    
    # 4. Clustering: group elements close together
    for base in range(0, 120, 10):
        cluster1 = [base + i for i in range(6)]
        cluster2 = [base + 20 + i for i in range(6)]
        cand = cluster1 + cluster2
        if all(0 <= x <= 128 for x in cand):
            candidates.append(cand)
    
    # Process candidates
    for candidate in candidates:
        if time.time() > deadline - 5:
            break
            
        # Local search to maximize collisions
        improved_candidate = _local_search_exhaustive(candidate, deadline, max_passes=5)
        
        # Evaluate
        score = _max_sidon_subset_size_fast(improved_candidate, deadline)
        
        if score < best_score:
            best_score = score
            best_subset = improved_candidate
            if best_score <= 4: # Theoretical min is likely 4 or 5
                break
                
    # If we have time, do a final polish on the best subset
    if time.time() < deadline - 10 and best_score > 4:
        # Try swapping each element with every other value to see if score drops
        final_subset = list(best_subset)
        n = 12
        for i in range(n):
            if time.time() > deadline - 2:
                break
            old_val = final_subset[i]
            for new_val in range(129):
                if new_val == old_val:
                    continue
                if any(final_subset[j] == new_val for j in range(n) if j != i):
                    continue
                
                final_subset[i] = new_val
                new_dups = _count_duplicate_sums(final_subset)
                if new_dups > _count_duplicate_sums(best_subset):
                    score = _max_sidon_subset_size_fast(final_subset, deadline)
                    if score < best_score:
                        best_score = score
                        best_subset = final_subset
                final_subset[i] = old_val

    return sorted(best_subset)
# EVOLVE-BLOCK-END
