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

def _max_sidon_subset_size(subset, deadline=None):
    """Find largest subset with all unordered pair sums unique (including diagonals).
    Uses branch-and-bound with pruning."""
    n = len(subset)
    if n <= 1:
        return n
    
    # Precompute all pair sums for quick checking
    # We'll use a recursive approach with memoization on the "used sums" set
    # For n=12, we can try all subsets but prune early
    
    best = 0
    # Try subsets in descending size order
    for size in range(n, 0, -1):
        if deadline and time.time() > deadline:
            break
        found = False
        for combo in combinations(range(n), size):
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

def _greedy_collision_swap(subset, deadline, max_swaps=10000):
    """Greedily swap elements to maximize duplicate pair sums."""
    subset = list(subset)
    n = len(subset)
    best_dups = _count_duplicate_sums(subset)
    
    for k in range(max_swaps):
        if k % 50 == 0 and time.time() > deadline:
            break
            
        # Try replacing element i with a random value in [0, 128]
        i = random.randint(0, n - 1)
        old_val = subset[i]
        new_val = random.randint(0, 128)
        
        if new_val in subset and new_val != old_val:
            continue
            
        subset[i] = new_val
        new_dups = _count_duplicate_sums(subset)
        if new_dups > best_dups:
            best_dups = new_dups
        else:
            subset[i] = old_val
    
    return subset, best_dups

def _structured_candidate(base, offsets, deadline):
    """Generate a structured candidate with controlled additive collisions.
    Uses multiples of a small base with offsets to create dense sum collisions."""
    # Try to create pairs that sum to the same value
    # e.g., a + (2b - a) = 2b for all a in a symmetric set
    candidates = []
    
    # Strategy 1: Arithmetic progression (maximizes duplicate sums)
    for start in range(0, 129 - 11 * base, 1):
        cand = [start + i * base for i in range(12)]
        if max(cand) <= 128:
            candidates.append(cand)
    
    # Strategy 2: Two interleaved arithmetic progressions
    for start1 in range(0, 60):
        for start2 in range(0, 60):
            cand = []
            for i in range(6):
                cand.append(start1 + i * base)
                cand.append(start2 + i * base)
            if len(set(cand)) == 12 and max(cand) <= 128:
                candidates.append(cand)
    
    # Strategy 3: Symmetric pairs around a center
    center = 64
    for d1 in range(1, 20):
        for d2 in range(d1 + 1, 20):
            for d3 in range(d2 + 1, 20):
                cand = [center - 3*d3, center - 2*d3, center - d3,
                        center - 2*d2, center - d2, center,
                        center + d2, center + 2*d2,
                        center - d1, center + d1,
                        center + 2*d3, center + 3*d3]
                if len(set(cand)) == 12 and min(cand) >= 0 and max(cand) <= 128:
                    candidates.append(cand)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    random.seed(42)
    
    best_subset = None
    best_score = 12  # Worst case: entire set is Sidon
    
    # Phase 1: Try structured candidates
    for base in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 10:
            break
        candidates = _structured_candidate(base, None, deadline)
        for cand in candidates[:100]:  # Limit per base
            if time.time() > deadline - 10:
                break
            cand = list(cand)
            # Improve via greedy
            improved, _ = _greedy_collision_swap(cand, deadline, max_swaps=2000)
            if time.time() > deadline - 5:
                break
            score = _max_sidon_subset_size(improved, deadline)
            if score < best_score:
                best_score = score
                best_subset = improved
                if best_score <= 4:
                    return sorted(best_subset)
    
    # Phase 2: Random restarts with greedy improvement
    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        candidate = random.sample(range(129), 12)
        
        time_left = deadline - time.time()
        if time_left < 15:
            max_swaps = 1000
        elif time_left < 40:
            max_swaps = 5000
        else:
            max_swaps = 10000
            
        improved, _ = _greedy_collision_swap(candidate, deadline, max_swaps=max_swaps)
        
        if time.time() > deadline - 2:
            break
            
        score = _max_sidon_subset_size(improved, deadline)
        if score < best_score:
            best_score = score
            best_subset = improved
            if best_score <= 4:
                break
    
    if best_subset is None:
        best_subset = list(range(12))
    
    return sorted(best_subset)
# EVOLVE-BLOCK-END
