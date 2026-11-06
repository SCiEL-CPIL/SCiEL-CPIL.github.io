"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _fast_is_sidon(lst):
    """Fast Sidon check for sorted list (all unordered pair sums unique including diagonals)."""
    n = len(lst)
    sums = set()
    for i in range(n):
        a = lst[i]
        for j in range(i, n):
            s = a + lst[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(ambient):
    """Compute max Sidon subset size for sorted 12-element list."""
    n = len(ambient)
    # Check from largest possible k down to 3.
    for k in range(min(n, 7), 2, -1):
        for combo in itertools.combinations(ambient, k):
            if _fast_is_sidon(list(combo)):
                return k
    return 2

def _count_sidon_subsets_of_size(ambient, k):
    """Count how many k-subsets are Sidon."""
    cnt = 0
    for combo in itertools.combinations(ambient, k):
        if _fast_is_sidon(list(combo)):
            cnt += 1
    return cnt

def _construct_candidate():
    """
    Generates a candidate with a small max Sidon subset size.
    Strategy:
    1. Start with dense arithmetic progressions.
    2. Perform local search to minimize the number of 5-Sidon subsets, 
       then minimize max Sidon size.
    """
    deadline = time.time() + 55  # Safety margin for 60s limit
    rng = random.Random(123)
    
    # Initialize with a dense AP
    best_cand = list(range(12))
    best_k = _max_sidon_size(best_cand)
    best_5_count = _count_sidon_subsets_of_size(best_cand, 5)
    
    # Local search
    # We want to minimize best_k, and secondarily best_5_count.
    # Actually, minimizing 5_count is a good proxy for reducing max Sidon size.
    
    current = best_cand[:]
    current_k = best_k
    current_5_count = best_5_count
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Pick a random index to perturb
        idx = rng.randint(0, 11)
        old_val = current[idx]
        
        # Try a random new value in [0, 128]
        # Bias towards dense regions?
        new_val = rng.randint(0, 128)
        
        if new_val == old_val:
            continue
            
        # Check uniqueness
        if new_val in [x for i, x in enumerate(current) if i != idx]:
            continue
            
        # Apply perturbation
        current[idx] = new_val
        current.sort()
        
        # Evaluate
        # First check if we still have a 5-Sidon? 
        # If we remove all 5-Sidons, max is <= 4.
        # Let's just compute max_k and 5_count if max_k is high.
        
        # Quick check: does it have a 5-Sidon?
        has_5 = False
        for combo in itertools.combinations(current, 5):
            if _fast_is_sidon(list(combo)):
                has_5 = True
                break
        
        if has_5:
            k = _max_sidon_size(current)
            # If k is 5 or more, count 5-sidons to guide search
            c5 = _count_sidon_subsets_of_size(current, 5)
        else:
            k = 4 if _count_sidon_subsets_of_size(current, 4) > 0 else 3 # approx
            # Actually if no 5-sidon, max is <= 4.
            # Let's be precise:
            k = _max_sidon_size(current)
            c5 = 0
            
        # Compare
        # We want to minimize k, then c5.
        if (k < current_k) or (k == current_k and c5 < current_5_count):
            current_k = k
            current_5_count = c5
            no_improve_count = 0
            if k < best_k or (k == best_k and c5 < best_5_count):
                best_k = k
                best_5_count = c5
                best_cand = current[:]
        else:
            # Revert
            current[idx] = old_val
            current.sort()
            no_improve_count += 1
            
    return best_cand

def construct():
    return _construct_candidate()
# EVOLVE-BLOCK-END
