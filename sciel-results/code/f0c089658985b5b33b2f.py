"""Stronger algorithm for reciprocal weight of 3-progression-free sets on [1,256]."""
import time
import random
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def _can_add(x, S_set, S_list):
    """Check if x can be added to S without creating a 3-AP. O(K) time."""
    for a in S_list:
        if a == x:
            continue
        # Case 1: x is the middle term. a, x, c. c = 2x - a.
        c = 2 * x - a
        if 1 <= c <= 256 and c != x and c in S_set:
            return False
        # Case 2: x is an endpoint. a, b, x or x, b, a. b = (a+x)/2.
        s_sum = a + x
        if s_sum % 2 == 0:
            b = s_sum // 2
            if b != x and b in S_set:
                return False
    return True

def _is_3fast(selected):
    """Check if a set is 3-AP free. O(K^2) but used only for initial verification if needed."""
    s = set(selected)
    for a in selected:
        if not _can_add(a, s - {a}, list(s - {a})):
             # This logic is slightly off for full check, better to just use the standard check
             # Actually, _can_add checks if adding a to (S\{a}) creates AP.
             # If S is 3-free, adding a back should not create AP *unless* the AP involves a and two other elements.
             # If _can_add returns False, it means there is an AP involving a.
             return False
    return True

def _recip_sum(selected):
    """Compute exact sum of reciprocals."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _greedy_by_reciprocal(n=256):
    """Greedy: add elements in order of decreasing 1/x (i.e., increasing x is wrong; 
    we want largest reciprocals first = smallest x first), but check 3-free constraint."""
    selected = []
    selected_set = set()
    # Try adding in order of smallest x (largest reciprocal) first
    for x in range(1, n+1):
        if _can_add(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    return selected

def _greedy_randomized(n=256, seed=0):
    """Greedy with randomization: try multiple random orderings weighted by reciprocal."""
    rng = random.Random(seed)
    best = None
    best_val = Fraction(0, 1)
    
    for trial in range(50):
        # Create weighted random ordering: prefer smaller numbers (larger reciprocals)
        # but with some randomness
        candidates = list(range(1, n+1))
        # Assign weights: weight = 1/x with some noise
        weights = [1.0/x for x in candidates]
        # Add noise
        noisy_weights = [w * (0.5 + rng.random()) for w in weights]
        
        # Sort by noisy weight descending (largest reciprocal first)
        order = sorted(candidates, key=lambda x: noisy_weights[x-1], reverse=True)
        
        selected = []
        selected_set = set()
        for x in order:
            if _can_add(x, selected_set, selected):
                selected.append(x)
                selected_set.add(x)
        
        val = _recip_sum(selected)
        if val > best_val:
            best_val = val
            best = selected
    
    return best if best else [1]

def _local_search(start_set, n=256, max_iters=10000, seed=0):
    """Local search: try swapping elements to improve reciprocal sum. Uses O(K) check."""
    rng = random.Random(seed)
    current = list(start_set)
    current_set = set(current)
    current_val = _recip_sum(current)
    best = list(current)
    best_val = current_val
    
    for it in range(max_iters):
        # Pick a random element to remove and a random candidate to add
        if not current:
            break
        remove_idx = rng.randrange(len(current))
        removed = current[remove_idx]
        
        # Find a candidate to add
        # Optimization: Pre-compute list of candidates outside current_set
        # This is O(N) per iteration, which might be slow if N=256 and iters=10000.
        # 256 * 10000 = 2.5M ops. Acceptable.
        candidates = [x for x in range(1, n+1) if x not in current_set]
        if not candidates:
            break
        
        # Prefer candidates with large reciprocal (small x)
        # Weighted random choice
        weights = [1.0/x for x in candidates]
        total_w = sum(weights)
        r = rng.random() * total_w
        cum = 0
        add_candidate = candidates[0]
        for c in candidates:
            cum += 1.0/c
            if cum >= r:
                add_candidate = c
                break
        
        # Try the swap
        # We remove 'removed' and add 'add_candidate'
        # We need to check if the new set is 3-free.
        # Instead of building the new set and checking from scratch (O(K^2)),
        # we can check if 'add_candidate' is valid in the set (current_set - {removed}).
        
        temp_set = current_set - {removed}
        temp_list = [x for x in current if x != removed]
        
        if _can_add(add_candidate, temp_set, temp_list):
            # Valid swap
            new_set = temp_set | {add_candidate}
            new_list = temp_list + [add_candidate]
            new_val = current_val - Fraction(1, removed) + Fraction(1, add_candidate)
            
            if new_val > current_val:
                current = new_list
                current_set = new_set
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best = list(current)
    
    return best

def construct():
    """Main entry point: find a 3-free subset of [1,256] maximizing sum of reciprocals."""
    start_time = time.time()
    deadline = 58.0  # safety margin before 60s limit
    
    # Start with greedy by reciprocal (smallest x first)
    best_set = _greedy_by_reciprocal(256)
    best_val = _recip_sum(best_set)
    
    # Try randomized greedy
    # Greedy is fast, so we can run many seeds
    seed = 0
    while time.time() - start_time < deadline - 10:
        candidate = _greedy_randomized(256, seed=seed)
        if candidate:
            val = _recip_sum(candidate)
            if val > best_val:
                best_val = val
                best_set = candidate
        seed += 1
    
    # Local search from best
    # Local search is now O(K) per check, so we can run more iterations
    ls_seed = 0
    while time.time() - start_time < deadline - 2:
        candidate = _local_search(best_set, 256, max_iters=20000, seed=ls_seed)
        if candidate:
            val = _recip_sum(candidate)
            if val > best_val:
                best_val = val
                best_set = candidate
        ls_seed += 1
    
    return best_set

# EVOLVE-BLOCK-END
