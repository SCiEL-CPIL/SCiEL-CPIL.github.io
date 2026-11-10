"""Rewritten program for B2[2] set construction in {1..1000}."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _greedy_build(order, deadline):
    """Greedy construction following a given order of candidate integers.
    
    Returns (chosen, counts) where chosen is the list of selected integers
    and counts is a dict mapping sum -> count of representations.
    """
    chosen = []
    counts = {}
    n = len(order)
    
    for i, x in enumerate(order):
        if time.time() > deadline:
            break
        # Compute new sums
        new_sums = [x + y for y in chosen] + [2 * x]
        
        # Check if adding x keeps all counts <= 2
        valid = True
        for s in new_sums:
            c = counts.get(s, 0) + 1
            if c > 2:
                valid = False
                break
        
        if valid:
            chosen.append(x)
            for s in new_sums:
                counts[s] = counts.get(s, 0) + 1
    
    return chosen, counts


def _vectorized_greedy_build(order, deadline):
    """Vectorized greedy construction using NumPy for speed.
    
    Returns (chosen, counts) where chosen is the list of selected integers
    and counts is a dict mapping sum -> count of representations.
    """
    chosen = []
    # Use a numpy array for sum counts (indexed by sum value 2..2000)
    sum_counts = np.zeros(2001, dtype=np.int32)
    
    for x in order:
        if time.time() > deadline:
            break
        
        if not chosen:
            chosen.append(x)
            sum_counts[2 * x] = 1
            continue
        
        # Compute all new sums with existing elements
        chosen_arr = np.array(chosen)
        new_sums = np.concatenate([chosen_arr + x, [2 * x]])
        
        # Check if any sum would exceed 2
        current_counts = sum_counts[new_sums]
        if np.any(current_counts >= 2):
            continue
        
        # Valid: add x
        chosen.append(x)
        sum_counts[new_sums] += 1
    
    # Convert to dict for compatibility
    counts = {}
    for s in range(2, 2001):
        if sum_counts[s] > 0:
            counts[s] = int(sum_counts[s])
    
    return chosen, counts


def _local_search_swap(chosen, counts, deadline, max_swaps=5000):
    """Improve the set by trying swap operations: remove one element, add another.
    
    Returns improved (chosen, counts).
    """
    chosen_set = set(chosen)
    improved = True
    swap_count = 0
    n = len(chosen)
    
    # Precompute a numpy array of chosen for fast operations
    chosen_arr = np.array(chosen)
    
    while improved and swap_count < max_swaps and time.time() < deadline:
        improved = False
        # Try removing each element and adding a better one
        for i, x in enumerate(chosen):
            if time.time() > deadline:
                break
            
            # Compute counts after removing x
            # Sums involving x: x+y for y in chosen, y != x, and 2*x
            other_elements = [y for y in chosen if y != x]
            
            # Build a temporary count array
            temp_counts = np.zeros(2001, dtype=np.int32)
            for s, c in counts.items():
                temp_counts[s] = c
            
            # Remove contributions from x
            for y in other_elements:
                s = x + y
                temp_counts[s] -= 1
            s = 2 * x
            temp_counts[s] -= 1
            
            # Try to find a valid candidate to add
            best_add = None
            for cand in range(1, 1001):
                if cand in chosen_set:
                    continue
                
                # Check if cand can be added: all new sums must have count < 2
                new_sums_cand = np.array([cand + y for y in other_elements] + [2 * cand])
                if np.any(temp_counts[new_sums_cand] >= 2):
                    continue
                
                best_add = cand
                break
            
            if best_add is not None:
                # Perform the swap
                chosen[i] = best_add
                chosen_set.discard(x)
                chosen_set.add(best_add)
                chosen_arr[i] = best_add
                
                # Update counts
                for y in other_elements:
                    s = best_add + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * best_add
                counts[s] = counts.get(s, 0) + 1
                
                improved = True
                swap_count += 1
                break  # Restart the loop after a swap
    
    return chosen, counts


def _sum_space_packing(deadline):
    """Sum-space packing heuristic: greedily assign integers to minimize sum conflicts.
    
    Strategy: process integers in an order that spreads out their pairwise sums.
    Use a priority-based approach where we prefer integers that create new sum values.
    """
    chosen = []
    sum_counts = np.zeros(2001, dtype=np.int32)
    
    # Strategy: start with well-spread integers
    # Use a golden-ratio-like spacing to maximize initial spread
    base = 1
    # Generate candidates sorted by a heuristic: prefer those far from existing sums
    candidates = list(range(1, 1001))
    
    # First pass: greedy with spread preference
    # Sort by distance from 500 (center) to encourage spread
    candidates.sort(key=lambda x: abs(x - 500))
    
    for x in candidates:
        if time.time() > deadline:
            break
        
        if not chosen:
            chosen.append(x)
            sum_counts[2 * x] = 1
            continue
        
        chosen_arr = np.array(chosen)
        new_sums = np.concatenate([chosen_arr + x, [2 * x]])
        
        current_counts = sum_counts[new_sums]
        if np.any(current_counts >= 2):
            continue
        
        chosen.append(x)
        sum_counts[new_sums] += 1
    
    # Convert to dict
    counts = {}
    for s in range(2, 2001):
        if sum_counts[s] > 0:
            counts[s] = int(sum_counts[s])
    
    return chosen, counts


def _try_orders(deadline):
    """Try multiple greedy orders and local search improvements.
    
    Returns the best (chosen, counts) found.
    """
    best_chosen = []
    best_counts = {}
    best_size = 0
    
    # Define several order strategies
    strategies = []
    
    # 1. Ascending
    strategies.append(list(range(1, 1001)))
    
    # 2. Descending
    strategies.append(list(range(1000, 0, -1)))
    
    # 3. Sparse (every 2nd)
    strategies.append(list(range(1, 1001, 2)))
    
    # 4. Random (seeded)
    rng = random.Random(42)
    rand_order = list(range(1, 1001))
    rng.shuffle(rand_order)
    strategies.append(rand_order)
    
    # 5. Another random
    rng2 = random.Random(123)
    rand_order2 = list(range(1, 1001))
    rng2.shuffle(rand_order2)
    strategies.append(rand_order2)
    
    # 6. Sparse descending
    strategies.append(list(range(1000, 0, -2)))
    
    # 7. Mix: evens then odds
    strategies.append(list(range(2, 1001, 2)) + list(range(1, 1001, 2)))
    
    # 8. Odds then evens
    strategies.append(list(range(1, 1001, 2)) + list(range(2, 1001, 2)))
    
    # 9. Spread from center
    center_order = list(range(500, 1001)) + list(range(499, 0, -1))
    strategies.append(center_order)
    
    # 10. Alternating low/high
    alt_order = []
    for i in range(500):
        alt_order.append(i + 1)
        alt_order.append(1000 - i)
    strategies.append(alt_order)
    
    for order in strategies:
        if time.time() > deadline:
            break
        chosen, counts = _vectorized_greedy_build(order, deadline)
        size = len(chosen)
        
        if size > best_size:
            best_size = size
            best_chosen = list(chosen)
            best_counts = dict(counts)
    
    # Also try sum-space packing
    if time.time() < deadline:
        sp_chosen, sp_counts = _sum_space_packing(deadline)
        if len(sp_chosen) > best_size:
            best_size = len(sp_chosen)
            best_chosen = sp_chosen
            best_counts = sp_counts
    
    # Local search on the best result
    if best_chosen and time.time() < deadline:
        best_chosen, best_counts = _local_search_swap(
            best_chosen, best_counts, deadline, max_swaps=5000
        )
        best_size = len(best_chosen)
    
    return best_chosen, best_counts


def construct():
    start_time = time.time()
    # Leave 15s safety margin before the 150s kill
    deadline = start_time + 135.0
    
    chosen, counts = _try_orders(deadline)
    
    return chosen

# EVOLVE-BLOCK-END
