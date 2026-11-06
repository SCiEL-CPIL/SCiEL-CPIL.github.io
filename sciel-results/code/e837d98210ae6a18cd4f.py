"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_valid(selected):
    """Check if selected set is valid: no element divides sum of any 3 distinct others."""
    n = len(selected)
    if n < 4:
        return True
    sel_set = set(selected)
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _can_add(selected, x):
    """Check if x can be added to selected without violating constraints."""
    n = len(selected)
    if n < 3:
        return True
    # Check if x divides sum of any 3 existing elements
    for t in combinations(selected, 3):
        if sum(t) % x == 0:
            return False
    # Check if any existing element divides sum of x + any 2 other existing elements
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        for pair in combinations(others, 2):
            s = x + pair[0] + pair[1]
            if s % a == 0:
                return False
    return True

def _compute_conflict_scores(n=192):
    """
    Compute a conflict score for each number 1..n.
    Score = number of triples (b, c, d) from 1..n \ {a} such that a | b+c+d.
    Higher score means 'more dangerous' to include.
    """
    scores = [0] * (n + 1)
    # Precompute all triples sums? No, too many.
    # For each a, count how many triples sum to multiple of a.
    # Sum range: min 1+2+3=6, max (n-2)+(n-1)+n = 3n-3.
    # For n=192, max sum = 573.
    # For a given a, multiples of a in [6, 573] are candidates.
    # For each multiple M, count number of triples (b,c,d) distinct from a, distinct from each other, summing to M.
    
    # Optimization:
    # We can iterate a from 1 to n.
    # For each a, we want to count triples from S = {1..n} \ {a}.
    # This is O(n * n^3) if done naively. 192 * 192^3 ~ 1.3 billion. Too slow in Python.
    
    # Alternative:
    # Just use a simpler heuristic: a is dangerous if it is small (divides many things).
    # Or: Count divisors of sums of pairs?
    # Let's use a proxy: 1/a is not a good proxy.
    # Let's try: For each a, count how many pairs (b,c) exist such that there is a d with a | b+c+d.
    # This is still heavy.
    
    # Let's use a simpler structural heuristic:
    # Numbers with many divisors in the range [1, n] are more likely to divide sums?
    # No, the number a itself is the divisor.
    # Small a divides more sums.
    # So score = 1/a? Or just index a?
    # Let's try sorting by a (ascending) as "most dangerous first"?
    # Or descending?
    
    # Let's try a different structural approach:
    # Partition numbers into "Small" (1..50) and "Large" (51..192).
    # Small numbers are likely to divide sums.
    # Large numbers are less likely to divide sums (sums are < 573, large a > 573/2 = 286.5 can only divide sums if sum is 2a or 3a.
    # If a > 191, 2a > 382. Sum can be up to 573.
    # So a in (191, 286] can divide 2a (if 2a <= 573) or 3a (if 3a <= 573 -> a <= 191).
    # So a > 191 can only divide 2a.
    # So large numbers are "safer".
    
    # Strategy: Prioritize adding Large numbers, then Small numbers.
    # But we need a specific order.
    
    # Let's compute a "danger" score based on the number of multiples of a in the possible sum range [6, 3n-3].
    # More multiples -> more potential conflicts.
    max_sum = 3 * n - 3
    scores = [0] * (n + 1)
    for a in range(1, n + 1):
        # Count multiples of a in [6, max_sum]
        # First multiple >= 6: ceil(6/a) * a
        # Last multiple <= max_sum: floor(max_sum/a) * a
        if a == 0: continue
        start_mult = (6 + a - 1) // a
        end_mult = max_sum // a
        if start_mult <= end_mult:
            scores[a] = end_mult - start_mult + 1
        else:
            scores[a] = 0
            
    # Higher score = more dangerous
    return scores

def _greedy(order):
    """Greedy construction using given order of candidates."""
    selected = []
    for x in order:
        if _can_add(selected, x):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Try to improve selected set via swaps and additions."""
    best = list(selected)
    best_size = len(best)
    all_nums = set(range(1, 193))
    
    while time.time() < deadline:
        improved = False
        
        # Try adding any missing number
        # Sort candidates by conflict score (lowest first? or highest first?)
        # We want to add "safe" numbers first.
        # Safe numbers have low conflict scores.
        scores = _compute_conflict_scores(192)
        missing = [x for x in all_nums - set(best)]
        missing.sort(key=lambda x: scores[x]) # Lowest danger first
        
        for x in missing:
            if _can_add(best, x):
                best.append(x)
                best_size = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one, add another
        # Remove "dangerous" numbers, add "safe" ones
        # Sort current best by danger (highest first) to remove
        best_sorted = sorted(best, key=lambda x: scores[x], reverse=True)
        
        for i, removed in enumerate(best_sorted):
            if time.time() >= deadline:
                break
            # Find index in best
            idx = best.index(removed)
            
            # Try to add a safer number
            # Candidates: missing numbers with lower danger than removed
            candidates = [x for x in missing if scores[x] < scores[removed]]
            # Shuffle to avoid bias? Or sort by danger?
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                trial = best[:idx] + best[idx+1:] + [x]
                if _check_valid(trial):
                    best = trial
                    best_size = len(best)
                    improved = True
                    added = True
                    break
            if added:
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin
    
    # Compute conflict scores
    scores = _compute_conflict_scores(192)
    
    # Create orders
    # Order 1: Ascending danger (safest first)
    order_safe = list(range(1, 193))
    order_safe.sort(key=lambda x: scores[x])
    
    # Order 2: Descending danger (most dangerous first, to prune?)
    order_danger = list(range(1, 193))
    order_danger.sort(key=lambda x: scores[x], reverse=True)
    
    # Order 3: Descending value (Large numbers first, as they are often safer)
    order_desc_val = list(range(192, 0, -1))
    
    # Order 4: Ascending value
    order_asc_val = list(range(1, 193))
    
    best_result = []
    
    # Try greedy with these orders
    for order in [order_safe, order_danger, order_desc_val, order_asc_val]:
        if time.time() > deadline - 20:
            break
        result = _greedy(order)
        if len(result) > len(best_result):
            best_result = result
    
    # Local search
    if time.time() < deadline - 5:
        improved = _local_search(best_result, deadline)
        if len(improved) > len(best_result):
            best_result = improved
            
    return best_result
# EVOLVE-BLOCK-END
