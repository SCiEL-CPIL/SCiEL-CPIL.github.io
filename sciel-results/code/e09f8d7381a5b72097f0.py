"""Large integer sets avoiding equal-gcd triples.
Uses hypergraph MIS heuristic with greedy min-degree initialization and local search swaps.
"""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _build_gcd_table(n):
    """Precompute gcd table for 1..n."""
    table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            table[i][j] = gcd(i, j)
    return table

def _is_bad_triple(a, b, c, gcd_table):
    """Check if three distinct elements have all three pairwise gcds equal."""
    g1 = gcd_table[a][b]
    g2 = gcd_table[a][c]
    g3 = gcd_table[b][c]
    return g1 == g2 == g3

def _hypergraph_degree(selected, all_elements, gcd_table):
    """Count for each candidate element how many bad triples it would form with pairs in selected."""
    degrees = {}
    for x in all_elements:
        if x in selected:
            continue
        count = 0
        for a, b in combinations(selected, 2):
            if _is_bad_triple(a, b, x, gcd_table):
                count += 1
        degrees[x] = count
    return degrees

def _greedy_by_degree(n, gcd_table, time_limit):
    """Greedy construction: repeatedly add the element with fewest bad triples."""
    selected = set()
    remaining = set(range(1, n + 1))
    deadline = time.time() + time_limit
    
    while remaining and time.time() < deadline:
        # Find element with minimum degree (fewest bad triples with existing pairs)
        best_elem = None
        best_deg = float('inf')
        for x in remaining:
            deg = 0
            for a, b in combinations(selected, 2):
                if _is_bad_triple(a, b, x, gcd_table):
                    deg += 1
                    if deg >= best_deg:
                        break
            if deg < best_deg:
                best_deg = deg
                best_elem = x
                if deg == 0:
                    break
        if best_elem is None:
            break
        selected.add(best_elem)
        remaining.discard(best_elem)
    
    return selected

def _local_search(selected, n, gcd_table, time_limit):
    """Local search: try swaps (remove one, add one) and additions to improve size."""
    selected = set(selected)
    deadline = time.time() + time_limit
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any element that doesn't create a bad triple
        candidates = [x for x in range(1, n + 1) if x not in selected]
        for x in candidates:
            if time.time() >= deadline:
                break
            valid = True
            for a, b in combinations(selected, 2):
                if _is_bad_triple(a, b, x, gcd_table):
                    valid = False
                    break
            if valid:
                selected.add(x)
                improved = True
                # Try to add more after this addition
                candidates = [y for y in range(1, n + 1) if y not in selected]
        
        # Try swaps: remove one element, see if we can add two or more
        for to_remove in list(selected):
            if time.time() >= deadline:
                break
            trial = selected - {to_remove}
            # Try to add elements back
            added = 0
            for x in range(1, n + 1):
                if x in trial or x == to_remove:
                    continue
                valid = True
                for a, b in combinations(trial, 2):
                    if _is_bad_triple(a, b, x, gcd_table):
                        valid = False
                        break
                if valid:
                    trial.add(x)
                    added += 1
            if added > 1:  # Net gain
                selected = trial
                improved = True
    
    return selected

def construct():
    """Return nonempty subset of [1,128] with no three distinct elements whose
    three pairwise greatest common divisors are equal. Maximize cardinality."""
    start_time = time.time()
    DEADLINE = 80.0  # Safety margin before 90s kill
    n = 128
    
    # Precompute gcd table
    gcd_table = _build_gcd_table(n)
    
    # Phase 1: Greedy construction with multiple random orderings
    best_set = set()
    remaining_time = DEADLINE - (time.time() - start_time)
    
    # Try greedy from different starting points
    import random
    random.seed(42)
    
    # First, try standard greedy (min degree)
    time_per_greedy = min(10.0, remaining_time / 5)
    greedy_result = _greedy_by_degree(n, gcd_table, time_per_greedy)
    if len(greedy_result) > len(best_set):
        best_set = greedy_result
    
    # Try greedy with different orderings
    for attempt in range(5):
        if time.time() - start_time > DEADLINE - 20:
            break
        # Random subset of elements to start with
        random.shuffle_order = list(range(1, n + 1))
        random.shuffle(random.shuffle_order)
        
        selected = set()
        for x in random.shuffle_order:
            if time.time() - start_time > DEADLINE - 20:
                break
            valid = True
            for a, b in combinations(selected, 2):
                if _is_bad_triple(a, b, x, gcd_table):
                    valid = False
                    break
            if valid:
                selected.add(x)
        
        if len(selected) > len(best_set):
            best_set = selected
    
    # Phase 2: Local search on the best set found
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5:
        local_result = _local_search(best_set, n, gcd_table, remaining_time - 5)
        if len(local_result) > len(best_set):
            best_set = local_result
    
    # Ensure nonempty
    if not best_set:
        best_set = {1}
    
    return list(best_set)
# EVOLVE-BLOCK-END
