"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _is_bad_triple(a, b, c, gcd_table):
    """Check if three distinct elements have all three pairwise gcds equal."""
    g_ab = gcd_table[a][b]
    g_ac = gcd_table[a][c]
    g_bc = gcd_table[b][c]
    return g_ab == g_ac == g_bc

def _build_gcd_table(n=128):
    """Precompute gcd table for 1..n."""
    table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            table[i][j] = g
            table[j][i] = g
    return table

def _hypergraph_degree(x, selected, gcd_table):
    """Count how many bad triples would be formed if x were added to selected."""
    count = 0
    for a, b in combinations(selected, 2):
        if _is_bad_triple(a, b, x, gcd_table):
            count += 1
    return count

def _greedy_by_degree(gcd_table, order=None, deadline=None):
    """Weighted greedy: prioritize vertices with lowest conflict degree."""
    if order is None:
        order = list(range(1, 129))
    selected = []
    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        if _hypergraph_degree(x, selected, gcd_table) == 0:
            selected.append(x)
    return selected

def _local_search(selected, gcd_table, max_iters=5000, deadline=None):
    """Try to improve by swapping elements in/out."""
    selected_set = set(selected)
    available = [x for x in range(1, 129) if x not in selected_set]
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if deadline is not None and time.time() > deadline:
            break
        for i, out_x in enumerate(selected):
            if deadline is not None and time.time() > deadline:
                break
            # Try removing out_x
            remaining = [x for j, x in enumerate(selected) if j != i]
            for in_x in available:
                if in_x in selected_set:
                    continue
                new_set = remaining + [in_x]
                # Check validity
                valid = True
                for a, b, c in combinations(new_set, 3):
                    if _is_bad_triple(a, b, c, gcd_table):
                        valid = False
                        break
                if valid and len(new_set) > len(selected):
                    selected = new_set
                    selected_set = set(selected)
                    available = [x for x in range(1, 129) if x not in selected_set]
                    improved = True
                    break
            if improved:
                break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    gcd_table = _build_gcd_table(128)
    
    best_set = []
    best_size = 0
    
    # Multiple randomized restarts with different orderings
    random.seed(42)
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different ordering strategies
        if r % 3 == 0:
            order = list(range(1, 129))
            random.shuffle(order)
        elif r % 3 == 1:
            # Prioritize numbers with fewer prime factors (less likely to share gcds)
            def count_prime_factors(n):
                cnt = 0
                d = 2
                while d * d <= n:
                    while n % d == 0:
                        cnt += 1
                        n //= d
                    d += 1
                if n > 1:
                    cnt += 1
                return cnt
            order = sorted(range(1, 129), key=lambda x: count_prime_factors(x))
        else:
            # Random with some bias toward larger numbers
            order = list(range(1, 129))
            random.shuffle(order)
        
        greedy_set = _greedy_by_degree(gcd_table, order, deadline)
        
        if len(greedy_set) > best_size:
            best_set = greedy_set[:]
            best_size = len(greedy_set)
        
        # Local search on the best greedy result
        if len(greedy_set) > 0 and time.time() < deadline - 5:
            ls_set = _local_search(greedy_set, gcd_table, max_iters=2000, deadline=deadline)
            if len(ls_set) > best_size:
                best_set = ls_set[:]
                best_size = len(ls_set)
    
    # Final local search on best
    if len(best_set) > 0 and time.time() < deadline - 2:
        final_set = _local_search(best_set, gcd_table, max_iters=3000, deadline=deadline)
        if len(final_set) > best_size:
            best_set = final_set
    
    return best_set if best_set else [1]
# EVOLVE-BLOCK-END
