import time
import random
from math import gcd

# EVOLVE-BLOCK-START

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    """
    n_max = 128
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Precompute GCD table
    gcd_table = [[0] * (n_max + 1) for _ in range(n_max + 1)]
    for i in range(1, n_max + 1):
        for j in range(i, n_max + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute all bad triples (a, b, c) where a < b < c and
    # gcd(a,b) == gcd(a,c) == gcd(b,c)
    # Store as a set of frozensets for fast lookup, and also as a 
    # mapping from element to set of bad triples containing it
    
    bad_triples = []
    # For each triple (a,b,c), check if it's bad
    # There are C(128,3) = 341,376 triples, so this is feasible
    for a in range(1, n_max + 1):
        for b in range(a + 1, n_max + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, n_max + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    bad_triples.append((a, b, c))
    
    # Build a conflict structure: for each element x, store the set of 
    # pairs (i, j) such that (x, i, j) is a bad triple
    # This allows fast checking: x can be added to S iff no pair (i,j) in S 
    # is in conflicts[x]
    
    # For each x, conflicts[x] = set of frozenset({i,j}) for bad triples containing x
    conflicts = [set() for _ in range(n_max + 1)]
    for (a, b, c) in bad_triples:
        conflicts[a].add((b, c))
        conflicts[b].add((a, c))
        conflicts[c].add((a, b))
    
    def can_add(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        # For each pair (i, j) in selected, check if (i, j) is in conflicts[x]
        sel_list = list(selected)
        n = len(sel_list)
        for i in range(n):
            a = sel_list[i]
            for j in range(i + 1, n):
                b = sel_list[j]
                # Check if (a, b) is in conflicts[x]
                if a < b:
                    if (a, b) in conflicts[x]:
                        return False
                else:
                    if (b, a) in conflicts[x]:
                        return False
        return True

    def greedy_build(order):
        """Build a valid set by adding elements in the given order."""
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected

    def local_search(selected, max_iters=200, timeout=30.0):
        """
        Improve the set using 1-add, 1-swap, and 2-for-1 exchange moves.
        """
        selected = set(selected)
        improved = True
        iter_count = 0
        start_ls = time.time()
        
        while improved and iter_count < max_iters:
            if time.time() > start_ls + timeout:
                break
            improved = False
            iter_count += 1
            
            # 1. Try adding any excluded element
            excluded = [x for x in range(1, n_max + 1) if x not in selected]
            for x in excluded:
                if can_add(x, selected):
                    selected.add(x)
                    improved = True
                    break
            
            # 2. If no add, try 1-swap (remove 1, add 1) - only if it enables future adds
            #    Or try 2-for-1 (remove 1, add 2)
            if not improved:
                # Try 2-for-1 exchange: remove x, add y1 and y2
                best_exchange = None
                best_new_size = len(selected)
                
                for x in list(selected):
                    temp = selected - {x}
                    # Try to add as many as possible from excluded
                    added = []
                    for y in excluded:
                        if y == x:
                            continue
                        if can_add(y, temp | set(added)):
                            added.append(y)
                    
                    new_size = len(temp) + len(added)
                    if new_size > best_new_size:
                        best_new_size = new_size
                        best_exchange = (x, added)
                
                if best_exchange and best_new_size > len(selected):
                    x, added = best_exchange
                    selected.remove(x)
                    selected.update(added)
                    improved = True
                elif best_exchange and best_new_size == len(selected) and len(added) >= 1:
                    # Size-neutral swap; accept to potentially escape local optimum
                    # But only if it changes the set (to avoid cycling)
                    x, added = best_exchange
                    old_set = set(selected)
                    selected.remove(x)
                    selected.update(added)
                    if selected != old_set:
                        improved = True
                        # Don't mark as improved if we want to continue searching for better
                        # Actually, let's not accept size-neutral swaps to ensure progress
                        # Undo
                        selected = old_set
        
        return selected

    # Strategy: Use multiple randomized greedy starts, then local search
    random.seed(42)
    best_set = set()
    best_size = 0
    
    # Strategy 1: Natural order greedy
    if time.time() < deadline:
        order1 = list(range(1, n_max + 1))
        s1 = greedy_build(order1)
        s1 = local_search(s1, max_iters=100, timeout=15.0)
        if len(s1) > best_size:
            best_size = len(s1)
            best_set = s1
    
    # Strategy 2: Descending order greedy
    if time.time() < deadline:
        order2 = list(range(n_max, 0, -1))
        s2 = greedy_build(order2)
        s2 = local_search(s2, max_iters=100, timeout=15.0)
        if len(s2) > best_size:
            best_size = len(s2)
            best_set = s2
    
    # Strategy 3: Randomized greedy multiple times
    if time.time() < deadline:
        remaining = deadline - time.time()
        num_trials = min(100, int(remaining / 0.5))  # ~0.5s per trial
        for trial in range(num_trials):
            if time.time() >= deadline:
                break
            order3 = list(range(1, n_max + 1))
            random.shuffle(order3)
            s3 = greedy_build(order3)
            s3 = local_search(s3, max_iters=50, timeout=5.0)
            if len(s3) > best_size:
                best_size = len(s3)
                best_set = s3
    
    # Strategy 4: More aggressive local search on best
    if time.time() < deadline and best_set:
        remaining = deadline - time.time()
        if remaining > 10:
            s4 = local_search(best_set, max_iters=500, timeout=remaining - 5)
            if len(s4) > best_size:
                best_size = len(s4)
                best_set = s4
    
    if not best_set:
        best_set = {1}
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
