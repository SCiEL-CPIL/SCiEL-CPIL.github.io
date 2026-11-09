import time
import random
from math import gcd
from itertools import combinations
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_lcms(n_max):
    """Precompute LCM table for all pairs in [1, n_max]."""
    n = n_max
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    return lcm_table

def _build_conflict_table(lcm_table, n_max):
    """
    Build conflict structure: for each element x, store the set of pairs (a,b)
    such that {x,a,b} is a bad triple.
    """
    n = n_max
    bad_pairs = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bc = (min(b,c), max(b,c))
                    ac = (min(a,c), max(a,c))
                    ab = (a, b)
                    bad_pairs[a].add(bc)
                    bad_pairs[b].add(ac)
                    bad_pairs[c].add(ab)
    
    return bad_pairs

def _greedy_construct_fast(order, bad_pairs, n_max):
    """Greedily add numbers in the given order if they don't form a bad triple."""
    selected = set()
    selected_list = []
    
    for x in order:
        ok = True
        for (a, b) in bad_pairs[x]:
            if a in selected and b in selected:
                ok = False
                break
        if ok:
            selected.add(x)
            selected_list.append(x)
    
    return selected_list

def _branch_and_bound(bad_pairs, n_max, initial_solution, time_limit=60.0):
    """
    Branch-and-bound solver for maximum independent set in 3-uniform hypergraph.
    Uses greedy coloring bound on the conflict graph.
    """
    start_time = time.time()
    best_solution = list(initial_solution)
    best_size = len(best_solution)
    
    # Build conflict graph: edge between x,y if there exists z such that {x,y,z} is bad
    # Actually, we need: x and y conflict if adding both would prevent some z from being added
    # More precisely: x and y are in conflict if there exists z such that (x,y) in bad_pairs[z]
    # This means: if we have x and y selected, then z cannot be added.
    # For upper bound: if we have a set S, the max we can add is bounded by
    # the independence number of the conflict graph on remaining vertices.
    
    # Build adjacency: adj[x] = set of y such that x and y "conflict" (i.e., there's a z with (x,y) in bad_pairs[z])
    adj = [set() for _ in range(n_max + 1)]
    for z in range(1, n_max + 1):
        for (a, b) in bad_pairs[z]:
            adj[a].add(b)
            adj[b].add(a)
    
    # Also build: for each vertex x, the set of vertices that conflict with x
    # conf_with[x] = set of y such that (x,y) appears in some bad_pairs[z]
    # This is the same as adj[x]
    
    def _greedy_color_bound(vertices, adj):
        """
        Greedy coloring upper bound for independence number.
        Returns upper bound on independence number of the subgraph induced by vertices.
        """
        if not vertices:
            return 0
        
        remaining = set(vertices)
        color = {}
        num_colors = 0
        
        while remaining:
            num_colors += 1
            # Pick a vertex with highest degree in remaining
            v = max(remaining, key=lambda x: len(adj[x] & remaining))
            # Assign this color to v and all non-adjacent vertices
            color_class = {v}
            for u in list(remaining):
                if u != v and u not in adj[v]:
                    color_class.add(u)
            remaining -= color_class
        
        return num_colors
    
    def _can_add(x, selected_set):
        """Check if x can be added to selected_set."""
        for (a, b) in bad_pairs[x]:
            if a in selected_set and b in selected_set:
                return False
        return True
    
    def _solve(selected, available, depth, current_size):
        nonlocal best_solution, best_size, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        # Update best if current is better
        if current_size > best_size:
            best_size = current_size
            best_solution = list(selected)
        
        if not available:
            return
        
        # Upper bound: current_size + independence number of conflict graph on available
        bound = current_size + _greedy_color_bound(available, adj)
        if bound <= best_size:
            return
        
        # Pick a vertex to branch on
        # Choose vertex with highest degree in available
        v = max(available, key=lambda x: len(adj[x] & available))
        
        # Branch 1: Include v
        if _can_add(v, selected):
            new_selected = selected | {v}
            # Remove v and all vertices that conflict with v from available
            # Actually, we should remove vertices that would form bad triples with v
            # But for safety, just remove v and continue
            new_available = available - {v}
            _solve(new_selected, new_available, depth + 1, current_size + 1)
        
        # Branch 2: Exclude v
        _solve(selected, available - {v}, depth + 1, current_size)
    
    # Start with empty selection and all vertices available
    # But we can use the initial solution as a starting point
    # For efficiency, start with the initial solution's elements as pre-selected
    
    # Actually, let's start fresh but use the bound to prune
    all_vertices = set(range(1, n_max + 1))
    _solve(set(), all_vertices, 0, 0)
    
    return best_solution

def _local_search_improve(subset, bad_pairs, n_max, time_limit=10.0):
    """
    Try to improve the subset by swapping elements or adding/removing.
    """
    start_time = time.time()
    current = set(subset)
    current_list = list(subset)
    all_numbers = set(range(1, n_max + 1))
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for x in all_numbers - current:
            can_add = True
            for (a, b) in bad_pairs[x]:
                if a in current and b in current:
                    can_add = False
                    break
            if can_add:
                current.add(x)
                current_list = list(current)
                improved = True
                break
        
        if improved:
            for x in list(all_numbers - current):
                can_add = True
                for (a, b) in bad_pairs[x]:
                    if a in current and b in current:
                        can_add = False
                        break
                if can_add:
                    current.add(x)
                    current_list = list(current)
        
        if not improved:
            for x in all_numbers - current:
                if time.time() - start_time > time_limit:
                    break
                for y in list(current):
                    new_current = current - {y}
                    can_add = True
                    for (a, b) in bad_pairs[x]:
                        if a in new_current and b in new_current:
                            can_add = False
                            break
                    if can_add:
                        current = new_current
                        current.add(x)
                        current_list = list(current)
                        improved = True
                        break
                if improved:
                    break
    
    return current_list

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    n_max = 128
    numbers = list(range(1, n_max + 1))
    
    # Precompute LCM table
    lcm_table = _precompute_lcms(n_max)
    
    # Build conflict table
    bad_pairs = _build_conflict_table(lcm_table, n_max)
    
    # Strategy 1: Primes and Prime Powers first
    def is_prime_power(n):
        if n < 2:
            return False
        p = n
        while p > 1:
            if p * p > n:
                break
            if n % p == 0:
                n //= p
            else:
                return False
        return True

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in numbers if is_prime(x)]
    prime_powers = [x for x in numbers if is_prime_power(x) and x not in primes]
    others = [x for x in numbers if x not in primes and x not in prime_powers]
    
    order1 = primes + prime_powers + others
    result1 = _greedy_construct_fast(order1, bad_pairs, n_max)
    
    order2 = numbers
    result2 = _greedy_construct_fast(order2, bad_pairs, n_max)
    
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    order3 = sorted(numbers, key=num_divisors)
    result3 = _greedy_construct_fast(order3, bad_pairs, n_max)
    
    order4 = sorted(numbers, key=lambda x: len(bad_pairs[x]))
    result4 = _greedy_construct_fast(order4, bad_pairs, n_max)
    
    best_result = max([result1, result2, result3, result4], key=len)
    
    # Try random shuffles
    random.seed(42)
    for i in range(200):
        if time.time() > deadline:
            break
        shuffled = numbers[:]
        random.shuffle(shuffled)
        res = _greedy_construct_fast(shuffled, bad_pairs, n_max)
        if len(res) > len(best_result):
            best_result = res
    
    # Local search improvement
    if time.time() < deadline - 20:
        improved = _local_search_improve(best_result, bad_pairs, n_max, time_limit=15.0)
        if len(improved) > len(best_result):
            best_result = improved
    
    # Branch and bound for exact optimization
    if time.time() < deadline - 10:
        remaining_time = deadline - time.time() - 5
        if remaining_time > 5:
            bb_result = _branch_and_bound(bad_pairs, n_max, best_result, time_limit=remaining_time)
            if len(bb_result) > len(best_result):
                best_result = bb_result
    
    # Final verification
    def _is_valid_subset(subset):
        n = len(subset)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = subset[i], subset[j], subset[k]
                    if _lcm(a, b) == _lcm(a, c) == _lcm(b, c):
                        return False
        return True
    
    if not _is_valid_subset(best_result):
        best_result = result1
    
    return best_result
