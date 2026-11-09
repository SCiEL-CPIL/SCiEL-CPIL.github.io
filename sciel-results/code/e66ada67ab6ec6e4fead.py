"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt
from itertools import combinations

P = {'n': 256}
_DEADLINE = time.time() + 75  # 75s safety margin before 90s kill

def _valid_set(s):
    """Check if set s satisfies: for all distinct a,b,c in s, a does not divide b*c."""
    lst = sorted(s)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(n):
            if j == i:
                continue
            b = lst[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = lst[k]
                if (b * c) % a == 0:
                    return False
    return True

def _can_add(s, x):
    """Check if adding x to set s maintains the constraint."""
    lst = sorted(s)
    n = len(lst)
    # Check all triples involving x
    for i in range(n):
        a = lst[i]
        # x and another element as b,c with a as divisor
        for j in range(n):
            if j == i:
                continue
            b = lst[j]
            if (b * x) % a == 0:
                return False
        # a = x, b and c from s
        for j in range(n):
            b = lst[j]
            for k in range(n):
                if k == j:
                    continue
                c = lst[k]
                if (b * c) % x == 0:
                    return False
    return True

def _greedy_add(order, n_max):
    """Greedy: try to add elements in given order."""
    s = set()
    for x in order:
        if x > n_max:
            continue
        if _can_add(s, x):
            s.add(x)
    return s

def _build_incompatibility_graph(n_max):
    """
    Build a graph where vertices are 2..n_max and an edge between x,y exists
    if x and y are incompatible (cannot both be in a valid set).
    
    Two elements x,y are incompatible if there exists some z (distinct from x,y, in 2..n_max)
    such that x divides y*z OR y divides x*z.
    
    Note: This is a sufficient condition for incompatibility but not necessary.
    The true constraint is 3-uniform, but this pairwise approximation helps guide search.
    """
    nums = list(range(2, n_max + 1))
    num_set = set(nums)
    n = len(nums)
    
    # adjacency as bitmasks or sets
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        x = nums[i]
        for j in range(i + 1, n):
            y = nums[j]
            # Check if x and y are incompatible
            # x divides y*z for some z != x,y
            # y divides x*z for some z != x,y
            incompatible = False
            
            # Check if x | y*z for some z in 2..n_max, z != x, z != y
            # x | y*z means (y*z) % x == 0
            # We need to check if there exists z such that z % (x / gcd(x,y)) == 0 and z != x, z != y
            g = _gcd(x, y)
            x_reduced = x // g
            # We need z such that x_reduced | z, and z != x, z != y, 2 <= z <= n_max
            # z must be a multiple of x_reduced
            # Check if any multiple of x_reduced in [2, n_max] is not x and not y
            found_z = False
            for z in range(x_reduced, n_max + 1, x_reduced):
                if z != x and z != y and z >= 2:
                    found_z = True
                    break
            if found_z:
                incompatible = True
            
            if not incompatible:
                # Check if y | x*z for some z != x, z != y
                y_reduced = y // g
                found_z = False
                for z in range(y_reduced, n_max + 1, y_reduced):
                    if z != x and z != y and z >= 2:
                        found_z = True
                        break
                if found_z:
                    incompatible = True
            
            if incompatible:
                adj[i].add(j)
                adj[j].add(i)
    
    return nums, adj

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _mis_greedy(nums, adj, order=None):
    """
    Greedy Maximum Independent Set on the incompatibility graph.
    Returns a set of indices that form an independent set (valid set).
    """
    n = len(nums)
    if order is None:
        # Sort by degree (ascending) - prefer low-degree vertices
        order = sorted(range(n), key=lambda i: len(adj[i]))
    
    selected = set()
    for i in order:
        # Check if i can be added: no conflict with selected vertices
        can_add = True
        for j in selected:
            if j in adj[i]:
                can_add = False
                break
        if can_add:
            selected.add(i)
    
    return selected

def _mis_branch_and_bound(nums, adj, deadline):
    """
    Branch and bound for Maximum Independent Set.
    Uses coloring-based upper bound.
    """
    n = len(nums)
    
    # Compute initial upper bound using greedy coloring
    def _color_bound(candidates):
        """Compute an upper bound on MIS size using greedy coloring."""
        if not candidates:
            return 0
        # Greedy coloring: assign colors to minimize max color
        color_of = {}
        next_color = 0
        for v in sorted(candidates, key=lambda x: -len(adj[x] & set(candidates))):
            used = set()
            for u in adj[v]:
                if u in color_of:
                    used.add(color_of[u])
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
            next_color = max(next_color, c + 1)
        return next_color
    
    best_set = set()
    best_size = 0
    
    def _bb(candidates, current_set):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        # Upper bound
        ub = _color_bound(candidates)
        if len(current_set) + ub <= best_size:
            return
        
        # Choose vertex with max degree in candidates
        cand_list = list(candidates)
        v = max(cand_list, key=lambda x: len(adj[x] & set(candidates)))
        
        # Branch 1: include v
        new_cand = candidates - {v} - adj[v]
        _bb(new_cand, current_set | {v})
        
        # Branch 2: exclude v
        _bb(candidates - {v}, current_set)
    
    _bb(set(range(n)), set())
    return best_set

def _mis_with_seed(nums, adj, seed_set, deadline):
    """
    Find MIS that includes all vertices in seed_set.
    """
    n = len(nums)
    # Remove all vertices that conflict with seed_set
    forbidden = set()
    for v in seed_set:
        forbidden.add(v)
        forbidden.update(adj[v])
    
    candidates = set(range(n)) - forbidden
    current_set = set(seed_set)
    
    best_set = set(current_set)
    best_size = len(current_set)
    
    def _color_bound(candidates):
        if not candidates:
            return 0
        color_of = {}
        next_color = 0
        for v in sorted(candidates, key=lambda x: -len(adj[x] & set(candidates))):
            used = set()
            for u in adj[v]:
                if u in color_of:
                    used.add(color_of[u])
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
            next_color = max(next_color, c + 1)
        return next_color
    
    def _bb(candidates, current_set):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        ub = _color_bound(candidates)
        if len(current_set) + ub <= best_size:
            return
        
        cand_list = list(candidates)
        v = max(cand_list, key=lambda x: len(adj[x] & set(candidates)))
        
        # Branch 1: include v
        new_cand = candidates - {v} - adj[v]
        _bb(new_cand, current_set | {v})
        
        # Branch 2: exclude v
        _bb(candidates - {v}, current_set)
    
    _bb(candidates, current_set)
    return best_set

def construct():
    start = time.time()
    n_max = P['n']
    deadline = start + 70  # Safety margin
    
    # Get primes
    primes = [a for a in range(2, n_max+1) if all(a%d for d in range(2, isqrt(a)+1))]
    
    # Build incompatibility graph
    nums, adj = _build_incompatibility_graph(n_max)
    num_to_idx = {nums[i]: i for i in range(n_max - 1)}
    
    best = set()
    best_size = 0
    
    # Strategy 1: Greedy MIS with different orderings
    for order_func in [
        lambda: sorted(range(n_max - 1), key=lambda i: len(adj[i])),  # min degree
        lambda: sorted(range(n_max - 1), key=lambda i: -len(adj[i])),  # max degree
        lambda: sorted(range(n_max - 1), key=lambda i: -nums[i]),  # large numbers
        lambda: sorted(range(n_max - 1), key=lambda i: nums[i]),  # small numbers
        lambda: sorted(range(n_max - 1), key=lambda i: -nums[i].bit_length()),
    ]:
        if time.time() > deadline:
            break
        order = order_func()
        sel_idx = _mis_greedy(nums, adj, order)
        sel_set = {nums[i] for i in sel_idx}
        if len(sel_set) > best_size:
            best_size = len(sel_set)
            best = sel_set
    
    # Strategy 2: Branch and bound (with time limit)
    if time.time() < deadline - 10:
        bb_deadline = min(time.time() + 15, deadline - 2)
        sel_idx = _mis_branch_and_bound(nums, adj, bb_deadline)
        sel_set = {nums[i] for i in sel_idx}
        if len(sel_set) > best_size:
            best_size = len(sel_set)
            best = sel_set
    
    # Strategy 3: MIS with prime seed
    if time.time() < deadline - 5:
        prime_indices = {num_to_idx[p] for p in primes}
        bb_deadline = min(time.time() + 10, deadline - 2)
        sel_idx = _mis_with_seed(nums, adj, prime_indices, bb_deadline)
        sel_set = {nums[i] for i in sel_idx}
        if len(sel_set) > best_size:
            best_size = len(sel_set)
            best = sel_set
    
    # Strategy 4: Original greedy approaches as fallback
    if time.time() < deadline - 3:
        composites = [a for a in range(2, n_max+1) if a not in set(primes)]
        for order in [
            sorted(composites, reverse=True),
            sorted(composites),
        ]:
            if time.time() > deadline:
                break
            s = set(primes)
            for x in order:
                if time.time() > deadline:
                    break
                if _can_add(s, x):
                    s.add(x)
            if len(s) > best_size:
                best_size = len(s)
                best = s
    
    # Strategy 5: Local improvement via swaps
    if time.time() < deadline - 1:
        current = list(best)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                trial = set(current)
                trial.remove(current[i])
                for x in range(2, n_max+1):
                    if x in trial:
                        continue
                    if _can_add(trial, x):
                        trial.add(x)
                        if len(trial) > len(current):
                            current = list(trial)
                            best = set(trial)
                            best_size = len(current)
                            improved = True
                        break
    
    # Verify the result
    if not _valid_set(best):
        # Fallback to primes if something went wrong
        best = set(primes)
        best_size = len(primes)
    
    result = sorted(best)
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
