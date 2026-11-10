# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses branch-and-bound with bitset operations, greedy initialization, and local search.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit
    
    # Compute forbidden differences (nonzero quadratic residues mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build conflict graph using bitsets for efficiency
    # conflicts[i] is a bitmask of all j such that (i-j) mod 205 is in forbidden
    conflicts = [0] * m
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                conflicts[i] |= (1 << j)
    
    # Greedy construction to get initial solution
    def greedy_construct(order):
        selected = set()
        available = (1 << m) - 1  # all nodes available
        for node in order:
            if not (available & (1 << node)):
                continue
            selected.add(node)
            available &= ~conflicts[node]
            available &= ~(1 << node)
        return selected
    
    # Get a good initial ordering by sorting by degree (conflict count)
    degrees = [bin(conflicts[i]).count('1') for i in range(m)]
    order = sorted(range(m), key=lambda x: degrees[x])
    best_set = greedy_construct(order)
    best_size = len(best_set)
    
    # Local search: try to add nodes or swap to improve
    def local_search(s):
        s = set(s)
        improved = True
        while improved:
            improved = False
            # Try adding any node that doesn't conflict
            for node in range(m):
                if node not in s:
                    ok = True
                    for v in s:
                        if (1 << v) & conflicts[node]:
                            ok = False
                            break
                    if ok:
                        s.add(node)
                        improved = True
                        break
            # Try swapping: remove one, add two
            if not improved:
                for remove in list(s):
                    s.remove(remove)
                    # Now try to add two nodes
                    added = []
                    for n1 in range(m):
                        if n1 in s or n1 == remove:
                            continue
                        ok1 = True
                        for v in s:
                            if (1 << v) & conflicts[n1]:
                                ok1 = False
                                break
                        if ok1:
                            s.add(n1)
                            added.append(n1)
                            # Try to add one more
                            for n2 in range(m):
                                if n2 in s or n2 == remove or n2 == n1:
                                    continue
                                ok2 = True
                                for v in s:
                                    if (1 << v) & conflicts[n2]:
                                        ok2 = False
                                        break
                                if ok2:
                                    s.add(n2)
                                    added.append(n2)
                                    break
                            break
                    if len(added) >= 2:
                        improved = True
                    else:
                        # Undo
                        s.add(remove)
                        for a in added:
                            s.discard(a)
        return s
    
    best_set = local_search(best_set)
    best_size = len(best_set)
    
    # Branch and bound for maximum independent set
    # Use recursive search with bitsets
    def branch_and_bound(candidates, current_set, current_size, upper_bound):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        if current_size + upper_bound <= best_size:
            return
        if not candidates:
            return
        # Try to add each candidate
        for i in range(m):
            if not (candidates & (1 << i)):
                continue
            if current_size + 1 > best_size:
                new_set = set(current_set)
                new_set.add(i)
                new_candidates = candidates & ~conflicts[i] & ~(1 << i)
                # Compute upper bound: count of remaining candidates
                ub = bin(new_candidates).count('1')
                if current_size + 1 + ub > best_size:
                    branch_and_bound(new_candidates, new_set, current_size + 1, ub)
            candidates &= ~(1 << i)
    
    # Initial B&B from empty set with all candidates
    all_candidates = (1 << m) - 1
    # Use a heuristic: process nodes in order of decreasing degree to prune earlier
    bnb_order = sorted(range(m), key=lambda x: -degrees[x])
    
    # Smarter B&B: build up the solution
    def bnb_build(candidates, current_set, current_size):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return
        if not candidates:
            if current_size > best_size:
                best_set = set(current_set)
                best_size = current_size
            return
        # Upper bound: current_size + number of candidates
        ub = bin(candidates).count('1')
        if current_size + ub <= best_size:
            return
        # Pick a node to branch on (highest degree in candidates for better pruning)
        # Find node with max conflicts within candidates
        best_node = -1
        max_conf = -1
        for i in range(m):
            if candidates & (1 << i):
                conf_count = bin(candidates & conflicts[i]).count('1')
                if conf_count > max_conf:
                    max_conf = conf_count
                    best_node = i
        if best_node == -1:
            return
        
        # Branch 1: include best_node
        new_candidates = candidates & ~conflicts[best_node] & ~(1 << best_node)
        new_set = set(current_set)
        new_set.add(best_node)
        bnb_build(new_candidates, new_set, current_size + 1)
        
        # Branch 2: exclude best_node
        candidates &= ~(1 << best_node)
        bnb_build(candidates, current_set, current_size)
    
    # Run B&B with time limit
    bnb_start = time.time()
    bnb_build(all_candidates, set(), 0)
    
    # Additional local search rounds with different orderings
    for _ in range(50):
        if time.time() > deadline:
            break
        random.shuffle(order)
        s = greedy_construct(order)
        s = local_search(s)
        if len(s) > best_size:
            best_set = s
            best_size = len(best_set)
    
    # Final local search on best
    best_set = local_search(best_set)
    best_size = len(best_set)
    
    return np.array(sorted(best_set), dtype=np.int64)


# EVOLVE-BLOCK-END
