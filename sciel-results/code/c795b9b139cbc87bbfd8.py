"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
            
    # For each pair (a,b), find all z such that {a,b,z} is a bad triple
    pair_witness = defaultdict(set)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            for z in range(1, n + 1):
                if z == a or z == b:
                    continue
                if lcm_table[a][z] == L_ab and lcm_table[b][z] == L_ab:
                    pair_witness[(a, b)].add(z)
    
    # For each element x, precompute which pairs (a,b) would be "blocked" by x
    blocked_by = [set() for _ in range(n + 1)]
    for (a, b), witnesses in pair_witness.items():
        for z in witnesses:
            blocked_by[z].add((a, b))
    
    def can_add(x, current_set):
        """Check if x can be added to current_set without forming a bad triple."""
        lst = sorted(current_set)
        k = len(lst)
        for i in range(k):
            a = lst[i]
            for j in range(i + 1, k):
                b = lst[j]
                if (a, b) in blocked_by[x]:
                    return False
        return True
    
    def greedy_add(order):
        """Greedy: add elements in given order if valid."""
        selected = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, selected):
                selected.add(x)
        return selected
    
    def local_search(initial_set, max_iters=200):
        """Local search: try to improve by removing one and adding another, or just adding."""
        current_set = set(initial_set)
        improved = True
        iter_count = 0
        while improved and time.time() < deadline - 2 and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Try to add an element first (cheaper)
            for in_elem in range(1, n + 1):
                if in_elem in current_set:
                    continue
                if can_add(in_elem, current_set):
                    current_set.add(in_elem)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add another
            for out_elem in list(current_set):
                if time.time() > deadline - 2:
                    break
                trial_base = current_set - {out_elem}
                for in_elem in range(1, n + 1):
                    if in_elem in current_set:
                        continue
                    if can_add(in_elem, trial_base):
                        current_set = trial_base | {in_elem}
                        improved = True
                        break
                if improved:
                    break
        return current_set
    
    # Branch and bound for maximum independent set in the hypergraph
    # We model this as: find largest subset with no bad triples
    # Use a greedy upper bound: for remaining candidates, at most ceil(len/candidates_that_conflict)
    
    def branch_and_bound(candidates, current_set, best_size, best_set_holder):
        """
        Branch and bound to find maximum valid set.
        candidates: set of elements not yet decided
        current_set: elements already included
        best_size: current best size found
        best_set_holder: [set] to hold the best set found
        """
        if time.time() > deadline - 1:
            return
        
        # Upper bound: current_size + len(candidates)
        if len(current_set) + len(candidates) <= best_size[0]:
            return
        
        if not candidates:
            if len(current_set) > best_size[0]:
                best_size[0] = len(current_set)
                best_set_holder[0] = set(current_set)
            return
        
        # Choose an element to branch on
        # Heuristic: pick element with most conflicts in candidates
        def conflict_in_candidates(x):
            count = 0
            lst = sorted(candidates)
            k = len(lst)
            for i in range(k):
                a = lst[i]
                for j in range(i + 1, k):
                    b = lst[j]
                    if (a, b) in blocked_by[x]:
                        count += 1
            return count
        
        # Pick element with highest conflict count
        best_elem = None
        best_conflict = -1
        for x in candidates:
            c = conflict_in_candidates(x)
            if c > best_conflict:
                best_conflict = c
                best_elem = x
        
        # Branch 1: include best_elem
        if can_add(best_elem, current_set):
            new_current = current_set | {best_elem}
            # Remove candidates that conflict with best_elem
            new_candidates = set()
            for x in candidates:
                if x == best_elem:
                    continue
                # Check if x can be added with best_elem in current_set
                if can_add(x, new_current):
                    new_candidates.add(x)
            branch_and_bound(new_candidates, new_current, best_size, best_set_holder)
        
        # Branch 2: exclude best_elem
        new_candidates = candidates - {best_elem}
        branch_and_bound(new_candidates, current_set, best_size, best_set_holder)
    
    random.seed(42)
    best_set = set()
    
    # Strategy: multiple restarts with different heuristics
    conflict_counts = [0] * (n + 1)
    for x in range(1, n + 1):
        conflict_counts[x] = len(blocked_by[x])
    
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different ordering strategies
        if i % 3 == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif i % 3 == 1:
            order = sorted(range(1, n + 1), key=lambda x: conflict_counts[x])
        else:
            order = sorted(range(1, n + 1), key=lambda x: -conflict_counts[x])
        
        s = greedy_add(order)
        s_improved = local_search(s)
        if len(s_improved) > len(best_set):
            best_set = s_improved
    
    # Use branch and bound starting from the best greedy solution
    if time.time() < deadline - 5 and best_set:
        best_size = [len(best_set)]
        best_set_holder = [set(best_set)]
        candidates = set(range(1, n + 1)) - best_set
        branch_and_bound(candidates, best_set, best_size, best_set_holder)
        if len(best_set_holder[0]) > len(best_set):
            best_set = best_set_holder[0]
    
    # Final pass: try to add any remaining elements
    if time.time() < deadline - 2:
        for in_elem in range(1, n + 1):
            if in_elem in best_set:
                continue
            if can_add(in_elem, best_set):
                best_set.add(in_elem)
    
    # Verify and return
    if not best_set:
        best_set = {1}
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
