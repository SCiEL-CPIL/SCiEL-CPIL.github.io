import time
import sys
from itertools import combinations
from math import gcd

_START_TIME = time.time()
_DEADLINE = 85.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _compute_lcm():
    result = 1
    for i in range(1, 23):
        result = result * i // gcd(result, i)
    return result

_LCM = _compute_lcm()
_W = [0] + [_LCM // i for i in range(1, 23)]

def construct():
    # 1. Find minimal dependent sets (bad sets)
    # A set of indices is "bad" if there exists a non-trivial partition
    # into two parts with equal sum of W_i.
    
    bad_sets = []
    
    # Check sizes 3 to 6. 
    # Size 2: W_i == W_j? No, W_i are distinct for i != j.
    # Size 1: No.
    
    max_check_size = 6
    
    for k in range(3, max_check_size + 1):
        if _time_left() < 5.0:
            break
            
        for combo in combinations(range(1, 23), k):
            # Check all partitions of this combo
            # We can iterate over masks from 1 to (1 << k) // 2 - 1
            # to avoid duplicates (A, B) and (B, A)
            # And exclude empty set.
            
            # Sum of all elements in combo
            total_sum = sum(_W[i] for i in combo)
            
            # We look for a subset A of combo such that sum(A) == total_sum // 2
            # This requires total_sum to be even.
            if total_sum % 2 != 0:
                continue
            
            target = total_sum // 2
            
            # Iterate subsets of combo
            # k is small (<=6), so 2^k is small.
            found = False
            for mask in range(1, (1 << k) // 2):
                s = 0
                for j in range(k):
                    if mask & (1 << j):
                        s += _W[combo[j]]
                if s == target:
                    # Found a partition. The whole combo is a bad set.
                    # Is it minimal?
                    # A bad set is minimal if no proper subset is bad.
                    # Since we iterate k from small to large, 
                    # if this set contains a smaller bad set, it's not minimal.
                    # We can check if it contains any existing bad set.
                    is_minimal = True
                    for bs in bad_sets:
                        if bs.issubset(set(combo)):
                            is_minimal = False
                            break
                    
                    if is_minimal:
                        bad_sets.append(frozenset(combo))
                    found = True
                    break
            
            # Optimization: if we found a bad set of size k, 
            # any superset is not minimal. 
            # But we still need to find other minimal sets of size k 
            # that don't contain previous bad sets.
            
    # 2. Solve Maximum Independent Set on the hypergraph
    # Vertices: 1..22
    # Constraint: For each bad_set B, not all vertices in B can be selected.
    # i.e., selected_set cannot contain B.
    
    # Backtracking search
    # State: current selected set, available vertices
    # We want to maximize size.
    
    best_size = 0
    best_set = []
    
    # Sort vertices? 
    # Vertices involved in more bad sets are "more constrained".
    # Let's just use 1..22 order.
    
    def backtrack(available, current_set, current_size):
        nonlocal best_size, best_set
        
        if _time_left() < 1.0:
            return
        
        # Upper bound
        if current_size + len(available) <= best_size:
            return
            
        if not available:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
            
        # Choose a vertex to branch on
        # Heuristic: pick a vertex that is in the most bad sets?
        # Or just the first one.
        v = available[0]
        rest = available[1:]
        
        # Option 1: Exclude v
        backtrack(rest, current_set, current_size)
        
        # Option 2: Include v
        # Check if including v creates a conflict with any bad set
        # i.e., is there a bad set B such that B \ {v} is subset of current_set?
        # If so, adding v completes B, which is invalid.
        
        conflict = False
        for bs in bad_sets:
            if v in bs:
                # Check if all other elements in bs are in current_set
                # If yes, we cannot add v.
                # Actually, we need to check if (bs - {v}) is subset of current_set.
                # If so, adding v makes current_set contain bs.
                # Wait, current_set is the set of *selected* vertices so far.
                # If we add v, new_set = current_set U {v}.
                # We need to ensure no bad_set is subset of new_set.
                # Since current_set was valid, no bad_set is subset of current_set.
                # The only new bad_sets that could be formed are those containing v.
                # So we check if there exists a bad_set bs such that:
                # 1. v in bs
                # 2. bs - {v} is subset of current_set
                
                if (bs - {v}).issubset(current_set):
                    conflict = True
                    break
        
        if not conflict:
            new_set = current_set | {v}
            backtrack(rest, new_set, current_size + 1)
            
    # Initial call
    # We can prune the available list? No, we want to consider all.
    # But we can sort available to improve pruning?
    # Let's just pass the full range.
    
    # To speed up, we can use a greedy lower bound first.
    # Greedy: pick vertices one by one if they don't complete a bad set.
    greedy_set = set()
    for v in range(1, 23):
        conflict = False
        for bs in bad_sets:
            if v in bs and (bs - {v}).issubset(greedy_set):
                conflict = True
                break
        if not conflict:
            greedy_set.add(v)
            
    best_size = len(greedy_set)
    best_set = list(greedy_set)
    
    # Now backtrack to find if we can do better
    # We can restrict the search to vertices not in greedy_set? 
    # No, we might need to swap.
    # But the space is small (22 vertices).
    
    # Optimization: 
    # If we have a candidate set, we can check if it's valid.
    # The backtrack function maintains validity.
    
    # Run backtrack
    # To make it faster, we can order vertices by "degree" (number of bad sets they are in).
    # Higher degree vertices are more likely to cause conflicts, so branching on them might prune earlier?
    # Or the opposite?
    # Let's just run it.
    
    all_vertices = list(range(1, 23))
    # Sort by number of bad sets containing them
    def degree(v):
        return sum(1 for bs in bad_sets if v in bs)
    
    all_vertices.sort(key=degree, reverse=True)
    
    backtrack(all_vertices, set(), 0)
    
    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set: {s}")
    print(f"Size: {len(s)}")
    # Verify
    # Check all subset sums distinct
    sums = {0}
    valid = True
    for x in s:
        w = _W[x]
        new_sums = {ss + w for ss in sums}
        if new_sums & sums:
            valid = False
            break
        sums = sums | new_sums
    print(f"Valid: {valid}")
    print(f"Time: {time.time() - _START_TIME:.2f}s")
