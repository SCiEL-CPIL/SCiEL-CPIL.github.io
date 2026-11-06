<<<<<<< SEARCH
import time
from itertools import combinations
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Greedy: try to build a large set
    # Start with all numbers, then remove those that form bad triples
    # Or greedy add in an order that maximizes size
    
    # Try greedy adding from 128 down to 1
    selected = []
    for x in range(128, 0, -1):
        if time.time() > deadline:
            break
        # Check if adding x creates a bad triple with any two existing elements
        valid = True
        n = len(selected)
        for i in range(n):
            if time.time() > deadline:
                break
            a = selected[i]
            lcm_ax = lcm(a, x)
            for j in range(i+1, n):
                b = selected[j]
                if lcm(a, b) == lcm_ax and lcm(b, x) == lcm_ax:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
    
    if not selected:
        selected = [1]
    
    return selected
=======
import time
import random
from math import lcm
from collections import defaultdict

def _precompute_conflicts():
    """Precompute for each element x, the set of pairs (i,j) in [1,128]
    such that {x, i, j} forms a bad triple."""
    N = 128
    # For each x, find all pairs (a,b) with a<b, a!=x, b!=x,
    # such that lcm(a,b)==lcm(a,x)==lcm(b,x)
    conflict_pairs = [set() for _ in range(N+1)]
    # Also precompute lcm table for speed
    lcm_table = [[0]*(N+1) for _ in range(N+1)]
    for a in range(1, N+1):
        for b in range(a, N+1):
            lcm_table[a][b] = lcm(a, b)
            lcm_table[b][a] = lcm(a, b)
    
    for x in range(1, N+1):
        for a in range(1, N+1):
            if a == x:
                continue
            lcm_ax = lcm_table[a][x]
            for b in range(a+1, N+1):
                if b == x:
                    continue
                if lcm_table[a][b] == lcm_ax and lcm_table[b][x] == lcm_ax:
                    conflict_pairs[x].add((a, b))
    return conflict_pairs, lcm_table

def _is_valid_set(s, conflict_pairs, lcm_table):
    """Check if set s contains no bad triple."""
    s_set = set(s)
    for x in s:
        for (a, b) in conflict_pairs[x]:
            if a in s_set and b in s_set:
                return False
    return True

def _greedy_build(order, conflict_pairs, deadline):
    """Greedy build given an order of elements to try."""
    selected = []
    selected_set = set()
    for x in order:
        if time.time() > deadline:
            break
        valid = True
        for (a, b) in conflict_pairs[x]:
            if a in selected_set and b in selected_set:
                valid = False
                break
        if valid:
            selected.append(x)
            selected_set.add(x)
    return selected, selected_set

def _local_search(selected, selected_set, conflict_pairs, lcm_table, deadline):
    """Try single-swap and double-swap moves to improve the set."""
    N = 128
    all_elements = list(range(1, N+1))
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Single swap: remove x from selected, add y not in selected
        # Try removing each x and see if we can add more
        for x in list(selected):
            if time.time() > deadline:
                return selected, selected_set
            # Remove x
            selected_set.remove(x)
            selected.remove(x)
            
            # Try adding elements not in selected_set
            added = False
            for y in all_elements:
                if y in selected_set:
                    continue
                # Check if y can be added
                valid = True
                for (a, b) in conflict_pairs[y]:
                    if a in selected_set and b in selected_set:
                        valid = False
                        break
                if valid:
                    selected_set.add(y)
                    selected.append(y)
                    added = True
                    break
            
            if not added:
                # Put x back
                selected_set.add(x)
                selected.append(x)
        
        # Double swap: remove x, y from selected, add u, v not in selected
        # Only try if we can find a beneficial swap
        n = len(selected)
        if n >= 2:
            for i in range(n):
                if time.time() > deadline:
                    return selected, selected_set
                x = selected[i]
                for j in range(i+1, n):
                    y = selected[j]
                    # Remove x and y
                    selected_set.remove(x)
                    selected_set.remove(y)
                    selected.remove(x)
                    selected.remove(y)
                    
                    # Try adding two elements
                    added_count = 0
                    added_elements = []
                    for u in all_elements:
                        if u in selected_set:
                            continue
                        valid = True
                        for (a, b) in conflict_pairs[u]:
                            if a in selected_set and b in selected_set:
                                valid = False
                                break
                        if valid:
                            selected_set.add(u)
                            added_elements.append(u)
                            added_count += 1
                            if added_count == 2:
                                break
                    
                    if added_count == 2:
                        # We swapped 2 out for 2 in - no net gain, but maybe better structure
                        # Actually this doesn't help cardinality. Skip.
                        # Undo
                        for u in added_elements:
                            selected_set.remove(u)
                        selected_set.add(x)
                        selected_set.add(y)
                        selected.append(x)
                        selected.append(y)
                    else:
                        # Undo
                        for u in added_elements:
                            selected_set.remove(u)
                        selected_set.add(x)
                        selected_set.add(y)
                        selected.append(x)
                        selected.append(y)
    
    return selected, selected_set

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Precompute conflicts
    conflict_pairs, lcm_table = _precompute_conflicts()
    
    best_selected = []
    best_size = 0
    
    # Multiple restarts with different random orders
    random.seed(42)
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            order = list(range(128, 0, -1))
        elif restart == 1:
            order = list(range(1, 129))
        elif restart == 2:
            # Prime-heavy order
            order = []
            for x in range(1, 129):
                if x > 1:
                    is_prime = True
                    for p in range(2, int(x**0.5)+1):
                        if x % p == 0:
                            is_prime = False
                            break
                    if is_prime:
                        order.append(x)
            for x in range(1, 129):
                if x not in order:
                    order.append(x)
        elif restart == 3:
            # Power-of-2 heavy
            order = []
            for k in range(7, -1, -1):
                order.append(2**k)
            for x in range(1, 129):
                if x not in order:
                    order.append(x)
        else:
            order = list(range(1, 129))
            random.shuffle(order)
        
        selected, selected_set = _greedy_build(order, conflict_pairs, deadline)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = list(selected)
        
        # Local search on this result
        if time.time() < deadline:
            selected_ls, selected_set_ls = _local_search(
                list(selected), set(selected_set), conflict_pairs, lcm_table, deadline
            )
            if len(selected_ls) > best_size:
                best_size = len(selected_ls)
                best_selected = list(selected_ls)
        
        # If we've found a very large set, we can stop early
        if best_size >= 100 and time.time() > start_time + 60:
            break
    
    if not best_selected:
        best_selected = [1]
    
    return best_selected
>>>>>>> REPLACE
