import time
from math import gcd
from collections import defaultdict
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    random.seed(12345)
    
    # Precompute LCM table
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) with a<b<c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    num_bad = len(bad_triples)
    
    # For each element, which bad triples does it appear in?
    elem_triples = [set() for _ in range(n+1)]
    for i, (a, b, c) in enumerate(bad_triples):
        elem_triples[a].add(i)
        elem_triples[b].add(i)
        elem_triples[c].add(i)
    
    # For each bad triple, store the three elements
    triple_elems = bad_triples  # (a,b,c)
    
    # Fast violation check: for element x, given a bitmask of selected elements,
    # count how many bad triples containing x are fully selected
    # We use integer bitmasks for speed
    
    # Precompute for each element x and each bad triple t containing x:
    # the other two elements
    x_other_pairs = [None] * (n+1)
    for x in range(1, n+1):
        x_other_pairs[x] = []
        for t in elem_triples[x]:
            a, b, c = triple_elems[t]
            if a == x:
                x_other_pairs[x].append((b, c))
            elif b == x:
                x_other_pairs[x].append((a, c))
            else:
                x_other_pairs[x].append((a, b))
    
    # Greedy hitting set: remove elements to hit all bad triples
    # Start with all selected, remove elements that cover the most active triples
    selected = [True] * (n + 1)
    active = set(range(num_bad))
    
    while active:
        if time.time() > deadline - 35:
            break
        best_x = -1
        best_cov = 0
        for x in range(1, n+1):
            if not selected[x]:
                continue
            cov = sum(1 for t in elem_triples[x] if t in active)
            if cov > best_cov:
                best_cov = cov
                best_x = x
        if best_x == -1 or best_cov == 0:
            break
        selected[best_x] = False
        active -= elem_triples[best_x]
    
    # Verify validity
    def is_valid(sel):
        for t in range(num_bad):
            a, b, c = triple_elems[t]
            if sel[a] and sel[b] and sel[c]:
                return False
        return True
    
    assert is_valid(selected), "Initial solution invalid"
    
    best_size = sum(selected[1:])
    best_sel = selected[:]
    
    # Fast add_violations using precomputed pairs
    def add_violations(sel, x):
        cnt = 0
        for (y, z) in x_other_pairs[x]:
            if sel[y] and sel[z]:
                cnt += 1
        return cnt
    
    # Phase 1: Greedy addition - add any element with 0 violations
    changed = True
    while changed and time.time() < deadline - 30:
        changed = False
        for x in range(1, n+1):
            if not selected[x] and add_violations(selected, x) == 0:
                selected[x] = True
                changed = True
    
    best_size = sum(selected[1:])
    best_sel = selected[:]
    
    # Phase 2: Local search with 2-opt moves (remove 1, add 1 or remove 1, add 2)
    # Use bitmask for faster operations
    # Represent selection as a set for clarity, but use lists for speed
    
    def can_add(sel, x):
        return add_violations(sel, x) == 0
    
    # 2-opt: remove element r, then try to add one or more elements
    for iteration in range(500):
        if time.time() > deadline - 25:
            break
        improved = False
        
        # Try removing each selected element and greedily adding back
        for r in range(1, n+1):
            if not selected[r]:
                continue
            if time.time() > deadline - 25:
                break
            
            saved = selected[r]
            selected[r] = False
            
            # Greedily add elements in order of fewest conflicts
            candidates = [x for x in range(1, n+1) if not selected[x]]
            # Sort by number of violations (fewer first)
            candidates.sort(key=lambda x: add_violations(selected, x))
            
            added = []
            for x in candidates:
                if add_violations(selected, x) == 0:
                    selected[x] = True
                    added.append(x)
            
            net_gain = len(added) - 1
            if net_gain > 0:
                best_size = sum(selected[1:])
                best_sel = selected[:]
                improved = True
            else:
                # Revert
                for x in added:
                    selected[x] = False
                selected[r] = True
        
        if not improved and time.time() > deadline - 20:
            break
    
    # Phase 3: Simulated annealing with bitmask
    # Use a different approach: random restarts with greedy + local search
    
    best_overall_size = best_size
    best_overall_sel = best_sel[:]
    
    num_restarts = 30
    restart_deadline = deadline - 5
    
    for restart in range(num_restarts):
        if time.time() > restart_deadline:
            break
        
        # Random initial solution: start with random subset, then greedily add
        random.seed(12345 + restart * 7919)
        cur_sel = [False] * (n + 1)
        
        # Randomly select some elements
        for x in range(1, n+1):
            if random.random() < 0.5:
                cur_sel[x] = True
        
        # Remove violations
        changed = True
        while changed and time.time() < restart_deadline:
            changed = False
            for t in range(num_bad):
                a, b, c = triple_elems[t]
                if cur_sel[a] and cur_sel[b] and cur_sel[c]:
                    # Remove one of them (random choice)
                    victim = random.choice([a, b, c])
                    cur_sel[victim] = False
                    changed = True
        
        # Greedy addition
        changed = True
        while changed and time.time() < restart_deadline:
            changed = False
            for x in range(1, n+1):
                if not cur_sel[x] and add_violations(cur_sel, x) == 0:
                    cur_sel[x] = True
                    changed = True
        
        cur_size = sum(cur_sel[1:])
        if cur_size > best_overall_size:
            best_overall_size = cur_size
            best_overall_sel = cur_sel[:]
        
        # Local search on this solution
        for ls_iter in range(200):
            if time.time() > restart_deadline:
                break
            ls_improved = False
            
            # Try 1-remove, multi-add
            for r in range(1, n+1):
                if not cur_sel[r]:
                    continue
                if time.time() > restart_deadline:
                    break
                
                cur_sel[r] = False
                candidates = [x for x in range(1, n+1) if not cur_sel[x]]
                candidates.sort(key=lambda x: add_violations(cur_sel, x))
                
                added = []
                for x in candidates:
                    if add_violations(cur_sel, x) == 0:
                        cur_sel[x] = True
                        added.append(x)
                
                net_gain = len(added) - 1
                if net_gain > 0:
                    cur_size += net_gain
                    ls_improved = True
                    if cur_size > best_overall_size:
                        best_overall_size = cur_size
                        best_overall_sel = cur_sel[:]
                else:
                    for x in added:
                        cur_sel[x] = False
                    cur_sel[r] = True
            
            if not ls_improved:
                break
    
    # Final polish
    for x in range(1, n+1):
        if not best_overall_sel[x] and add_violations(best_overall_sel, x) == 0:
            best_overall_sel[x] = True
    
    result = [x for x in range(1, n+1) if best_overall_sel[x]]
    if not result:
        result = [1]
    
    return result
