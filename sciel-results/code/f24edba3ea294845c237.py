import random
from itertools import combinations
from math import gcd
import time

N = 128

def compute_conflicts():
    """
    Precompute for each x, the list of pairs (u, v) such that 
    the triple {x, u, v} is a bad triple (all pairwise GCDs equal).
    We store pairs as sorted tuples (min(u,v), max(u,v)).
    """
    conflicts = {i: [] for i in range(1, N + 1)}
    # Generate all bad triples
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    # Triple (a, b, c) is bad
                    # Add (b, c) to conflicts[a]
                    # Add (a, c) to conflicts[b]
                    # Add (a, b) to conflicts[c]
                    
                    p1 = (b, c)
                    p2 = (a, c)
                    p3 = (a, b)
                    
                    conflicts[a].append(p1)
                    conflicts[b].append(p2)
                    conflicts[c].append(p3)
                    
    return conflicts

CONFLICTS = compute_conflicts()

def can_add(x, current_set):
    """
    Check if x can be added to current_set.
    current_set is a set of ints.
    """
    # For each pair (u, v) in conflicts[x], if u in current_set and v in current_set,
    # then adding x creates a bad triple.
    for u, v in CONFLICTS[x]:
        if u in current_set and v in current_set:
            return False
    return True

def greedy_build(order):
    s = set()
    for x in order:
        if can_add(x, s):
            s.add(x)
    return s

def local_search(s, deadline):
    """
    Improve set s with multiple move types, respecting deadline.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any single element
        candidates = [x for x in range(1, N + 1) if x not in s]
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline:
                return s
            if can_add(x, s):
                s.add(x)
                improved = True
                break
        
        if improved or time.time() > deadline:
            continue
            
        # 2. Try 1-for-1 swap: remove one, add one
        list_s = list(s)
        random.shuffle(list_s)
        
        for u in list_s:
            if time.time() > deadline:
                return s
            
            s_without_u = s - {u}
            
            # Try to add an element from the complement
            addable = [x for x in range(1, N + 1) if x not in s]
            random.shuffle(addable)
            
            for x in addable:
                if time.time() > deadline:
                    return s
                if can_add(x, s_without_u):
                    s.remove(u)
                    s.add(x)
                    improved = True
                    break
            
            if improved:
                break
        
        if improved or time.time() > deadline:
            continue
        
        # 3. Try 2-for-1 swap: remove two, add one (net -1, but may enable future +1s)
        # Actually, let's try 2-for-2 or 1-for-2 instead to maintain or increase size
        # Try 1-for-2: remove one, add two
        list_s = list(s)
        random.shuffle(list_s)
        
        for u in list_s:
            if time.time() > deadline:
                return s
            
            s_without_u = s - {u}
            
            # Try to add two elements from the complement
            addable = [x for x in range(1, N + 1) if x not in s]
            random.shuffle(addable)
            
            # Try pairs of addable elements
            found_pair = False
            for i in range(len(addable)):
                if time.time() > deadline:
                    return s
                x1 = addable[i]
                if not can_add(x1, s_without_u):
                    continue
                
                s_with_x1 = s_without_u | {x1}
                
                for j in range(i + 1, len(addable)):
                    if time.time() > deadline:
                        return s
                    x2 = addable[j]
                    if can_add(x2, s_with_x1):
                        s.remove(u)
                        s.add(x1)
                        s.add(x2)
                        improved = True
                        found_pair = True
                        break
                
                if found_pair:
                    break
            
            if improved:
                break

    return s

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_set = set()
    
    # Initial greedy with sorted order
    initial_order = list(range(1, N + 1))
    s = greedy_build(initial_order)
    s = local_search(s, deadline)
    if len(s) > len(best_set):
        best_set = s.copy()
        
    # Try reverse order
    if time.time() < deadline:
        initial_order_rev = list(range(N, 0, -1))
        s = greedy_build(initial_order_rev)
        s = local_search(s, deadline)
        if len(s) > len(best_set):
            best_set = s.copy()

    # Try order by number of conflicts (fewer conflicts first)
    if time.time() < deadline:
        conflict_counts = {x: len(CONFLICTS[x]) for x in range(1, N + 1)}
        initial_order_conf = sorted(range(1, N + 1), key=lambda x: conflict_counts[x])
        s = greedy_build(initial_order_conf)
        s = local_search(s, deadline)
        if len(s) > len(best_set):
            best_set = s.copy()
        
    # Random restarts
    while time.time() < deadline:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        
        s = greedy_build(random_order)
        s = local_search(s, deadline)
        
        if len(s) > len(best_set):
            best_set = s.copy()
            
    return list(best_set)
