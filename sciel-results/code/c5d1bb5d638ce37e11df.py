import random
from math import gcd
import time
from itertools import combinations

N = 128

def compute_bad_triples():
    """
    Precompute all bad triples (a, b, c) with a < b < c
    where gcd(a,b) = gcd(b,c) = gcd(a,c)
    """
    bad_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    bad_triples.add((a, b, c))
    return bad_triples

BAD_TRIPLES = compute_bad_triples()

# Build for each element, the set of pairs that form a bad triple with it
element_pairs = {i: set() for i in range(1, N + 1)}
for (a, b, c) in BAD_TRIPLES:
    element_pairs[a].add((b, c))
    element_pairs[b].add((a, c))
    element_pairs[c].add((a, b))

def can_add(x, current_set):
    """Check if x can be added to current_set without forming a bad triple."""
    for u, v in element_pairs[x]:
        if u in current_set and v in current_set:
            return False
    return True

def greedy_build(order):
    """Build a set greedily following the given order."""
    s = set()
    for x in order:
        if can_add(x, s):
            s.add(x)
    return s

def local_search(s, deadline, max_iter=100):
    """
    Improve set s using local search with multiple move types.
    """
    s = set(s)
    improved = True
    iter_count = 0
    while improved and time.time() < deadline and iter_count < max_iter:
        improved = False
        iter_count += 1
        
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
        
        if improved:
            continue
            
        # 2. Try 1-for-1 swap: remove one, add one
        list_s = list(s)
        random.shuffle(list_s)
        
        for u in list_s:
            if time.time() > deadline:
                return s
            
            s_without_u = s - {u}
            
            addable = [x for x in range(1, N + 1) if x not in s and x != u]
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
        
        # 3. Try 2-for-3 swap: remove two, add three (increase size by 1)
        if not improved and time.time() < deadline:
            list_s = list(s)
            random.shuffle(list_s)
            for i in range(min(len(list_s), 15)):
                u = list_s[i]
                for j in range(i + 1, min(len(list_s), 15)):
                    v = list_s[j]
                    if time.time() > deadline:
                        return s
                    s_without_uv = s - {u, v}
                    addable = [x for x in range(1, N + 1) if x not in s and x != u and x != v]
                    random.shuffle(addable)
                    
                    # Try to find three elements that can all be added
                    found_triple = False
                    for i1 in range(len(addable)):
                        if time.time() > deadline:
                            return s
                        x1 = addable[i1]
                        if can_add(x1, s_without_uv):
                            s_with_x1 = s_without_uv | {x1}
                            for i2 in range(i1 + 1, len(addable)):
                                x2 = addable[i2]
                                if can_add(x2, s_with_x1):
                                    s_with_x1x2 = s_with_x1 | {x2}
                                    for i3 in range(i2 + 1, len(addable)):
                                        x3 = addable[i3]
                                        if can_add(x3, s_with_x1x2):
                                            s.remove(u)
                                            s.remove(v)
                                            s.add(x1)
                                            s.add(x2)
                                            s.add(x3)
                                            found_triple = True
                                            improved = True
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # 4. Try 3-for-4 swap: remove three, add four (increase size by 1)
        if not improved and time.time() < deadline:
            list_s = list(s)
            random.shuffle(list_s)
            for i in range(min(len(list_s), 10)):
                u = list_s[i]
                for j in range(i + 1, min(len(list_s), 10)):
                    v = list_s[j]
                    for k in range(j + 1, min(len(list_s), 10)):
                        w = list_s[k]
                        if time.time() > deadline:
                            return s
                        s_without_uvw = s - {u, v, w}
                        addable = [x for x in range(1, N + 1) if x not in s and x != u and x != v and x != w]
                        random.shuffle(addable)
                        
                        # Try to find four elements that can all be added
                        found_quad = False
                        for i1 in range(len(addable)):
                            if time.time() > deadline:
                                return s
                            x1 = addable[i1]
                            if can_add(x1, s_without_uvw):
                                s_with_x1 = s_without_uvw | {x1}
                                for i2 in range(i1 + 1, len(addable)):
                                    x2 = addable[i2]
                                    if can_add(x2, s_with_x1):
                                        s_with_x1x2 = s_with_x1 | {x2}
                                        for i3 in range(i2 + 1, len(addable)):
                                            x3 = addable[i3]
                                            if can_add(x3, s_with_x1x2):
                                                s_with_x1x2x3 = s_with_x1x2 | {x3}
                                                for i4 in range(i3 + 1, len(addable)):
                                                    x4 = addable[i4]
                                                    if can_add(x4, s_with_x1x2x3):
                                                        s.remove(u)
                                                        s.remove(v)
                                                        s.remove(w)
                                                        s.add(x1)
                                                        s.add(x2)
                                                        s.add(x3)
                                                        s.add(x4)
                                                        found_quad = True
                                                        improved = True
                                                        break
                                                if improved:
                                                    break
                                            if improved:
                                                break
                                        if improved:
                                            break
                                    if improved:
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    return s

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_set = set()
    
    # Strategy 1: Greedy with different orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: len(element_pairs[x])),
        sorted(range(1, N + 1), key=lambda x: -len(element_pairs[x])),
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        s = greedy_build(order)
        s = local_search(s, deadline, max_iter=50)
        if len(s) > len(best_set):
            best_set = s.copy()
    
    # Strategy 2: Random restarts with local search
    while time.time() < deadline:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        
        s = greedy_build(random_order)
        s = local_search(s, deadline, max_iter=30)
        
        if len(s) > len(best_set):
            best_set = s.copy()
    
    return list(best_set)
