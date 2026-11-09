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
# conflicts[x] = set of (u, v) such that (min(x,u,v), mid, max) is a bad triple
# Actually, simpler: for each x, store a set of pairs (u, v) such that {x, u, v} is a bad triple.
# To check if x can be added to S, we check if any (u, v) in conflicts[x] has u in S and v in S.
element_conflicts = {i: set() for i in range(1, N + 1)}
for (a, b, c) in BAD_TRIPLES:
    element_conflicts[a].add((b, c))
    element_conflicts[b].add((a, c))
    element_conflicts[c].add((a, b))

def can_add(x, current_set):
    """Check if x can be added to current_set without forming a bad triple."""
    for u, v in element_conflicts[x]:
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
    Improve set s using local search.
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
            
        # 2. Try 1-for-1 swap: remove one, add one (size stays same, might enable future additions)
        # Actually, 1-for-1 doesn't increase size. We want to increase size.
        # So we should look for moves that increase size.
        # If no single add works, we need to remove some elements to free up space for more elements.
        
        # Strategy: Try to remove k elements and add k+1 elements.
        # Start with k=1 (remove 1, add 2).
        
        list_s = list(s)
        random.shuffle(list_s)
        
        # Try removing 1 element and adding 2
        found_improvement = False
        for u in list_s:
            if time.time() > deadline:
                return s
            
            s_without_u = s - {u}
            
            # Find pairs of elements that can be added to s_without_u
            # This is expensive. Instead, try to add one element to s_without_u, then another.
            
            addable_1 = []
            for x in range(1, N + 1):
                if x not in s_without_u:
                    if can_add(x, s_without_u):
                        addable_1.append(x)
            
            if not addable_1:
                continue
                
            random.shuffle(addable_1)
            
            for x1 in addable_1:
                if time.time() > deadline:
                    return s
                
                s_with_x1 = s_without_u | {x1}
                
                addable_2 = []
                for x2 in range(1, N + 1):
                    if x2 not in s_with_x1:
                        if can_add(x2, s_with_x1):
                            addable_2.append(x2)
                            
                if addable_2:
                    # We found x1, x2 that can be added if u is removed.
                    # This increases size by 1 (remove 1, add 2).
                    s.remove(u)
                    s.add(x1)
                    s.add(addable_2[0])
                    found_improvement = True
                    break
            
            if found_improvement:
                break
        
        if found_improvement:
            improved = True
            continue
            
        # Try removing 2 elements and adding 3
        list_s = list(s)
        random.shuffle(list_s)
        found_improvement = False
        
        # Limit the search space for pairs to avoid O(N^2) explosion in inner loops
        # We only try the first few random pairs
        for i in range(min(len(list_s), 10)):
            u = list_s[i]
            for j in range(i + 1, min(len(list_s), 10)):
                v = list_s[j]
                if time.time() > deadline:
                    return s
                
                s_without_uv = s - {u, v}
                
                # Greedily add elements to s_without_uv
                temp_s = set(s_without_uv)
                added = []
                
                for x in range(1, N + 1):
                    if x not in temp_s:
                        if can_add(x, temp_s):
                            temp_s.add(x)
                            added.append(x)
                            if len(added) >= 3:
                                break
                            
                if len(added) >= 3:
                    # We removed 2, added 3. Net gain +1.
                    s.remove(u)
                    s.remove(v)
                    for x in added[:3]:
                        s.add(x)
                    found_improvement = True
                    break
            
            if found_improvement:
                break
        
        if found_improvement:
            improved = True
            continue
            
        # If stuck, try a kick: remove 3 random elements, then try to fill
        if time.time() < deadline:
             # Simple kick: remove 3 random elements
             if len(s) >= 3:
                 to_remove = random.sample(list(s), 3)
                 for x in to_remove:
                     s.remove(x)
                 # Try to add back
                 for x in range(1, N + 1):
                     if x not in s:
                         if can_add(x, s):
                             s.add(x)
                 improved = True # Force another iteration to see if we can improve from this new state

    return s

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s limit
    
    best_set = set()
    
    # Strategy 1: Greedy with different orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: len(element_conflicts[x])),
        sorted(range(1, N + 1), key=lambda x: -len(element_conflicts[x])),
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
