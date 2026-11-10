import time
import random
import math
from itertools import combinations

def lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin for safety

    # 1. Precompute conflicts
    # conflicts[x] is a list of tuples (a, b) such that {a,b,x} is a bad triple
    conflicts = {i: [] for i in range(1, 129)}
    for a in range(1, 129):
        for b in range(a + 1, 129):
            for c in range(b + 1, 129):
                if lcm(a, b) == lcm(a, c) == lcm(b, c):
                    # Add to conflicts for a, b, c
                    conflicts[a].append((b, c))
                    conflicts[b].append((a, c))
                    conflicts[c].append((a, b))
                    
    def can_add(x, S):
        # S is a set
        for a, b in conflicts[x]:
            if a in S and b in S:
                return False
        return True

    def local_search(initial_set):
        S = set(initial_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # 1. Try to add any element
            for x in range(1, 129):
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
                    break
            
            # 2. If no element could be added, try swaps
            if not improved:
                # 1-Swap: remove a, add b
                swap_found = False
                for a in list(S):
                    for b in range(1, 129):
                        if b in S:
                            continue
                        S_new = S - {a}
                        if can_add(b, S_new):
                            S_new.add(b)
                            # Check if this swap allows adding at least one more element
                            # to avoid cycling
                            added = False
                            for y in range(1, 129):
                                if y not in S_new and can_add(y, S_new):
                                    S_new.add(y)
                                    added = True
                                    break
                            if added:
                                S = S_new
                                improved = True
                                swap_found = True
                                break
                    if swap_found:
                        break
                
                # 2-Swap (Kicks): remove a,b, add c,d
                if not improved and time.time() < deadline - 5:
                    kick_found = False
                    # Sample a few kicks to save time
                    # Or try all if fast enough
                    # |S| ~ 60, pairs ~ 1800. Candidates ~ 68, pairs ~ 2300.
                    # 1800 * 2300 is too many.
                    # Let's try random kicks
                    for _ in range(100):
                        if time.time() > deadline:
                            break
                        a, b = random.sample(list(S), 2)
                        S_new = S - {a, b}
                        # Try to find c, d
                        candidates = [x for x in range(1, 129) if x not in S_new]
                        random.shuffle(candidates)
                        for c in candidates:
                            if not can_add(c, S_new):
                                continue
                            S_temp = S_new | {c}
                            for d in candidates:
                                if d == c:
                                    continue
                                if not can_add(d, S_temp):
                                    continue
                                S_new = S_temp | {d}
                                # Check if this allows more additions
                                added = False
                                for y in range(1, 129):
                                    if y not in S_new and can_add(y, S_new):
                                        S_new.add(y)
                                        added = True
                                        break
                                if added:
                                    S = S_new
                                    kick_found = True
                                    break
                            if kick_found:
                                break
                        if kick_found:
                            break
                    if kick_found:
                        improved = True

        return S

    best_set = set()
    
    # Strategies for initial order
    # 1. Descending
    # 2. Ascending
    # 3. Safety (fewest conflicts)
    # 4. Random
    
    safety_order = sorted(range(1, 129), key=lambda x: len(conflicts[x]))
    
    strategies = [
        list(range(128, 0, -1)),
        list(range(1, 129)),
        safety_order
    ]
    
    for i in range(100): # 100 restarts
        if time.time() > deadline:
            break
            
        if i < 3:
            order = strategies[i % len(strategies)]
        else:
            order = list(range(1, 129))
            random.shuffle(order)
            
        # Greedy construction
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
                
        # Local search
        S = local_search(S)
        
        if len(S) > len(best_set):
            best_set = S
            if time.time() > deadline:
                break

    return list(best_set)
