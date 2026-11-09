import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_bad_triples(max_n):
    """
    Precompute all bad triples {a,b,c} with a<b<c<=max_n
    where lcm(a,b)==lcm(a,c)==lcm(b,c).
    """
    n = max_n
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            l_ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == l_ab and lcm_table[b][c] == l_ab:
                    bad_triples.append((a, b, c))
    return bad_triples

def _greedy_removal(bad_triples, max_n, tie_breaker='random'):
    """
    Greedy removal of elements to eliminate all bad triples.
    Returns the remaining set.
    """
    in_set = [True] * (max_n + 1)
    # degree[x] = number of bad triples involving x that are fully in_set
    degree = [0] * (max_n + 1)
    
    # Map element to list of indices of triples containing it
    elem_to_triples = {i: [] for i in range(1, max_n + 1)}
    for idx, (a, b, c) in enumerate(bad_triples):
        elem_to_triples[a].append(idx)
        elem_to_triples[b].append(idx)
        elem_to_triples[c].append(idx)
        
    # Initialize degrees
    for t in bad_triples:
        degree[t[0]] += 1
        degree[t[1]] += 1
        degree[t[2]] += 1
        
    # active_triples: set of indices of triples that are currently "bad" (all 3 in set)
    active_triples = set(range(len(bad_triples)))
    
    removed = []
    
    while active_triples:
        # Find element with max degree
        max_deg = -1
        candidates = []
        for x in range(1, max_n + 1):
            if in_set[x]:
                if degree[x] > max_deg:
                    max_deg = degree[x]
                    candidates = [x]
                elif degree[x] == max_deg:
                    candidates.append(x)
        
        if not candidates:
            break
            
        if tie_breaker == 'random':
            x = random.choice(candidates)
        elif tie_breaker == 'min':
            x = min(candidates)
        elif tie_breaker == 'max':
            x = max(candidates)
        else:
            x = candidates[0]
            
        # Remove x
        in_set[x] = False
        removed.append(x)
        
        # Update degrees for neighbors
        for t_idx in elem_to_triples[x]:
            a, b, c = bad_triples[t_idx]
            if in_set[a] and in_set[b] and in_set[c]:
                # This triple was active, now it's not because x is removed
                # But wait, if x is removed, the triple is no longer "in the set".
                # We need to decrement the degree of the other two elements
                # ONLY IF the triple was previously counted in their degrees.
                # The triple is counted in degree[y] if y is in the triple AND the triple is active.
                # Since we are removing x, the triple becomes inactive.
                # So we decrement degree for the other two.
                
                # Identify the other two
                others = [y for y in (a, b, c) if y != x]
                for y in others:
                    if in_set[y]:
                        degree[y] -= 1
                active_triples.remove(t_idx)
            # If the triple was not active (some other element already removed), 
            # it wasn't contributing to degrees, so no change needed.
            
    return [i for i in range(1, max_n + 1) if in_set[i]]

def _greedy_inclusion(bad_triples, max_n, order='random'):
    """
    Greedy inclusion: add elements if they don't form bad triples with existing set.
    """
    # Precompute for fast check: 
    # For a candidate x, and existing set S, check if any (a,b) in S forms bad triple with x.
    # Bad triple (a,b,x) requires lcm(a,b)==lcm(a,x)==lcm(b,x).
    # We can precompute for each pair (a,b) the LCM, and check x.
    
    # Faster check:
    # For each x, we need to ensure no pair (a,b) in S is "conflicting" with x.
    # Let's precompute conflict sets?
    # Conflicts for x: pairs (a,b) such that (a,b,x) is bad.
    # Since max_n is small, we can just check against current S.
    
    if order == 'random':
        elements = list(range(1, max_n + 1))
        random.shuffle(elements)
    elif order == 'asc':
        elements = list(range(1, max_n + 1))
    elif order == 'desc':
        elements = list(range(max_n, 0, -1))
    else:
        elements = list(range(1, max_n + 1))
        
    S = []
    # To speed up, keep track of "blocked" pairs?
    # When we add x, we block any pair (a,b) that forms a bad triple with x.
    # If a pair is blocked, we cannot add any third element that completes it?
    # No, the constraint is on triples.
    # If S contains a,b, and we add x, we must ensure no bad triple.
    # If S contains a,b,c, it's already checked.
    
    # Simple check:
    for x in elements:
        valid = True
        for a, b in combinations(S, 2):
            if _lcm(a,b) == _lcm(a,x) == _lcm(b,x):
                valid = False
                break
        if valid:
            S.append(x)
            
    return S

def construct():
    start_time = time.time()
    max_n = 128
    bad_triples = _precompute_bad_triples(max_n)
    
    best_set = []
    
    # Strategy 1: Greedy Removal
    # Run multiple times with different tie breakers
    for tb in ['random', 'min', 'max']:
        for _ in range(5):
            if time.time() - start_time > 50:
                break
            s = _greedy_removal(bad_triples, max_n, tb)
            if len(s) > len(best_set):
                best_set = s
                
    # Strategy 2: Greedy Inclusion
    for order in ['random', 'asc', 'desc']:
        for _ in range(5):
            if time.time() - start_time > 50:
                break
            s = _greedy_inclusion(bad_triples, max_n, order)
            if len(s) > len(best_set):
                best_set = s
                
    # Strategy 3: Local Search / Swap on best_set
    # Try to replace an element in best_set with one not in it, if it allows adding more later?
    # Or just check if we can add any element not in best_set?
    # If best_set is maximal, we can't add.
    # We can try to "shake": remove a random element, then greedily add back?
    
    # Simple improvement: try to add elements to best_set
    current_set = set(best_set)
    improved = True
    while improved:
        improved = False
        for x in range(1, max_n + 1):
            if x not in current_set:
                # Check if x can be added
                valid = True
                for a, b in combinations(list(current_set), 2):
                    if _lcm(a,b) == _lcm(a,x) == _lcm(b,x):
                        valid = False
                        break
                if valid:
                    current_set.add(x)
                    improved = True
        if time.time() - start_time > 55:
            break
            
    best_set = list(current_set)
    
    return best_set
