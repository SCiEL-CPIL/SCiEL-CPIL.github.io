"""Improved program: greedy + local search for max independent set in 3-uniform hypergraph."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
P = {'n': 128}

def _precompute_conflicts():
    """
    Precompute for each vertex v, the set of pairs (u, w) such that {u, v, w} is a forbidden triple.
    A triple {a,b,c} is forbidden if a+b, a+c, b+c are all in the selected set.
    But we need: for a set S, no three distinct a,b,c in S have a+b, a+c, b+c all in S.
    
    Actually, let's re-read the problem:
    "Forbid THREE DISTINCT members a,b,c for which a+b, a+c and b+c are also selected."
    
    So the forbidden configuration is: a,b,c are in S, AND a+b, a+c, b+c are all in S.
    Note: a+b, a+c, b+c may or may not be distinct from a,b,c and each other.
    
    The hyperedge is over the 6 values {a,b,c,a+b,a+c,b+c} but the constraint is that 
    if a,b,c are chosen, then we must NOT also have a+b, a+c, b+c all chosen.
    
    This is not a standard hypergraph independence problem because the "edge" involves 
    both the triple and their sums. Let me think differently.
    
    A set S is valid iff there do NOT exist distinct a,b,c in S such that 
    (a+b) in S and (a+c) in S and (b+c) in S.
    
    So we can model this as: for each triple {a,b,c} with a<b<c, if all three are in S,
    then at least one of a+b, a+c, b+c must NOT be in S.
    
    This is a constraint satisfaction problem. Let's use a different approach:
    We'll use a greedy construction with backtracking/local search.
    """
    n = P['n']
    # For each triple (a,b,c), store the pair sums
    triple_sums = {}
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            for c in range(b+1, n+1):
                s1, s2, s3 = a+b, a+c, b+c
                # Only relevant if sums are within 1..n
                if s1 <= n and s2 <= n and s3 <= n:
                    triple_sums[(a,b,c)] = (s1, s2, s3)
    return triple_sums

def _is_valid(S, triple_sums):
    """Check if set S is valid."""
    S_set = set(S)
    for (a,b,c), (s1,s2,s3) in triple_sums.items():
        if a in S_set and b in S_set and c in S_set:
            if s1 in S_set and s2 in S_set and s3 in S_set:
                return False
    return True

def _greedy_construct(triple_sums, order, deadline):
    """Greedy construction: add vertices in given order if valid."""
    S = set()
    S_list = []
    n = P['n']
    for v in order:
        if time.time() > deadline:
            break
        # Check if adding v creates a forbidden configuration
        valid = True
        # v could be part of a triple (a,b,c) where v is one of a,b,c
        # or v could be one of the sums s1,s2,s3
        # Case 1: v is one of a,b,c in some triple
        # We need to check: are there a,b in S such that {a,b,v} with their sums all in S?
        # Case 2: v is one of the sums, e.g., v = a+b for some a,b in S
        # Then we need a+c and b+c also in S for some c in S
        
        # More efficient: for each pair (a,b) in S, check if v completes a forbidden config
        # Sub-case A: v is the third element c, and a+b, a+v, b+v are in S
        #   Need a+b in S, a+v in S, b+v in S
        # Sub-case B: v = a+b (sum), need a+c and b+c in S for some c in S
        #   So for each c in S, check if a+c in S and b+c in S
        
        S_list_set = set(S)
        for a in S_list:
            for b in S_list:
                if a == b:
                    continue
                # Sub-case A: v is the third distinct element
                # a, b, v are distinct
                if v == a or v == b:
                    continue
                s_ab = a + b
                s_av = a + v
                s_bv = b + v
                if s_ab in S_list_set and s_av in S_list_set and s_bv in S_list_set:
                    valid = False
                    break
                # Sub-case B: v is a sum, e.g., v = a + b
                if a + b == v:
                    # Need to find c in S such that a+c in S and b+c in S
                    # c must be distinct from a, b, v
                    for c in S_list:
                        if c == a or c == b or c == v:
                            continue
                        if (a+c) in S_list_set and (b+c) in S_list_set:
                            valid = False
                            break
                    if not valid:
                        break
            if not valid:
                break
        if valid:
            S.add(v)
            S_list.append(v)
    return S_list

def _local_search(S_list, triple_sums, deadline, max_iter=500):
    """Local search: try to add/remove elements to improve."""
    S = set(S_list)
    S_set = set(S)
    n = P['n']
    best = list(S)
    best_size = len(best)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        # Try adding each vertex not in S
        for v in range(1, n+1):
            if v in S_set:
                continue
            # Check if adding v is valid
            valid = True
            for a in list(S):
                for b in list(S):
                    if a == b:
                        continue
                    if v == a or v == b:
                        continue
                    s_ab = a + b
                    s_av = a + v
                    s_bv = b + v
                    if s_ab in S_set and s_av in S_set and s_bv in S_set:
                        valid = False
                        break
                    if a + b == v:
                        for c in list(S):
                            if c == a or c == b or c == v:
                                continue
                            if (a+c) in S_set and (b+c) in S_set:
                                valid = False
                                break
                        if not valid:
                            break
                if not valid:
                    break
            if valid:
                S.add(v)
                S_set.add(v)
                improved = True
                break
        if improved:
            if len(S) > best_size:
                best = list(S)
                best_size = len(S)
            continue
        # Try removing one element and adding two (swap)
        for v in list(S):
            S.remove(v)
            S_set.remove(v)
            # Try to add two new elements
            added = 0
            for u in range(1, n+1):
                if u in S_set or u == v:
                    continue
                valid = True
                for a in list(S):
                    for b in list(S):
                        if a == b:
                            continue
                        if u == a or u == b:
                            continue
                        s_ab = a + b
                        s_au = a + u
                        s_bu = b + u
                        if s_ab in S_set and s_au in S_set and s_bu in S_set:
                            valid = False
                            break
                        if a + b == u:
                            for c in list(S):
                                if c == a or c == b or c == u:
                                    continue
                                if (a+c) in S_set and (b+c) in S_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                    if not valid:
                        break
                if valid:
                    S.add(u)
                    S_set.add(u)
                    added += 1
                    if added == 2:
                        break
            if added > 1:
                if len(S) > best_size:
                    best = list(S)
                    best_size = len(S)
                improved = True
            else:
                # Put v back
                S.add(v)
                S_set.add(v)
            if improved:
                break
        if not improved:
            break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline, 5s margin
    
    triple_sums = _precompute_conflicts()
    
    # Try multiple random orderings with greedy + local search
    best_set = []
    best_size = 0
    
    # Start with the seed solution
    seed_set = list(range(P['n']//2+1, P['n']+1))
    best_set = seed_set[:]
    best_size = len(best_set)
    
    random.seed(42)
    
    # Try several random orderings
    for trial in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Different ordering strategies
        if trial % 3 == 0:
            order = list(range(1, P['n']+1))
            random.shuffle(order)
        elif trial % 3 == 1:
            # Prefer larger numbers first
            order = list(range(P['n'], 0, -1))
            random.shuffle(order)
        else:
            # Prefer numbers that are not sums of two others
            order = list(range(1, P['n']+1))
            random.shuffle(order)
        
        S_list = _greedy_construct(triple_sums, order, deadline - 5.0)
        
        if len(S_list) > best_size:
            best_set = S_list[:]
            best_size = len(S_list)
        
        # Local search
        if time.time() < deadline - 2.0:
            improved = _local_search(S_list, triple_sums, deadline - 2.0, max_iter=100)
            if len(improved) > best_size:
                best_set = improved[:]
                best_size = len(improved)
    
    # Final polish: try to add any remaining elements
    if time.time() < deadline - 1.0:
        S_set = set(best_set)
        for v in range(1, P['n']+1):
            if v in S_set:
                continue
            valid = True
            for a in best_set:
                for b in best_set:
                    if a == b:
                        continue
                    if v == a or v == b:
                        continue
                    s_ab = a + b
                    s_av = a + v
                    s_bv = b + v
                    if s_ab in S_set and s_av in S_set and s_bv in S_set:
                        valid = False
                        break
                    if a + b == v:
                        for c in best_set:
                            if c == a or c == b or c == v:
                                continue
                            if (a+c) in S_set and (b+c) in S_set:
                                valid = False
                                break
                        if not valid:
                            break
                if not valid:
                    break
            if valid:
                best_set.append(v)
                S_set.add(v)
                best_size = len(best_set)
    
    # Ensure all are Python ints
    best_set = [int(x) for x in best_set]
    return list(best_set)
# EVOLVE-BLOCK-END
