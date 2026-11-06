import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 180
    numbers = list(range(1, N + 1))
    
    # 1. Precompute all forbidden triples
    # 1/a = 1/b + 1/c => a = bc / (b+c)
    # a, b, c distinct.
    # Since a = bc/(b+c), a < b and a < c.
    # So a is the smallest.
    # Iterate b, c.
    triples = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if a < b and a < c and a != b and a != c and 1 <= a <= N:
                    # Ensure distinct
                    if len({a, b, c}) == 3:
                        triples.add((a, b, c))
    
    # Convert to list for easier manipulation
    triples_list = list(triples)
    
    # Precompute degree of each vertex
    degree = {v: 0 for v in numbers}
    for t in triples_list:
        for v in t:
            degree[v] += 1
    
    best_set = set()
    
    # Strategy A: Greedy Add
    # Sort by degree ascending
    sorted_numbers = sorted(numbers, key=lambda x: degree[x])
    S_A = set()
    for x in sorted_numbers:
        if time.time() > deadline:
            break
        valid = True
        # Check if adding x creates a triple with any two elements in S_A
        # We need to check pairs (u, v) in S_A
        # Optimization: Precompute for each x, the set of pairs (u,v) that form a triple with x
        # But doing it on the fly is O(|S|^2). 180^2 is small.
        for u in S_A:
            for v in S_A:
                if u == v:
                    continue
                # Check if {x, u, v} is in triples
                # We can use a set of frozensets or just check the specific tuple
                # Since triples are stored as (a,b,c) with a<b<c, we can sort the triple
                t = tuple(sorted((x, u, v)))
                if t in triples:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S_A.add(x)
    
    if len(S_A) > len(best_set):
        best_set = S_A
    
    # Strategy B: Greedy Remove
    # Start with all numbers, remove highest degree vertex involved in a remaining triple
    active_vertices = set(numbers)
    # Map vertex to list of triples containing it
    # We need to efficiently remove triples.
    # Let's keep a list of active triples.
    active_triples = [set(t) for t in triples_list]
    
    # Re-calculate degrees based on active_triples
    deg = {v: 0 for v in numbers}
    for t in active_triples:
        for v in t:
            deg[v] += 1
            
    while active_triples:
        if time.time() > deadline:
            break
        # Find vertex with max degree among those in active triples
        max_deg = -1
        max_v = -1
        # We only care about vertices in active triples
        # Iterate through active triples to find candidates? 
        # Or just check all active_vertices?
        # Checking all active_vertices is O(N). N=180.
        for v in active_vertices:
            if deg[v] > max_deg:
                max_deg = deg[v]
                max_v = v
        
        if max_v == -1:
            break
            
        # Remove max_v
        active_vertices.remove(max_v)
        
        # Remove all triples containing max_v
        new_active_triples = []
        for t in active_triples:
            if max_v in t:
                # Update degrees for other vertices in t
                for v in t:
                    if v != max_v:
                        deg[v] -= 1
            else:
                new_active_triples.append(t)
        active_triples = new_active_triples
        
    S_B = active_vertices
    if len(S_B) > len(best_set):
        best_set = S_B

    # Strategy C: Randomized Greedy Add
    # Try a few random orders
    for _ in range(10):
        if time.time() > deadline:
            break
        random.shuffle(sorted_numbers)
        S_C = set()
        for x in sorted_numbers:
            valid = True
            for u in S_C:
                for v in S_C:
                    if u == v:
                        continue
                    t = tuple(sorted((x, u, v)))
                    if t in triples:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S_C.add(x)
        if len(S_C) > len(best_set):
            best_set = S_C

    # Ensure non-empty
    if not best_set:
        best_set = {180}
        
    return list(best_set)
