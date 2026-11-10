import time
import random
from math import gcd
from itertools import combinations

# Precomputation
N = 128
gcd_table = [[0]*(N+1) for _ in range(N+1)]
for i in range(1, N+1):
    for j in range(i, N+1):
        g = gcd(i, j)
        gcd_table[i][j] = g
        gcd_table[j][i] = g

# Precompute bad triples
bad_triples = []
bad_triples_set = set()
for i in range(1, N+1):
    for j in range(i+1, N+1):
        for k in range(j+1, N+1):
            g1 = gcd_table[i][j]
            g2 = gcd_table[j][k]
            g3 = gcd_table[k][i]
            if g1 == g2 == g3:
                bad_triples.append((i, j, k))
                bad_triples_set.add((i, j, k))

# For add-back check, we don't need to precompute more, just use gcd_table.

def can_add(v, S):
    """
    Check if v can be added to S without creating a bad triple.
    S is a set of integers.
    """
    # Group elements in S by gcd(v, a)
    groups = {}
    for a in S:
        g = gcd_table[v][a]
        if g not in groups:
            groups[g] = []
        groups[g].append(a)
    
    # For each group G_g, check if there exist a, b in G_g such that gcd(a, b) == g
    for g, members in groups.items():
        if len(members) < 2:
            continue
        # Check pairs in members
        found = False
        for i in range(len(members)):
            a = members[i]
            for j in range(i+1, len(members)):
                b = members[j]
                if gcd_table[a][b] == g:
                    return False
        # If no pair found, this group is safe
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin
    
    best_set = set()
    best_size = 0
    
    # Initial greedy to get a baseline
    # Just use all numbers and remove conflicts
    # But let's run the heuristic directly.
    
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        S = set(range(1, N+1))
        R = set()
        
        # Phase 1: Remove vertices to resolve conflicts
        # We need to know which triples are still "active" (all 3 vertices in S)
        # Instead of maintaining active triples list, we can just check conflicts on the fly?
        # No, that's slow.
        # Let's maintain a count of conflicts for each vertex.
        
        # Better: 
        # While there is a bad triple in S:
        #   Find one.
        #   Remove a vertex from it.
        
        # To find a bad triple quickly, we can iterate through a list of all bad triples and check if all 3 are in S.
        # This is O(|T|) per check. |T| is ~300k.
        # If we do this 100 times, 30M ops. Might be slow in Python.
        
        # Optimization: Maintain a list of "active" triples.
        active_triples = list(bad_triples)
        # But we need to update it as we remove vertices.
        # If we remove v, we remove all triples containing v.
        
        # Let's use a different strategy for Phase 1:
        # Calculate degree of each vertex in the hypergraph.
        degrees = {i: 0 for i in range(1, N+1)}
        for t in bad_triples:
            for v in t:
                degrees[v] += 1
        
        # To find a conflict, we can just pick a random bad triple and check if it's in S.
        # If yes, resolve it.
        # This is lazy evaluation.
        
        # Let's try:
        # While True:
        #   Find a bad triple T subset of S.
        #   If none, break.
        #   Pick v in T with max degree (in current S? or original?).
        #   Remove v from S, add to R.
        #   Update degrees.
        
        # Finding a bad triple T subset of S:
        # Iterate through bad_triples.
        # If t[0] in S and t[1] in S and t[2] in S: return t.
        # This is fast if conflicts are few.
        
        while True:
            # Find a bad triple
            conflict_found = None
            # Shuffle to avoid bias? Or just take first found.
            # Taking first found might be deterministic.
            # Let's iterate.
            for t in bad_triples:
                if t[0] in S and t[1] in S and t[2] in S:
                    conflict_found = t
                    break
            
            if conflict_found is None:
                break
            
            # Pick vertex to remove
            # Pick one with highest degree among the 3?
            # Or random?
            # Highest degree is generally better for hitting set.
            v_to_remove = max(conflict_found, key=lambda x: degrees[x])
            
            S.remove(v_to_remove)
            R.add(v_to_remove)
            
            # Update degrees
            # We need to decrement degree of neighbors?
            # Actually, degrees are used for heuristic.
            # If we remove v, the "effective" degree of others doesn't change in the original hypergraph,
            # but the number of active triples they are part of decreases.
            # For simplicity, let's keep original degrees.
            # Or update: for each triple containing v, decrement degree of other 2 vertices.
            # This requires iterating through triples containing v.
            # Precompute list of triples for each vertex.
            
        # Phase 2: Add back
        # Shuffle R
        r_list = list(R)
        random.shuffle(r_list)
        
        for v in r_list:
            if can_add(v, S):
                S.add(v)
                R.remove(v)
                
        # Update best
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Optional: Local Search / Kicks
        # Try to swap elements
        # If time permits
        
    return list(best_set)

# Precompute triples per vertex for faster degree update if needed
# But for now, simple version.
