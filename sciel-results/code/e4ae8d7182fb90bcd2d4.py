"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we aim to finish by 130s.
    deadline = start_time + 120.0
    
    # We know n=9 is valid (seed).
    # We want to maximize n. Known bound is 34.
    # We will try to find a valid coloring for n=34, then 33, etc.
    
    best_result = {
        'n': 9,
        'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    }
    
    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 10:
            break
            
        # 1. Precomputation for this n
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Map vertex to list of triple indices containing it
        vertex_to_triples = [[] for _ in range(n)]
        for i, (a, b, c) in enumerate(triples):
            vertex_to_triples[a].append(i)
            vertex_to_triples[b].append(i)
            vertex_to_triples[c].append(i)
            
        # Precompute K4s and K5s as list of (vertices, triple_indices)
        # This allows us to quickly check validity and pick a vertex to flip
        k4s = []
        for quad in combinations(range(n), 4):
            idxs = [triple_to_idx[t] for t in combinations(quad, 3)]
            k4s.append((quad, idxs))
            
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = [triple_to_idx[t] for t in combinations(quint, 3)]
            k5s.append((quint, idxs))
            
        # 2. Local Search
        # We will perform several random restarts
        max_restarts = 20
        found_valid = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize random coloring
            colors = [random.randint(0, 1) for _ in range(len(triples))]
            
            # WalkSAT-like loop
            max_iters = 1000
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                    
                # Check for violations
                # We check K4s first (red), then K5s (blue)
                # To speed up, we could randomize the order, but checking all is ~0.2s
                # Let's just find the first one.
                
                violation = None
                viol_type = None # 0 for Red K4, 1 for Blue K5
                
                # Check Red K4
                for quad, idxs in k4s:
                    if colors[idxs[0]] == 0 and colors[idxs[1]] == 0 and colors[idxs[2]] == 0 and colors[idxs[3]] == 0:
                        violation = quad
                        viol_type = 0
                        break
                        
                if violation is None:
                    # Check Blue K5
                    for quint, idxs in k5s:
                        # All 10 must be 1
                        if all(colors[i] == 1 for i in idxs):
                            violation = quint
                            viol_type = 1
                            break
                
                if violation is None:
                    # Valid coloring found
                    found_valid = True
                    best_result = {'n': n, 'colors': colors}
                    break
                
                # Flip a random vertex from the violating clique
                v = random.choice(violation)
                for idx in vertex_to_triples[v]:
                    colors[idx] = 1 - colors[idx]
                    
            if found_valid:
                break
                
        if found_valid:
            break

    return best_result
# EVOLVE-BLOCK-END
