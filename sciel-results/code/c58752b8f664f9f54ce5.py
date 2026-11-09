import numpy as np
from itertools import combinations
import random

def construct():
    random.seed(777)
    
    # Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = np.zeros((q, q), dtype=np.int8)
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj[i, j] = 1
                adj[j, i] = 1
    
    n = q
    
    def find_4_sets(adj, n):
        """Find all cliques of size 4 and independent sets of size 4."""
        k4s = []
        i4s = []
        for combo in combinations(range(n), 4):
            is_clique = True
            is_indep = True
            for idx in range(4):
                for jdx in range(idx+1, 4):
                    u, v = combo[idx], combo[jdx]
                    if adj[u, v]:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique:
                k4s.append(combo)
            if is_indep:
                i4s.append(combo)
        return k4s, i4s
    
    # Initial 4-sets
    k4s, i4s = find_4_sets(adj, n)
    
    # Try to add vertices
    for target_n in range(q+1, 47):
        added = False
        # We might need to retry with different random seeds for the repair process
        for attempt in range(50):
            # Solve for new vertex edges
            x = np.random.randint(0, 2, size=n)
            
            valid = True
            # Repair loop
            for step in range(1000):
                # Check constraints
                violated_k4 = None
                violated_i4 = None
                
                # Check independent 4-sets: need at least 1 edge
                for s in i4s:
                    if np.sum(x[list(s)]) == 0:
                        violated_i4 = s
                        break
                
                if violated_i4 is not None:
                    # Pick a random vertex in the set to connect
                    idx = random.choice(violated_i4)
                    x[idx] = 1
                    continue # Re-check
                    
                # Check clique 4-sets: need at most 3 edges
                for s in k4s:
                    if np.sum(x[list(s)]) == 4:
                        violated_k4 = s
                        break
                        
                if violated_k4 is not None:
                    # Pick a random vertex in the set to disconnect
                    idx = random.choice(violated_k4)
                    x[idx] = 0
                    continue # Re-check
                    
                # If no violations found in this pass
                valid = True
                break
            
            # Final verification of the candidate x
            # We need to be sure no constraints are violated
            # The repair loop might oscillate, so we check explicitly
            final_valid = True
            for s in i4s:
                if np.sum(x[list(s)]) == 0:
                    final_valid = False
                    break
            if final_valid:
                for s in k4s:
                    if np.sum(x[list(s)]) == 4:
                        final_valid = False
                        break
            
            if final_valid:
                # Add the vertex
                new_adj_full = np.zeros(n+1, dtype=np.int8)
                new_adj_full[:n] = x
                adj = np.vstack([np.hstack([adj, x.reshape(-1,1)]), np.hstack([x, np.zeros(1)])])
                n += 1
                added = True
                # Update 4-sets for the new graph
                k4s, i4s = find_4_sets(adj, n)
                break
        
        if not added:
            break
    
    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges.append((i, j))
    
    return n, edges
