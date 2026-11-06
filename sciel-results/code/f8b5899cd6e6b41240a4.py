import time
import random
from itertools import combinations
import numpy as np

def construct():
    deadline = time.time() + 290
    
    # Helper to check if a subset of vertices in a graph (represented by adj_mask)
    # contains a K4 or independent 4-set.
    def has_K4_or_I4(adj_mask, subset):
        # subset is a list of vertices
        # We need to check if the induced subgraph on subset has a K4 or I4
        # We can just check all combinations of 4 in subset
        if len(subset) < 4:
            return False, None
        
        # Optimization: if subset is small, brute force is fine
        for comb in combinations(subset, 4):
            # Check if comb is a clique in the original graph
            is_clique = True
            is_indep = True
            for i in range(4):
                for j in range(i+1, 4):
                    u, v = comb[i], comb[j]
                    if (adj_mask[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
            if is_clique:
                return True, comb
            if is_indep:
                return True, comb
        return False, None

    # Build Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj_mask = [0] * q
    for i in range(q):
        for j in range(i+1, q):
            if (j - i) % q in residues:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    current_n = q
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Try to add vertices up to 42
    for target_n in range(38, 43):
        if time.time() > deadline - 10:
            break
            
        # Try to add vertex target_n - 1
        # We have a graph on target_n - 1 vertices
        # We need to find a valid connection set S for the new vertex
        found_valid = False
        attempts = 0
        max_attempts = 100000 # Adjust based on speed
        
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            # Randomly choose neighbors
            # We can bias towards 50%
            S = []
            for i in range(target_n - 1):
                if random.random() < 0.5:
                    S.append(i)
            
            # Check conditions
            valid = True
            # Check G[S]
            if has_K4_or_I4(adj_mask, S)[0]:
                valid = False
            # Check G[bar_S]
            if valid:
                bar_S = [i for i in range(target_n - 1) if i not in S]
                if has_K4_or_I4(adj_mask, bar_S)[0]:
                    valid = False
            
            if valid:
                # Add the vertex
                # Update adj_mask
                adj_mask.append(0)
                for i in S:
                    adj_mask[i] |= (1 << (target_n - 1))
                    adj_mask[target_n - 1] |= (1 << i)
                
                # Update edges
                for i in S:
                    edges.append((i, target_n - 1))
                    if i > target_n - 1:
                        edges[-1] = (target_n - 1, i)
                    # Ensure i < target_n - 1 is true since i < target_n - 1
                    # So (i, target_n - 1) is correct order if i < target_n - 1
                    # Wait, i is in range(target_n - 1), so i < target_n - 1.
                    # So (i, target_n - 1) is valid.
                
                current_n = target_n
                found_valid = True
                break
        
        if not found_valid:
            # Could not add this vertex, stop
            break
    
    return current_n, edges
