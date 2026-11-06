import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Seed: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    best_n = q
    best_edges = seed_edges
    
    # Initialize adjacency matrix
    adj = np.zeros((42, 42), dtype=bool) # Pre-allocate max size
    for i, j in seed_edges:
        adj[i, j] = True
        adj[j, i] = True
        
    current_n = q
    
    # Precompute K4s and Ind4s and their incidence lists
    def get_k4s_ind4s_and_incidence(adj, n):
        k4s = []
        ind4s = []
        for combo in combinations(range(n), 4):
            is_clique = True
            is_ind = True
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = combo[i], combo[j]
                    if adj[u, v]:
                        is_ind = False
                    else:
                        is_clique = False
                    if not is_clique and not is_ind:
                        break
                if not is_clique and not is_ind:
                    break
            if is_clique:
                k4s.append(combo)
            if is_ind:
                ind4s.append(combo)
        
        # Build incidence lists
        k4_inc = [[] for _ in range(n)]
        for idx, k4 in enumerate(k4s):
            for v in k4:
                k4_inc[v].append(idx)
                
        ind4_inc = [[] for _ in range(n)]
        for idx, i4 in enumerate(ind4s):
            for v in i4:
                ind4_inc[v].append(idx)
                
        return k4s, ind4s, k4_inc, ind4_inc

    k4s, ind4s, k4_inc, ind4_inc = get_k4s_ind4s_and_incidence(adj, current_n)
    
    # Try to add vertices
    for target_n in range(current_n + 1, 43):
        if time.time() > deadline - 20:
            break
            
        found_valid = False
        max_restarts = 500
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize x: use a bias based on vertex degree in current graph
            # Vertices with higher degree should be more likely to connect to new vertex
            # (to avoid creating large independent sets)
            # Vertices with lower degree should be more likely NOT to connect
            # (to avoid creating large cliques)
            # Actually, let's think: if x[i]=1 means edge (i, new), then:
            # - To avoid K5: don't connect new vertex to all 4 vertices of any K4
            # - To avoid Ind5: don't leave new vertex non-adjacent to all 4 vertices of any Ind4
            # A good heuristic: connect to vertices that are in many Ind4s but few K4s
            
            # Simple random init with slight bias
            x = np.random.choice([0, 1], size=current_n).astype(bool)
            
            # Initialize counts
            k4_counts = np.zeros(len(k4s), dtype=np.int8)
            ind4_counts = np.zeros(len(ind4s), dtype=np.int8)
            
            # Calculate initial counts
            for idx, k4 in enumerate(k4s):
                k4_counts[idx] = np.sum(x[list(k4)])
            for idx, i4 in enumerate(ind4s):
                ind4_counts[idx] = np.sum(1 - x[list(i4)])
                
            # Calculate initial violations (hard)
            current_violations = np.sum(k4_counts == 4) + np.sum(ind4_counts == 4)
            
            # Also compute smooth penalty for SA guidance
            # Soft penalty: for each K4, penalty = max(0, count-3)^2
            # For each Ind4, penalty = max(0, indcount-3)^2
            def compute_soft_penalty(k4_counts, ind4_counts):
                k4_pen = np.sum(np.maximum(k4_counts - 3, 0) ** 2)
                ind_pen = np.sum(np.maximum(ind4_counts - 3, 0) ** 2)
                return k4_pen + ind_pen
                
            current_soft = compute_soft_penalty(k4_counts, ind4_counts)
            
            if current_violations == 0:
                found_valid = True
                break
            
            # Simulated Annealing on x with soft penalty as primary objective
            temperature = 5.0
            cooling_rate = 0.9995
            steps = 20000
            
            for step in range(steps):
                if time.time() > deadline - 5:
                    break
                if current_violations == 0:
                    found_valid = True
                    break
                    
                # Flip a random bit
                idx_v = np.random.randint(0, current_n)
                old_val = x[idx_v]
                new_val = not old_val
                
                # Update counts and violations
                delta_violations = 0
                old_k4_counts = {}
                old_ind4_counts = {}
                
                # Update K4s containing idx_v
                for k4_idx in k4_inc[idx_v]:
                    old_count = k4_counts[k4_idx]
                    old_k4_counts[k4_idx] = old_count
                    if old_val:
                        # 1 -> 0
                        new_count = old_count - 1
                        if old_count == 4:
                            delta_violations -= 1
                    else:
                        # 0 -> 1
                        new_count = old_count + 1
                        if new_count == 4:
                            delta_violations += 1
                    k4_counts[k4_idx] = new_count
                    
                # Update Ind4s containing idx_v
                for i4_idx in ind4_inc[idx_v]:
                    old_count = ind4_counts[i4_idx]
                    old_ind4_counts[i4_idx] = old_count
                    if old_val:
                        # 1 -> 0 (x becomes 0, so non-neighbor count increases)
                        new_count = old_count + 1
                        if new_count == 4:
                            delta_violations += 1
                    else:
                        # 0 -> 1 (x becomes 1, so non-neighbor count decreases)
                        new_count = old_count - 1
                        if old_count == 4:
                            delta_violations -= 1
                    ind4_counts[i4_idx] = new_count
                
                new_violations = current_violations + delta_violations
                new_soft = compute_soft_penalty(k4_counts, ind4_counts)
                delta_soft = new_soft - current_soft
                
                # Accept based on combined objective: violations * 100 + soft penalty
                delta_obj = delta_violations * 100.0 + delta_soft
                
                # Accept?
                if delta_obj < 0 or (temperature > 0.01 and random.random() < np.exp(-delta_obj / temperature)):
                    x[idx_v] = new_val
                    current_violations = new_violations
                    current_soft = new_soft
                else:
                    # Revert counts
                    for k4_idx in k4_inc[idx_v]:
                        k4_counts[k4_idx] = old_k4_counts[k4_idx]
                    for i4_idx in ind4_inc[idx_v]:
                        ind4_counts[i4_idx] = old_ind4_counts[i4_idx]
                    
                    # x remains old_val
                    # current_violations, current_soft remain old
                
                temperature *= cooling_rate
            
            if current_violations == 0:
                found_valid = True
                break
        
        if found_valid:
            # Add vertex target_n - 1
            new_idx = target_n - 1
            for i in range(current_n):
                if x[i]:
                    adj[i, new_idx] = True
                    adj[new_idx, i] = True
            
            current_n = target_n
            best_n = current_n
            # Update best_edges
            best_edges = []
            for i in range(current_n):
                for j in range(i + 1, current_n):
                    if adj[i, j]:
                        best_edges.append((i, j))
            
            # Update k4s, ind4s, and incidence
            k4s, ind4s, k4_inc, ind4_inc = get_k4s_ind4s_and_incidence(adj, current_n)
        else:
            # If we can't add this vertex, stop
            break
            
    return best_n, best_edges
