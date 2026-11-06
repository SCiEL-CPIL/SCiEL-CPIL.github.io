import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 320  # Safety margin (evaluator kills at 330s)

    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # adj_np is a numpy bool array
    adj_np = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj_np[i, j] = True
                adj_np[j, i] = True

    best_n = n
    best_adj_np = adj_np.copy()

    def check_validity(new_mask, adj_np, n):
        """Check if adding a vertex with neighbor mask new_mask keeps graph (5,5)-free.
        Returns True if valid."""
        S = np.where(new_mask)[0]
        T = np.where(~new_mask)[0]
        
        # Check S for K4
        if len(S) >= 4:
            for i_idx in range(len(S)):
                i = S[i_idx]
                for j_idx in range(i_idx + 1, len(S)):
                    j = S[j_idx]
                    if not adj_np[i, j]:
                        continue
                    # Common neighbors of i and j in S
                    common = adj_np[i] & adj_np[j] & new_mask
                    cand = np.where(common)[0]
                    if len(cand) < 2:
                        continue
                    # Check if any pair in cand is an edge
                    for k in cand:
                        if (adj_np[k] & common).any():
                            return False

        # Check T for Independent Set of size 4
        if len(T) >= 4:
            T_mask = ~new_mask
            for i_idx in range(len(T)):
                i = T[i_idx]
                for j_idx in range(i_idx + 1, len(T)):
                    j = T[j_idx]
                    # i and j must be non-connected
                    if adj_np[i, j]:
                        continue
                    # Common non-neighbors of i and j in T
                    common = (~adj_np[i]) & (~adj_np[j]) & T_mask
                    cand = np.where(common)[0]
                    if len(cand) < 2:
                        continue
                    # Check if subgraph on cand is a clique in A
                    # If not, there is a non-edge, so independent set exists
                    for k in cand:
                        mask_others = common.copy()
                        mask_others[k] = 0
                        if not (adj_np[k] & mask_others).all():
                            return False

        return True

    def add_vertex(adj_np, new_mask):
        n = len(adj_np)
        new_adj = np.zeros((n + 1, n + 1), dtype=bool)
        new_adj[:n, :n] = adj_np
        new_adj[:n, n] = new_mask
        new_adj[n, :n] = new_mask
        return new_adj

    best_overall_n = best_n
    best_overall_adj = best_adj_np.copy()
    
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        current_adj = best_overall_adj.copy()
        current_n = best_overall_n
        
        random.seed(42 + restart * 7)
        
        max_add_attempts = 500
        add_success_count = 0
        
        while current_n < 46 and add_success_count < max_add_attempts:
            if time.time() > deadline:
                break
            
            found = False
            max_attempts = 2000
            attempts = 0
            
            while attempts < max_attempts and not found:
                if time.time() > deadline:
                    break
                
                mask_np = np.random.random(current_n) < 0.5
                
                if check_validity(mask_np, current_adj, current_n):
                    current_adj = add_vertex(current_adj, mask_np)
                    current_n += 1
                    found = True
                    if current_n > best_overall_n:
                        best_overall_n = current_n
                        best_overall_adj = current_adj.copy()
                    add_success_count += 1
                    break
                
                attempts += 1
            
            if not found:
                if current_n > best_overall_n:
                    break
                elif current_n > q:
                    current_adj = current_adj[:current_n-1, :current_n-1]
                    current_n -= 1
                else:
                    break

    best_edges = []
    for i in range(best_overall_n):
        for j in range(i + 1, best_overall_n):
            if best_overall_adj[i, j]:
                best_edges.append((i, j))
    
    return best_overall_n, best_edges
