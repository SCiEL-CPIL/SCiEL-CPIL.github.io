import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # All 2x2x2 sub-blocks
    def all_blocks(n, m, p):
        blocks = []
        for i0 in range(n - 1):
            for j0 in range(m - 1):
                for k0 in range(p - 1):
                    blocks.append((i0, i0 + 1, j0, j0 + 1, k0, k0 + 1))
        return blocks
    
    all_blks = all_blocks(n, m, p)
    num_blocks = len(all_blks)
    
    # Precompute conflict graph
    # conflict[i][j] is True if block i and block j share any entry
    conflict = [[False] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        i0, i1, j0, j1, k0, k1 = all_blks[i]
        for j in range(i + 1, num_blocks):
            i2, i3, j2, j3, k2, k3 = all_blks[j]
            # Ranges overlap if:
            # i ranges: [i0, i1] and [i2, i3] overlap -> max(i0, i2) <= min(i1, i3)
            # j ranges: [j0, j1] and [j2, j3] overlap -> max(j0, j2) <= min(j1, j3)
            # k ranges: [k0, k1] and [k2, k3] overlap -> max(k0, k2) <= min(k1, k3)
            i_overlap = max(i0, i2) <= min(i1, i3)
            j_overlap = max(j0, j2) <= min(j1, j3)
            k_overlap = max(k0, k2) <= min(k1, k3)
            if i_overlap and j_overlap and k_overlap:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Find Maximum Independent Set using backtracking
    # This is a standard MIS problem.
    # Heuristic: order vertices by degree (highest first) to prune early?
    # Or just simple backtracking with a good initial solution.
    
    # Let's use a simple branch and bound.
    # State: current set, candidate set.
    # To speed up, we can use a greedy initial solution and then improve.
    
    # Greedy initial solution
    def greedy_mis():
        selected = []
        candidates = list(range(num_blocks))
        while candidates:
            # Pick vertex with min degree in current candidate subgraph?
            # Or just first one.
            # Let's pick one with fewest conflicts with other candidates.
            min_deg = float('inf')
            best_idx = -1
            for i_idx, i in enumerate(candidates):
                deg = sum(1 for j_idx, j in enumerate(candidates) if i != j and conflict[i][j])
                if deg < min_deg:
                    min_deg = deg
                    best_idx = i_idx
            v = candidates.pop(best_idx)
            selected.append(v)
            # Remove v and its neighbors from candidates
            new_candidates = []
            for j in candidates:
                if not conflict[v][j]:
                    new_candidates.append(j)
            candidates = new_candidates
        return selected
    
    best_solution = greedy_mis()
    best_size = len(best_solution)
    
    # Backtracking search for improvement
    # We can use a recursive function that tries to add vertices.
    # Given the small size (48), we can be aggressive.
    
    def backtrack(candidates, current_set, current_size):
        nonlocal best_solution, best_size
        if time.time() > deadline - 5:
            return
        
        # Pruning: if current_size + len(candidates) <= best_size, return
        if current_size + len(candidates) <= best_size:
            return
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_solution = list(current_set)
            return
        
        # Heuristic: pick a vertex to branch on.
        # Pick one with high degree?
        # Let's just pick the first one.
        v = candidates[0]
        remaining = candidates[1:]
        
        # Branch 1: Include v
        new_candidates = [u for u in remaining if not conflict[v][u]]
        current_set.add(v)
        backtrack(new_candidates, current_set, current_size + 1)
        current_set.remove(v)
        
        # Branch 2: Exclude v
        backtrack(remaining, current_set, current_size)

    # To make backtracking faster, we can order vertices.
    # Let's sort all blocks by degree descending.
    degrees = [sum(conflict[i]) for i in range(num_blocks)]
    sorted_indices = sorted(range(num_blocks), key=lambda x: degrees[x], reverse=True)
    
    # We need to map sorted indices back to original for conflict checks?
    # Easier: just use the sorted list as the "candidates" pool initially, 
    # but the conflict graph is defined on original indices.
    # The backtrack function above uses original indices.
    # Let's just run backtrack on the full set of indices.
    # It might be slow if not pruned well.
    
    # Alternative: Use a library? No.
    # Let's try a simpler local search: Simulated Annealing or Hill Climbing on the set of blocks.
    # Objective: Maximize number of blocks subject to no conflicts.
    # This is hard with constraints.
    
    # Let's stick to the backtracking but with a time limit and better ordering.
    # We will pass the sorted list of indices to start with.
    # But the conflict check uses original indices.
    
    # Let's optimize the backtrack by using bitsets or just lists.
    # For 48 nodes, lists are fine.
    
    # Let's restart the backtracking with the sorted order.
    # We modify backtrack to take a list of candidate indices.
    
    def backtrack_sorted(candidates, current_set, current_size):
        nonlocal best_solution, best_size
        if time.time() > deadline - 5:
            return
        
        if current_size + len(candidates) <= best_size:
            return
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_solution = list(current_set)
            return
        
        # Pick the first candidate
        v = candidates[0]
        remaining = candidates[1:]
        
        # Branch 1: Include v
        new_candidates = [u for u in remaining if not conflict[v][u]]
        current_set.add(v)
        backtrack_sorted(new_candidates, current_set, current_size + 1)
        current_set.remove(v)
        
        # Branch 2: Exclude v
        backtrack_sorted(remaining, current_set, current_size)

    # Start with all blocks sorted by degree
    initial_candidates = sorted_indices
    backtrack_sorted(initial_candidates, set(best_solution), best_size)
    
    # Use best_solution
    selected_blocks = [all_blks[idx] for idx in best_solution]
    
    covered = set()
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i0, i1, j0, j1, k0, k1 in selected_blocks:
        for i in (i0, i1):
            for j in (j0, j1):
                for k in (k0, k1):
                    covered.add((i, j, k))
        
        # Global indices
        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1
        
        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1
        
        c00 = k0 * n + i0
        c01 = k1 * n + i0
        c10 = k0 * n + i1
        c11 = k1 * n + i1
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        U_cols.append({a00: 1, a11: 1})
        V_cols.append({b00: 1, b11: 1})
        W_cols.append({c00: 1, c11: 1})
        
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        U_cols.append({a10: 1, a11: 1})
        V_cols.append({b00: 1})
        W_cols.append({c10: 1, c11: -1})
        
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        U_cols.append({a00: 1})
        V_cols.append({b01: 1, b11: -1})
        W_cols.append({c01: 1, c11: 1})
        
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        U_cols.append({a11: 1})
        V_cols.append({b10: 1, b00: -1})
        W_cols.append({c00: 1, c10: 1})
        
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        U_cols.append({a00: 1, a01: 1})
        V_cols.append({b11: 1})
        W_cols.append({c00: -1, c01: 1})
        
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        U_cols.append({a10: 1, a00: -1})
        V_cols.append({b00: 1, b01: 1})
        W_cols.append({c11: 1})
        
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        U_cols.append({a01: 1, a11: -1})
        V_cols.append({b10: 1, b11: 1})
        W_cols.append({c00: 1})
    
    # Add schoolbook for remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U_cols.append({i * m + j: 1})
                V_cols.append({j * p + k: 1})
                W_cols.append({k * n + i: 1})
    
    # Convert to dense matrices
    rank = len(U_cols)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx, c] = val
        for idx, val in V_cols[c].items():
            V[idx, c] = val
        for idx, val in W_cols[c].items():
            W[idx, c] = val
    
    # Verify
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.sum((T_comp - T) ** 2)
    
    if error > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
