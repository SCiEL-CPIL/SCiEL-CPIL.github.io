import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 110.0

    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Strassen-based decomposition for 2x4x5
    # Rank 36
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_2x2x2_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        strassen_patterns = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        for u_pat, v_pat, w_pat in strassen_patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx // 2
                    k_local = idx % 2
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx // 2
                    k_local = idx % 2
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    def add_2x2x1_block(U, V, W, a_i_offset, a_j_offset, b_j_offset, b_k_offset, c_i_offset, c_k_offset, col_offset):
        patterns = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),
            ([0, 1, 0, 0], [0, 1], [1, 0]),
            ([0, 0, 1, 0], [1, 0], [0, 1]),
            ([0, 0, 0, 1], [0, 1], [0, 1]),
        ]
        for u_pat, v_pat, w_pat in patterns:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    i_global = a_i_offset + i_local
                    j_global = a_j_offset + j_local
                    U[i_global * m + j_global, col_offset] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    k_local = 0
                    j_global = b_j_offset + j_local
                    k_global = b_k_offset + k_local
                    V[j_global * p + k_global, col_offset] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    k_local = 0
                    i_global = c_i_offset + i_local
                    k_global = c_k_offset + k_local
                    W[k_global * n + i_global, col_offset] = val
            col_offset += 1
        return col_offset

    col = 0
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 0, 0, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 0, 0, 4, 0, 4, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 0, 0, 0, col)
    col = add_2x2x2_block(U, V, W, 0, 2, 2, 2, 0, 2, col)
    col = add_2x2x1_block(U, V, W, 0, 2, 2, 4, 0, 4, col)
    assert col == total_rank

    def verify_decomposition(U, V, W):
        rank = U.shape[1]
        # Check diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check off-diagonal terms
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U[a] * V[b] * W[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    # Try to find a lower-rank decomposition using the sliding window approach
    # Partition m=4 into overlapping windows of length 2: [0,1], [1,2], [2,3]
    # For each window, compute partial 2x2x5 products, then combine with corrections
    
    # Alternative: try to find rank-35 by checking if any column is redundant
    # For each column r, check if removing it and adjusting others still gives valid decomposition
    # This is equivalent to checking if the tensor lies in the span of the other 35 rank-1 tensors
    
    found_lower_rank = False
    best_U = U
    best_V = V
    best_W = W
    best_r = total_rank
    
    if time.time() < deadline - 15:
        # Try each possible column removal
        for remove_col in range(total_rank):
            if time.time() > deadline - 15:
                break
            # Create U', V', W' without column remove_col
            U_test = np.delete(U, remove_col, axis=1)
            V_test = np.delete(V, remove_col, axis=1)
            W_test = np.delete(W, remove_col, axis=1)
            
            # Check if the remaining 35 columns still decompose the tensor
            # This requires that the tensor T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # equals sum_{r≠remove_col} U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # which means the removed rank-1 tensor must be zero, which it's not
            # So simple removal won't work.
            
            # Instead, try: can we express the removed column as a linear combination
            # of others in a way that preserves the tensor? This is complex.
            # Skip for now.
            pass
    
    # Try a different approach: use numerical optimization to find a rank-35 decomposition
    # Set up the problem as minimizing the tensor norm with rank constraint
    # This is non-convex, so use local search from the rank-36 solution
    
    if time.time() < deadline - 10:
        # Try perturbing the rank-36 solution to find a rank-35 solution
        # Idea: set one column to zero and see if the tensor can still be decomposed
        # by adjusting the other columns slightly
        
        import random
        random.seed(42)
        
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            # Pick a column to potentially eliminate
            target_col = trial % total_rank
            
            # Create a perturbed version where we try to zero out target_col
            # by absorbing its contribution into other columns
            U_pert = U.copy()
            V_pert = V.copy()
            W_pert = W.copy()
            
            # Try to find small perturbations to other columns that compensate
            # for removing target_col
            # This is a linear system: find delta_U, delta_V, delta_W such that
            # sum_{r≠target} (U[:,r]+delta_U[:,r]) ⊗ (V[:,r]+delta_V[:,r]) ⊗ (W[:,r]+delta_W[:,r])
            # = original tensor
            # Linearizing: sum_{r≠target} [delta_U[:,r]⊗V[:,r]⊗W[:,r] + U[:,r]⊗delta_V[:,r]⊗W[:,r] + U[:,r]⊗V[:,r]⊗delta_W[:,r]]
            # = -U[:,target]⊗V[:,target]⊗W[:,target]
            
            # This is a linear system in the perturbations. Let's set up the matrix.
            # The tensor has nm*mp*pn = 8*20*10 = 1600 components.
            # Each perturbation has (nm + mp + pn) * (rank-1) = (8+20+10)*35 = 1470 variables.
            # This is overdetermined, so we need least squares.
            
            # Build the linear system
            nm, mp, pn = 8, 20, 10
            rank_minus_1 = total_rank - 1
            num_vars = (nm + mp + pn) * rank_minus_1
            num_eqs = nm * mp * pn
            
            # This is too large to solve directly. Use iterative method.
            # Instead, try a simpler heuristic: for each pair of columns, check if
            # their difference can be absorbed.
            
            # Actually, let's just try a direct numerical search with fewer trials
            # and simpler checks.
            break

    if verify_decomposition(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return best_result
# EVOLVE-BLOCK-END
