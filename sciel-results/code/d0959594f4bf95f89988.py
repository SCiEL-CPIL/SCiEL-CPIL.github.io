import time
import sys
import random
import numpy as np
from itertools import combinations, permutations

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    """
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Known rank 44 solution
    U44, V44, W44 = _get_rank_44_solution()
    
    best_rank = 44
    best_U, best_V, best_W = U44, V44, W44
    
    # Try to find rank 43 and 42
    for target_rank in [43, 42]:
        if time.time() > deadline - 30:
            break
        
        found = _search_for_rank(target_rank, deadline, start_time)
        if found:
            best_rank = target_rank
            best_U, best_V, best_W = found
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _get_rank_44_solution():
    """
    Returns a rank-44 solution using Strassen's algorithm on 2x2 blocks.
    
    The 2x3 * 3x8 multiplication can be split into two 2x2 * 2x4 blocks
    (left columns 0-3, right columns 4-7), each using Strassen (rank 7)
    plus 2 direct terms for the remaining columns.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 44)
    V = [[0] * rank for _ in range(m * p)]  # shape (24, 44)
    W = [[0] * rank for _ in range(p * n)]  # shape (16, 44)
    
    col = 0
    
    def add_strassen_block(u_map, v_map, w_map):
        """Add 7 Strassen terms for a 2x2 * 2x2 sub-problem."""
        nonlocal col
        
        # M1: (A00+A11)(B00+B11)
        U[u_map[0]][col] += 1
        U[u_map[3]][col] += 1
        V[v_map[0]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[u_map[2]][col] += 1
        U[u_map[3]][col] += 1
        V[v_map[0]][col] += 1
        W[w_map[2]][col] += 1
        W[w_map[3]][col] += -1
        col += 1
        
        # M3: A00(B01-B11)
        U[u_map[0]][col] += 1
        V[v_map[1]][col] += 1
        V[v_map[3]][col] += -1
        W[w_map[1]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[u_map[3]][col] += 1
        V[v_map[2]][col] += 1
        V[v_map[0]][col] += -1
        W[w_map[0]][col] += 1
        W[w_map[2]][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[u_map[0]][col] += 1
        U[u_map[1]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += -1
        W[w_map[1]][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[u_map[2]][col] += 1
        U[u_map[0]][col] += -1
        V[v_map[0]][col] += 1
        V[v_map[1]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[u_map[1]][col] += 1
        U[u_map[3]][col] += -1
        V[v_map[2]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += 1
        col += 1

    # For each half (left: cols 0-3, right: cols 4-7)
    for half in range(2):
        # u_map maps Strassen indices to U indices
        # U indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1], 4=A[0,2], 5=A[1,2]
        u_map = {0: 0, 1: 1, 2: 3, 3: 4}
        
        # First 2x2 block: B columns 0,1 (within this half)
        b_start = half * 4
        v_map_b1 = {
            0: 0 * p + (b_start + 0),  # B[0, col]
            1: 0 * p + (b_start + 1),  # B[0, col+1]
            2: 1 * p + (b_start + 0),  # B[1, col]
            3: 1 * p + (b_start + 1)   # B[1, col+1]
        }
        w_map_c1 = {
            0: (b_start + 0) * n + 0,  # C[row0, col]
            1: (b_start + 1) * n + 0,  # C[row0, col+1]
            2: (b_start + 0) * n + 1,  # C[row1, col]
            3: (b_start + 1) * n + 1   # C[row1, col+1]
        }
        add_strassen_block(u_map, v_map_b1, w_map_c1)
        
        # Second 2x2 block: B columns 2,3 (within this half)
        b_start2 = half * 4 + 2
        v_map_b2 = {
            0: 0 * p + (b_start2 + 0),
            1: 0 * p + (b_start2 + 1),
            2: 1 * p + (b_start2 + 0),
            3: 1 * p + (b_start2 + 1)
        }
        w_map_c2 = {
            0: (b_start2 + 0) * n + 0,
            1: (b_start2 + 1) * n + 0,
            2: (b_start2 + 0) * n + 1,
            3: (b_start2 + 1) * n + 1
        }
        add_strassen_block(u_map, v_map_b2, w_map_c2)
        
        # Direct terms for B column 2 (the middle column not covered by Strassen)
        # Actually, we used cols 0,1 and 2,3. We need col 2 separately?
        # Wait, let me reconsider. For 2x2 * 2x2, we cover 2 B columns.
        # For 2x3 * 2x4, we have 3 B rows and 4 B columns.
        # Strassen covers 2x2, so we need to handle the extra row and column.
        
        # Let me redo this more carefully.
        # Actually, the standard approach: 2x3 * 3x4 can be done with
        # Strassen on 2x2 subblocks plus direct terms.
        
        # For each half (4 B columns):
        # Use Strassen for B[0:2, 0:2] and B[0:2, 2:4] (using A[0:2, 0:2])
        # Then direct terms for A[0:2, 2] * B[2, 0:4]
        
        for k in range(4):
            # Direct term: A[i, 2] * B[2, half*4+k] -> C[i, half*4+k]
            for i in range(2):
                u_idx = i * m + 2  # A[i, 2]
                v_idx = 2 * p + (half * 4 + k)  # B[2, col]
                w_idx = (half * 4 + k) * n + i  # C[i, col]
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return U, V, W

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor T[i,j,k]."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n*m):
        i_row = i // m
        i_col = i % m
        for j in range(m*p):
            j_row = j // p
            j_col = j % p
            if i_col == j_row:
                k_row = i_row
                k_col = j_col
                k = k_col * n + k_row
                T[i, j, k] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-8):
    """Verify that the decomposition exactly matches the target tensor."""
    n, m, p = T.shape[0] // m, m, T.shape[2] // n
    nm, mp, pn = T.shape
    
    max_err = 0.0
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                computed = 0.0
                for t in range(len(U[0])):
                    computed += U[i][t] * V[j][t] * W[k][t]
                err = abs(computed - T[i, j, k])
                if err > max_err:
                    max_err = err
    return max_err < tol

def _search_for_rank(target_rank, deadline, start_time):
    """
    Try to find a decomposition of the given rank by eliminating terms
    from the rank-44 solution.
    
    For each candidate term to eliminate, we solve for coefficients c
    such that the contribution of the eliminated term equals a linear
    combination of the kept terms' contributions.
    """
    n, m, p = 2, 3, 8
    U44, V44, W44 = _get_rank_44_solution()
    r44 = 44
    num_eliminate = r44 - target_rank
    
    if num_eliminate <= 0:
        return None
    
    T = _build_target_tensor(n, m, p)
    
    # Convert to numpy arrays
    U_arr = np.array(U44, dtype=float)
    V_arr = np.array(V44, dtype=float)
    W_arr = np.array(W44, dtype=float)
    
    nm, mp, pn = n*m, m*p, p*n
    
    # Precompute all (i,j,k) indices where T != 0
    nonzero_ijk = []
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                if T[i, j, k] != 0:
                    nonzero_ijk.append((i, j, k))
    
    # For each subset of terms to eliminate, try to find a valid elimination
    # Strategy: try single-term eliminations first, then pairs
    
    if num_eliminate == 1:
        # Try eliminating each single term
        for t_e in range(r44):
            if time.time() > deadline - 10:
                return None
            
            keep_set = [t for t in range(r44) if t != t_e]
            
            # Solve for c such that:
            # U[:,t_e] * V[:,t_e] * W[:,t_e] = sum_{t_k in keep} c[t_k] * U[:,t_k] * V[:,t_k] * W[:,t_k]
            
            # Build linear system
            A = np.zeros((len(nonzero_ijk), len(keep_set)))
            b = np.zeros(len(nonzero_ijk))
            
            for idx, (i, j, k) in enumerate(nonzero_ijk):
                b[idx] = U_arr[i, t_e] * V_arr[j, t_e] * W_arr[k, t_e]
                for k_idx, t_k in enumerate(keep_set):
                    A[idx, k_idx] = U_arr[i, t_k] * V_arr[j, t_k] * W_arr[k, t_k]
            
            # Solve least squares
            try:
                c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if the solution is good
            if residuals.size > 0:
                if residuals[0] > 1e-6 * (np.sum(b**2) + 1e-10):
                    continue
            
            # Verify the solution
            new_U = U_arr.copy()
            new_V = V_arr.copy()
            new_W = W_arr.copy()
            
            # The eliminated term's contribution should be zero in the new decomposition
            # We don't actually modify U, V, W - we just remove the term
            # But we need to verify that the remaining terms still give the correct result
            
            error = 0.0
            for i in range(nm):
                for j in range(mp):
                    for k in range(pn):
                        computed = 0.0
                        for t_k in keep_set:
                            computed += new_U[i, t_k] * new_V[j, t_k] * new_W[k, t_k]
                        error += (computed - T[i, j, k]) ** 2
            
            if error < 1e-4:
                # Found a valid solution!
                final_U = []
                final_V = []
                final_W = []
                for t in keep_set:
                    final_U.append(new_U[:, t].tolist())
                    final_V.append(new_V[:, t].tolist())
                    final_W.append(new_W[:, t].tolist())
                
                # Transpose to get shape (nm, rank), (mp, rank), (pn, rank)
                final_U = np.array(final_U).T.tolist()
                final_V = np.array(final_V).T.tolist()
                final_W = np.array(final_W).T.tolist()
                
                return final_U, final_V, final_W
    
    elif num_eliminate == 2:
        # Try eliminating pairs of terms
        # This is more complex - we need to eliminate both terms simultaneously
        for t_e1, t_e2 in combinations(range(r44), 2):
            if time.time() > deadline - 10:
                return None
            
            keep_set = [t for t in range(r44) if t not in (t_e1, t_e2)]
            
            # We need to find c1, c2 such that:
            # U[:,t_e1]*V[:,t_e1]*W[:,t_e1] + U[:,t_e2]*V[:,t_e2]*W[:,t_e2]
            # = sum_{t_k in keep} (c1[t_k]*U[:,t_e1]*V[:,t_e1]*W[:,t_e1] + c2[t_k]*U[:,t_e2]*V[:,t_e2]*W[:,t_e2])
            # Wait, that's not right. We need:
            # The contribution of t_e1 and t_e2 should be expressible as a linear combination of kept terms
            
            # Actually, for simultaneous elimination, we need:
            # For all (i,j,k): U[i,t_e1]*V[j,t_e1]*W[k,t_e1] + U[i,t_e2]*V[j,t_e2]*W[k,t_e2]
            # = sum_{t_k in keep} c[t_k] * U[i,t_k]*V[j,t_k]*W[k,t_k]
            
            # This is a single linear system with unknowns c[t_k]
            
            A = np.zeros((len(nonzero_ijk), len(keep_set)))
            b = np.zeros(len(nonzero_ijk))
            
            for idx, (i, j, k) in enumerate(nonzero_ijk):
                b[idx] = (U_arr[i, t_e1] * V_arr[j, t_e1] * W_arr[k, t_e1] +
                         U_arr[i, t_e2] * V_arr[j, t_e2] * W_arr[k, t_e2])
                for k_idx, t_k in enumerate(keep_set):
                    A[idx, k_idx] = U_arr[i, t_k] * V_arr[j, t_k] * W_arr[k, t_k]
            
            try:
                c, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            if residuals.size > 0:
                if residuals[0] > 1e-6 * (np.sum(b**2) + 1e-10):
                    continue
            
            # Verify
            error = 0.0
            for i in range(nm):
                for j in range(mp):
                    for k in range(pn):
                        computed = 0.0
                        for t_k in keep_set:
                            computed += U_arr[i, t_k] * V_arr[j, t_k] * W_arr[k, t_k]
                        error += (computed - T[i, j, k]) ** 2
            
            if error < 1e-4:
                final_U = []
                final_V = []
                final_W = []
                for t in keep_set:
                    final_U.append(U_arr[:, t].tolist())
                    final_V.append(V_arr[:, t].tolist())
                    final_W.append(W_arr[:, t].tolist())
                
                final_U = np.array(final_U).T.tolist()
                final_V = np.array(final_V).T.tolist()
                final_W = np.array(final_W).T.tolist()
                
                return final_U, final_V, final_W
    
    return None
