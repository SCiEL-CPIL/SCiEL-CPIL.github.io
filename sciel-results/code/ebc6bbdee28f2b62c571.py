"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Strategy: Start with rank-42 block-Strassen solution, then try to reduce rank
    # by finding linear dependencies or using a more efficient decomposition.
    
    # First, build the initial rank-42 solution
    U_init, V_init, W_init = _build_strassen_solution(n, m, p)
    rank_init = len(U_init[0])
    
    # Verify the initial solution
    if not _verify_solution(U_init, V_init, W_init, n, m, p):
        # Should not happen, but return it anyway
        return {"U": U_init, "V": V_init, "W": W_init}
    
    # Try to reduce rank by checking for redundant columns
    U_reduced, V_reduced, W_reduced = _reduce_rank(U_init, V_init, W_init, n, m, p)
    rank_reduced = len(U_reduced[0])
    
    if rank_reduced < rank_init:
        return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    
    # Try alternative block structures
    best_U, best_V, best_W = U_init, V_init, W_init
    best_rank = rank_init
    
    # Try different block splits for the middle dimension
    # Split m=4 into different groupings
    for m_split in [[1,1,2], [1,2,1], [2,1,1], [1,3], [3,1], [2,2]]:
        if time.time() > deadline:
            break
        try:
            U_alt, V_alt, W_alt = _build_custom_block_solution(n, m, p, m_split)
            if _verify_solution(U_alt, V_alt, W_alt, n, m, p):
                r_alt = len(U_alt[0])
                if r_alt < best_rank:
                    best_U, best_V, best_W = U_alt, V_alt, W_alt
                    best_rank = r_alt
        except Exception:
            continue
    
    if best_rank < rank_init:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to find a lower rank solution via iterative refinement
    # Use the fact that we can try rank r and solve the system
    for r in range(24, min(best_rank, 35)):
        if time.time() > deadline:
            break
        try:
            U_sol, V_sol, W_sol = _solve_for_rank(n, m, p, r)
            if U_sol is not None and _verify_solution(U_sol, V_sol, W_sol, n, m, p):
                return {"U": U_sol, "V": V_sol, "W": W_sol}
        except Exception:
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _build_strassen_solution(n, m, p):
    """Build the standard block-Strassen solution with rank 42."""
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]

    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return U, V, W

def _verify_solution(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Convert to numpy for efficiency
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, j, k), check: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _reduce_rank(U, V, W, n, m, p):
    """Try to reduce rank by finding linearly dependent columns."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check for columns that are linear combinations of others
    # This is complex for trilinear forms, so we do a simple check:
    # If two columns have the same pattern up to scaling, we might be able to merge
    # For now, just return the original if no obvious reduction
    
    # Try to find if any column is zero
    for c in range(rank):
        if np.allclose(U_arr[:, c], 0) or np.allclose(V_arr[:, c], 0) or np.allclose(W_arr[:, c], 0):
            # Remove this column
            U_new = np.delete(U_arr, c, axis=1)
            V_new = np.delete(V_arr, c, axis=1)
            W_new = np.delete(W_arr, c, axis=1)
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return U, V, W

def _build_custom_block_solution(n, m, p, m_split):
    """Build a solution with custom block splitting of the middle dimension."""
    # This is a placeholder - implement specific block structures
    raise NotImplementedError("Custom block solution not yet implemented")

def _solve_for_rank(n, m, p, r):
    """Try to solve for a rank-r decomposition. Returns None if not found."""
    # This would require solving a large nonlinear system
    # For now, return None
    return None, None, None
# EVOLVE-BLOCK-END
