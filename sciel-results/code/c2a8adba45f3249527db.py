"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _get_rank17_solution():
    """Returns the known rank-17 solution using Strassen block."""
    n, m, p = 2, 3, 3
    n_a, n_b, n_c = 6, 9, 6
    rank = 17
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    
    def set_term(r, u_dict, v_dict, w_dict):
        for a, val in u_dict.items():
            U[a, r] = val
        for b, val in v_dict.items():
            V[b, r] = val
        for c, val in w_dict.items():
            W[c, r] = val
            
    # Part 1: Strassen for A1(2x2) x B1_left(2x2) -> C[0:2, 0:2]
    set_term(0, {0:1, 4:1}, {0:1, 4:1}, {0:1, 3:1})
    set_term(1, {3:1, 4:1}, {0:1}, {1:1, 3:-1})
    set_term(2, {0:1}, {1:1, 4:-1}, {2:1, 3:1})
    set_term(3, {4:1}, {3:1, 0:-1}, {0:1, 1:1})
    set_term(4, {0:1, 1:1}, {4:1}, {0:-1, 2:1})
    set_term(5, {3:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_term(6, {1:1, 4:-1}, {3:1, 4:1}, {0:1})
    
    # Part 2: Direct for A1(2x2) x B1_right(2x1) -> C[:, 2]
    set_term(7, {0:1}, {2:1}, {4:1})
    set_term(8, {1:1}, {5:1}, {4:1})
    set_term(9, {3:1}, {2:1}, {5:1})
    set_term(10, {4:1}, {5:1}, {5:1})
    
    # Part 3: Direct for A2(2x1) x B2(1x3) -> C[0:2, 0:3]
    set_term(11, {2:1}, {6:1}, {0:1})
    set_term(12, {2:1}, {7:1}, {2:1})
    set_term(13, {2:1}, {8:1}, {4:1})
    set_term(14, {5:1}, {6:1}, {1:1})
    set_term(15, {5:1}, {7:1}, {3:1})
    set_term(16, {5:1}, {8:1}, {5:1})
    
    return U, V, W

def _try_absorb_term_systematic(U17, V17, W17, T, remove_idx, deadline):
    """
    Systematically try to absorb term remove_idx into the other 16 terms
    by trying all single-entry support modifications.
    """
    n_a, n_b, n_c = U17.shape[0], V17.shape[0], W17.shape[0]
    rank17 = U17.shape[1]
    keep_indices = [r for r in range(rank17) if r != remove_idx]
    T_flat = T.flatten()
    num_entries = T_flat.size
    
    # First try: no modification, just solve
    A = np.zeros((num_entries, 16))
    for j, r in enumerate(keep_indices):
        A[:, j] = np.einsum('a,b,c->abc', U17[:, r], V17[:, r], W17[:, r]).flatten()
    try:
        alpha, _, _, _ = np.linalg.lstsq(A, T_flat, rcond=None)
        if np.allclose(A @ alpha, T_flat, atol=1e-9):
            U16 = np.zeros((n_a, 16))
            V16 = np.zeros((n_b, 16))
            W16 = np.zeros((n_c, 16))
            for j, r in enumerate(keep_indices):
                U16[:, j] = alpha[j] * U17[:, r]
                V16[:, j] = alpha[j] * V17[:, r]
                W16[:, j] = alpha[j] * W17[:, r]
            return U16, V16, W16
    except:
        pass
    
    # Get base supports
    base_supU = [set(np.nonzero(U17[:, r])[0]) for r in keep_indices]
    base_supV = [set(np.nonzero(V17[:, r])[0]) for r in keep_indices]
    base_supW = [set(np.nonzero(W17[:, r])[0]) for r in keep_indices]
    
    # Systematically try adding one entry to U, V, or W of each term
    for j in range(16):
        if time.time() > deadline:
            return None
        # Try adding each possible U entry
        for new_a in range(n_a):
            if new_a not in base_supU[j]:
                supU = list(base_supU[j]) + [new_a]
                supV = list(base_supV[j])
                supW = list(base_supW[j])
                U_test = np.zeros((n_a, 16))
                V_test = np.zeros((n_b, 16))
                W_test = np.zeros((n_c, 16))
                for jj, rr in enumerate(keep_indices):
                    if jj == j:
                        for idx in supU:
                            U_test[idx, jj] = 1
                        for idx in supV:
                            V_test[idx, jj] = 1
                        for idx in supW:
                            W_test[idx, jj] = 1
                    else:
                        U_test[:, jj] = U17[:, rr]
                        V_test[:, jj] = V17[:, rr]
                        W_test[:, jj] = W17[:, rr]
                A_test = np.zeros((num_entries, 16))
                for jj in range(16):
                    A_test[:, jj] = np.einsum('a,b,c->abc', U_test[:, jj], V_test[:, jj], W_test[:, jj]).flatten()
                try:
                    alpha_test, _, _, _ = np.linalg.lstsq(A_test, T_flat, rcond=None)
                    if np.allclose(A_test @ alpha_test, T_flat, atol=1e-9):
                        U16 = np.zeros((n_a, 16))
                        V16 = np.zeros((n_b, 16))
                        W16 = np.zeros((n_c, 16))
                        for jj in range(16):
                            U16[:, jj] = alpha_test[jj] * U_test[:, jj]
                            V16[:, jj] = alpha_test[jj] * V_test[:, jj]
                            W16[:, jj] = alpha_test[jj] * W_test[:, jj]
                        return U16, V16, W16
                except:
                    pass
        # Try adding each possible V entry
        for new_b in range(n_b):
            if new_b not in base_supV[j]:
                supU = list(base_supU[j])
                supV = list(base_supV[j]) + [new_b]
                supW = list(base_supW[j])
                U_test = np.zeros((n_a, 16))
                V_test = np.zeros((n_b, 16))
                W_test = np.zeros((n_c, 16))
                for jj, rr in enumerate(keep_indices):
                    if jj == j:
                        for idx in supU:
                            U_test[idx, jj] = 1
                        for idx in supV:
                            V_test[idx, jj] = 1
                        for idx in supW:
                            W_test[idx, jj] = 1
                    else:
                        U_test[:, jj] = U17[:, rr]
                        V_test[:, jj] = V17[:, rr]
                        W_test[:, jj] = W17[:, rr]
                A_test = np.zeros((num_entries, 16))
                for jj in range(16):
                    A_test[:, jj] = np.einsum('a,b,c->abc', U_test[:, jj], V_test[:, jj], W_test[:, jj]).flatten()
                try:
                    alpha_test, _, _, _ = np.linalg.lstsq(A_test, T_flat, rcond=None)
                    if np.allclose(A_test @ alpha_test, T_flat, atol=1e-9):
                        U16 = np.zeros((n_a, 16))
                        V16 = np.zeros((n_b, 16))
                        W16 = np.zeros((n_c, 16))
                        for jj in range(16):
                            U16[:, jj] = alpha_test[jj] * U_test[:, jj]
                            V16[:, jj] = alpha_test[jj] * V_test[:, jj]
                            W16[:, jj] = alpha_test[jj] * W_test[:, jj]
                        return U16, V16, W16
                except:
                    pass
        # Try adding each possible W entry
        for new_c in range(n_c):
            if new_c not in base_supW[j]:
                supU = list(base_supU[j])
                supV = list(base_supV[j])
                supW = list(base_supW[j]) + [new_c]
                U_test = np.zeros((n_a, 16))
                V_test = np.zeros((n_b, 16))
                W_test = np.zeros((n_c, 16))
                for jj, rr in enumerate(keep_indices):
                    if jj == j:
                        for idx in supU:
                            U_test[idx, jj] = 1
                        for idx in supV:
                            V_test[idx, jj] = 1
                        for idx in supW:
                            W_test[idx, jj] = 1
                    else:
                        U_test[:, jj] = U17[:, rr]
                        V_test[:, jj] = V17[:, rr]
                        W_test[:, jj] = W17[:, rr]
                A_test = np.zeros((num_entries, 16))
                for jj in range(16):
                    A_test[:, jj] = np.einsum('a,b,c->abc', U_test[:, jj], V_test[:, jj], W_test[:, jj]).flatten()
                try:
                    alpha_test, _, _, _ = np.linalg.lstsq(A_test, T_flat, rcond=None)
                    if np.allclose(A_test @ alpha_test, T_flat, atol=1e-9):
                        U16 = np.zeros((n_a, 16))
                        V16 = np.zeros((n_b, 16))
                        W16 = np.zeros((n_c, 16))
                        for jj in range(16):
                            U16[:, jj] = alpha_test[jj] * U_test[:, jj]
                            V16[:, jj] = alpha_test[jj] * V_test[:, jj]
                            W16[:, jj] = alpha_test[jj] * W_test[:, jj]
                        return U16, V16, W16
                except:
                    pass
    return None

def _laser_search_rank16(T, deadline):
    """
    Laser method: fix a hyperplane by setting one entry of each factor vector to 1,
    then solve for the remaining free variables.
    """
    n_a, n_b, n_c = 6, 9, 6
    T_flat = T.flatten()
    num_entries = T_flat.size
    rank = 16
    
    # Try different laser hyperplanes
    # A laser hyperplane fixes: for each term r, one entry in U, V, W is set to 1
    # We try systematic choices
    
    # Strategy: for each choice of (a_fixed, b_fixed, c_fixed) pattern,
    # try to find a valid decomposition
    
    # Simpler approach: try fixing specific entries and solving
    # For term r, fix U[r_idx, r] = 1, V[v_idx, r] = 1, W[w_idx, r] = 1
    
    # Try a few structured patterns
    patterns = [
        # (U_fixed_indices, V_fixed_indices, W_fixed_indices)
        (list(range(16)), list(range(16)), list(range(16))),  # identity-like
    ]
    
    # Alternative: use the rank-17 structure as a starting point
    # and try to find a rank-16 by solving a larger system
    
    # More practical: try all single-entry removals from rank-17
    U17, V17, W17 = _get_rank17_solution()
    
    for remove_idx in range(17):
        if time.time() > deadline:
            return None
        
        keep_indices = [r for r in range(17) if r != remove_idx]
        
        # Try to solve with modified supports
        result = _try_absorb_term_systematic(U17, V17, W17, T, remove_idx, deadline)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline with safety margin
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    T = _build_tensor(n, m, p)
    
    # 1. Attempt systematic rank-16 search
    result_16 = _laser_search_rank16(T, deadline)
    if result_16 is not None:
        U, V, W = result_16
        if _verify_decomposition(U, V, W, T):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # 2. Fallback to verified Rank 17
    U, V, W = _get_rank17_solution()
    
    if not _verify_decomposition(U, V, W, T):
        # Final fallback: Schoolbook (Rank 18)
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
