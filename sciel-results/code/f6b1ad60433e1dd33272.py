"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x7 matrix.
    
    Strategy:
    1. Build the matrix multiplication tensor T[a,b,c]
    2. Try to find a low-rank decomposition with integer/half-integer entries
    3. Use structured search based on the tensor's algebraic properties
    4. Fall back to schoolbook if no better decomposition is found
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p  # 84

    # Build the tensor T[a, b, c] where:
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a, b, c] = 1 if the indices are consistent (same i, j, k), else 0
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]"""
        T_recon = np.zeros((nm, mp, pn), dtype=float)
        for r in range(U.shape[1]):
            T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        return np.max(np.abs(T_recon - T)) < tol

    def round_to_half_integer(x):
        """Round to nearest multiple of 0.5"""
        return np.round(2 * x) / 2.0

    def is_half_integer_array(x, tol=1e-6):
        """Check if all entries are close to half-integers"""
        return np.max(np.abs(x * 2 - np.round(x * 2))) < tol

    def try_refine_with_scaling(U, V, W, n_attempts=200):
        """
        Try to find scalings alpha, beta, gamma (alpha*beta*gamma=1)
        such that after rounding to half-integers, the decomposition is exact.
        """
        # First check if already half-integer
        if is_half_integer_array(U) and is_half_integer_array(V) and is_half_integer_array(W):
            if verify_decomposition(U, V, W):
                return U, V, W
        
        best_err = float('inf')
        best_result = None
        
        for _ in range(n_attempts):
            if time.time() > deadline:
                break
            
            # Random scaling factors
            alpha = np.random.uniform(0.3, 3.0)
            beta = np.random.uniform(0.3, 3.0)
            gamma = 1.0 / (alpha * beta)
            
            U_s = U * alpha
            V_s = V * beta
            W_s = W * gamma
            
            U_r = round_to_half_integer(U_s)
            V_r = round_to_half_integer(V_s)
            W_r = round_to_half_integer(W_s)
            
            if verify_decomposition(U_r, V_r, W_r):
                return U_r, V_r, W_r
        
        return None

    def structured_search(rank, max_iter=5000):
        """
        Try to find a decomposition of given rank using structured approach.
        The tensor has a special structure: T[a,b,c] = 1 iff indices are consistent.
        
        For matrix multiplication C = A * B where A is n×m and B is m×p:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
        
        We look for decompositions of the form:
        T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        Key insight: For each (i,j,k), we need:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        And for all other (a,b,c) combinations, the sum should be 0.
        """
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Try multiple random initializations
        n_init = min(20, max(5, max_iter // 250))
        
        for init in range(n_init):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Gradient-free optimization: iterative refinement
            # We want to minimize ||T - sum_r U[:,r] V[:,r] W[:,r]||^2
            # subject to entries being half-integers (or close)
            
            # Use a simple iterative approach:
            # For each r, try to improve the reconstruction
            
            current_err = float('inf')
            for it in range(max_iter // n_init):
                if time.time() > deadline:
                    break
                
                # Compute current error
                T_recon = np.zeros((nm, mp, pn), dtype=float)
                for r in range(rank):
                    T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
                
                err = np.max(np.abs(T_recon - T))
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if err < 1e-10:
                    break
                
                # Try to improve: for each r, adjust U[:,r], V[:,r], W[:,r]
                # to better match the residual
                residual = T - T_recon
                
                for r in range(rank):
                    # For this rank component, we want to reduce the residual
                    # The contribution of component r is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
                    
                    # Try adjusting W first (it's the "output" side)
                    # For each (a,b), the residual at (a,b,:) should be matched by
                    # U[a,r] * V[b,r] * W[:,r]
                    
                    # Simplified: try random perturbations
                    if np.random.rand() < 0.3:
                        # Perturb U
                        idx = np.random.randint(nm)
                        U[idx, r] += np.random.randn() * 0.1
                    elif np.random.rand() < 0.6:
                        # Perturb V
                        idx = np.random.randint(mp)
                        V[idx, r] += np.random.randn() * 0.1
                    else:
                        # Perturb W
                        idx = np.random.randint(pn)
                        W[idx, r] += np.random.randn() * 0.1
            
            # After optimization, try to round
            result = try_refine_with_scaling(best_U if best_U is not None else U,
                                             best_V if best_V is not None else V,
                                             best_W if best_W is not None else W,
                                             n_attempts=100)
            if result is not None:
                return result
        
        return None

    def svd_based_search(rank):
        """
        Use SVD on unfoldings to get a starting point, then refine.
        """
        if time.time() > deadline:
            return None
        
        try:
            # Unfolding 2: (nm*mp, pn)
            T_unfold2 = T.reshape(nm * mp, pn)
            
            U_svd, s_vals, Vt_svd = np.linalg.svd(T_unfold2, full_matrices=False)
            
            if rank >= len(s_vals):
                rank = len(s_vals)
            if rank < 1:
                return None
                
            U_trunc = U_svd[:, :rank] * s_vals[:rank]
            Vt_trunc = Vt_svd[:rank, :]
            
            # Reshape U_trunc to (nm, mp, rank)
            U_reshaped = U_trunc.reshape(nm, mp, rank)
            
            # For each r, factor U_reshaped[:, :, r] as U[:, r] * V[:, r]^T
            U_fact = np.zeros((nm, rank))
            V_fact = np.zeros((mp, rank))
            
            feasible = True
            for r in range(rank):
                mat_r = U_reshaped[:, :, r]
                if np.linalg.matrix_rank(mat_r, tol=1e-10) > 1:
                    feasible = False
                    break
                try:
                    u_r, s_r, vt_r = np.linalg.svd(mat_r, full_matrices=False)
                    if len(s_r) > 0 and s_r[0] > 1e-12:
                        U_fact[:, r] = u_r[:, 0] * np.sqrt(s_r[0])
                        V_fact[:, r] = vt_r[0, :] * np.sqrt(s_r[0])
                    else:
                        U_fact[:, r] = 0
                        V_fact[:, r] = 0
                except Exception:
                    feasible = False
                    break
            
            if not feasible:
                return None
            
            W_fact = Vt_trunc.T  # shape (pn, rank)
            
            # Check if they reconstruct T
            if verify_decomposition(U_fact, V_fact, W_fact):
                result = try_refine_with_scaling(U_fact, V_fact, W_fact, n_attempts=200)
                if result is not None:
                    return result
            
            # Try structured refinement
            result = structured_search(rank, max_iter=2000)
            if result is not None:
                return result
                
        except Exception:
            pass
        
        return None

    # Try to find minimal rank
    best_result = None
    
    # Lower bound from matrix ranks of unfoldings
    T_unfold1 = T.reshape(nm, mp * pn)
    T_unfold2 = T.reshape(nm * mp, pn)
    T_unfold3 = T.reshape(mp, nm * pn)
    
    try:
        rank_lb1 = np.linalg.matrix_rank(T_unfold1, tol=1e-10)
        rank_lb2 = np.linalg.matrix_rank(T_unfold2, tol=1e-10)
        rank_lb3 = np.linalg.matrix_rank(T_unfold3, tol=1e-10)
        rank_lb = max(rank_lb1, rank_lb2, rank_lb3)
    except Exception:
        rank_lb = 1

    rank_lb = max(1, int(rank_lb))
    
    # For 3x4x7 matrix multiplication, known lower bounds suggest rank >= 3*4*7 = 84
    # but there might be better decompositions
    # Try ranks from lower bound up to schoolbook
    max_rank_to_try = min(schoolbook_rank, rank_lb + 40)
    
    # First try SVD-based approach for lower ranks
    for rank in range(rank_lb, min(rank_lb + 10, max_rank_to_try + 1)):
        if time.time() > deadline:
            break
        
        result = svd_based_search(rank)
        if result is not None:
            U_r, V_r, W_r = result
            best_result = (U_r, V_r, W_r, rank)
            break
    
    # If not found, try structured search for a few more ranks
    if best_result is None:
        for rank in range(max(rank_lb, 20), min(rank_lb + 20, max_rank_to_try + 1)):
            if time.time() > deadline:
                break
            
            result = structured_search(rank, max_iter=3000)
            if result is not None:
                U_r, V_r, W_r = result
                best_result = (U_r, V_r, W_r, rank)
                break

    # If no better decomposition found, fall back to schoolbook
    if best_result is None:
        rank = schoolbook_rank
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        
        best_result = (U, V, W, rank)

    U, V, W, rank = best_result
    
    # Convert to lists of lists for compatibility
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
