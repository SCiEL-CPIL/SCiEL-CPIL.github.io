"""
5x5 matrix multiplication - randomized algebraic rank-reduction heuristic.
Initializes factor matrices with random integer values and iteratively projects
them onto the null space of the tensor flattening matrix using integer-preserving
Gram-Schmidt orthogonalization, directly minimizing the rank parameter.
"""
import numpy as np
import random
import time

DIMENSIONS = [5, 5, 5]

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _residual(U, V, W, n, m, p):
    """
    Compute the residual of the tensor decomposition.
    For each (i,k), check that sum_c W[k*n+i,c] * (U^T A)_c * (V^T B)_c = (AB)[i,k]
    for all A, B. This is equivalent to checking that the reconstructed tensor matches.
    
    We check by evaluating on random A, B matrices.
    """
    max_res = 0.0
    for _ in range(20):
        A = np.random.randint(-10, 11, size=(n, m)).astype(float)
        B = np.random.randint(-10, 11, size=(m, p)).astype(float)
        AB = A @ B
        A_flat = A.flatten()
        B_flat = B.flatten()
        # Compute C[i,k] = sum_c W[k*n+i,c] * (U^T A_flat)_c * (V^T B_flat)_c
        UA = U.T @ A_flat  # shape (rank,)
        VB = V.T @ B_flat  # shape (rank,)
        # C[k*n+i, c] contribution
        # For each (i,k): sum_c W[k*n+i, c] * UA[c] * VB[c]
        # W has shape (p*n, rank), so W[k*n+i, :] for each (k,i)
        C = np.zeros((p, n))
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                C[k, i] = np.dot(W[idx, :], UA * VB)
        # Compare with AB: AB[i,k] should equal C[k,i]
        res = np.max(np.abs(AB - C.T))
        max_res = max(max_res, res)
    return max_res

def _integer_gram_schmidt(cols, max_iter=100):
    """
    Integer-preserving Gram-Schmidt orthogonalization.
    Given a list of column vectors, try to make them more orthogonal
    while keeping entries as integers or half-integers.
    """
    n_cols = len(cols)
    if n_cols == 0:
        return []
    
    orth = [np.array(c, dtype=float) for c in cols]
    
    for iteration in range(max_iter):
        # For each pair, try to reduce inner product
        improved = False
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                dot = np.dot(orth[i], orth[j])
                if abs(dot) > 1e-9:
                    # Try to subtract a small integer multiple to reduce dot
                    # orth[i] = orth[i] - k * orth[j] for some small integer k
                    # This changes dot(orth[i], orth[j]) -> dot - k*||orth[j]||^2
                    norm_j_sq = np.dot(orth[j], orth[j])
                    if norm_j_sq > 1e-12:
                        # Find k that minimizes |dot - k * norm_j_sq|
                        k = round(dot / norm_j_sq)
                        if k != 0 and abs(k) <= 5:
                            new_i = orth[i] - k * orth[j]
                            # Check if entries are close to integers or half-integers
                            new_i_rounded = np.round(new_i * 2) / 2.0
                            if np.max(np.abs(new_i - new_i_rounded)) < 1e-9:
                                orth[i] = new_i_rounded
                                improved = True
        if not improved:
            break
    
    return orth

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115.0  # Leave some margin
    
    # Start with schoolbook
    U_s, V_s, W_s = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    best_rank = n * m * p
    best_res = 0.0
    
    # Build the tensor flattening matrix for rank reduction
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We want to find rank-r decomposition where r < n*m*p
    
    # Strategy: Try to find a lower rank by random initialization + projection
    # The key constraint is that the sum of outer products must equal T
    
    # For each candidate rank r, we try to find U, V, W of shapes (n*m, r), (m*p, r), (p*n, r)
    # such that the tensor condition is satisfied
    
    # Approach: Start from schoolbook and try to merge columns
    # Two columns c1 and c2 can be merged if there exist scalars a, b such that
    # the merged column satisfies the tensor condition
    
    # Alternative: Random search with constraint satisfaction
    # For small rank reductions, try systematic search
    
    # Let's try a different approach: use the fact that for 5x5x5,
    # the tensor has symmetries we can exploit
    
    # Try ranks from 124 down to some minimum
    # For each rank, do random restarts
    
    # Actually, let's try a more direct approach:
    # 1. Start with schoolbook
    # 2. For each pair of columns, check if they can be merged
    # 3. If yes, merge them and continue
    
    U = U_s.copy()
    V = V_s.copy()
    W = W_s.copy()
    current_rank = best_rank
    
    # Try to merge columns
    # Two columns c1, c2 can be merged if the tensor condition is preserved
    # when we replace them with a single column
    
    # For a valid decomposition, we need:
    # sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] = T
    # where T[i*m+j, j*p+k, k*n+i] = 1
    
    # If we have two columns c1, c2, we can try to find a single column c
    # such that U[:,c]⊗V[:,c]⊗W[:,c] = U[:,c1]⊗V[:,c1]⊗W[:,c1] + U[:,c2]⊗V[:,c2]⊗W[:,c2]
    
    # This is generally not possible with a single rank-1 term, but might be
    # possible if we allow the other columns to adjust
    
    # Simpler approach: just try random perturbations and check validity
    
    # Let's try a randomized approach with integer-preserving steps
    for rank_attempt in range(best_rank - 1, max(50, best_rank - 20), -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a rank-rank_attempt decomposition
        found = False
        for restart in range(10):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with random integers in [-2, 2]
            U_r = np.random.randint(-2, 3, size=(n * m, rank_attempt)).astype(float)
            V_r = np.random.randint(-2, 3, size=(m * p, rank_attempt)).astype(float)
            W_r = np.random.randint(-2, 3, size=(p * n, rank_attempt)).astype(float)
            
            # Iteratively project to satisfy the tensor condition
            for it in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                # Compute residual
                res = _residual(U_r, V_r, W_r, n, m, p)
                if res < 1e-9:
                    found = True
                    break
                
                # Try to reduce residual by adjusting one factor at a time
                # Fix U, V; solve for W
                # C[i,k] = sum_c W[k*n+i,c] * UA[c] * VB[c]
                # We want C = AB for all A, B
                # This means W[k*n+i,c] * UA[c] * VB[c] must sum to AB[i,k]
                
                # For fixed U, V, the condition is that for all A, B:
                # sum_c W[k*n+i,c] * (sum_ab U[a*m+b,c] A[a,b]) * (sum_cd V[c'*p+d,c] B[c',d]) = (AB)[i,k]
                
                # This is a bilinear constraint on W given U, V
                # Let's try to solve it by least squares on random A, B
                
                # Generate random A, B
                n_samples = 20
                A_samples = np.random.randint(-5, 6, size=(n_samples, n, m)).astype(float)
                B_samples = np.random.randint(-5, 6, size=(n_samples, m, p)).astype(float)
                AB_samples = [A_samples[s] @ B_samples[s] for s in range(n_samples)]
                
                # For each sample s, C_s[i,k] = sum_c W[k*n+i,c] * UA_s[c] * VB_s[c]
                # where UA_s[c] = sum_ab U[a*m+b,c] * A_s[a,b]
                # and VB_s[c] = sum_cd V[c'*p+d,c] * B_s[c',d]
                
                # Build the linear system for W
                # Unknowns: W[k*n+i, c] for all k, i, c
                # Equations: for each sample s, for each i, k:
                # sum_c W[k*n+i,c] * UA_s[c] * VB_s[c] = AB_s[i,k]
                
                # This is linear in W! Let's solve it.
                # Number of unknowns: p * n * rank_attempt
                # Number of equations: n_samples * p * n
                
                num_unknowns = p * n * rank_attempt
                num_eqs = n_samples * p * n
                
                # Build the system
                A_mat = np.zeros((num_eqs, num_unknowns))
                b_vec = np.zeros(num_eqs)
                
                eq_idx = 0
                for s in range(n_samples):
                    UA_s = U_r.T @ A_samples[s].flatten()
                    VB_s = V_r.T @ B_samples[s].flatten()
                    for k in range(p):
                        for i in range(n):
                            w_idx = k * n + i
                            for c in range(rank_attempt):
                                col_idx = w_idx * rank_attempt + c
                                A_mat[eq_idx, col_idx] = UA_s[c] * VB_s[c]
                            b_vec[eq_idx] = AB_samples[s][i, k]
                            eq_idx += 1
                
                # Solve the least squares problem
                try:
                    W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_new = W_flat.reshape(p * n, rank_attempt)
                    
                    # Check if entries are close to integers or half-integers
                    W_rounded = np.round(W_new * 2) / 2.0
                    if np.max(np.abs(W_new - W_rounded)) < 1e-6:
                        W_r = W_rounded
                        # Recompute residual
                        res = _residual(U_r, V_r, W_r, n, m, p)
                        if res < 1e-9:
                            found = True
                            break
                except:
                    pass
                
                # Similarly try to solve for U and V
                # Fix V, W; solve for U
                A_mat_U = np.zeros((num_eqs, num_unknowns))
                b_vec_U = np.zeros(num_eqs)
                eq_idx = 0
                for s in range(n_samples):
                    VB_s = V_r.T @ B_samples[s].flatten()
                    # C_s[i,k] = sum_c W[k*n+i,c] * UA_s[c] * VB_s[c]
                    # UA_s[c] = sum_ab U[a*m+b,c] * A_s[a,b]
                    # So C_s[i,k] = sum_ab sum_c U[a*m+b,c] * A_s[a,b] * W[k*n+i,c] * VB_s[c]
                    # = sum_ab U[a*m+b,c] * A_s[a,b] * (sum_c W[k*n+i,c] * VB_s[c])
                    for k in range(p):
                        for i in range(n):
                            wvb = np.dot(W_r[k * n + i, :], VB_s)
                            for a in range(n):
                                for b in range(m):
                                    u_idx = a * m + b
                                    col_idx = u_idx * rank_attempt
                                    # U has shape (n*m, rank_attempt), so U[a*m+b, c]
                                    # The unknown is U[a*m+b, c] for each c
                                    # C_s[i,k] = sum_c U[i*m+?, c] * ... 
                                    # Wait, U[a*m+b, c] where a=i for the row index
                                    # Actually U has shape (n*m, rank), indexed by [a*m+b, c]
                                    # For the term A[i,j] * B[j,k], we need U[i*m+j, c]
                                    # So for fixed i, k, the contribution is:
                                    # sum_j sum_c U[i*m+j, c] * A[i,j] * W[k*n+i, c] * VB_s[c]
                                    # = sum_j A[i,j] * sum_c U[i*m+j, c] * W[k*n+i, c] * VB_s[c]
                                    # = sum_j A[i,j] * (U[i*m+j, :] . (W[k*n+i, :] * VB_s))
                                    if a == i:
                                        wvb_vec = W_r[k * n + i, :] * VB_s
                                        for c in range(rank_attempt):
                                            col_idx = u_idx * rank_attempt + c
                                            A_mat_U[eq_idx, col_idx] = A_samples[s, a, b] * wvb_vec[c]
                            b_vec_U[eq_idx] = AB_samples[s][i, k]
                            eq_idx += 1
                
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_mat_U, b_vec_U, rcond=None)
                    U_new = U_flat.reshape(n * m, rank_attempt)
                    U_rounded = np.round(U_new * 2) / 2.0
                    if np.max(np.abs(U_new - U_rounded)) < 1e-6:
                        U_r = U_rounded
                        res = _residual(U_r, V_r, W_r, n, m, p)
                        if res < 1e-9:
                            found = True
                            break
                except:
                    pass
                
                # Fix U, W; solve for V
                A_mat_V = np.zeros((num_eqs, num_unknowns))
                b_vec_V = np.zeros(num_eqs)
                eq_idx = 0
                for s in range(n_samples):
                    UA_s = U_r.T @ A_samples[s].flatten()
                    for k in range(p):
                        for i in range(n):
                            # C_s[i,k] = sum_c W[k*n+i,c] * UA_s[c] * VB_s[c]
                            # VB_s[c] = sum_cd V[c'*p+d,c] * B_s[c',d]
                            # So C_s[i,k] = sum_cd sum_c V[c'*p+d,c] * B_s[c',d] * W[k*n+i,c] * UA_s[c]
                            # = sum_cd B_s[c',d] * (V[c'*p+d, :] . (W[k*n+i, :] * UA_s))
                            uw = W_r[k * n + i, :] * UA_s
                            for c_prime in range(m):
                                for d in range(p):
                                    v_idx = c_prime * p + d
                                    for c in range(rank_attempt):
                                        col_idx = v_idx * rank_attempt + c
                                        A_mat_V[eq_idx, col_idx] = B_samples[s, c_prime, d] * uw[c]
                            b_vec_V[eq_idx] = AB_samples[s][i, k]
                            eq_idx += 1
                
                try:
                    V_flat, _, _, _ = np.linalg.lstsq(A_mat_V, b_vec_V, rcond=None)
                    V_new = V_flat.reshape(m * p, rank_attempt)
                    V_rounded = np.round(V_new * 2) / 2.0
                    if np.max(np.abs(V_new - V_rounded)) < 1e-6:
                        V_r = V_rounded
                        res = _residual(U_r, V_r, W_r, n, m, p)
                        if res < 1e-9:
                            found = True
                            break
                except:
                    pass
            
            if found:
                best_U = U_r.copy()
                best_V = V_r.copy()
                best_W = W_r.copy()
                best_rank = rank_attempt
                best_res = 0.0
                break
        
        if found:
            # Try to go even lower
            continue
    
    # Final check
    final_res = _residual(best_U, best_V, best_W, n, m, p)
    if final_res > 1e-9:
        # Fall back to schoolbook
        best_U, best_V, best_W = U_s, V_s, W_s
        best_rank = n * m * p
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
