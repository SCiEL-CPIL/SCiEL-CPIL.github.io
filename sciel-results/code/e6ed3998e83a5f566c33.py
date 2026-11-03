"""
Strategy: Data-driven greedy rank-1 extraction followed by local search.
We treat the matrix multiplication tensor T as a 3-way array.
We start with a random (or schoolbook) decomposition and iteratively:
1. Try to replace a rank-1 term with a better one (minimizing residual).
2. Try to extract a new rank-1 term from the residual (greedy).
3. Try to remove a term if the residual is small enough (rank reduction).

This is a heuristic approach that may find lower rank than the structural 98.
"""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _matmul_tensor_to_decomp(n, m, p, U, V, W):
    """Check if U, V, W decompose the tensor."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_true = _build_tensor(n, m, p)
    return np.allclose(T_recon, T_true, atol=1e-9)

def _greedy_extract_residual(R, n, m, p, max_iter=1000, tol=1e-10):
    """
    Greedy rank-1 extraction from residual tensor R.
    Returns list of (u, v, w) vectors.
    """
    terms = []
    R_work = R.copy()
    for _ in range(max_iter):
        if np.linalg.norm(R_work) < tol:
            break
        
        # Find best rank-1 approximation of R_work
        # Use SVD on different modes
        best_u, best_v, best_w = None, None, None
        best_score = -1
        
        # Mode 1 SVD
        M1 = R_work.reshape(n*m, -1)
        U1, S1, Vt1 = np.linalg.svd(M1, full_matrices=False)
        u1 = U1[:, 0]
        sv = S1[0]
        v1 = Vt1[0, :].reshape(m*p, p*n)
        # Now find best rank-1 in v1
        M2 = v1
        U2, S2, Vt2 = np.linalg.svd(M2, full_matrices=False)
        v2 = U2[:, 0]
        w2 = Vt2[0, :]
        
        score1 = np.abs(np.einsum('i,j,k->', u1, v2, w2) * sv)
        
        # Mode 2 SVD
        M2_full = R_work.reshape(m*p, -1)
        U2f, S2f, Vt2f = np.linalg.svd(M2_full, full_matrices=False)
        v2f = U2f[:, 0]
        sv2f = S2f[0]
        u2f = Vt2f[0, :].reshape(n*m, p*n)
        M3 = u2f
        U3, S3, Vt3 = np.linalg.svd(M3, full_matrices=False)
        u3 = U3[:, 0]
        w3 = Vt3[0, :]
        
        score2 = np.abs(np.einsum('i,j,k->', u3, v2f, w3) * sv2f)
        
        # Mode 3 SVD
        M3_full = R_work.reshape(p*n, -1)
        U3f, S3f, Vt3f = np.linalg.svd(M3_full, full_matrices=False)
        w3f = U3f[:, 0]
        sv3f = S3f[0]
        v3f = Vt3f[0, :].reshape(n*m, m*p)
        M4 = v3f
        U4, S4, Vt4 = np.linalg.svd(M4, full_matrices=False)
        v4 = U4[:, 0]
        u4 = Vt4[0, :]
        
        score3 = np.abs(np.einsum('i,j,k->', u4, v4, w3f) * sv3f)
        
        # Pick best
        if score1 >= score2 and score1 >= score3:
            best_u, best_v, best_w = u1, v2, w2
        elif score2 >= score1 and score2 >= score3:
            best_u, best_v, best_w = u3, v2f, w3
        else:
            best_u, best_v, best_w = u4, v4, w3f
        
        # Normalize
        norm_u = np.linalg.norm(best_u)
        norm_v = np.linalg.norm(best_v)
        norm_w = np.linalg.norm(best_w)
        if norm_u < 1e-12 or norm_v < 1e-12 or norm_w < 1e-12:
            break
        
        best_u = best_u / norm_u
        best_v = best_v / norm_v
        best_w = best_w / norm_w
        
        # Compute coefficient
        coeff = np.einsum('i,j,k->', best_u, best_v, best_w)
        if abs(coeff) < 1e-12:
            break
            
        # Update residual
        R_work -= coeff * np.einsum('i,j,k->ijk', best_u, best_v, best_w)
        terms.append((best_u, best_v, best_w, coeff))
        
    return terms

def _local_search(U, V, W, T, n, m, p, max_iter=500, tol=1e-9):
    """
    Local search: try to improve rank by replacing or removing terms.
    """
    r = U.shape[1]
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    
    current_rank = r
    no_improve = 0
    max_no_improve = 50
    
    for it in range(max_iter):
        if no_improve > max_no_improve:
            break
            
        improved = False
        
        # Try to remove a term
        for r_idx in range(current_rank):
            # Compute residual without this term
            R = T - np.einsum('ar,br,cr->abc', 
                              np.delete(U_work, r_idx, axis=1), 
                              np.delete(V_work, r_idx, axis=1), 
                              np.delete(W_work, r_idx, axis=1))
            if np.linalg.norm(R) < tol:
                U_work = np.delete(U_work, r_idx, axis=1)
                V_work = np.delete(V_work, r_idx, axis=1)
                W_work = np.delete(W_work, r_idx, axis=1)
                current_rank -= 1
                improved = True
                no_improve = 0
                break
        
        if improved:
            continue
            
        # Try to replace a term with a better one
        for r_idx in range(current_rank):
            # Remove term
            U_temp = np.delete(U_work, r_idx, axis=1)
            V_temp = np.delete(V_work, r_idx, axis=1)
            W_temp = np.delete(W_work, r_idx, axis=1)
            
            R = T - np.einsum('ar,br,cr->abc', U_temp, V_temp, W_temp)
            
            # Extract best rank-1 from R
            M1 = R.reshape(n*m, -1)
            U1, S1, Vt1 = np.linalg.svd(M1, full_matrices=False)
            u1 = U1[:, 0]
            v1 = Vt1[0, :].reshape(m*p, p*n)
            M2 = v1
            U2, S2, Vt2 = np.linalg.svd(M2, full_matrices=False)
            v2 = U2[:, 0]
            w2 = Vt2[0, :]
            
            norm_u = np.linalg.norm(u1)
            norm_v = np.linalg.norm(v2)
            norm_w = np.linalg.norm(w2)
            if norm_u < 1e-12 or norm_v < 1e-12 or norm_w < 1e-12:
                continue
                
            u1 = u1 / norm_u
            v2 = v2 / norm_v
            w2 = w2 / norm_w
            coeff = np.einsum('i,j,k->', u1, v2, w2)
            
            if abs(coeff) < 1e-12:
                continue
                
            # Check if this replacement improves
            U_new = np.insert(U_temp, r_idx, u1 * coeff, axis=1)
            V_new = np.insert(V_temp, r_idx, v2, axis=1)
            W_new = np.insert(W_temp, r_idx, w2, axis=1)
            
            R_new = T - np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
            R_old = T - np.einsum('ar,br,cr->abc', U_work, V_work, W_work)
            
            if np.linalg.norm(R_new) < np.linalg.norm(R_old) - 1e-10:
                U_work = U_new
                V_work = V_new
                W_work = W_new
                improved = True
                no_improve = 0
                break
        
        if not improved:
            no_improve += 1
            
    return U_work, V_work, W_work, current_rank

def _schoolbook_decomp(n, m, p):
    U = np.zeros((n*m, n*m*p), dtype=np.float64)
    V = np.zeros((m*p, n*m*p), dtype=np.float64)
    W = np.zeros((p*n, n*m*p), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomp(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try greedy extraction from zero
    if time.time() < deadline - 20:
        terms = _greedy_extract_residual(T, n, m, p, max_iter=200)
        if terms:
            r_greedy = len(terms)
            U_g = np.zeros((n*m, r_greedy), dtype=np.float64)
            V_g = np.zeros((m*p, r_greedy), dtype=np.float64)
            W_g = np.zeros((p*n, r_greedy), dtype=np.float64)
            for idx, (u, v, w, coeff) in enumerate(terms):
                U_g[:, idx] = u * coeff
                V_g[:, idx] = v
                W_g[:, idx] = w
            
            if _matmul_tensor_to_decomp(n, m, p, U_g, V_g, W_g):
                if r_greedy < best_rank:
                    best_rank = r_greedy
                    best_U, best_V, best_W = U_g, V_g, W_g
                    print(f"Greedy found rank {r_greedy}")
            
            # Local search on greedy result
            if time.time() < deadline - 10:
                U_ls, V_ls, W_ls, r_ls = _local_search(U_g, V_g, W_g, T, n, m, p, max_iter=200)
                if _matmul_tensor_to_decomp(n, m, p, U_ls, V_ls, W_ls):
                    if r_ls < best_rank:
                        best_rank = r_ls
                        best_U, best_V, best_W = U_ls, V_ls, W_ls
                        print(f"Local search on greedy found rank {r_ls}")
    
    # Local search on schoolbook
    if time.time() < deadline - 10:
        U_ls, V_ls, W_ls, r_ls = _local_search(U, V, W, T, n, m, p, max_iter=300)
        if _matmul_tensor_to_decomp(n, m, p, U_ls, V_ls, W_ls):
            if r_ls < best_rank:
                best_rank = r_ls
                best_U, best_V, best_W = U_ls, V_ls, W_ls
                print(f"Local search on schoolbook found rank {r_ls}")
    
    # Convert to half-integer lists
    def to_half_integer(mat):
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-6):
            return (rounded / 2).tolist()
        rounded_int = np.round(mat)
        if np.allclose(mat, rounded_int, atol=1e-6):
            return rounded_int.astype(int).tolist()
        return mat.tolist()

    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
