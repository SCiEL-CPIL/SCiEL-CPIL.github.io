import numpy as np
from fractions import Fraction
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_tensor_from_factors(U, V, W, n, m, p):
    """Build the full tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[i, j, k] = np.sum(U_arr[idx_u, :] * V_arr[idx_v, :] * W_arr[idx_w, :])
    return T

def _residual_tensor(T, U, V, W, n, m, p):
    """Compute residual = T - sum of rank-1 terms from U,V,W"""
    T_decomp = _build_tensor_from_factors(U, V, W, n, m, p)
    return T - T_decomp

def _try_peel_rank1(T, n, m, p, time_limit):
    """
    Try to find a rank-1 summand in the tensor T.
    A rank-1 tensor has form T[i,j,k] = a[i,j] * b[j,k] * c[k,i]
    We search for such a decomposition.
    """
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Reshape T to matrix form for rank analysis
    # T is (n, m, p). For rank-1, T[i,j,k] = u[i,j] * v[j,k] * w[k,i]
    # This is a 3D tensor. We can try to find if it's rank-1 by checking
    # if all 2D slices are rank-1 and consistent.
    
    # Alternative approach: try to find vectors u (nm x 1), v (mp x 1), w (pn x 1)
    # such that T[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    # This is a multiplicative problem. Let's take logs where possible.
    # But entries can be zero or negative, so we need a different approach.
    
    # Strategy: try all possible sign patterns and magnitudes
    # For a rank-1 tensor, for any fixed j, the matrix T[:,j,:] should be outer product of u[:,j] and w[:,j]
    # i.e., T[i,j,k] = u_ij * w_ki for fixed j, which means T[:,j,:] = u_j * w_j^T where u_j is n-vector and w_j is p-vector
    
    # Actually for rank-1: T[i,j,k] = a[i,j] * b[j,k] * c[k,i]
    # For fixed j: T[i,j,k] = (a[i,j]) * (b[j,k] * c[k,i])
    # The matrix M_j[i,k] = b[j,k] * c[k,i] should be rank-1
    
    # Check if each slice T[:,j,:] is rank-1
    for j in range(m):
        M = T[:, j, :]  # (n, p) matrix
        # Check rank
        try:
            svd_ranks = np.linalg.matrix_rank(M, tol=1e-10)
            if svd_ranks > 1:
                return None
        except:
            return None
    
    # If all slices are rank-1, try to reconstruct
    # T[i,j,k] = a[i,j] * b[j,k] * c[k,i]
    # From slice j=0: T[i,0,k] = a[i,0] * b[0,k] * c[k,i]
    # Let's set b[0,k] = 1 for all k (normalization), then T[i,0,k] = a[i,0] * c[k,i]
    # So c[k,i] = T[i,0,k] / a[i,0]
    
    # We need to find a[i,j], b[j,k], c[k,i] consistently
    # This is a system of equations. Let's use the fact that:
    # T[i,j,k] * T[i',j',k'] * T[i'',j'',k''] = T[i,j',k'] * T[i',j'',k] * T[i'',j,k]
    # for rank-1 tensors (multiplicative consistency)
    
    # Simpler: fix normalization b[0,0] = 1
    # Then for j=0: T[i,0,k] = a[i,0] * c[k,i]
    # For k=0: T[i,j,0] = a[i,j] * b[j,0] * c[0,i]
    
    # Let's try a direct approach: assume the rank-1 form and solve
    # T[i,j,k] = a[i,j] * b[j,k] * c[k,i]
    
    # Take the element T[0,0,0] = a[0,0] * b[0,0] * c[0,0]
    # Normalize: set a[0,0] = 1, then b[0,0] * c[0,0] = T[0,0,0]
    
    # This is complex. Let's try a numerical optimization approach.
    
    def _rank1_residual(params, n=n, m=m, p=p, T=T):
        """Compute residual for given rank-1 parameters"""
        nm = n * m
        mp = m * p
        pn = p * n
        a = params[:nm].reshape(n, m)
        b = params[nm:nm+mp].reshape(m, p)
        c = params[nm+mp:nm+mp+pn].reshape(p, n)
        
        T_recon = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] = a[i, j] * b[j, k] * c[k, i]
        return np.sum((T - T_recon) ** 2)
    
    # Initialize with zeros
    nm = n * m
    mp = m * p
    pn = p * n
    x0 = np.zeros(nm + mp + pn)
    
    # Try multiple initializations
    best_res = np.inf
    best_params = None
    
    for trial in range(10):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        np.random.seed(trial)
        x0 = np.random.randn(nm + mp + pn) * 0.1
        
        try:
            result = minimize(_rank1_residual, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if result.fun < best_res:
                best_res = result.fun
                best_params = result.x
        except:
            continue
    
    if best_res > 1e-10:
        return None
    
    # Extract parameters
    a = best_params[:nm].reshape(n, m)
    b = best_params[nm:nm+mp].reshape(m, p)
    c = best_params[nm+mp:nm+mp+pn].reshape(p, n)
    
    # Check if entries are half-integers (within tolerance)
    all_vals = np.concatenate([a.flatten(), b.flatten(), c.flatten()])
    for val in all_vals:
        # Check if val is close to a half-integer
        nearest_half = np.round(val * 2) / 2
        if abs(val - nearest_half) > 1e-8:
            return None
    
    # Convert to fractions
    a_frac = [[Fraction(int(np.round(a[i,j] * 2)), 2) for j in range(m)] for i in range(n)]
    b_frac = [[Fraction(int(np.round(b[j,k] * 2)), 2) for k in range(p)] for j in range(m)]
    c_frac = [[Fraction(int(np.round(c[k,i] * 2)), 2) for i in range(n)] for k in range(p)]
    
    return (a_frac, b_frac, c_frac)

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, all k (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: j=0,1, k=0,1 using Strassen (7 terms)
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: j=0,1, k=2,3 using Strassen (7 terms)
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    # Now try to peel off rank-1 summands to reduce rank
    # Build the target tensor
    T_target = np.ones((n, m, p))
    
    # Try peeling
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_rank = rank
    
    start_time = time.time()
    deadline = 130  # Leave 20s margin
    
    # Try to find if we can reduce rank by finding linear dependencies
    # or by peeling off rank-1 components
    
    # First, check if the current factor matrices have linear dependencies
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check rank of U
    rank_U = np.linalg.matrix_rank(U_arr, tol=1e-10)
    rank_V = np.linalg.matrix_rank(V_arr, tol=1e-10)
    rank_W = np.linalg.matrix_rank(W_arr, tol=1e-10)
    
    # If any factor has rank less than the stated rank, we can reduce
    # But this doesn't directly help since the decomposition is valid
    
    # Try a different approach: search for lower rank decompositions
    # by trying to combine terms
    
    # For now, return the known rank-28 solution
    # Future work: implement actual peeling algorithm
    
    return {"U": best_U, "V": best_V, "W": best_W}
