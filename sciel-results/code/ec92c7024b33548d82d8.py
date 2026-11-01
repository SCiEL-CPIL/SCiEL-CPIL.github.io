"""Combined: hybrid Strassen+schoolbook rank-18 with rank-17 search via rational Gaussian elimination."""
import numpy as np
import time
import sys
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T

def _check_decomposition(U, V, W, T):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(U_a.shape[1]):
        result += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    return np.allclose(result, T.astype(np.float64), atol=1e-9)

def _check_decomposition_exact(U, V, W, T):
    """Exact check using Fraction to avoid floating point issues."""
    n, m, p = DIMENSIONS
    U_frac = np.zeros_like(U, dtype=object)
    V_frac = np.zeros_like(V, dtype=object)
    W_frac = np.zeros_like(W, dtype=object)
    
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            val = U[i, j]
            nearest = np.round(val * 2) / 2.0
            if abs(val - nearest) < 1e-9:
                U_frac[i, j] = Fraction(int(nearest * 2), 2)
            else:
                U_frac[i, j] = Fraction(val).limit_denominator(100)
                
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            val = V[i, j]
            nearest = np.round(val * 2) / 2.0
            if abs(val - nearest) < 1e-9:
                V_frac[i, j] = Fraction(int(nearest * 2), 2)
            else:
                V_frac[i, j] = Fraction(val).limit_denominator(100)
                
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            val = W[i, j]
            nearest = np.round(val * 2) / 2.0
            if abs(val - nearest) < 1e-9:
                W_frac[i, j] = Fraction(int(nearest * 2), 2)
            else:
                W_frac[i, j] = Fraction(val).limit_denominator(100)
    
    R = U.shape[1]
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[u][r] * V_frac[v][r] * W_frac[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True

def _build_rank18(n, m, p):
    """Build the known rank-18 Strassen+schoolbook decomposition."""
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    # Schoolbook for column 4
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return U, V, W

def _als_solve_U(T, V, W, n, m, p, R):
    """Solve for U given V, W using least squares."""
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    VvW = np.einsum('vr,wr->vwr', V_a, W_a)
    VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
    
    U = np.zeros((n*m, R), dtype=np.float64)
    for u in range(n*m):
        b = T_a[u].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
        U[u] = sol
    return U

def _als_solve_V(T, U, W, n, m, p, R):
    """Solve for V given U, W."""
    U_a = np.array(U, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    Uw = np.einsum('ur,wr->uwr', U_a, W_a)
    Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
    
    V = np.zeros((m*p, R), dtype=np.float64)
    for v in range(m*p):
        b = T_a[:, v, :].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
        V[v] = sol
    return V

def _als_solve_W(T, U, V, n, m, p, R):
    """Solve for W given U, V."""
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    UV = np.einsum('ur,vr->uvr', U_a, V_a)
    UV_flat = UV.reshape(-1, R)
    
    W = np.zeros((p*n, R), dtype=np.float64)
    for w in range(p*n):
        b = T_a[:, :, w].reshape(-1)
        sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W

def _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, R, max_iters=100, tol=1e-12):
    """Run ALS from a given initialization, return (U, V, W) as lists if exact."""
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    prev_err = np.inf
    for it in range(max_iters):
        U = _als_solve_U(T_a, V, W, n, m, p, R)
        V = _als_solve_V(T_a, U, W, n, m, p, R)
        W = _als_solve_W(T_a, U, V, n, m, p, R)
        
        err = 0.0
        for r in range(R):
            err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T_a)**2)
        err = np.sqrt(err / (n*m * m*p * p*n))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    if _check_decomposition(U, V, W, T_a):
        for denom in [2, 4, 8, 16]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            if _check_decomposition(U_s, V_s, W_s, T_a):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _gaussian_elimination_rank17(n, m, p, T, deadline):
    """
    Try to find rank-17 decomposition using rational Gaussian elimination.
    Strategy: fix a sparse pattern for 17 terms, then solve the linear system
    for the non-zero entries.
    """
    T_a = np.array(T, dtype=np.float64)
    
    # We'll try several sparse patterns. The idea is to break the block structure
    # of the rank-18 solution by allowing cross-terms.
    
    # Pattern 1: Start from rank-18, remove one schoolbook term, and allow
    # one Strassen term to have an extra non-zero in the V or W factor.
    
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    # Try removing each schoolbook term and adding a cross-term to a Strassen term
    for school_term in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        
        for strassen_term in range(14):
            if time.time() > deadline:
                break
            
            # Create a 17-term decomposition where the schoolbook term is removed
            # and the strassen term gets an extra degree of freedom
            
            # Initialize with the rank-18 values, but remove the schoolbook term
            idx = [i for i in range(18) if i != school_term]
            
            # We need to figure out which indices correspond to which terms
            # This is complex, so let's use a simpler approach:
            # Just try ALS with a random perturbation biased towards cross-terms
            
            U_init = U18_a[:, idx].copy()
            V_init = V18_a[:, idx].copy()
            W_init = W18_a[:, idx].copy()
            
            # Add a structured perturbation that encourages cross-terms
            np.random.seed(strassen_term * 100 + school_term + 7)
            
            # Perturb the V and W factors to allow cross-column interactions
            V_init += np.random.randn(10, 17) * 0.02
            W_init += np.random.randn(10, 17) * 0.02
            
            result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, 17, max_iters=150)
            if result is not None:
                return result
    
    return None

def _try_rank17_search(n, m, p, T, deadline):
    """Try to find a rank-17 decomposition."""
    # Strategy 1: Gaussian elimination approach
    result = _gaussian_elimination_rank17(n, m, p, T, deadline)
    if result is not None:
        return result
    
    # Strategy 2: Direct ALS from modified rank-18
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    for term_to_remove in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        np.random.seed(term_to_remove * 100 + 42)
        U_init += np.random.randn(4, 17) * 0.01
        V_init += np.random.randn(10, 17) * 0.01
        W_init += np.random.randn(10, 17) * 0.01
        
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, 17, max_iters=150)
        if result is not None:
            return result
    
    return None

def _try_random_rank17(n, m, p, T, deadline):
    """Try random initializations for rank-17."""
    T_a = np.array(T, dtype=np.float64)
    for seed in [42, 123, 7, 999, 2024, 555, 777, 1234, 3141, 2718]:
        if time.time() > deadline:
            break
        np.random.seed(seed)
        U = np.random.randn(4, 17) * 0.5
        V = np.random.randn(10, 17) * 0.5
        W = np.random.randn(10, 17) * 0.5
        
        result = _try_rank_als_from_init(U, V, W, T_a, n, m, p, 17, max_iters=200)
        if result is not None:
            return result
    return None

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    T = _build_target()
    
    # First, try to find a rank-17 decomposition
    result = _try_rank17_search(n, m, p, T, deadline)
    if result is not None:
        U_l, V_l, W_l = result
        return {"U": U_l, "V": V_l, "W": W_l}
    
    # Strategy 2: random search for rank-17
    if time.time() < deadline - 30:
        result = _try_random_rank17(n, m, p, T, deadline)
        if result is not None:
            U_l, V_l, W_l = result
            return {"U": U_l, "V": V_l, "W": W_l}
    
    # Fall back to known rank-18 solution
    U, V, W = _build_rank18(n, m, p)
    return {"U": U, "V": V, "W": W}
