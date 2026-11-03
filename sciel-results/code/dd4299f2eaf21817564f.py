"""
Tensor decomposition for 3x3 * 3x4 matrix multiplication.
Uses a tensor-train-inspired search with exact rational recovery to find
a rank-minimal exact decomposition.
"""

import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]

# Deadline guard: evaluator kills at 150s, use 140s safety margin
DEADLINE = 140.0
START_TIME = None

def _time_left():
    if START_TIME is None:
        return DEADLINE
    return DEADLINE - (time.time() - START_TIME)

def _frac_from_float(x, max_den=10000):
    """Convert a float to a Fraction, trying small denominators first."""
    if abs(x) < 1e-12:
        return Fraction(0)
    f = Fraction(x).limit_denominator(max_den)
    # Check if it's close to the original
    if abs(float(f) - x) < 1e-9:
        return f
    # Try a few more
    f2 = Fraction(x).limit_denominator(max_den * 10)
    if abs(float(f2) - x) < 1e-9:
        return f2
    return Fraction(x).limit_denominator(max_den * 100)

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact tensor decomposition."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # The multiplication tensor T[i,j,k] = delta(i,j,k) means:
    # T = sum_c U[:,c] * V[:,c] * W[:,c] (outer product)
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix mult: (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 if this term contributes, which is always 1
    # Actually the multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # because (AB)_{ik} = sum_j A_{ij} B_{jk}
    # Wait, no. The tensor for matrix multiplication C = A*B where A is n x m, B is m x p:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k (it's the all-ones tensor in this case? No!)
    # Actually the multiplication tensor is defined as:
    # T = sum_{i,j,k} e_i^A * e_j^B * e_k^C * delta_{ij, jk} ... 
    # Let me think more carefully.
    # The bilinear map is (A, B) -> A*B.
    # A is n x m, B is m x p, C = A*B is n x p.
    # C_{ik} = sum_j A_{ij} B_{jk}
    # So the tensor T has entries T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait no, that's not right either. The tensor represents the bilinear form.
    # T(A, B) = A*B where (A*B)_{ik} = sum_j A_{ij} B_{jk}
    # As a tensor in (R^{nm}) * (R^{mp}) * (R^{np}):
    # T[A_idx, B_idx, C_idx] = 1 if the term A_{ij} * B_{jk} contributes to C_{ik}
    # A_idx = i*m + j, B_idx = j*p + k, C_idx = k*n + i
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # And all other entries are 0.
    #
    # A rank-r decomposition: T = sum_c u_c * v_c * w_c (outer products)
    # means T[a, b, c] = sum_d U[a,d] * V[b,d] * W[c,d]
    #
    # So we need: for all i,j,k: U[i*m+j, d] * V[j*p+k, d] * W[k*n+i, d] summed over d = 1
    # and for all other (a,b,c) not of that form, the sum = 0.
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute the decomposition
    T_decomp = np.zeros_like(T)
    for d in range(rank):
        T_decomp += np.outer(np.outer(U[:, d], V[:, d]), W[:, d])
    
    # Check exactness
    return np.allclose(T_decomp, T, atol=1e-9)

def _verify_exact_rational(U, V, W, n, m, p):
    """Verify using exact rational arithmetic."""
    rank = len(U[0])
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i*m+j, j*p+k, k*n+i)] = Fraction(1)
    
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                s = Fraction(0)
                for d in range(rank):
                    s += Fraction(U[a][d]) * Fraction(V[b][d]) * Fraction(W[c][d])
                expected = T.get((a, b, c), Fraction(0))
                if s != expected:
                    return False
    return True

def _try_rank_schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _search_low_rank(n, m, p, max_rank, time_budget):
    """
    Search for a low-rank exact decomposition using a tensor-train-inspired approach.
    We try ranks from low to high, using random initialization + local optimization
    with exact rational recovery.
    """
    global START_TIME
    
    best = None
    
    # Strategy: For small ranks, try to find integer/half-integer solutions
    # using a structured search inspired by tensor trains.
    
    # The multiplication tensor for n x m * m x p has a known structure.
    # For 3x3 * 3x4, let's try ranks from 10 upward.
    
    for rank in range(10, min(max_rank + 1, n * m * p)):
        if _time_left() < 5:
            break
        
        # Try multiple random initializations
        max_iters = 50
        found = False
        
        for trial in range(max_iters):
            if _time_left() < 3:
                break
            
            # Random initialization with small integers
            np.random.seed(trial * 1000 + rank)
            U = np.random.randint(-2, 3, size=(n*m, rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, rank)).astype(float)
            W = np.random.randint(-2, 3, size=(p*n, rank)).astype(float)
            
            # Local optimization: minimize ||T - sum U_c V_c W_c||^2
            # Use a simple coordinate descent / gradient step approach
            T_target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m+j, j*p+k, k*n+i] = 1.0
            
            # Precompute for efficiency
            # T_decomp[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
            # Gradient wrt U[a,d]: sum_{b,c} T_decomp_residual[a,b,c] * V[b,d]*W[c,d]
            
            for it in range(100):
                if _time_left() < 3:
                    break
                
                # Compute current tensor
                T_curr = np.zeros((n*m, m*p, p*n))
                for d in range(rank):
                    T_curr += np.einsum('a,b,c->abc', U[:, d], V[:, d], W[:, d])
                
                residual = T_target - T_curr
                err = np.sum(residual**2)
                
                if err < 1e-12:
                    # Found exact solution! Try rational recovery
                    U_frac = [[_frac_from_float(U[a, d]) for d in range(rank)] for a in range(n*m)]
                    V_frac = [[_frac_from_float(V[b, d]) for d in range(rank)] for b in range(m*p)]
                    W_frac = [[_frac_from_float(W[c, d]) for d in range(rank)] for c in range(p*n)]
                    
                    # Verify exactly
                    if _verify_exact_rational(U_frac, V_frac, W_frac, n, m, p):
                        found = True
                        best = {"U": U_frac, "V": V_frac, "W": W_frac}
                        return best
                    break
                
                # Gradient descent step
                # d/dU[a,d] = sum_{b,c} residual[a,b,c] * V[b,d] * W[c,d]
                # d/dV[b,d] = sum_{a,c} residual[a,b,c] * U[a,d] * W[c,d]
                # d/dW[c,d] = sum_{a,b} residual[a,b,c] * U[a,d] * V[b,d]
                
                lr = 0.01
                for d in range(rank):
                    # Gradient for U[:, d]
                    grad_U = np.zeros(n*m)
                    for a in range(n*m):
                        # sum_{b,c} residual[a,b,c] * V[b,d] * W[c,d]
                        s = 0.0
                        for b in range(m*p):
                            for c in range(p*n):
                                s += residual[a, b, c] * V[b, d] * W[c, d]
                        grad_U[a] = s
                    U[:, d] -= lr * grad_U
                    
                    # Gradient for V[:, d]
                    grad_V = np.zeros(m*p)
                    for b in range(m*p):
                        s = 0.0
                        for a in range(n*m):
                            for c in range(p*n):
                                s += residual[a, b, c] * U[a, d] * W[c, d]
                        grad_V[b] = s
                    V[:, d] -= lr * grad_V
                    
                    # Gradient for W[:, d]
                    grad_W = np.zeros(p*n)
                    for c in range(p*n):
                        s = 0.0
                        for a in range(n*m):
                            for b in range(m*p):
                                s += residual[a, b, c] * U[a, d] * V[b, d]
                        grad_W[c] = s
                    W[:, c] -= lr * grad_W
                
                # Snap to nearest half-integer
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
            
            # Check if we converged
            T_curr = np.zeros((n*m, m*p, p*n))
            for d in range(rank):
                T_curr += np.einsum('a,b,c->abc', U[:, d], V[:, d], W[:, d])
            if np.allclose(T_curr, T_target, atol=1e-9):
                U_frac = [[_frac_from_float(U[a, d]) for d in range(rank)] for a in range(n*m)]
                V_frac = [[_frac_from_float(V[b, d]) for d in range(rank)] for b in range(m*p)]
                W_frac = [[_frac_from_float(W[c, d]) for d in range(rank)] for c in range(p*n)]
                if _verify_exact_rational(U_frac, V_frac, W_frac, n, m, p):
                    found = True
                    best = {"U": U_frac, "V": V_frac, "W": W_frac}
                    return best
    
    return best

def _structured_search(n, m, p):
    """
    Use the known structure of matrix multiplication to search for low-rank decompositions.
    
    For C = A*B where A is n x m, B is m x p:
    C_{ik} = sum_j A_{ij} B_{jk}
    
    A rank-r decomposition means we can write:
    C_{ik} = sum_d (sum_i' A_{i'j'} * u_{i'j',d}) * (sum_j' B_{j'k'} * v_{j'k',d}) * w_{k'i',d}
    
    More precisely, T[i,j,k] = 1 iff the term exists.
    T = sum_d u_d * v_d * w_d where:
    u_d[i*m+j] = U[i*m+j, d]
    v_d[j*p+k] = V[j*p+k, d]  
    w_d[k*n+i] = W[k*n+i, d]
    
    And sum_d U[i*m+j,d] * V[j*p+k,d] * W[k*n+i,d] = 1 for all i,j,k.
    """
    global START_TIME
    
    # Try a structured approach: assume U, V, W have small integer entries
    # and search over a restricted space.
    
    # For 3x3 * 3x4: n=3, m=3, p=4
    # U is 9 x rank, V is 12 x rank, W is 12 x rank
    
    # Try ranks from 10 to 35
    for rank in range(10, 36):
        if _time_left() < 10:
            break
        
        # Use a smarter initialization based on the structure
        # Each rank-1 term should "cover" some subset of the (i,j,k) triples
        
        # Try a greedy/structured approach
        # The key insight: we need sum_d U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d] = 1
        
        # Let's try to use the fact that for fixed j, we have n*p = 12 terms
        # that all involve the same j index.
        
        # Alternative: use the known Strassen-like structure
        # For 2x2: rank 7. For 3x3: best known is 23 (Lader).
        # For 3x3 * 3x4: this is less standard.
        
        # Let's try a direct search with structured initialization
        np.random.seed(42 + rank)
        
        # Initialize with a pattern that respects the j-index structure
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Try to assign each rank-1 component to a specific j
        # With m=3, we have 3 j-values. If we use ~rank/3 components per j...
        
        # Actually, let's just do a more careful random search with better initialization
        for attempt in range(20):
            if _time_left() < 5:
                break
            
            seed = rank * 100 + attempt
            np.random.seed(seed)
            
            # Initialize with small random integers
            U = np.random.randint(-1, 2, size=(n*m, rank)).astype(float)
            V = np.random.randint(-1, 2, size=(m*p, rank)).astype(float)
            W = np.random.randint(-1, 2, size=(p*n, rank)).astype(float)
            
            # Ensure at least some non-zero entries
            for d in range(rank):
                if np.all(U[:, d] == 0):
                    U[np.random.randint(n*m), d] = 1
                if np.all(V[:, d] == 0):
                    V[np.random.randint(m*p), d] = 1
                if np.all(W[:, d] == 0):
                    W[np.random.randint(p*n), d] = 1
            
            # Local search: try flipping individual entries
            T_target = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_target[i*m+j, j*p+k, k*n+i] = 1.0
            
            best_err = float('inf')
            for it in range(200):
                if _time_left() < 3:
                    break
                
                T_curr = np.zeros((n*m, m*p, p*n))
                for d in range(rank):
                    T_curr += np.einsum('a,b,c->abc', U[:, d], V[:, d], W[:, d])
                err = np.sum((T_curr - T_target)**2)
                
                if err < 1e-10:
                    U_frac = [[_frac_from_float(U[a, d]) for d in range(rank)] for a in range(n*m)]
                    V_frac = [[_frac_from_float(V[b, d]) for d in range(rank)] for b in range(m*p)]
                    W_frac = [[_frac_from_float(W[c, d]) for d in range(rank)] for c in range(p*n)]
                    if _verify_exact_rational(U_frac, V_frac, W_frac, n, m, p):
                        return {"U": U_frac, "V": V_frac, "W": W_frac}
                    break
                
                if err < best_err:
                    best_err = err
                
                # Random perturbation: pick a random entry and change it by +-1
                if np.random.random() < 0.33:
                    a = np.random.randint(n*m)
                    d = np.random.randint(rank)
                    U[a, d] += np.random.choice([-1, 1])
                elif np.random.random() < 0.66:
                    b = np.random.randint(m*p)
                    d = np.random.randint(rank)
                    V[b, d] += np.random.choice([-1, 1])
                else:
                    c = np.random.randint(p*n)
                    d = np.random.randint(rank)
                    W[c, d] += np.random.choice([-1, 1])
                
                # Keep entries bounded
                U = np.clip(U, -3, 3)
                V = np.clip(V, -3, 3)
                W = np.clip(W, -3, 3)
    
    return None

def construct():
    global START_TIME
    START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    
    # First, always have the schoolbook as fallback
    schoolbook = _try_rank_schoolbook(n, m, p)
    
    # Try to find a better decomposition
    if _time_left() > 20:
        result = _structured_search(n, m, p)
        if result is not None:
            return result
    
    if _time_left() > 10:
        result = _search_low_rank(n, m, p, n*m*p - 1, _time_left() - 5)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return schoolbook
