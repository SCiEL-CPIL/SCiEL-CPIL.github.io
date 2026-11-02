"""
Matrix multiplication 2x3 by 3x3 - rank-minimal exact tensor decomposition.

The task is to find the minimum rank R such that we can write the matrix
multiplication tensor as a sum of R rank-1 tensors:
    T[i,j,k] = sum_{c=0}^{R-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

For a 2x3 by 3x3 matrix multiply:
- n=2 (rows of A, columns of C)
- m=3 (columns of A, rows of B)  
- p=3 (columns of B, columns of C)

The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k] for i in [0,1], j in [0,1,2], k in [0,1,2]

The tensor T has entries T[i,j,k] = 1 for all valid (i,j,k) since each product A[i,j]*B[j,k]
contributes to C[i,k].

Known bounds for rank of matrix multiplication tensor:
- Lower bound: max(n*m, m*p, p*n) = max(6, 9, 9) = 9
- Schoolbook: n*m*p = 18

For 2x3 by 3x3, we search for decompositions starting from rank 9 upward.
We use a systematic search approach with random initialization and gradient-based refinement,
plus a final exact verification step.

Given the 150s time budget, we need to be efficient. We'll:
1. Try known constructions if any exist for small cases
2. Use randomized search with local optimization
3. Verify exactness with rational arithmetic
"""

import time
import random
import math
from fractions import Fraction

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _verify_decomposition(U, V, W, rank, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    
    For each output entry C[i,k], we need:
        sum_j A[i,j] * B[j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    More precisely, the tensor entry T[i,j,k] (coefficient of A[i,j]*B[j,k] in C[i,k])
    must equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    Since T[i,j,k] = 1 for all valid (i,j,k), we check:
        for all i in [0,n), j in [0,m), k in [0,p):
            sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(U[i][c]) for c in range(rank)] for i in range(n*m)]
    V_frac = [[Fraction(V[j][c]) for c in range(rank)] for j in range(m*p)]
    W_frac = [[Fraction(W[k][c]) for c in range(rank)] for k in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U_frac[i*m + j][c] * V_frac[j*p + k][c] * W_frac[k*n + i][c]
                if s != 1:
                    return False
    return True

def _verify_decomposition_float(U, V, W, rank, n, m, p, tol=1e-8):
    """Float verification (faster, for initial screening)."""
    import numpy as np
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rational_polish(U, V, W, rank, n, m, p, max_iter=1000):
    """
    Try to find exact rational solutions near the given float solution.
    Returns (U, V, W) with rational (Fraction) entries if found, else None.
    """
    # Round to nearby rationals with small denominators
    def round_to_fraction(x, max_denom=20):
        """Find the best rational approximation with denominator <= max_denom."""
        if x == 0:
            return Fraction(0)
        best = None
        best_err = float('inf')
        for d in range(1, max_denom + 1):
            n_val = round(x * d)
            frac = Fraction(n_val, d)
            err = abs(float(frac) - x)
            if err < best_err:
                best_err = err
                best = frac
        return best
    
    U_frac = [[round_to_fraction(U[i][c]) for c in range(rank)] for i in range(n*m)]
    V_frac = [[round_to_fraction(V[j][c]) for c in range(rank)] for j in range(m*p)]
    W_frac = [[round_to_fraction(W[k][c]) for c in range(rank)] for k in range(p*n)]
    
    if _verify_decomposition(U_frac, V_frac, W_frac, rank, n, m, p):
        return U_frac, V_frac, W_frac
    
    # Try larger denominators
    for max_denom in [50, 100, 200]:
        U_frac = [[round_to_fraction(U[i][c], max_denom) for c in range(rank)] for i in range(n*m)]
        V_frac = [[round_to_fraction(V[j][c], max_denom) for c in range(rank)] for j in range(m*p)]
        W_frac = [[round_to_fraction(W[k][c], max_denom) for c in range(rank)] for k in range(p*n)]
        
        if _verify_decomposition(U_frac, V_frac, W_frac, rank, n, m, p):
            return U_frac, V_frac, W_frac
        
        if _time_remaining() < 10:
            break
    
    return None

def _random_decomposition(rank, n, m, p, scale=1.0):
    """Generate a random decomposition."""
    U = [[random.uniform(-scale, scale) for _ in range(rank)] for _ in range(n*m)]
    V = [[random.uniform(-scale, scale) for _ in range(rank)] for _ in range(m*p)]
    W = [[random.uniform(-scale, scale) for _ in range(rank)] for _ in range(p*n)]
    return U, V, W

def _compute_residuals(U, V, W, rank, n, m, p):
    """Compute residual: for each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1."""
    residuals = {}
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                res = s - 1.0
                residuals[(i,j,k)] = res
                max_res = max(max_res, abs(res))
    return residuals, max_res

def _gradient_descent(U, V, W, rank, n, m, p, max_iter=5000, lr=0.01):
    """
    Gradient descent to minimize sum of squared residuals.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    for it in range(max_iter):
        if _time_remaining() < 5:
            break
        
        # Compute residuals and gradients
        # f = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        # df/dU[a,c] = 2 * sum_{j,k: i*m+j=a} res[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
        
        # We'll compute gradients efficiently
        grad_U = [[0.0]*rank for _ in range(n*m)]
        grad_V = [[0.0]*rank for _ in range(m*p)]
        grad_W = [[0.0]*rank for _ in range(p*n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    res = s - 1.0
                    
                    for c in range(rank):
                        # dU
                        grad_U[i*m+j][c] += 2.0 * res * V[j*p+k][c] * W[k*n+i][c]
                        # dV
                        grad_V[j*p+k][c] += 2.0 * res * U[i*m+j][c] * W[k*n+i][c]
                        # dW
                        grad_W[k*n+i][c] += 2.0 * res * U[i*m+j][c] * V[j*p+k][c]
        
        # Update with adaptive learning rate
        # Scale learning rate based on gradient magnitude
        grad_norm = 0.0
        for a in range(n*m):
            for c in range(rank):
                grad_norm += grad_U[a][c]**2
        for a in range(m*p):
            for c in range(rank):
                grad_norm += grad_V[a][c]**2
        for a in range(p*n):
            for c in range(rank):
                grad_norm += grad_W[a][c]**2
        grad_norm = math.sqrt(grad_norm) if grad_norm > 0 else 1.0
        
        # Adaptive lr
        eff_lr = lr * 100.0 / (grad_norm + 1.0)
        eff_lr = min(eff_lr, 0.1)
        
        for a in range(n*m):
            for c in range(rank):
                U[a][c] -= eff_lr * grad_U[a][c]
        for a in range(m*p):
            for c in range(rank):
                V[a][c] -= eff_lr * grad_V[a][c]
        for a in range(p*n):
            for c in range(rank):
                W[a][c] -= eff_lr * grad_W[a][c]
        
        # Check convergence
        if it % 100 == 0:
            _, max_res = _compute_residuals(U, V, W, rank, n, m, p)
            if max_res < 1e-10:
                break
    
    return U, V, W

def _try_find_decomposition(rank, n, m, p, time_budget=20.0):
    """
    Try to find a decomposition of given rank using multiple strategies.
    Returns (U, V, W) with exact values if found, else None.
    """
    start_time = time.time()
    
    # Strategy 1: Try structured solutions
    # For rank = m*p = 9, one approach:
    # Use U[i*m+j, c] where c indexes (j,k) pairs
    # But this gives rank 9 only if we can factor appropriately
    
    # Let's try a more systematic approach:
    # Initialize with structured patterns
    
    # Pattern 1: U depends on i and j, V on j and k, W on k and i
    # Try: U[i*m+j, c] = a_i * b_j for some structured a, b
    # This is hard to get right, so let's use random init + GD
    
    for attempt in range(200):
        if time.time() - start_time > time_budget:
            break
        if _time_remaining() < 10:
            break
        
        # Random initialization with various scales
        scale = random.choice([0.1, 0.5, 1.0, 2.0])
        U, V, W = _random_decomposition(rank, n, m, p, scale)
        
        # Gradient descent
        U, V, W = _gradient_descent(U, V, W, rank, n, m, p, max_iter=2000, lr=0.05)
        
        # Check if we're close
        _, max_res = _compute_residuals(U, V, W, rank, n, m, p)
        
        if max_res < 1e-6:
            # Try rational polish
            result = _try_rational_polish(U, V, W, rank, n, m, p)
            if result is not None:
                return result
        
        # Try local perturbations if close
        if max_res < 1e-3:
            for _ in range(100):
                if _time_remaining() < 5:
                    break
                # Perturb one entry
                matrix = random.choice(['U', 'V', 'W'])
                if matrix == 'U':
                    idx = random.randint(0, n*m-1)
                    col = random.randint(0, rank-1)
                    U[idx][col] += random.uniform(-0.1, 0.1)
                elif matrix == 'V':
                    idx = random.randint(0, m*p-1)
                    col = random.randint(0, rank-1)
                    V[idx][col] += random.uniform(-0.1, 0.1)
                else:
                    idx = random.randint(0, p*n-1)
                    col = random.randint(0, rank-1)
                    W[idx][col] += random.uniform(-0.1, 0.1)
                
                # Quick GD
                U, V, W = _gradient_descent(U, V, W, rank, n, m, p, max_iter=200, lr=0.02)
                _, max_res = _compute_residuals(U, V, W, rank, n, m, p)
                
                if max_res < 1e-6:
                    result = _try_rational_polish(U, V, W, rank, n, m, p)
                    if result is not None:
                        return result
    
    return None

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = 2, 3, 3
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 9, 9) = 9
    # Schoolbook: 18
    # Try ranks from lower bound to schoolbook
    
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    for rank in range(9, best_rank):
        if _time_remaining() < 15:
            break
        
        # Allocate time based on remaining budget
        time_budget = min(20.0, _time_remaining() / max(1, best_rank - rank - 1))
        time_budget = max(5.0, time_budget)
        
        result = _try_find_decomposition(rank, n, m, p, time_budget)
        if result is not None:
            U_frac, V_frac, W_frac = result
            # Convert Fractions to their float representations for the output
            # But we need to output the exact values
            U_out = [[float(U_frac[i][c]) for c in range(rank)] for i in range(n*m)]
            V_out = [[float(V_frac[j][c]) for c in range(rank)] for j in range(m*p)]
            W_out = [[float(W_frac[k][c]) for c in range(rank)] for k in range(p*n)]
            
            # Verify with exact arithmetic one more time
            if _verify_decomposition(U_frac, V_frac, W_frac, rank, n, m, p):
                best_result = {"U": U_out, "V": V_out, "W": W_out}
                best_rank = rank
                break  # Can't do better than this rank
    
    return best_result

# EVOLVE-BLOCK-END
