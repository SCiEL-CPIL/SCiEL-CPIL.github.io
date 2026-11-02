"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for arr in [U, V, W]:
        if len(arr) == 0:
            return False
        if len(arr[0]) != rank:
            return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Verify exactness with Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != Fraction(1):
                    return False
    return True

def _schoolbook(n, m, p):
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

def _constructive_decomposition(n, m, p):
    """
    Try to find a lower-rank decomposition using a structured algebraic approach.
    
    Key idea: The matrix multiplication tensor T[i,j,k] = 1 (for all i,j,k in the 
    appropriate ranges) can be decomposed as sum_c U[i,j,c] * V[j,k,c] * W[i,k,c].
    
    We use a "block" approach: partition the m dimension into groups and use 
    polynomial/interpolation tricks to reduce the number of multiplications.
    
    For n=3, m=4, p=7:
    - Schoolbook rank = 84
    - We try to find a rank < 84 decomposition using structured patterns.
    
    Strategy: Use the fact that for fixed (i,k), we need to compute a dot product 
    of length m. We can use the "middle product" or "Karatsuba-like" tricks.
    
    Actually, let's try a more direct approach: random initialization + 
    alternating least squares / Newton's method on the nonlinear system.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for verification and safety
    
    nm, mp, pn = n * m, m * p, p * n
    school = _schoolbook(n, m, p)
    best_result = school
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Start from reasonable lower bounds
    # For matrix multiplication, known lower bounds suggest rank >= n*p + m*p + n*m - something
    # But let's try various ranks
    
    candidate_ranks = list(range(50, 84))
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        # Try multiple random initializations
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            np.random.seed(attempt * 1000 + target_rank)
            U_try = np.random.randn(nm, target_rank) * 0.1
            V_try = np.random.randn(mp, target_rank) * 0.1
            W_try = np.random.randn(pn, target_rank) * 0.1
            
            # Use alternating optimization:
            # Fix V and W, solve for U (linear system)
            # Fix U and W, solve for V (linear system)  
            # Fix U and V, solve for W (linear system)
            
            success = False
            for iter_num in range(200):
                if time.time() > deadline:
                    break
                
                # Compute current reconstruction error
                # S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
                # We want S[i,j,k] = 1 for all valid (i,j,k)
                
                # Instead of computing full tensor, check error at target points
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U_try[i*m+j, c] * V_try[j*p+k, c] * W_try[k*n+i, c]
                            error += (s - 1.0) ** 2
                
                if error < 1e-12:
                    success = True
                    break
                
                # Update U: fix V and W, solve linear system for U
                # For each (i,j), we need sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = 1 for all k
                # This is a system of p equations in target_rank unknowns for each (i,j)
                # But it's coupled across (i,j) through the same c index
                
                # Let's use a gradient-based update instead
                lr = 0.05 / (1 + iter_num * 0.05)
                
                # Compute gradients
                grad_U = np.zeros_like(U_try)
                grad_V = np.zeros_like(V_try)
                grad_W = np.zeros_like(W_try)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U_try[i*m+j, c] * V_try[j*p+k, c] * W_try[k*n+i, c]
                            diff = s - 1.0
                            for c in range(target_rank):
                                grad_U[i*m+j, c] += diff * V_try[j*p+k, c] * W_try[k*n+i, c]
                                grad_V[j*p+k, c] += diff * U_try[i*m+j, c] * W_try[k*n+i, c]
                                grad_W[k*n+i, c] += diff * U_try[i*m+j, c] * V_try[j*p+k, c]
                
                U_try -= lr * grad_U
                V_try -= lr * grad_V
                W_try -= lr * grad_W
            
            if success:
                # Verify with exact arithmetic
                U_list = [[float(U_try[i][c]) for c in range(target_rank)] for i in range(nm)]
                V_list = [[float(V_try[j][c]) for c in range(target_rank)] for j in range(mp)]
                W_list = [[float(W_try[k][c]) for c in range(target_rank)] for k in range(pn)]
                
                if _verify_exact(U_list, V_list, W_list, n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
        
        if best_result is not school:
            break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    
    # Try the constructive approach first
    result = _constructive_decomposition(n, m, p)
    
    # Verify the result
    if _verify(result['U'], result['V'], result['W'], n, m, p):
        return result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
