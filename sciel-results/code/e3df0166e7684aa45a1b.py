"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the factor matrices exactly decompose the matrix multiplication tensor.
    For each (i,j,k), the tensor entry T[i*m+j, j*p+k, k*n+i] should equal
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    Returns True if all entries match exactly (using Fraction arithmetic).
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[u_idx][c]) * Fraction(V[v_idx][c]) * Fraction(W[w_idx][c])
                if total != 1:
                    return False
    return True

def _try_low_rank_search(n, m, p, time_limit=130):
    """
    Attempt to find a lower-rank exact decomposition using iterative refinement.
    Starts from random initializations and tries to converge to an exact solution.
    Returns a valid decomposition dict if found, else None.
    """
    start_time = time.time()
    best_solution = None
    best_rank = n * m * p
    
    # Lower bound for matrix multiplication rank is max(n*m, m*p, p*n) for rectangular cases
    # For 4x4 * 4x9: max(16, 36, 36) = 36
    min_possible_rank = max(n * m, m * p, p * n)
    
    # We will try ranks from min_possible_rank up to best_rank - 1
    for target_rank in range(min_possible_rank, best_rank):
        if time.time() - start_time > time_limit - 15:
            break
        
        # Multiple random restarts for each rank
        # Allocate time per rank based on remaining time
        remaining_time = time_limit - (time.time() - start_time)
        max_restarts = max(1, int(remaining_time / 5))  # ~5s per restart
        max_restarts = min(max_restarts, 20)  # Cap restarts
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Initialize with random small values
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Random initialization - try different scales
            scale = 0.1 * (1 + restart * 0.5)
            U[:, :] = np.random.randn(n * m, target_rank) * scale
            V[:, :] = np.random.randn(m * p, target_rank) * scale
            W[:, :] = np.random.randn(p * n, target_rank) * scale
            
            # Iterative refinement: alternate updating each factor
            converged = False
            max_iterations = 100
            for iteration in range(max_iterations):
                if time.time() - start_time > time_limit - 10:
                    break
                
                # Update U: for each (i,j) row, solve linear system
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        # Build A matrix: A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            v_idx = j * p + k
                            w_idx = k * n + i
                            A[k, :] = V[v_idx, :] * W[w_idx, :]
                        b = np.ones(p)
                        
                        try:
                            U[u_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each (j,k) row, solve linear system
                for j in range(m):
                    for k in range(p):
                        v_idx = j * p + k
                        # Build A matrix: A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            u_idx = i * m + j
                            w_idx = k * n + i
                            A[i, :] = U[u_idx, :] * W[w_idx, :]
                        b = np.ones(n)
                        
                        try:
                            V[v_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each (k,i) row, solve linear system
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        # Build A matrix: A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            A[j, :] = U[u_idx, :] * V[v_idx, :]
                        b = np.ones(m)
                        
                        try:
                            W[w_idx, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = 0
                            for c in range(target_rank):
                                val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Convert to exact fractions if possible
                # Try with increasing precision
                for denom in [1000, 10000, 100000]:
                    U_frac = [[Fraction(U[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(n * m)]
                    V_frac = [[Fraction(V[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(m * p)]
                    W_frac = [[Fraction(W[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(p * n)]
                    
                    # Verify exactness
                    if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                        best_solution = {"U": U_frac, "V": V_frac, "W": W_frac}
                        best_rank = target_rank
                        break
                if best_solution is not None and best_rank == target_rank:
                    break
    
    return best_solution

# EVOLVE-BLOCK-END
