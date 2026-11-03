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

def _compute_tensor_error(U, V, W, n, m, p, target_rank):
    """Compute max error of tensor decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(target_rank):
                    val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _try_low_rank_search(n, m, p, time_limit=135):
    """
    Attempt to find a lower-rank exact decomposition using iterative refinement.
    Starts from random initializations and tries to converge to an exact solution.
    Returns a valid decomposition dict if found, else None.
    """
    start_time = time.time()
    best_solution = None
    best_rank = n * m * p
    
    # Lower bound for matrix multiplication rank
    min_possible_rank = max(n * m, m * p, p * n)
    
    # Try ranks from min_possible_rank up to best_rank - 1
    for target_rank in range(min_possible_rank, best_rank):
        if time.time() - start_time > time_limit - 20:
            break
        
        # Allocate time per rank
        remaining_time = time_limit - (time.time() - start_time)
        
        # Determine number of restarts based on remaining time
        # Each restart gets ~2-3 seconds
        max_restarts = max(1, int(remaining_time / 3))
        max_restarts = min(max_restarts, 50)  # Cap restarts
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 15:
                break
            
            # Initialize with different strategies
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Different initialization strategies
            if restart % 3 == 0:
                # Random Gaussian
                scale = 0.1 * (1 + restart * 0.2)
                U[:, :] = np.random.randn(n * m, target_rank) * scale
                V[:, :] = np.random.randn(m * p, target_rank) * scale
                W[:, :] = np.random.randn(p * n, target_rank) * scale
            elif restart % 3 == 1:
                # Sparse random
                U[:, :] = np.random.choice([-1, 0, 1], size=(n * m, target_rank)) * 0.5
                V[:, :] = np.random.choice([-1, 0, 1], size=(m * p, target_rank)) * 0.5
                W[:, :] = np.random.choice([-1, 0, 1], size=(p * n, target_rank)) * 0.5
            else:
                # Structured initialization with some zeros
                U[:, :] = np.random.randn(n * m, target_rank) * 0.2
                V[:, :] = np.random.randn(m * p, target_rank) * 0.2
                W[:, :] = np.random.randn(p * n, target_rank) * 0.2
                # Add some structure
                for c in range(target_rank):
                    if c % 2 == 0:
                        U[:, c] *= 1.5
                        V[:, c] *= 1.5
                        W[:, c] *= 0.5
                    else:
                        U[:, c] *= 0.5
                        V[:, c] *= 0.5
                        W[:, c] *= 1.5
            
            # Iterative refinement with adaptive iterations
            converged = False
            max_iterations = 150
            prev_error = float('inf')
            stagnation_counter = 0
            
            for iteration in range(max_iterations):
                if time.time() - start_time > time_limit - 12:
                    break
                
                # Update U: for each (i,j) row, solve linear system
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
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
                current_error = _compute_tensor_error(U, V, W, n, m, p, target_rank)
                
                if current_error < 1e-12:
                    converged = True
                    break
                
                # Detect stagnation
                if current_error > prev_error * 0.999:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                
                if stagnation_counter > 10:
                    # Try a small perturbation to escape local minimum
                    U += np.random.randn(*U.shape) * 0.01
                    V += np.random.randn(*V.shape) * 0.01
                    W += np.random.randn(*W.shape) * 0.01
                    stagnation_counter = 0
                
                prev_error = current_error
            
            if converged:
                # Convert to exact fractions with multiple precision levels
                found_exact = False
                for denom in [100, 1000, 10000, 100000, 1000000]:
                    U_frac = [[Fraction(U[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(n * m)]
                    V_frac = [[Fraction(V[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(m * p)]
                    W_frac = [[Fraction(W[i, c]).limit_denominator(denom) for c in range(target_rank)] for i in range(p * n)]
                    
                    if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                        best_solution = {"U": U_frac, "V": V_frac, "W": W_frac}
                        best_rank = target_rank
                        found_exact = True
                        break
                
                if found_exact:
                    break
    
    return best_solution

# EVOLVE-BLOCK-END
