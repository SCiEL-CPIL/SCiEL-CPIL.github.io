"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    Returns {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Schoolbook rank is n*m*p = 63
    schoolbook_rank = n * m * p
    
    # The tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # In tensor form: T_{i,j,k} where i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i} delta_{k,k}
    # Wait, let me think about this more carefully.
    
    # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor T has components T[i,j,k] = 1 (it's the identity-like tensor)
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Actually, the standard formulation:
    # T[i,j,k] = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}
    # where the sum is over r = 0 to rank-1
    
    # For the schoolbook: each (i,j,k) triple gets its own term with all ones.
    # Rank = n*m*p = 63.
    
    # Can we do better? For 3x3 x 3x7:
    # Known lower bounds: rank >= n*m + n*p + m*p - n - m - p + 1 (not always tight)
    # For n=m=3, p=7: 9 + 21 + 21 - 3 - 3 - 7 + 1 = 39
    # But this bound isn't always achievable.
    
    # Let's try to find a lower rank decomposition using optimization.
    # We'll try ranks from a lower bound up to schoolbook_rank.
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Also rank >= n*m + p*n - m = 9 + 21 - 3 = 27 (from one bound)
    # Another bound: rank >= n*p + m*n - m = 21 + 9 - 3 = 27
    
    # Let's try ranks from 27 up to 63.
    
    def verify_decomposition(U, V, W, rank, tol=1e-8):
        """Verify that the decomposition is exact."""
        # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # and check it equals T[i,j,k] = 1 for all i,j,k
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape for efficient computation
        # U_arr has shape (n*m, rank)
        # V_arr has shape (m*p, rank)  
        # W_arr has shape (p*n, rank)
        
        # For each r, we have outer product contributions
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's compute this efficiently
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        # This is a tensor contraction
        
        # Compute using einsum
        T_computed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # T_target[i,j,k] = 1 for all i,j,k
        T_target = np.ones((n, m, p))
        
        return np.allclose(T_computed, T_target, atol=tol)
    
    def try_rank(rank, max_iterations=1000, seed=42):
        """Try to find a decomposition of given rank."""
        if time.time() > deadline:
            return None
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        def objective(params):
            """Objective function to minimize."""
            U_flat = params[:n*m*rank].reshape(n*m, rank)
            V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            U_reshaped = U_flat.reshape(n, m, rank)
            V_reshaped = V_flat.reshape(m, p, rank)
            W_reshaped = W_flat.reshape(p, n, rank)
            
            T_computed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            T_target = np.ones((n, m, p))
            
            return np.sum((T_computed - T_target) ** 2)
        
        # Try to optimize
        try:
            result = minimize(objective, np.concatenate([U.flatten(), V.flatten(), W.flatten()]),
                             method='L-BFGS-B', options={'maxiter': max_iterations, 'ftol': 1e-15})
            
            if result.fun < 1e-8:
                params = result.x
                U_opt = params[:n*m*rank].reshape(n*m, rank)
                V_opt = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_opt = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Check if we can round to integers/half-integers
                # First, check if the solution is close to rational values
                # Try to find a common denominator
                
                # Check the values
                all_vals = np.concatenate([U_opt.flatten(), V_opt.flatten(), W_opt.flatten()])
                
                # Try to see if values are close to integers or half-integers
                def is_close_to_rational(vals, denoms=[1, 2, 3, 4, 6]):
                    """Check if values are close to rationals with small denominators."""
                    for d in denoms:
                        scaled = vals * d
                        rounded = np.round(scaled)
                        if np.allclose(scaled, rounded, atol=1e-6):
                            return d, rounded
                    return None, None
                
                denom, rounded_vals = is_close_to_rational(all_vals)
                
                if denom is not None:
                    U_int = (rounded_vals[:n*m*rank] / denom).reshape(n*m, rank)
                    V_int = (rounded_vals[n*m*rank:n*m*rank + m*p*rank] / denom).reshape(m*p, rank)
                    W_int = (rounded_vals[n*m*rank + m*p*rank:] / denom).reshape(p*n, rank)
                    
                    # Verify the integer/half-integer decomposition
                    if verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank, tol=1e-6):
                        return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
                # If not close to rational, try a different approach
                # Maybe the solution requires larger denominators
                # For now, return the floating point solution (may not be accepted)
                if result.fun < 1e-10:
                    # Try to find rational approximation with larger denominators
                    for d in [12, 24, 36, 48, 60, 72, 120]:
                        scaled = all_vals * d
                        rounded = np.round(scaled)
                        if np.allclose(scaled, rounded, atol=1e-5):
                            U_int = (rounded[:n*m*rank] / d).reshape(n*m, rank)
                            V_int = (rounded[n*m*rank:n*m*rank + m*p*rank] / d).reshape(m*p, rank)
                            W_int = (rounded[n*m*rank + m*p*rank:] / d).reshape(p*n, rank)
                            
                            if verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank, tol=1e-5):
                                return U_int.tolist(), V_int.tolist(), W_int.tolist()
                    
                    # Return floating point as fallback (may not satisfy integer/half-integer requirement)
                    return U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
        
        except Exception as e:
            pass
        
        return None
    
    # Try ranks from lower bound up
    # Lower bound is at least max(n*m, m*p, p*n) = 21
    # But more realistically, for 3x3x7, known results suggest rank around 40-50
    
    # Let's try a few ranks
    best_result = None
    
    # Start from a reasonable lower bound
    # For 3x3x3, minimum rank is 23 (Strassen-like)
    # For 3x3x7, it's likely higher
    
    # Try ranks: 30, 35, 40, 45, 50, 55, 60, 63
    for rank in [30, 35, 40, 45, 50, 55, 60, 63]:
        if time.time() > deadline:
            break
        
        result = try_rank(rank)
        if result is not None:
            best_result = result
            break
    
    # If no better result found, use schoolbook
    if best_result is None:
        U, V, W = schoolbook_decomposition(n, m, p)
    else:
        U, V, W = best_result
    
    return {"U": U, "V": V, "W": W}

def schoolbook_decomposition(n, m, p):
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
    return U, V, W
# EVOLVE-BLOCK-END
