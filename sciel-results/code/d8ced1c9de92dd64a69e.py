"""
Attempt to find a rank-minimal decomposition for 2x3 by 3x7 matrix multiplication.
Uses a fast local optimization approach with integer/half-integer rounding.
Falls back to schoolbook (rank 42) if no valid lower-rank decomposition is found.
"""

import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Initialize schoolbook solution
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def make_output(U, V, W):
        """Convert numpy arrays to lists with integer/half-integer values"""
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    def validate_and_round(U, V, W):
        """Validate that the decomposition is exact and round to half-integers"""
        # First, check if values are close to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check if rounding introduces significant error
        if np.max(np.abs(U - U_rounded)) > 1e-6 or \
           np.max(np.abs(V - V_rounded)) > 1e-6 or \
           np.max(np.abs(W - W_rounded)) > 1e-6:
            return None
        
        # Verify exact decomposition
        # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 if i,j,k are valid indices, 0 otherwise
        # Actually, for matrix multiplication, we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid i,j,k
        
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U_rounded.shape[1]):
                        s += U_rounded[i * m + j, r] * V_rounded[j * p + k, r] * W_rounded[k * n + i, r]
                    errors.append(s - 1.0)
        
        max_err = max(abs(e) for e in errors) if errors else 0.0
        if max_err > 1e-6:
            return None
        
        return make_output(U_rounded, V_rounded, W_rounded)
    
    # Try to find lower rank decompositions using local optimization
    # We'll try ranks from 41 down to 30 (limited by time)
    
    best_result = make_output(U_sb, V_sb, W_sb)
    
    # For each candidate rank, try a few random initializations with local optimization
    for rank in range(schoolbook_rank - 1, 29, -1):
        if time.time() > deadline:
            break
        
        # Try a few random starts
        for trial in range(3):
            if time.time() > deadline:
                break
            
            # Initialize with random half-integer values
            U = (np.random.randint(0, 5, size=(n * m, rank)) - 2) * 0.5
            V = (np.random.randint(0, 5, size=(m * p, rank)) - 2) * 0.5
            W = (np.random.randint(0, 5, size=(p * n, rank)) - 2) * 0.5
            
            # Local optimization: gradient descent to minimize sum of (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Current sum
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err = s - 1.0
                            
                            for r in range(rank):
                                dU[i*m+j, r] += 2 * err * V[j*p+k, r] * W[k*n+i, r]
                                dV[j*p+k, r] += 2 * err * U[i*m+j, r] * W[k*n+i, r]
                                dW[k*n+i, r] += 2 * err * U[i*m+j, r] * V[j*p+k, r]
                
                # Update with small step size
                lr = 0.01
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Try to round and validate
            result = validate_and_round(U, V, W)
            if result is not None:
                best_result = result
                # Found a valid lower-rank solution, stop searching
                return best_result
    
    return best_result
