"""Tensor decomposition search for 4x5 * 5x5 matrix multiplication.
Uses a structured search for lower-rank decompositions than schoolbook."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]
n, m, p = DIMENSIONS

# Precompute the standard tensor T[i,j,k] = delta (A[i,j], B[j,k], C[k,i])
# Actually the tensor is T[i,j,k] where C[k,i] = sum_j A[i,j]*B[j,k]
# We need U (n*m x r), V (m*p x r), W (p*n x r) such that
# sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is the "multiplication" term
# i.e., the (i,j,k) entry of the matmul tensor is 1 for all i,j,k (since C[k,i] = sum_j A[i,j] B[j,k])
# Wait, let me re-read the convention.
# The matmul tensor T has entries T[i,j,k] = 1 for all i,j,k in the standard decomposition.
# The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal T[i,j,k] = 1.

def construct():
    """Search for a lower-rank tensor decomposition of 4x5 * 5x5 matmul."""
    t0 = time.time()
    deadline = 130.0  # safety margin before 150s limit
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 100
    
    # Known lower bounds / results:
    # For 4x5*5x5, the rank is at most 100 (schoolbook).
    # There are known results that rank can be reduced.
    # Let's try to find a decomposition with rank < 100.
    
    # Strategy: Try to find a valid decomposition by solving for entries.
    # For a given rank r, we have 20*r + 25*r + 20*r = 65*r variables.
    # We need 100 constraints (one for each (i,j,k) tuple).
    # For r < 100, this is an underdetermined system, but the equations are cubic (trilinear).
    
    # Let's try a constructive approach:
    # Group the indices to find rank-1 terms that cover multiple (i,j,k) entries.
    
    # Actually, let's try a simple improvement: 
    # For the 4x5 * 5x5 case, we can potentially reduce rank by exploiting structure.
    
    # Let me try to verify if rank 99 is possible by solving a small system.
    # Actually, let's just try to do a numerical optimization.
    
    # For a candidate rank r, randomly initialize U, V, W and minimize
    # sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    
    # But this is non-convex. Let's try with r close to 100 first.
    
    # Try ranks from schoolbook_rank-1 downward
    best_rank = schoolbook_rank
    best_result = None
    
    # Build the constraint vectors
    # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    def try_rank(r, max_iter=5000, seed=42):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline:
            return None
        np.random.seed(seed)
        
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        lr = 0.01
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradients
            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        
                        prod = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                        diff = prod - 1.0
                        loss += diff * diff
                        
                        # d/dU[idx_u,c] = 2*diff * V[idx_v,c] * W[idx_w,c]
                        dU[idx_u] += 2 * diff * V[idx_v] * W[idx_w]
                        dV[idx_v] += 2 * diff * U[idx_u] * W[idx_w]
                        dW[idx_w] += 2 * diff * U[idx_u] * V[idx_v]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if loss < 1e-10:
                break
            if it % 500 == 0 and it > 0:
                # Adaptive learning rate
                if loss > 1.0:
                    lr = max(0.001, lr * 0.5)
                else:
                    lr = min(0.05, lr * 1.2)
        
        if time.time() > deadline:
            return None
            
        # Check if solution is valid
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-8:
            # Convert to list of lists with rounding to half-integers if possible
            U_list = np.round(U * 2) / 2  # round to nearest 0.5
            V_list = np.round(V * 2) / 2
            W_list = np.round(W * 2) / 2
            
            # Verify the rounded version
            max_err2 = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        val = np.sum(U_list[idx_u] * V_list[idx_v] * W_list[idx_w])
                        max_err2 = max(max_err2, abs(val - 1.0))
            
            if max_err2 < 1e-8:
                return {
                    "U": U_list.tolist(),
                    "V": V_list.tolist(),
                    "W": W_list.tolist()
                }
            else:
                # Return unrounded if it works
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        return None
    
    # Try ranks from 99 down to 95 (limited search)
    for r in range(schoolbook_rank - 1, max(80, schoolbook_rank - 20), -1):
        if time.time() > deadline:
            break
        result = try_rank(r, max_iter=3000)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook
    rank = schoolbook_rank
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
