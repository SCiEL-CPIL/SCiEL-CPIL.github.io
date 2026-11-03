"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _validate_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # T[i,j,k,l] = delta_{j,l} * delta_{i,k}
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j,k} * delta_{i,?}
    # Actually for 3x3 * 3x5: A is 3x3, B is 3x5, C = A*B is 3x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = delta_{i,j} (output index i, input A index j, output B index k)
    # Wait, let me reconsider the tensor structure.
    
    # The task says: W[k*n+i, r] follows the official trace convention
    # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = delta_{j,k}... no wait.
    
    # Let me think about this more carefully.
    # U has shape (n*m, rank) - indexed by (i,j) where i is row of A, j is column of A/row of B
    # V has shape (m*p, rank) - indexed by (j,k) where j is row of B, k is column of B
    # W has shape (p*n, rank) - indexed by (k,i) where k is column of B, i is row of A
    
    # The decomposition should satisfy:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j,j}... 
    # Actually it should equal the matrix multiplication tensor.
    
    # For exact matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # But the tensor value should be 1 for all valid (i,j,k) combinations where j is the shared index.
    
    # Actually, the tensor for matrix multiplication is:
    # T[i,j,k] = delta_{j,j} which is always 1, but that's not right.
    
    # Let me reconsider. The matrix multiplication tensor T satisfies:
    # T[i,j,k] = 1 if and only if this term contributes to C[i,k]
    # For C = A*B, C[i,k] = sum_j A[i,j] * B[j,k]
    # So every (i,j,k) triple contributes with coefficient 1.
    # Thus T[i,j,k] = 1 for all i,j,k.
    
    # Wait, that can't be right either because then the rank would be n*m*p.
    
    # Let me re-examine. The tensor T is defined such that:
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k.
    # So the tensor is all ones, and its rank is indeed n*m*p for the schoolbook algorithm.
    
    # But we can do better! The rank of the matrix multiplication tensor is what we're looking for.
    # For 3x3 x 3x5, the minimum rank is not known to be less than 45 (the schoolbook rank).
    
    # Let's verify: the decomposition should satisfy
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Let's check this condition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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

def _try_low_rank_search(n, m, p, max_rank, deadline):
    """Try to find a low-rank exact decomposition using structured patterns."""
    # Start from known lower bounds and try to verify or improve
    # For 3x3 x 3x5, the schoolbook rank is 45
    # Let's try ranks from 40 down to see if we can find an exact decomposition
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank
    
    for rank in range(max_rank, best_rank - 1, -1):
        if time.time() > deadline:
            break
        
        # Try a few structured patterns for this rank
        # Pattern 1: Try to merge some schoolbook terms
        # For example, if we have terms (i1,j1,k1) and (i2,j2,k2), can we combine them?
        
        # Simple approach: try random initializations with refinement
        np.random.seed(42)
        
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Try gradient descent to minimize the error
            # Error = sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
            
            lr = 0.01
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Compute current values
                values = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            values[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                
                # Compute gradients
                error = values - 1.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = error[i, j, k]
                            for c in range(rank):
                                grad_U[i*m+j, c] += 2 * e * V[j*p+k, c] * W[k*n+i, c]
                                grad_V[j*p+k, c] += 2 * e * U[i*m+j, c] * W[k*n+i, c]
                                grad_W[k*n+i, c] += 2 * e * U[i*m+j, c] * V[j*p+k, c]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if np.max(np.abs(error)) < 1e-8:
                    # Found exact solution!
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    
                    if _validate_decomposition(U_list, V_list, W_list, n, m, p):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
            
            if best_rank < max_rank:
                break
        
        if best_rank < max_rank:
            break
    
    return best_result, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with schoolbook
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Theoretical lower bound for 3x3 x 3x5 is at least max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But in practice, for small matrices, the minimum rank is often close to the schoolbook rank
    
    # Try ranks from 44 down to 30
    result, rank = _try_low_rank_search(n, m, p, 44, deadline)
    if result is not None and rank < best_rank:
        best_result = result
        best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
