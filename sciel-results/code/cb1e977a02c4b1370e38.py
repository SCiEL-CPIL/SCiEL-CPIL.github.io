"""Find rank-minimal exact tensor decomposition for 3x3 x 3x7 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 3, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor from decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for all valid (i,j,k) since each contributes A[i,j]*B[j,k]
    # Actually the tensor entry is 1 for all (i,j,k) in the range
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute the decomposed tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_fixed(r, n, m, p, max_iter=2000, tol=1e-8, seed=42):
    """Try to find a rank-r decomposition using gradient-based optimization."""
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(n*m, r) * 0.1
    V = rng.randn(m*p, r) * 0.1
    W = rng.randn(p*n, r) * 0.1
    
    # We want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all (i,j,k)
    # This is a tensor factorization problem
    
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        # Compute current tensor values
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:])
                    errors.append(val - 1.0)
        
        errors = np.array(errors)
        loss = np.sum(errors**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    err = errors[i*m*p + j*p + k]
                    for c in range(r):
                        dU[i*m+j, c] += 2 * err * V[j*p+k, c] * W[k*n+i, c]
                        dV[j*p+k, c] += 2 * err * U[i*m+j, c] * W[k*n+i, c]
                        dW[k*n+i, c] += 2 * err * U[i*m+j, c] * V[j*p+k, c]
        
        # Gradient descent with adaptive learning rate
        lr = 0.01 / (1 + iteration * 0.001)
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Check if we found a valid solution
    if best_loss < tol:
        return best_U, best_V, best_W, True
    return None, None, None, False

def _try_rank_structured(r, n, m, p, seed=42):
    """Try structured initialization based on Strassen-like patterns."""
    rng = np.random.RandomState(seed)
    
    # For 3x3x3, Strassen uses 7 terms. For 3x3x7, we might use
    # a combination of Strassen blocks for the 7 output columns.
    
    # The 7 output columns can be grouped: first 3 columns use one Strassen-like
    # structure, next 3 use another, last 1 uses a simpler structure.
    
    # Actually, let's think of it differently. The tensor T[i,j,k] for 3x3x7
    # is the "outer product" structure where for each output column k, we have
    # a 3x3 matrix multiplication.
    
    # A key insight: if we use the same U and V patterns but different W patterns
    # for different groups of k, we can share computation.
    
    # Let's try: use 7 Strassen-like decompositions, one for each k, but with
    # shared U and V factors.
    
    # Strassen's 7 products for 3x3:
    # M1 = (A00+A11+A22)(B00+B11+B22)
    # M2 = (A01-A11+A21)(B00-B11+B20)  -- wait, this isn't quite right
    # Let me use a known rank-21 decomposition for 3x3x3.
    
    # Actually, for 3x3x3, the best known rank is 21 (not 7; Strassen is for 2x2x2).
    # For 3x3x3, the schoolbook rank is 27, and the best known is 21.
    
    # Let me just do a numerical search with multiple random restarts.
    
    best_result = None
    for attempt in range(5):
        U, V, W, success = _try_rank_fixed(r, n, m, p, max_iter=3000, seed=seed+attempt)
        if success:
            return U, V, W, True
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 63
    
    # Start from schoolbook and try to reduce rank
    # Try ranks from low to high
    
    # First, let's establish the schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find lower rank decompositions
    # The theoretical lower bound for 3x3x7 is at least max(9, 21, 21) = 21
    # but likely higher. Let's try from rank 27 upward.
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Try various ranks
    for rank in range(27, schoolbook_rank + 1):
        if time.time() > deadline - 30:  # Leave 30s margin
            break
        
        # Try with different seeds
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline - 30:
                break
            
            U, V, W, success = _try_rank_fixed(rank, n, m, p, max_iter=1000, seed=seed)
            if success:
                # Convert to list format with integer/half-integer if possible
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Try to round to nearest 0.5 if close
                for mat in [U_list, V_list, W_list]:
                    for i in range(len(mat)):
                        for j in range(len(mat[i])):
                            val = mat[i][j]
                            rounded = round(val * 2) / 2
                            if abs(val - rounded) < 1e-6:
                                mat[i][j] = rounded
                
                # Verify
                if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()
