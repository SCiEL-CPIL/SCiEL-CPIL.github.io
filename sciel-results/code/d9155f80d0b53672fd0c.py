"""Matrix multiplication 3,3,6 rank-minimal decomposition search."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 6]

def _build_target_tensor():
    """Build the 3x3x6 matrix multiplication tensor T[i,j,k] = 1 if k is in range."""
    n, m, p = 3, 3, 6
    # T[i,j,k] = delta for multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T[i,j,k] where the rank decomposition gives C[i,k] = sum_r U[i,j,r]*V[j,k,r]*W[r] (simplified)
    # Actually for matrix mult: T[i,j,k] = 1 means term A[i,j]*B[j,k]
    # The tensor is T[i,j,k] for i in [0,n), j in [0,m), k in [0,p)
    # where T[i,j,k] = 1 (it's just the structure of which products exist)
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _verify_decomposition(U, V, W):
    """Verify that U, V, W decompose the 3x3x6 matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    # Convert to numpy
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # Decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the convention is W[k*n+i, r]
    # So the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Check: for each (i,j,k), the sum should equal 1 (since T[i,j,k]=1 for all)
    # Actually no - the tensor for matrix multiplication has T[i,j,k] = 1 for all valid indices
    # and the decomposition must satisfy: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let me re-examine: the product C = A * B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 (it's the identity on the multiplication structure)
    # Rank decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _try_random_search(deadline, start_rank=40, max_rank=54):
    """Try to find a low-rank decomposition via random search."""
    n, m, p = 3, 3, 6
    best_rank = max_rank + 1
    best_result = None
    
    for rank in range(start_rank, max_rank + 1):
        if time.time() > deadline - 30:
            break
        # Try multiple random initializations
        for attempt in range(200):
            if time.time() > deadline - 30:
                break
            
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Scale to help convergence
            U *= 1.0 / np.sqrt(rank)
            V *= 1.0 / np.sqrt(rank)
            W *= 1.0 / np.sqrt(rank)
            
            # Iterate: for each tensor index, adjust the rank-1 components
            converged = False
            for iteration in range(200):
                if time.time() > deadline - 30:
                    break
                
                max_change = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Target: T[i,j,k] = 1
                            # Current: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                            current = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            error = 1.0 - current
                            if abs(error) < 1e-12:
                                continue
                            
                            # Adjust the rank-1 contributions
                            # For each r, we can adjust U, V, or W
                            # Use gradient-like update
                            for r in range(rank):
                                grad_U = V[j*p+k, r] * W[k*n+i, r]
                                grad_V = U[i*m+j, r] * W[k*n+i, r]
                                grad_W = U[i*m+j, r] * V[j*p+k, r]
                                
                                lr = 0.01 / (1 + iteration * 0.01)
                                U[i*m+j, r] += lr * error * grad_U
                                V[j*p+k, r] += lr * error * grad_V
                                W[k*n+i, r] += lr * error * grad_W
                                max_change = max(max_change, abs(lr * error))
                
                if max_change < 1e-10:
                    converged = True
                    break
            
            if time.time() > deadline - 30:
                break
            
            # Check if we found a valid decomposition
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U.tolist(), V.tolist(), W.tolist())
                    if best_rank <= start_rank:
                        return best_result
                break  # Found a valid one for this rank, move to next
    
    return best_result

def _try_structured_search(deadline):
    """Try structured decompositions based on block structure."""
    n, m, p = 3, 3, 6
    best_rank = 55
    best_result = None
    
    # Strategy: split the 6 columns into two groups of 3
    # Use two 3x3x3 decompositions and look for rank sharing
    
    # Try rank 42-50 with various structured patterns
    for rank in range(42, 51):
        if time.time() > deadline - 30:
            break
        
        # Try sparse structured matrices
        for pattern_seed in range(50):
            if time.time() > deadline - 30:
                break
            
            rng = np.random.RandomState(pattern_seed)
            
            # Create sparse random matrices
            density = 0.3
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            for idx in range(n*m):
                mask = rng.rand(rank) < density
                U[idx, mask] = rng.randn(mask.sum()) if mask.sum() > 0 else 0
            
            for idx in range(m*p):
                mask = rng.rand(rank) < density
                V[idx, mask] = rng.randn(mask.sum()) if mask.sum() > 0 else 0
            
            for idx in range(p*n):
                mask = rng.rand(rank) < density
                W[idx, mask] = rng.randn(mask.sum()) if mask.sum() > 0 else 0
            
            # Refine with local search
            for iteration in range(100):
                if time.time() > deadline - 30:
                    break
                
                max_change = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            current = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            error = 1.0 - current
                            if abs(error) < 1e-14:
                                continue
                            
                            for r in range(rank):
                                if abs(U[i*m+j, r]) > 1e-8 or abs(V[j*p+k, r]) > 1e-8 or abs(W[k*n+i, r]) > 1e-8:
                                    grad_U = V[j*p+k, r] * W[k*n+i, r]
                                    grad_V = U[i*m+j, r] * W[k*n+i, r]
                                    grad_W = U[i*m+j, r] * V[j*p+k, r]
                                    
                                    lr = 0.005 / (1 + iteration * 0.005)
                                    U[i*m+j, r] += lr * error * grad_U
                                    V[j*p+k, r] += lr * error * grad_V
                                    W[k*n+i, r] += lr * error * grad_W
                                    max_change = max(max_change, abs(lr * error))
                
                if max_change < 1e-12:
                    break
            
            if time.time() > deadline - 30:
                break
            
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U.tolist(), V.tolist(), W.tolist())
                break
    
    return best_result

def _schoolbook_decomposition():
    """Return the schoolbook rank-54 decomposition."""
    n, m, p = 3, 3, 6
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
    """Find a rank-minimal exact tensor decomposition for 3x3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Try structured search first (more likely to find low rank)
    result = _try_structured_search(deadline)
    
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Try random search
    if time.time() < deadline - 5:
        result = _try_random_search(deadline, start_rank=40, max_rank=53)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    sb = _schoolbook_decomposition()
    if _verify_decomposition(sb["U"], sb["V"], sb["W"]):
        return sb
    
    # Ultimate fallback
    return sb

# EVOLVE-BLOCK-END
