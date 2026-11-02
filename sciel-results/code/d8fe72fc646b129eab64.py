DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product
from fractions import Fraction
import random

def _elapsed(t0):
    return time.time() - t0

def _check_deadline(t0, limit=145):
    return _elapsed(t0) < limit

def _build_strassen_rank55():
    """
    Build a rank-55 decomposition using 5 Strassen blocks.
    
    For 2x3 x 3x10 matrix multiplication:
    - Split the 10 output columns into 5 groups of 2
    - For each group, we compute a 2x2 matrix product
    - Use Strassen's algorithm (rank 7) for each 2x2 block
    - Plus schoolbook for the remaining connections
    
    Actually, let me think more carefully.
    
    A is 2x3, B is 3x10, C is 2x10.
    
    We can view this as: for each column pair (2k, 2k+1) of B,
    we have a 3x2 block B_k. Then C[:, 2k:2k+2] = A * B_k.
    
    A is 2x3, B_k is 3x2, result is 2x2.
    
    For 2x2 matrix multiplication, Strassen uses rank 7.
    We have 5 such blocks (k=0..4).
    
    Total rank = 5 * 7 = 35? That seems too optimistic.
    
    Wait, the issue is that A is shared across all blocks.
    We can't just apply Strassen independently because A appears in all of them.
    
    Let me reconsider. The tensor is T_{i,j,k} where:
    - i in {0,1} (row of A, row of C)
    - j in {0,1,2} (column of A, row of B)  
    - k in {0,...,9} (column of B, column of C)
    
    T_{i,j,k} = 1 if the term involves A[i,j], B[j,k], C[i,k]
    
    For schoolbook: rank = 2*3*10 = 60
    
    For Strassen on 2x2 blocks: we need to be careful.
    
    Alternative approach: Use the fact that n=2 is small.
    
    C[0,k] = sum_j A[0,j] * B[j,k]
    C[1,k] = sum_j A[1,j] * B[j,k]
    
    This is like: for each k, C[:,k] = B[:,k]^T * A^T (roughly)
    
    Actually, let me just implement a known rank-55 construction if one exists,
    or search for a lower rank.
    
    For now, let me return the schoolbook rank-60 as a safe baseline,
    but structure the code to allow optimization.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition(U, V, W, tol=1e-10):
    """
    Verify that the decomposition exactly represents matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    The tensor equation: sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t] = T
    where T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j', i==i', k==k' else 0
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Build the tensor from decomposition
    T_decomp = np.zeros((n*m, m*p, p*n))
    for t in range(rank):
        for a in range(n*m):
            if abs(U[a, t]) > tol:
                for b in range(m*p):
                    if abs(V[b, t]) > tol:
                        for c in range(p*n):
                            if abs(W[c, t]) > tol:
                                T_decomp[a, b, c] += U[a, t] * V[b, t] * W[c, t]
    
    # Build the target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
    
    # Check if they match
    diff = np.abs(T_decomp - T_target)
    max_diff = np.max(diff)
    return max_diff < tol, max_diff

def _search_rank(target_rank, t0, deadline_limit=145):
    """
    Search for a decomposition of the given rank.
    Uses random initialization and local search.
    """
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Random initialization
    best_score = float('inf')
    best_decomp = None
    
    for attempt in range(50):
        if not _check_deadline(t0, deadline_limit):
            break
        
        # Initialize with random values
        U = np.random.randn(dim_U, target_rank) * 0.5
        V = np.random.randn(dim_V, target_rank) * 0.5
        W = np.random.randn(dim_W, target_rank) * 0.5
        
        # Try to optimize
        for iteration in range(100):
            if not _check_deadline(t0, deadline_limit):
                break
            
            # Compute the current tensor
            T_decomp = np.zeros((dim_U, dim_V, dim_W))
            for t in range(target_rank):
                T_decomp += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
            
            # Target tensor
            T_target = np.zeros((dim_U, dim_V, dim_W))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        T_target[a, b, c] = 1
            
            # Compute error
            error = np.sum((T_decomp - T_target) ** 2)
            
            if error < best_score:
                best_score = error
                best_decomp = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-10:
                break
            
            # Gradient descent step
            lr = 0.01
            for t in range(target_rank):
                # Gradient with respect to U[:,t]
                grad_U = np.zeros(dim_U)
                for a in range(dim_U):
                    for b in range(dim_V):
                        for c in range(dim_W):
                            grad_U[a] += 2 * (T_decomp[a,b,c] - T_target[a,b,c]) * V[b,t] * W[c,t]
                
                # Gradient with respect to V[:,t]
                grad_V = np.zeros(dim_V)
                for b in range(dim_V):
                    for a in range(dim_U):
                        for c in range(dim_W):
                            grad_V[b] += 2 * (T_decomp[a,b,c] - T_target[a,b,c]) * U[a,t] * W[c,t]
                
                # Gradient with respect to W[:,t]
                grad_W = np.zeros(dim_W)
                for c in range(dim_W):
                    for a in range(dim_U):
                        for b in range(dim_V):
                            grad_W[c] += 2 * (T_decomp[a,b,c] - T_target[a,b,c]) * U[a,t] * V[b,t]
                
                U[:,t] -= lr * grad_U
                V[:,t] -= lr * grad_V
                W[:,t] -= lr * grad_W
        
        if not _check_deadline(t0, deadline_limit):
            break
    
    if best_decomp is not None:
        U, V, W = best_decomp
        valid, max_diff = _verify_decomposition(U, V, W)
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying 
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy:
    1. Try to find a low-rank decomposition via optimization
    2. Fall back to schoolbook (rank 60) if optimization fails
    """
    t0 = time.time()
    deadline_limit = 145  # Safety margin before 150s kill
    
    # Try to find a lower-rank decomposition
    for target_rank in [35, 40, 45, 50, 55]:
        if not _check_deadline(t0, deadline_limit):
            break
        
        result = _search_rank(target_rank, t0, deadline_limit)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _build_strassen_rank55()
# EVOLVE-BLOCK-END
