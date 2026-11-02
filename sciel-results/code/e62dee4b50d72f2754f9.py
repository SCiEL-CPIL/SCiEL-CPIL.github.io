"""
Matrix multiplication tensor decomposition for 3x4 * 4x8.
Uses a rank-reduction strategy: try to find a decomposition with rank < 96.
The known upper bound is 96 (schoolbook). We attempt to find lower rank
decompositions using structured random search with integer/half-integer entries.
"""

import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 4, 8]
TIME_LIMIT = 120.0  # seconds
SAFETY_MARGIN = 10.0  # seconds before hard limit

def _verify(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = 1 if we multiply A[i,j]*B[j,k] to get C[k,i] (trace convention: W[k*n+i])
    
    Actually, let me re-derive the convention:
    - Input: A is n x m, B is m x p
    - Output: C = A @ B is n x p, C[i,k] = sum_j A[i,j] * B[j,k]
    - W has shape (p*n, rank), indexed by W[k*n + i, c]
    - U has shape (n*m, rank), indexed by U[i*m + j, c]
    - V has shape (m*p, rank), indexed by V[j*p + k, c]
    
    The decomposition says: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    And for all other (i,j,k) combinations that don't form a valid product term, the sum should be 0.
    
    Wait, actually the matrix multiplication tensor T is defined such that:
    T[i,j,k] = 1 if (i,j,k) corresponds to a term in the product, else 0.
    For C[i,k] = sum_j A[i,j] B[j,k], the tensor has T[i,j,k] = 1 for all i,j,k (since every (i,j,k) contributes).
    
    Hmm, let me reconsider. The matrix multiplication tensor of size (n,m,p) has entries:
    T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    No wait, that's not right either. Let me think about this more carefully.
    
    The matrix multiplication map is: (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k].
    
    This is a bilinear map. The associated tensor T has components T[i,j,k] such that:
    C[i,k] = sum_{j,c} A[i,j] * U... 
    
    Actually, the standard tensor decomposition for matrix multiplication:
    We want to express the multiplication tensor. The tensor T has T[i,j,k] = delta_{i,k}? No...
    
    Let me use the schoolbook as reference:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k)
    
    So for a given (i,j,k), the contribution is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    Summing over c: for the schoolbook, each (i,j,k) has its own c, so the sum is 1 for each (i,j,k).
    
    The tensor we're decomposing has T[i,j,k] = 1 for all valid (i,j,k) where i<n, j<m, k<p.
    
    So the verification is: for all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    rank = U.shape[1]
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m+j]
                v_val = V_arr[j*p+k]
                w_val = W_arr[k*n+i]
                result = np.dot(u_val, v_val * w_val)
                if abs(result - 1.0) > 1e-9:
                    return False
    return True

def _to_half_integer_matrix(arr):
    """Check if array entries are integer or half-integer, return clean version."""
    arr = np.array(arr)
    # Multiply by 2, check if integer
    doubled = arr * 2
    rounded = np.round(doubled)
    if np.allclose(doubled, rounded, atol=1e-8):
        # Convert to exact half-integer representation
        return rounded / 2.0
    return None

def _random_search(n, m, p, max_rank, time_budget, rng):
    """
    Random search for a low-rank decomposition.
    Tries various ranks and random initializations.
    """
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    start_time = time.time()
    
    # Try ranks from low to high
    for rank in range(10, max_rank + 1):
        if time.time() - start_time > time_budget:
            break
        
        # Multiple random restarts for this rank
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > time_budget:
                break
            
            # Initialize with small random values
            U = rng.uniform(-1, 1, size=(n*m, rank))
            V = rng.uniform(-1, 1, size=(m*p, rank))
            W = rng.uniform(-1, 1, size=(p*n, rank))
            
            # Gradient descent to minimize sum of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            lr = 0.01
            for step in range(200):
                if time.time() - start_time > time_budget:
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i*m+j
                            v_idx = j*p+k
                            w_idx = k*n+i
                            
                            prod = U[u_idx] * V[v_idx] * W[w_idx]
                            # Sum over c
                            val = np.sum(prod, axis=1)
                            err = val - 1.0
                            
                            # Gradient w.r.t. U[u_idx,c] = err * V[v_idx,c] * W[w_idx,c]
                            grad_U[u_idx] += err * V[v_idx] * W[w_idx]
                            grad_V[v_idx] += err * U[u_idx] * W[w_idx]
                            grad_W[w_idx] += err * U[u_idx] * V[v_idx]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i*m+j
                            v_idx = j*p+k
                            w_idx = k*n+i
                            val = np.sum(U[u_idx] * V[v_idx] * W[w_idx], axis=1)
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-8:
                    # Snap to half-integers
                    U_clean = _to_half_integer_matrix(U)
                    V_clean = _to_half_integer_matrix(V)
                    W_clean = _to_half_integer_matrix(W)
                    if U_clean is not None and V_clean is not None and W_clean is not None:
                        U_list = U_clean.tolist()
                        V_list = V_clean.tolist()
                        W_list = W_clean.tolist()
                        if _verify(U_list, V_list, W_list, n, m, p):
                            if rank < best_rank:
                                best_rank = rank
                                best_result = {'U': U_list, 'V': V_list, 'W': W_list}
                    break
            
            if best_result is not None and rank == best_rank:
                break
    
    return best_result, best_rank

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    result = {'U': U_school, 'V': V_school, 'W': W_school}
    best_rank = schoolbook_rank
    
    # Try to find better decomposition
    start_time = time.time()
    time_budget = TIME_LIMIT - SAFETY_MARGIN - (time.time() - start_time)
    
    if time_budget > 5:
        rng = np.random.default_rng(42)
        found, found_rank = _random_search(n, m, p, schoolbook_rank - 1, time_budget, rng)
        if found is not None:
            result = found
            best_rank = found_rank
    
    return result
