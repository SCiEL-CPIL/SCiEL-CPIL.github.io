"""Tensor decomposition for 3x5x6 matrix multiplication.
Attempts to find a rank-minimal exact decomposition.
Falls back to schoolbook (rank 90) if no better solution is found within time limit.
"""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(V[0])
    # Build the multiplication tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For correct decomposition, T should equal the identity-like tensor:
    # T[i, j, k] = 1 if the (i,k) entry of A*B comes from A[i,j]*B[j,k]
    # Actually, the tensor for matrix mult is: T[i,j,k] = 1 for all i,j,k
    # No wait. The matrix multiplication tensor has entries:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i, j, k] = 1 for all valid i, j, k.
    # And we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _verify_numpy(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = len(V[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # For each c, the rank-1 tensor is outer product of:
    #   u_c reshaped to (n, m): U_arr[:, c] reshaped
    #   v_c reshaped to (m, p): V_arr[:, c] reshaped  
    #   w_c reshaped to (p, n): W_arr[:, c] reshaped
    # The sum over c of u_c[i,j] * v_c[j,k] * w_c[k,i] should equal 1 for all i,j,k.
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Let's compute this efficiently.
    # For a fixed j: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is tricky to vectorize directly. Let's just do it in a loop over j.
    
    for j in range(m):
        # U[:, c] for rows i*m+j, i in 0..n-1
        U_slice = U_arr[np.array([i*m+j for i in range(n)]), :].T  # (rank, n)
        V_slice = V_arr[np.array([j*p+k for k in range(p)]), :].T  # (rank, p)
        # For each (i, k), W[k*n+i, c]
        W_slice = np.array([[W_arr[k*n+i, c] for c in range(rank)] for k in range(p) for i in range(n)]).reshape(p, n, rank).transpose(2, 0, 1)  # (rank, p, n)
        
        # T[i,k] = sum_c U_slice[c,i] * V_slice[c,k] * W_slice[c,k,i]
        # For each c: outer of U_slice[:,i], V_slice[:,k], W_slice[c,k,i]
        # T[i,k] = sum_c (U_slice[c,i] * V_slice[c,k] * W_slice[c,k,i])
        
        # Compute: for each c, form matrix M_c[i,k] = U_slice[c,i] * V_slice[c,k] * W_slice[c,k,i]
        # Sum over c.
        # This is: sum_c U_slice[:,i] * V_slice[:,k] * W_slice[c,k,i]
        
        # Let's compute it as: for each c, the contribution is a rank-1-ish thing but W depends on both i and k.
        # Actually W_slice[c, k, i] is a specific value.
        # T[i,k] = sum_c U_slice[c,i] * V_slice[c,k] * W_slice[c,k,i]
        
        # Compute for all (i,k):
        # T[i,k] = sum_c U_slice[c,i] * V_slice[c,k] * W_slice[c,k,i]
        
        # Let's do: for each c, add U_slice[c,:] * V_slice[c,:] * W_slice[c, :, :]
        # But W_slice[c, k, i] is indexed by (k, i) for fixed c.
        
        # T = zeros(n, p)
        T = np.zeros((n, p))
        for c in range(rank):
            T += np.outer(U_slice[c, :], V_slice[c, :]) * W_slice[c, :, :]
        
        if not np.allclose(T, 1.0, atol=1e-10):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    schoolbook = _build_schoolbook()
    
    # Try random search for lower rank decompositions
    # Start with rank slightly below schoolbook and try to find exact solutions
    
    best_result = schoolbook
    best_rank = n * m * p
    
    # We'll try ranks from some lower bound up to best_rank - 1
    # Lower bound: max(n*m, m*p, p*n, ...) but really the tensor rank is at least max(n*m, m*p, p*n)
    # For 3,5,6: max(15, 30, 18) = 30
    # But the actual rank is likely much higher.
    
    # Let's try a limited random search
    # Random initialization with small integer/half-integer entries
    
    min_rank_to_try = max(n*m, m*p, p*n)
    
    for rank in range(min_rank_to_try, best_rank):
        if time.time() > deadline:
            break
        
        # Try a few random initializations for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random entries from {-1, 0, 0.5, 1}
            entries = [-1, 0, 0.5, 1]
            U = [[random.choice(entries) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(entries) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(entries) for _ in range(rank)] for _ in range(p * n)]
            
            if _verify_numpy(U, V, W, n, m, p):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        
        if time.time() > deadline:
            break
    
    return best_result
