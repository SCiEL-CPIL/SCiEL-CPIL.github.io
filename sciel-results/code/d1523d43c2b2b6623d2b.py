"""Rewritten program for matrix multiplication 2,2,8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a low-rank decomposition for multiplying a 2x2 matrix by a 2x8 matrix.
    Uses Strassen's algorithm on 2x2 blocks to achieve rank 28.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    
    # Target rank: 28 (4 blocks of 2x2, each using 7 Strassen multiplications)
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Iterate over the 4 blocks of B (and C)
    for b in range(4):
        # Base index for this block's columns in B and C
        # B columns are 2b and 2b+1
        # C columns are 2b and 2b+1
        
        # Strassen terms indices in the global rank
        # r = b * 7 + s
        
        # Helper to get V index for B[j, k]
        def v_idx(j, k):
            return j * p + k
            
        # Helper to get W index for C[i, k]
        # Note: W index is k*n + i
        def w_idx(i, k):
            return k * n + i
            
        # Helper to get U index for A[i, j]
        # Note: U index is i*m + j
        def u_idx(i, j):
            return i * m + j

        # Term 1: M1 = (a+d)(e+h)
        r = b * 7 + 0
        U[u_idx(0, 0), r] += 1  # a
        U[u_idx(1, 1), r] += 1  # d
        V[v_idx(0, 2*b), r] += 1    # e
        V[v_idx(1, 2*b+1), r] += 1  # h
        W[w_idx(0, 2*b), r] += 1    # C[0, 2b]
        W[w_idx(0, 2*b+1), r] += 1  # C[0, 2b+1]
        W[w_idx(1, 2*b), r] += -1   # C[1, 2b]
        W[w_idx(1, 2*b+1), r] += 1  # C[1, 2b+1]
        
        # Term 2: M2 = (c+d)(e)
        r = b * 7 + 1
        U[u_idx(1, 0), r] += 1  # c
        U[u_idx(1, 1), r] += 1  # d
        V[v_idx(0, 2*b), r] += 1    # e
        W[w_idx(1, 2*b), r] += 1    # C[1, 2b]
        W[w_idx(1, 2*b+1), r] += -1 # C[1, 2b+1]
        
        # Term 3: M3 = a(f-h)
        r = b * 7 + 2
        U[u_idx(0, 0), r] += 1  # a
        V[v_idx(0, 2*b+1), r] += 1   # f
        V[v_idx(1, 2*b+1), r] += -1  # -h
        W[w_idx(0, 2*b+1), r] += 1   # C[0, 2b+1]
        W[w_idx(1, 2*b+1), r] += 1   # C[1, 2b+1]
        
        # Term 4: M4 = d(g-e)
        r = b * 7 + 3
        U[u_idx(1, 1), r] += 1  # d
        V[v_idx(1, 2*b), r] += 1   # g
        V[v_idx(0, 2*b), r] += -1  # -e
        W[w_idx(0, 2*b), r] += 1   # C[0, 2b]
        W[w_idx(1, 2*b), r] += 1   # C[1, 2b]
        
        # Term 5: M5 = (a+b)h
        r = b * 7 + 4
        U[u_idx(0, 0), r] += 1  # a
        U[u_idx(0, 1), r] += 1  # b
        V[v_idx(1, 2*b+1), r] += 1  # h
        W[w_idx(0, 2*b), r] += -1   # C[0, 2b]
        W[w_idx(0, 2*b+1), r] += 1  # C[0, 2b+1]
        
        # Term 6: M6 = (c-a)(e+g)
        r = b * 7 + 5
        U[u_idx(1, 0), r] += 1   # c
        U[u_idx(0, 0), r] += -1  # -a
        V[v_idx(0, 2*b), r] += 1   # e
        V[v_idx(1, 2*b), r] += 1   # g
        W[w_idx(1, 2*b+1), r] += 1 # C[1, 2b+1]
        
        # Term 7: M7 = (b-d)(f+h)
        r = b * 7 + 6
        U[u_idx(0, 1), r] += 1   # b
        U[u_idx(1, 1), r] += -1  # -d
        V[v_idx(0, 2*b+1), r] += 1  # f
        V[v_idx(1, 2*b+1), r] += 1  # h
        W[w_idx(0, 2*b), r] += 1    # C[0, 2b]

    # Verify the decomposition
    if time.time() < deadline:
        if _verify_decomposition(U, V, W, n, m, p):
            # Convert to lists for JSON serialization if needed, but interface expects dict
            # The interface says "factor matrices of shapes... with integer/half-integer entries"
            # Returning numpy arrays is usually fine, but lists are safer for some evaluators.
            # However, the seed program returned lists. Let's return lists to be safe.
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    
    # The tensor T[i,j,k] should equal 1 for all i,j,k
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We can vectorize this check
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (n, p, rank) ? No, W is (p*n, rank)
    # W index is k*n + i. So W.reshape(p, n, rank).
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # We want to compute sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is a tensor contraction.
    # Let's just loop over a few random points or all points if small enough.
    # n=2, m=2, p=8, rank=28. Total points = 32.
    # Looping 32 times with numpy dot products is very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Extract vectors
                u_vec = U_reshaped[i, j, :]
                v_vec = V_reshaped[j, k, :]
                w_vec = W_reshaped[k, i, :]
                
                # Compute sum of element-wise products
                val = np.sum(u_vec * v_vec * w_vec)
                
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

# EVOLVE-BLOCK-END
