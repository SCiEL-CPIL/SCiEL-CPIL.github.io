"""Matrix multiplication 2,4,5 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the target tensor T with shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    # Try to reduce rank by removing redundant columns
    # A column is redundant if the tensor can still be represented without it
    # We'll try greedy removal
    
    current_rank = rank
    U_curr = U.copy()
    V_curr = V.copy()
    W_curr = W.copy()
    
    def check_decomposition(U_mat, V_mat, W_mat, target_T, tol=1e-10):
        """Check if U @ V^T @ W^T approximates target_T"""
        # The decomposition is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We can check this by computing the tensor and comparing
        # For efficiency, check a few random slices
        n_a, n_b, n_c = target_T.shape
        # Check all entries (small enough for 8,20,10)
        reconstructed = np.zeros_like(target_T)
        for r in range(U_mat.shape[1]):
            reconstructed += np.outer(np.outer(U_mat[:, r], V_mat[:, r]), W_mat[:, r])
        return np.allclose(reconstructed, target_T, atol=tol)
    
    # Try removing columns one by one
    removed = []
    for attempt in range(100):  # Limit iterations
        if time.time() > deadline:
            break
        
        progress = False
        for r in range(current_rank):
            if time.time() > deadline:
                break
            
            # Try removing column r
            if r in removed:
                continue
            
            U_test = np.delete(U_curr, r, axis=1)
            V_test = np.delete(V_curr, r, axis=1)
            W_test = np.delete(W_curr, r, axis=1)
            
            if check_decomposition(U_test, V_test, W_test, T):
                # Column r is redundant
                removed.append(r)
                U_curr = U_test
                V_curr = V_test
                W_curr = W_test
                current_rank -= 1
                progress = True
                break  # Restart scanning from beginning
        
        if not progress:
            break
    
    # Convert to lists of integers/half-integers
    def to_rational_list(arr):
        result = []
        for row in arr:
            row_list = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2.0
                row_list.append(rounded)
            result.append(row_list)
        return result
    
    if time.time() > deadline:
        # Return schoolbook as fallback
        return {"U": to_rational_list(U), "V": to_rational_list(V), "W": to_rational_list(W)}
    
    return {"U": to_rational_list(U_curr), "V": to_rational_list(V_curr), "W": to_rational_list(W_curr)}
# EVOLVE-BLOCK-END
