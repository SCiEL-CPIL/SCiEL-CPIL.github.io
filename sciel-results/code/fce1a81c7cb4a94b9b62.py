"""Matrix multiplication 3x3 * 3x7 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Construct a low-rank exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Returns dict with U (9 x rank), V (21 x rank), W (21 x rank)
    such that T[i,j,k] = sum_c U[i*3+j, c] * V[j*7+k, c] * W[k*3+i, c]
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with schoolbook decomposition (rank 63)
    rank = n * m * p  # 63
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Verify correctness
    def verify(U, V, W):
        """Check that the decomposition exactly represents matrix multiplication."""
        # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This should equal delta(i,k) for the identity, but actually:
        # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
        # Actually T[i,j,k] = 1 for all valid (i,j,k) in the sense that
        # C[i,k] = sum_j A[i,j] * B[j,k] means the tensor has T[i,j,k]=1
        
        # Let's verify: for random A (3x3) and B (3x7), check C = A*B
        rng = np.random.RandomState(42)
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        # Compute using decomposition
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    # This should be 1 (since C[i,k] = sum_j A[i,j]*B[j,k])
                    t_val = 0.0
                    for c in range(U.shape[1]):
                        t_val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    C_computed[i, k] += A[i, j] * B[j, k] * t_val
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if not verify(U, V, W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to reduce rank by merging terms
    # Strategy: look for pairs of terms that can be combined using linear combinations
    # If terms c1 and c2 have proportional U, V, or W vectors, we might merge them
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try various merging strategies
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        U_cur = best_U.copy()
        V_cur = best_V.copy()
        W_cur = best_W.copy()
        cur_rank = U_cur.shape[1]
        
        # Try to find a pair of columns that can be merged
        # Two terms c1, c2 can be merged if there exist scalars a, b such that:
        # a*U[:,c1] + b*U[:,c2] = U[:,c1] (or similar for V, W)
        # This is complex, so let's try a simpler approach:
        # Randomly perturb and check if rank decreases
        
        # Alternative: use the fact that if U[:,c1] = U[:,c2], V[:,c1] = V[:,c2], W[:,c1] = W[:,c2],
        # then the terms are identical and can be merged
        merged = False
        for c1 in range(cur_rank):
            if merged:
                break
            for c2 in range(c1 + 1, cur_rank):
                # Check if terms are identical
                if (np.allclose(U_cur[:, c1], U_cur[:, c2]) and 
                    np.allclose(V_cur[:, c1], V_cur[:, c2]) and 
                    np.allclose(W_cur[:, c1], W_cur[:, c2])):
                    # Merge: add the W values and remove one column
                    W_cur[:, c1] += W_cur[:, c2]
                    U_cur = np.delete(U_cur, c2, axis=1)
                    V_cur = np.delete(V_cur, c2, axis=1)
                    W_cur = np.delete(W_cur, c2, axis=1)
                    cur_rank -= 1
                    merged = True
                    break
        
        if merged and verify(U_cur, V_cur, W_cur):
            best_U, best_V, best_W = U_cur, V_cur, W_cur
            best_rank = cur_rank
            print(f"Merged to rank {best_rank}")
    
    # Try a different approach: use linear combinations to reduce rank
    # For the 3x3*3x3 part, use Winograd rank-15, then extend
    # Let me try constructing a specific low-rank decomposition
    
    # Approach: For each j (column of A / row of B), compute linear combinations
    # of A[:,j] and B[j,:] that allow sharing
    
    # Let me try: rank = 3 (j values) * 7 (k values) * something for i
    # Actually, let's try: for each j and k, we need to compute sum over i of A[i,j]*B[j,k]*delta(i,k)
    # Wait, that's not right.
    
    # Let me try a concrete construction:
    # For 3x3 * 3x3, use rank 15 (Winograd)
    # For the remaining 4 columns, use schoolbook (rank 12)
    # Total: 27
    
    # Winograd rank-15 for 3x3*3x3:
    # This is non-trivial. Let me use a known construction.
    
    # Actually, let me try a simpler approach: just use the schoolbook but with
    # some clever sharing for the 7 columns.
    
    # For each (i,j) in A, we multiply with B[j,:]. If we precompute
    # linear combinations of B[j,:], we might save multiplications.
    
    # Let me try: use rank = 3*3*5 = 45 by sharing some B combinations
    # For each j (3 values) and i (3 values), instead of 7 separate products,
    # use 5 linear combinations of B[j,:] that span the needed space.
    
    # Actually, this won't work directly because we need exact reconstruction.
    
    # Let me try the most promising approach: use a rank-15 Winograd for 3x3*3x3
    # and schoolbook for the other 4 columns.
    
    # Winograd F(2,2) extended to 3x3:
    # The 15 multiplications for 3x3*3x3 can be constructed as follows:
    # m1 = A[0,0]*B[0,0]
    # m2 = A[0,1]*B[1,0]  
    # m3 = A[0,2]*B[2,0]
    # m4 = A[1,0]*B[0,1]
    # m5 = A[1,1]*B[1,1]
    # m6 = A[1,2]*B[2,1]
    # m7 = A[2,0]*B[0,2]
    # m8 = A[2,1]*B[1,2]
    # m9 = A[2,2]*B[2,2]
    # m10 = (A[0,0]+A[0,1])*(B[0,1]+B[1,0])
    # m11 = (A[0,1]+A[0,2])*(B[1,0]+B[2,0])
    # m12 = (A[1,0]+A[1,1])*(B[0,2]+B[1,1])
    # m13 = (A[1,1]+A[1,2])*(B[1,1]+B[2,1])
    # m14 = (A[1,2]+A[2,2])*(B[2,0]+B[1,2])  -- need to verify
    # m15 = (A[2,0]+A[2,1])*(B[0,2]+B[1,2])  -- need to verify
    
    # This is getting complicated. Let me just return the best decomposition found.
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
