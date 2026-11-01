DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Use block decomposition: partition middle dimension into two groups of 2
    # A is 2x4, B is 4x7, C is 2x7
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # Each sub-problem is 2x2 @ 2x7, which has rank 2*7 = 14
    # Total rank = 2 * 14 = 28
    
    # But wait, let me verify the rank of 2x2 @ 2xp
    # The tensor T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,...,p-1}
    # T[i,j,k] = delta_{i,j} ... no, T[i*m+j, j*p+k, k*n+i] = 1
    # For 2x2 @ 2xp: n=2, m=2, p=7
    # T[i*2+j, j*7+k, k*2+i] = 1 for i in {0,1}, j in {0,1}, k in {0..6}
    # That's 2*2*7 = 28 terms naively.
    
    # Can we do better? The rank of (2,2,p) MM tensor...
    # For (2,2,2) it's 7 (Strassen). For (2,2,p) it's known to be 2p for p>=3? 
    # Actually I'm not sure. Let me just use the naive 28 for each block, total 56.
    # That's not an improvement.
    
    # Let me try a different partition. What if I use Strassen-like tricks?
    
    # Alternative: use the fact that (2,4,7) can be decomposed using
    # known results. The rank of (2,4,7) MM is at most...
    
    # Let me just implement the naive decomposition and add a simple local search
    # to try to reduce rank.
    
    import time
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with naive rank 56
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
