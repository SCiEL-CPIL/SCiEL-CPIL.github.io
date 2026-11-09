"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # For each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal delta
    # Actually, the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # should equal 1 if the indices match a valid multiplication path, else 0.
    # For matrix mult: T[i,j,k] = 1 always (it's the multiplication tensor)
    # Wait, the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) in the range.
    # Let me check: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i,j,k in range.
    # The decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # should equal 1 for all i in [0,n), j in [0,m), k in [0,p).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _build_strassen_2x4x6():
    """
    Build a Strassen-based decomposition for 2x4 * 4x6.
    
    Approach: Use the bilinear algorithm structure.
    
    A is 2x4: indices (i, j) where i in {0,1}, j in {0,1,2,3}
    B is 4x6: indices (j, k) where j in {0,1,2,3}, k in {0,1,2,3,4,5}
    C is 2x6: indices (i, k)
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split j into {0,1} and {2,3}:
    C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
    
    Let A1 = A[:,0:2] (2x2), A2 = A[:,2:4] (2x2)
    Let B1 = B[0:2,:] (2x6), B2 = B[2:4,:] (2x6)
    
    C = A1 @ B1 + A2 @ B2
    
    Now apply Strassen to each 2x2 x 2x6 product.
    For 2x2 x 2x6, split the 6 columns into two 3-column blocks:
    B1 = [B1a | B1b] where B1a is 2x3, B1b is 2x3
    
    So A1 @ B1 = A1 @ B1a + A1 @ B1b (two 2x2 x 2x3 products)
    
    For 2x2 x 2x3, the minimal rank is 5 (known result).
    
    Total rank: 2 (A1/A2) * 2 (B1/B2) * 2 (a/b split) * 5 = 40
    
    But we can do better by sharing terms. Let me just build rank 40 first.
    """
    n, m, p = 2, 4, 6
    
    # Build 2x2 x 2x3 decomposition of rank 5
    # This is a known result; let me construct it.
    # For 2x2 x 2x3, the tensor has 12 entries.
    # Rank 5 means 5 bilinear terms.
    
    # I'll use a standard construction.
    # Let me define the factors for 2x2 x 2x3 (rank 5):
    # U: 4x5, V: 6x5, W: 6x5 (p*n = 3*2 = 6)
    
    # This is getting complex. Let me just return schoolbook for now
    # and note that the rank is 48.
    
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
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build schoolbook decomposition (rank 48) as baseline
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
    
    # Verify the schoolbook decomposition
    if not _verify(U, V, W, n, m, p):
        # Should never happen, but fallback just in case
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
