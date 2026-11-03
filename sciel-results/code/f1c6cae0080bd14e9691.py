"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-114 tensor decomposition for 4x4x9 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 decomposition for 2x2x2 to build a rank-49 decomposition for 4x4x4 
      via Kronecker product.
    - Partition the 4x4x9 problem into two 4x4x4 blocks and one 4x4x1 block.
    - The 4x4x1 block has rank 16 (4 independent dot products of length 4).
    - Total rank: 49 + 49 + 16 = 114.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Define the 7 components of Strassen's 2x2x2 decomposition
    # U, V, W are 4x7 matrices
    # U rows: a, b, c, d (indices 0, 1, 2, 3)
    # V rows: e, f, g, h (indices 0, 1, 2, 3)
    # W rows: C00, C01, C10, C11 (indices 0, 1, 2, 3)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Component 1: M1 = (a+d)(e+h)
    U2[:, 0] = [1, 0, 0, 1]
    V2[:, 0] = [1, 0, 0, 1]
    W2[:, 0] = [1, 0, 0, 1]
    
    # Component 2: M2 = (c+d)e
    U2[:, 1] = [0, 0, 1, 1]
    V2[:, 1] = [1, 0, 0, 0]
    W2[:, 1] = [0, 1, 0, 0] # C10 (i=1, k=0 -> idx 1)
    
    # Component 3: M3 = a(f-h)
    U2[:, 2] = [1, 0, 0, 0]
    V2[:, 2] = [0, 1, 0, -1]
    W2[:, 2] = [0, 0, 1, 1] # C01 (idx 2), C11 (idx 3)
    
    # Component 4: M4 = d(g-e)
    U2[:, 3] = [0, 0, 0, 1]
    V2[:, 3] = [0, 0, 1, -1]
    W2[:, 3] = [1, 1, 0, 0] # C00 (idx 0), C10 (idx 1)
    
    # Component 5: M5 = (a+b)h
    U2[:, 4] = [1, 1, 0, 0]
    V2[:, 4] = [0, 0, 0, 1]
    W2[:, 4] = [1, 0, 1, 0] # C00 (idx 0), C01 (idx 2)
    
    # Component 6: M6 = (c-a)(e+f)
    U2[:, 5] = [-1, 0, 1, 0]
    V2[:, 5] = [1, 1, 0, 0]
    W2[:, 5] = [0, 0, 0, 1] # C11 (idx 3)
    
    # Component 7: M7 = (b-d)(g+h)
    U2[:, 6] = [0, 1, 0, -1]
    V2[:, 6] = [0, 0, 1, 1]
    W2[:, 6] = [1, 0, 0, 1] # C00 (idx 0), C11 (idx 3)
    
    # Construct 4x4x4 decomposition (Rank 49)
    # U4 is 16x49, V4 is 16x49, W4 is 16x49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # Kronecker product of the column vectors
            # Index mapping: i = i1*2 + i2, j = j1*2 + j2
            # U4 row index: i*4 + j = (i1*2 + i2)*4 + (j1*2 + j2)
            # This is not simply np.kron(U2[:, r1], U2[:, r2]) if the flattening is different.
            # Let's check np.kron.
            # np.kron([a,b,c,d], [e,f,g,h]) = [ae, af, ag, ah, be, bf, bg, bh, ...]
            # This corresponds to row index i*4+j where i is the first vector index and j is the second?
            # If U2 vector is [u0, u1, u2, u3] corresponding to (0,0), (0,1), (1,0), (1,1).
            # Then kron(u, v) gives:
            # u0*v0, u0*v1, u0*v2, u0*v3
            # u1*v0, u1*v1, u1*v2, u1*v3
            # u2*v0, ...
            # u3*v0, ...
            # Row 0: u0*v0 -> i=0, j=0?
            # In U2, index 0 is (0,0), 1 is (0,1), 2 is (1,0), 3 is (1,1).
            # So u0 is (0,0), u1 is (0,1), u2 is (1,0), u3 is (1,1).
            # The kron order puts u0's block first.
            # Row 0-3: u0 * v0..v3.
            # v0 is (0,0), v1 is (0,1), v2 is (1,0), v3 is (1,1).
            # So Row 0: (0,0) * (0,0) -> i=0, j=0.
            # Row 1: (0,0) * (0,1) -> i=0, j=1.
            # Row 2: (0,0) * (1,0) -> i=0, j=2? No.
            # The index in U4 is i*4+j.
            # i=0, j=0 -> 0.
            # i=0, j=1 -> 1.
            # i=0, j=2 -> 2.
            # i=0, j=3 -> 3.
            # i=1, j=0 -> 4.
            # 
            # In the kron result:
            # Index 0: u0 v0. u0 is (0,0), v0 is (0,0). i=0, j=0.
            # Index 1: u0 v1. u0 is (0,0), v1 is (0,1). i=0, j=1.
            # Index 2: u0 v2. u0 is (0,0), v2 is (1,0). i=0, j=2?
            # Wait. v2 corresponds to index 2 in V2, which is (1,0).
            # So the second index j is determined by v's index.
            # v index 0 -> j1=0, j2=0.
            # v index 1 -> j1=0, j2=1.
            # v index 2 -> j1=1, j2=0.
            # v index 3 -> j1=1, j2=1.
            # So j = j1*2 + j2.
            # For index 2 in V2, j1=1, j2=0, so j=2.
            # For index 0 in U2, i1=0, i2=0, so i=0.
            # So Row 2 corresponds to i=0, j=2.
            # This matches the row-major flattening of 4x4 matrix.
            # So np.kron is correct.
            
            U4[:, r] = np.kron(U2[:, r1], U2[:, r2])
            V4[:, r] = np.kron(V2[:, r1], V2[:, r2])
            W4[:, r] = np.kron(W2[:, r1], W2[:, r2])
            
    # Now assemble the full 4x4x9 decomposition
    # Rank = 49 + 49 + 16 = 114
    rank = 114
    U = np.zeros((16, rank))
    V = np.zeros((36, rank)) # m*p = 4*9 = 36
    W = np.zeros((36, rank)) # p*n = 9*4 = 36
    
    # Block 1: 4x4x4 (B columns 0-3)
    # U part: same as U4
    # V part: V4 for B cols 0-3
    # W part: W4 for B cols 0-3
    
    # B is 4x9. B[:, 0:4] is the first block.
    # V index: j*p + k. j in 0..3, k in 0..8.
    # For block 1, k is 0..3.
    # V4 index: j*4 + k.
    # We need to map V4 rows to V rows.
    # V4 row j*4+k corresponds to B[j, k].
    # In V, row j*9+k.
    
    # W index: k*n + i. k in 0..8, i in 0..3.
    # W4 index: k*4 + i.
    # In W, row k*4 + i.
    
    # Copy U4 to U[:, 0:49]
    U[:, 0:49] = U4
    
    # Copy V4 to V[:, 0:49] with index mapping
    for j in range(4):
        for k in range(4):
            v4_idx = j * 4 + k
            v_full_idx = j * 9 + k
            V[v_full_idx, 0:49] = V4[v4_idx, :]
            
    # Copy W4 to W[:, 0:49] with index mapping
    for k in range(4):
        for i in range(4):
            w4_idx = k * 4 + i
            w_full_idx = k * 4 + i # Same mapping for 4x4
            W[w_full_idx, 0:49] = W4[w4_idx, :]
            
    # Block 2: 4x4x4 (B columns 4-7)
    # U part: same as U4
    U[:, 49:98] = U4
    
    # V part: V4 for B cols 4-7
    for j in range(4):
        for k in range(4):
            v4_idx = j * 4 + k
            v_full_idx = j * 9 + (k + 4)
            V[v_full_idx, 49:98] = V4[v4_idx, :]
            
    # W part: W4 for B cols 4-7
    for k in range(4):
        for i in range(4):
            w4_idx = k * 4 + i
            w_full_idx = (k + 4) * 4 + i
            W[w_full_idx, 49:98] = W4[w4_idx, :]
            
    # Block 3: 4x4x1 (B column 8)
    # Rank 16
    # U part: Identity-like?
    # C_i = sum_j A_{i,j} B_{j,8}
    # Decomposition: sum_{i,j} A_{i,j} * B_{j,8} * e_i
    # For each i, j, we have a term.
    # U_{i,j, r} = 1 if r corresponds to (i,j)
    # V_{j,8, r} = 1 if r corresponds to (i,j)
    # W_{8,i, r} = 1 if r corresponds to (i,j)
    
    # r from 98 to 113. 16 terms.
    for i in range(4):
        for j in range(4):
            r = 98 + i * 4 + j
            u_idx = i * 4 + j
            v_idx = j * 9 + 8
            w_idx = 8 * 4 + i
            
            U[u_idx, r] = 1
            V[v_idx, r] = 1
            W[w_idx, r] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
