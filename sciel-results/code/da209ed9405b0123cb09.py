"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Strassen 2x2x2 decomposition (rank 7)
    def get_strassen_222():
        U22 = np.zeros((4, 7))
        V22 = np.zeros((4, 7))
        W22 = np.zeros((4, 7))
        
        # M1 = (A11+A22)*(B11+B22) -> C11 + C22
        U22[0, 0] = 1; U22[3, 0] = 1
        V22[0, 0] = 1; V22[3, 0] = 1
        W22[0, 0] = 1; W22[3, 0] = 1
        
        # M2 = (A21+A22)*B11 -> C21
        U22[2, 1] = 1; U22[3, 1] = 1
        V22[0, 1] = 1
        W22[2, 1] = 1
        
        # M3 = A11*(B11-B21) -> C12 - C22
        U22[0, 2] = 1
        V22[0, 2] = 1; V22[2, 2] = -1
        W22[1, 2] = 1; W22[3, 2] = -1
        
        # M4 = A22*(B22-B12) -> -C11 + C12
        U22[3, 3] = 1
        V22[3, 3] = 1; V22[1, 3] = -1
        W22[0, 3] = -1; W22[1, 3] = 1
        
        # M5 = (A11+A12)*B22 -> C11 + C21
        U22[0, 4] = 1; U22[1, 4] = 1
        V22[3, 4] = 1
        W22[0, 4] = 1; W22[2, 4] = 1
        
        # M6 = (A21-A11)*(B11+B12) -> C22
        U22[2, 5] = 1; U22[0, 5] = -1
        V22[0, 5] = 1; V22[1, 5] = 1
        W22[3, 5] = 1
        
        # M7 = (A12-A22)*(B21+B22) -> C11 - C22
        U22[1, 6] = 1; U22[3, 6] = -1
        V22[2, 6] = 1; V22[3, 6] = 1
        W22[0, 6] = 1; W22[3, 6] = -1
        
        return U22, V22, W22
    
    def verify_decomp(U, V, W, n, m, p):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Direct construction: place 2x2x2 Strassen blocks optimally
    # i-dim (4): blocks at i=0 and i=2
    # j-dim (5): blocks at j=0 and j=3 (leaving j=2 uncovered)
    # k-dim (6): blocks at k=0, k=2, k=4
    # Total: 2*2*3 = 12 blocks
    
    block_positions = []
    for i0 in [0, 2]:
        for j0 in [0, 3]:
            for k0 in [0, 2, 4]:
                block_positions.append((i0, j0, k0))
    
    num_blocks = len(block_positions)
    strassen_rank_per_block = 7
    total_strassen_rank = num_blocks * strassen_rank_per_block
    
    # Determine covered entries
    covered = set()
    for (i0, j0, k0) in block_positions:
        for i in [i0, i0+1]:
            for j in [j0, j0+1]:
                for k in [k0, k0+1]:
                    covered.add((i, j, k))
    
    num_covered = len(covered)
    num_schoolbook = n * m * p - num_covered
    total_rank = total_strassen_rank + num_schoolbook
    
    # Build the decomposition
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Add Strassen blocks
    for (i0, j0, k0) in block_positions:
        U22, V22, W22 = get_strassen_222()
        
        for r in range(strassen_rank_per_block):
            # U: row index i*m + j, local index (i-i0)*2 + (j-j0)
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    val = U22[local_idx, r]
                    if val != 0:
                        i = i0 + li
                        j = j0 + lj
                        U[i * m + j, col + r] = val
            
            # V: row index j*p + k, local index (j-j0)*2 + (k-k0)
            for lj in range(2):
                for lk in range(2):
                    local_idx = lj * 2 + lk
                    val = V22[local_idx, r]
                    if val != 0:
                        j = j0 + lj
                        k = k0 + lk
                        V[j * p + k, col + r] = val
            
            # W: row index k*n + i, local index (k-k0)*2 + (i-i0)
            for lk in range(2):
                for li in range(2):
                    local_idx = lk * 2 + li
                    val = W22[local_idx, r]
                    if val != 0:
                        k = k0 + lk
                        i = i0 + li
                        W[k * n + i, col + r] = val
        
        col += strassen_rank_per_block
    
    # Add schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    # Verify the full decomposition
    if not verify_decomp(U, V, W, n, m, p):
        # Fall back to pure schoolbook
        schoolbook_rank = n * m * p
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
