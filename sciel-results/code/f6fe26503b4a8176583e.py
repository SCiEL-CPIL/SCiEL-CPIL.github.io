import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0

    # Schoolbook fallback
    def get_schoolbook():
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

    # Strassen 2x2x2 decomposition (rank 7)
    # U indexed by (i,j), V by (j,k), W by (k,i) for 2x2x2
    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]  # indexed by i*2+j
        V = [[0]*7 for _ in range(4)]  # indexed by j*2+k
        W = [[0]*7 for _ in range(4)]  # indexed by k*2+i
        
        # M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1  # a00, a11
        V[0][0] = 1; V[3][0] = 1  # b00, b11
        W[0][0] = 1; W[3][0] = 1  # c00, c11 (k=0,i=0 and k=1,i=1)
        
        # M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1  # a10, a11
        V[0][1] = 1  # b00
        W[2][1] = 1  # c10 (k=0,i=1)
        
        # M3 = a00(b10-b11)
        U[0][2] = 1  # a00
        V[2][2] = 1; V[3][2] = -1  # b10, b11
        W[1][2] = 1  # c01 (k=1,i=0)
        
        # M4 = a11(b01-b00)
        U[3][3] = 1  # a11
        V[1][3] = 1; V[0][3] = -1  # b01, b00
        W[1][3] = 1; W[0][3] = -1  # c01, c00
        
        # M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1  # a00, a01
        V[3][4] = 1  # b11
        W[0][4] = 1; W[2][4] = 1  # c00, c10
        
        # M6 = (a10-a00)(b00+b10)
        U[2][5] = 1; U[0][5] = -1  # a10, a00
        V[0][5] = 1; V[2][5] = 1  # b00, b10
        W[3][5] = 1  # c11 (k=1,i=1)
        
        # M7 = (a01-a11)(b01+b11)
        U[1][6] = 1; U[3][6] = -1  # a01, a11
        V[1][6] = 1; V[3][6] = 1  # b01, b11
        W[1][6] = -1; W[3][6] = 1  # c01, c11
        
        return U, V, W

    # Verify Strassen decomposition
    def verify_2x2x2(U, V, W):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r] for r in range(7))
                    expected = 1 if j == k else 0
                    if abs(s - expected) > 1e-10:
                        return False
        return True

    # Build n x m x p tensor from a 2x2x2 base using block structure
    # Split n, m, p into blocks of size 2 (with padding if needed)
    def build_from_strassen(n, m, p):
        base_U, base_V, base_W = strassen_2x2x2()
        
        # Number of blocks in each dimension
        n_blocks = (n + 1) // 2
        m_blocks = (m + 1) // 2
        p_blocks = (p + 1) // 2
        
        rank = 7 * n_blocks * m_blocks * p_blocks
        
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        for ib in range(n_blocks):
            for jb in range(m_blocks):
                for kb in range(p_blocks):
                    # Local indices within the 2x2x2 block
                    i_local = 0 if ib < n_blocks - 1 else 0  # We only use i_local=0 for last block if odd
                    # Actually, let's be more careful. For each block position, we map
                    # global indices to local 2x2x2 indices.
                    
                    # For block (ib, jb, kb), the local 2x2x2 decomposition covers:
                    # i in [2*ib, 2*ib+1] if 2*ib+1 < n, else just [2*ib]
                    # j in [2*jb, 2*jb+1] if 2*jb+1 < m, else just [2*jb]
                    # k in [2*kb, 2*kb+1] if 2*kb+1 < p, else just [2*kb]
                    
                    # For simplicity and correctness, let's only use full 2x2x2 blocks
                    # where all three dimensions have at least 2 elements.
                    # For the last block in any dimension, we need to handle the single element.
                    
                    # This approach gets complex. Let's use a different strategy:
                    # Pad n, m, p to multiples of 2, build the tensor, then remove padded rows/cols.
                    pass
        
        # Alternative: Direct construction using the tensor structure
        # The matrix multiplication tensor T[n,m,p] has T[i,j,k] = 1 iff j == k
        # We can decompose this using the 2x2x2 Strassen decomposition as a building block.
        
        # For 4x5x6:
        # Split n=4 into 2+2, m=5 into 2+2+1, p=6 into 2+2+2
        # This gives blocks: 2x2x2, 2x2x2, 2x1x2, 2x2x2, 2x2x2, 2x1x2
        # But 2x1x2 is not standard. Let's use a different split.
        
        # Better: Use the fact that we can compose decompositions.
        # 4x5x6 = (2+2) x 5 x 6
        # = 2x5x6 + 2x5x6
        # And 2x5x6 can be built from 2x2x2 blocks in the p dimension (6 = 2+2+2)
        # So 2x5x6 needs 5*3 = 15 blocks of 2x2x2? No, that's not right either.
        
        # Let me think more carefully. The tensor T[i,j,k] = delta_{j,k}.
        # For 2x2x2, Strassen gives rank 7.
        # For 2x2x6, we split p into 3 blocks of 2, giving 3 copies of 2x2x2, rank 21.
        # For 4x2x6, we split n into 2 blocks of 2, giving 2 copies of 2x2x6, rank 42.
        # For 4x5x6, we split m into 2+2+1. The 1-wide block is tricky.
        
        # Actually, for m=5, we can split into 2+3, and 3 can be handled as 2+1.
        # A 2x1x2 block has T[i,0,k] = delta_{0,k}, which is just 2x1x2 with rank 2 (schoolbook).
        # So 2x3x2 = 2x2x2 + 2x1x2, rank 7 + 2 = 9.
        # Then 4x3x6 = 2*(2x3x6) = 2*(2x2x6 + 2x1x6) = 2*(21 + 6) = 54.
        # And 4x5x6 = 4x2x6 + 4x3x6 = 42 + 54 = 96.
        
        # That's better than 120 but not great. Let me try a different approach.
        
        # Actually, I realize the issue. Let me just construct the 4x5x6 tensor
        # by combining 2x2x2 Strassen blocks where possible and schoolbook for the rest.
        
        # For 4x5x6:
        # n=4: 2 blocks of 2
        # m=5: 2 blocks of 2, 1 block of 1
        # p=6: 3 blocks of 2
        
        # Full 2x2x2 blocks: 2*2*3 = 12 blocks, rank 12*7 = 84
        # 2x1x2 blocks: 2*1*3 = 6 blocks, rank 6*2 = 12 (schoolbook for 2x1x2)
        # Total rank: 84 + 12 = 96
        
        # Let's implement this properly.
        
        U = [[0]*96 for _ in range(n*m)]
        V = [[0]*96 for _ in range(m*p)]
        W = [[0]*96 for _ in range(p*n)]
        
        col = 0
        for ib in range(n_blocks):
            for jb in range(m_blocks):
                for kb in range(p_blocks):
                    i_start = 2 * ib
                    j_start = 2 * jb
                    k_start = 2 * kb
                    
                    i_end = min(i_start + 2, n)
                    j_end = min(j_start + 2, m)
                    k_end = min(k_start + 2, p)
                    
                    i_size = i_end - i_start
                    j_size = j_end - j_start
                    k_size = k_end - k_start
                    
                    if i_size == 2 and j_size == 2 and k_size == 2:
                        # Use Strassen 2x2x2
                        for r in range(7):
                            for ii in range(2):
                                for jj in range(2):
                                    U[(i_start + ii) * m + (j_start + jj)][col + r] = base_U[ii * 2 + jj][r]
                            for jj in range(2):
                                for kk in range(2):
                                    V[(j_start + jj) * p + (k_start + kk)][col + r] = base_V[jj * 2 + kk][r]
                            for kk in range(2):
                                for ii in range(2):
                                    W[(k_start + kk) * n + (i_start + ii)][col + r] = base_W[kk * 2 + ii][r]
                        col += 7
                    else:
                        # Schoolbook for this sub-block
                        for ii in range(i_size):
                            for jj in range(j_size):
                                for kk in range(k_size):
                                    U[(i_start + ii) * m + (j_start + jj)][col] = 1
                                    V[(j_start + jj) * p + (k_start + kk)][col] = 1
                                    W[(k_start + kk) * n + (i_start + ii)][col] = 1
                                    col += 1
        
        # Trim if col < 96
        if col < 96:
            U = [row[:col] for row in U]
            V = [row[:col] for row in V]
            W = [row[:col] for row in W]
        
        return U, V, W

    # Verify the full decomposition
    def verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    expected = 1 if j == k else 0
                    if abs(s - expected) > 1e-10:
                        return False
        return True

    # Try the Strassen-based construction
    try:
        U, V, W = build_from_strassen(n, m, p)
        if verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback to schoolbook
    return get_schoolbook()

# Test the Strassen decomposition
if __name__ == "__main__":
    U, V, W = strassen_2x2x2()
    assert verify_2x2x2(U, V, W), "Strassen decomposition failed!"
    print("Strassen 2x2x2 verified!")
    
    # Test with random matrices
    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = np.array([[5, 6], [7, 8]], dtype=float)
    C_expected = A @ B
    
    # Reconstruct using decomposition
    C_reconstructed = np.zeros((2, 2))
    for i in range(2):
        for k in range(2):
            for r in range(7):
                # Sum over j
                s = 0
                for j in range(2):
                    s += U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r]
                C_reconstructed[i][k] = s
    
    # This should give delta_{j,k} pattern, not the actual product
    # The tensor decomposition gives the bilinear form, not the product directly
    print("Tensor check passed if verify_2x2x2 returned True")
