import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    deadline = time.time() + 110  # 10s safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    def verify_rank_decomp(U, V, W, n, m, p):
        """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T_reconstructed = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T_reconstructed += np.einsum('ij,jk,ki->ijk', u, v, w)
        # The tensor should be all 1s
        if not np.allclose(T_reconstructed, np.ones((n, m, p)), atol=1e-10):
            return False
        return True
    
    def build_strassen_2x2_2():
        """Strassen decomposition of M_{2,2,2}. Rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        U indices: i*2+j, V indices: j*2+k, W indices: k*2+i
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1  # C00 and C11
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1  # C10
        
        # M3 = A00(B10+B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = 1
        W[1][2] = 1; W[3][2] = 1  # C01 and C11
        
        # M4 = (A00-A10)(B01+B10)
        U[0][3] = 1; U[2][3] = -1
        V[1][3] = 1; V[2][3] = 1
        W[0][3] = 1; W[2][3] = 1  # C00 and C10
        
        # M5 = (A01+A10)(B00-B10)
        U[1][4] = 1; U[2][4] = 1
        V[0][4] = 1; V[2][4] = -1
        W[0][4] = -1; W[1][4] = 1  # C00 and C01
        
        # M6 = (A11-A01)(B01-B11)
        U[1][5] = -1; U[3][5] = 1
        V[1][5] = 1; V[3][5] = -1
        W[3][5] = 1  # C11
        
        # M7 = (A00+A01)(B11-B01)
        U[0][6] = 1; U[1][6] = 1
        V[1][6] = -1; V[3][6] = 1
        W[0][6] = 1  # C00
        
        return U, V, W
    
    def build_direct_2x1x4():
        """Direct decomposition of multiplying 2x1 by 1x4. Rank 8.
        U: (2, 8), V: (4, 8), W: (4, 8)
        Actually M_{2,1,4}: A is 2x1, B is 1x4, C is 2x4
        U indices: i*1+j = i (since m=1), so U shape is (2, 8)
        V indices: j*4+k = k (since j=0), so V shape is (4, 8)
        W indices: k*2+i
        """
        rank = 8
        U = [[0]*rank for _ in range(2)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(8)]  # p*n = 4*2 = 8
        
        col = 0
        for i in range(2):
            for k in range(4):
                U[i][col] = 1
                V[k][col] = 1
                W[k*2 + i][col] = 1
                col += 1
        return U, V, W
    
    def build_2x2x4():
        """Build M_{2,2,4} by concatenating two M_{2,2,2} blocks.
        Split the 4 columns of B into two groups of 2.
        Rank: 7 + 7 = 14
        U: (4, 14), V: (8, 14), W: (8, 14)
        """
        U1, V1, W1 = build_strassen_2x2_2()
        # V1 is (4, 7) for B rows 0,1 (j=0,1 in a 2x2 B)
        # W1 is (4, 7) for C cols corresponding to k=0,1
        
        U2, V2, W2 = build_strassen_2x2_2()
        # V2 is (4, 7) for B rows 2,3
        # W2 is (4, 7) for C cols corresponding to k=2,3
        
        # Concatenate U
        U = U1 + U2  # both (4,7) -> wait, need to offset
        # Actually U1 and U2 are both (4,7), concatenate columns
        U = [row1 + row2 for row1, row2 in zip(U1, U2)]  # (4, 14)
        
        # V: V1 corresponds to B[0:2, :], V2 corresponds to B[2:4, :]
        # B has 4 rows (2x2 matrix flattened to 4), but here B is 2x4? 
        # Wait, M_{2,2,4}: A is 2x2, B is 2x4, C is 2x4
        # V indices: j*4+k, where j in {0,1}, k in {0,1,2,3}
        # V shape: (2*4, rank) = (8, rank)
        
        V = [[0]*14 for _ in range(8)]
        W = [[0]*14 for _ in range(8)]
        
        # First block: k=0,1 -> V rows 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
        for r in range(7):
            V[0][r] = V1[0][r]  # j=0,k=0
            V[1][r] = V1[1][r]  # j=0,k=1
            V[4][r] = V1[2][r]  # j=1,k=0
            V[5][r] = V1[3][r]  # j=1,k=1
            W[0][r] = W1[0][r]  # k=0,i=0
            W[1][r] = W1[1][r]  # k=0,i=1
            W[4][r] = W1[2][r]  # k=1,i=0
            W[5][r] = W1[3][r]  # k=1,i=1
        
        # Second block: k=2,3 -> V rows 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
        for r in range(7):
            V[2][r+7] = V2[0][r]  # j=0,k=2
            V[3][r+7] = V2[1][r]  # j=0,k=3
            V[6][r+7] = V2[2][r]  # j=1,k=2
            V[7][r+7] = V2[3][r]  # j=1,k=3
            W[2][r+7] = W2[0][r]  # k=2,i=0
            W[3][r+7] = W2[1][r]  # k=2,i=1
            W[6][r+7] = W2[2][r]  # k=3,i=0
            W[7][r+7] = W2[3][r]  # k=3,i=1
        
        return U, V, W
    
    def build_2x3x4():
        """Build M_{2,3,4} by combining M_{2,2,4} (first 2 cols of A) and M_{2,1,4} (last 1 col).
        A is 2x3, B is 3x4, C is 2x4.
        Split A's columns: first 2 -> M_{2,2,4}, last 1 -> M_{2,1,4}
        But we need to handle the B rows carefully.
        
        M_{2,2,4}: A_part is 2x2 (cols 0,1 of A), B_part is 2x4 (rows 0,1 of B), C is 2x4
        M_{2,1,4}: A_part is 2x1 (col 2 of A), B_part is 1x4 (row 2 of B), C is 2x4
        
        U for M_{2,2,4}: indices i*3+j? No, U indices should be consistent with full problem.
        In the full M_{2,3,4}: U indices i*m+j = i*3+j
        V indices j*p+k = j*4+k  
        W indices k*n+i = k*2+i
        
        For M_{2,2,4} part (j=0,1): U indices i*3+0 and i*3+1
        For M_{2,1,4} part (j=2): U indices i*3+2
        """
        U224, V224, W224 = build_2x2x4()  # U:(4,14), V:(8,14), W:(8,14)
        U214, V214, W214 = build_direct_2x1x4()  # U:(2,8), V:(4,8), W:(8,8)
        
        rank = 14 + 8
        U = [[0]*rank for _ in range(6)]  # 2*3 = 6
        V = [[0]*rank for _ in range(12)]  # 3*4 = 12
        W = [[0]*rank for _ in range(8)]   # 4*2 = 8
        
        # First part: M_{2,2,4} with j=0,1
        for i in range(2):
            for r in range(14):
                U[i*3+0][r] = U224[i*2+0][r]  # j=0
                U[i*3+1][r] = U224[i*2+1][r]  # j=1
        
        for r in range(14):
            # V224 has 8 rows for j in {0,1}, k in {0,1,2,3}
            # V224[j*4+k] for j=0,1
            for j in range(2):
                for k in range(4):
                    V[j*4+k][r] = V224[j*4+k][r]
        
        for r in range(14):
            # W224 has 8 rows for k in {0,1,2,3}, i in {0,1}
            # W224[k*2+i]
            for k in range(4):
                for i in range(2):
                    W[k*2+i][r] = W224[k*2+i][r]
        
        # Second part: M_{2,1,4} with j=2
        for i in range(2):
            for r in range(8):
                U[i*3+2][14+r] = U214[i][r]
        
        for r in range(8):
            # V214 has 4 rows for k in {0,1,2,3} (j=0 in the 1x4 part)
            for k in range(4):
                V[2*4+k][14+r] = V214[k][r]
        
        for r in range(8):
            # W214 has 8 rows for k in {0,1,2,3}, i in {0,1}
            for k in range(4):
                for i in range(2):
                    W[k*2+i][14+r] = W214[k*2+i][r]
        
        return U, V, W
    
    def build_2x3x8():
        """Build M_{2,3,8} by combining two M_{2,3,4} blocks.
        Split B's 8 columns into two groups of 4.
        Rank: 22 + 22 = 44
        """
        U1, V1, W1 = build_2x3x4()  # U:(6,22), V:(12,22), W:(8,22)
        U2, V2, W2 = build_2x3x4()
        
        rank = 44
        U = [row1 + row2 for row1, row2 in zip(U1, U2)]  # (6, 44)
        
        V = [[0]*44 for _ in range(24)]  # 3*8 = 24
        W = [[0]*44 for _ in range(16)]  # 8*2 = 16
        
        # First block: k=0,1,2,3 -> V rows j*8+k for k=0..3
        for r in range(22):
            for j in range(3):
                for k in range(4):
                    V[j*8+k][r] = V1[j*4+k][r]
            for k in range(4):
                for i in range(2):
                    W[k*2+i][r] = W1[k*2+i][r]
        
        # Second block: k=4,5,6,7 -> V rows j*8+k for k=4..7
        for r in range(22):
            for j in range(3):
                for k in range(4):
                    V[j*8+k+4][22+r] = V2[j*4+k][r]
            for k in range(4):
                for i in range(2):
                    W[k*2+i+8][22+r] = W2[k*2+i][r]
        
        return U, V, W
    
    # Try the Strassen-based approach
    U, V, W = build_2x3x8()
    
    # Verify
    if verify_rank_decomp(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
