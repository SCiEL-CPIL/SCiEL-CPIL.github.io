"""Matrix multiplication 2x3 x 3x9 with explicit construction."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 s safety margin

    n, m, p = DIMENSIONS  # 2, 3, 9
    rank_nm = n * m * p   # 54

    # ---- Fallback: schoolbook ----
    def schoolbook():
        r = rank_nm
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Try to build a rank-51 decomposition
    # Using 3 blocks of 2x3x3 with rank 17 each.
    # 2x3x3 split into:
    # 1. 2x2x2 (Strassen, rank 7)
    # 2. 2x1x1x2 (rank 4)
    # 3. 2x2x1 (rank 4)
    # 4. 2x1x1 (rank 2)
    # Total 17.
    
    def get_2x3x3_block():
        r = 17
        U = [[0] * r for _ in range(6)]
        V = [[0] * r for _ in range(9)]
        W = [[0] * r for _ in range(6)]
        
        # Strassen for 2x2x2 part (a00,a01,a10,a11) x (b00,b01,b10,b11)
        # Indices in U: 0,1,2,3
        # Indices in V: 0,1,2,3
        # Indices in W: 0,1,2,3 (c00,c01,c10,c11)
        
        # S1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
        
        # S2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # S3 = a00(b10-b11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # S4 = a11(b01-b00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = -1; W[1][3] = 1
        
        # S5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[3][4] = -1
        
        # S6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # S7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = -1; W[2][6] = 1
        
        # Part 2: a02*b20, a02*b21, a12*b20, a12*b21
        # U indices: 4 (a02), 5 (a12)
        # V indices: 6 (b20), 7 (b21)
        # W indices: 0 (c00), 1 (c01), 2 (c10), 3 (c11)
        
        U[4][7] = 1; V[6][7] = 1; W[0][7] = 1
        U[4][8] = 1; V[7][8] = 1; W[1][8] = 1
        U[5][9] = 1; V[6][9] = 1; W[2][9] = 1
        U[5][10] = 1; V[7][10] = 1; W[3][10] = 1
        
        # Part 3: a00*b02, a01*b12, a10*b02, a11*b12
        # U indices: 0,1,2,3
        # V indices: 4 (b02), 5 (b12)
        # W indices: 4 (c02), 5 (c12)
        
        U[0][11] = 1; V[4][11] = 1; W[4][11] = 1
        U[1][12] = 1; V[5][12] = 1; W[4][12] = 1
        U[2][13] = 1; V[4][13] = 1; W[5][13] = 1
        U[3][14] = 1; V[5][14] = 1; W[5][14] = 1
        
        # Part 4: a02*b22, a12*b22
        # U indices: 4,5
        # V indices: 8 (b22)
        # W indices: 4,5
        
        U[4][15] = 1; V[8][15] = 1; W[4][15] = 1
        U[5][16] = 1; V[8][16] = 1; W[5][16] = 1
        
        return U, V, W

    # Verify the block
    def verify_block(U, V, W):
        # Random test
        np.random.seed(42)
        A = np.random.rand(2, 3)
        B = np.random.rand(3, 3)
        C_expected = A @ B
        
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # C = U A V B W^T ? No, C[i,k] = sum_t U[i,j]_t V[j,k]_t W[k,i]_t ?
        # The convention is W[k*n+i, r].
        # C[i,k] = sum_j sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        
        C_actual = np.zeros((2, 3))
        for i in range(2):
            for k in range(3):
                s = 0
                for j in range(3):
                    for t in range(17):
                        s += U_np[i*3+j, t] * V_np[j*3+k, t] * W_np[k*2+i, t]
                C_actual[i, k] = s
        
        return np.allclose(C_actual, C_expected)

    U_block, V_block, W_block = get_2x3x3_block()
    if verify_block(U_block, V_block, W_block):
        # Tile 3 times
        r_total = 51
        U = [[0] * r_total for _ in range(6)]
        V = [[0] * r_total for _ in range(27)]
        W = [[0] * r_total for _ in range(18)]
        
        for b in range(3):
            offset = b * 17
            for i in range(6):
                for t in range(17):
                    U[i][offset + t] = U_block[i][t]
            
            for j in range(3):
                for k in range(3):
                    for t in range(17):
                        V[j*9 + k][offset + t] = V_block[j*3 + k][t]
            
            for k in range(3):
                for i in range(2):
                    for t in range(17):
                        W[k*2 + i][offset + t] = W_block[k*3 + i][t] # Wait, W_block is 6x17, indices 0..5.
                        # W_block index is k*n+i? No, W_block was defined for 2x3x3, so n=2, p=3.
                        # W_block row index is k*n+i = k*2+i.
                        # Here W is for 2x3x9, so n=2, p=9.
                        # W row index is k_total * n + i.
                        # k_total = b*3 + k.
                        
                        # Correct mapping:
                        # W_block[k*2+i] corresponds to local k.
                        # Global k = b*3 + k.
                        # Global W index = (b*3+k)*2 + i.
                        
                        # Let's re-do the tiling carefully.
        
        # Re-implement tiling
        U = [[0] * r_total for _ in range(6)]
        V = [[0] * r_total for _ in range(27)]
        W = [[0] * r_total for _ in range(18)]
        
        for b in range(3):
            offset = b * 17
            for i in range(6):
                for t in range(17):
                    U[i][offset + t] = U_block[i][t]
            
            for j in range(3):
                for k in range(3):
                    for t in range(17):
                        V[j*9 + k][offset + t] = V_block[j*3 + k][t]
            
            for k in range(3):
                for i in range(2):
                    for t in range(17):
                        W[(b*3+k)*2 + i][offset + t] = W_block[k*2 + i][t]
        
        # Final verification
        np.random.seed(43)
        A = np.random.rand(2, 3)
        B = np.random.rand(3, 9)
        C_expected = A @ B
        
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        C_actual = np.zeros((2, 9))
        for i in range(2):
            for k in range(9):
                s = 0
                for j in range(3):
                    for t in range(51):
                        s += U_np[i*3+j, t] * V_np[j*9+k, t] * W_np[k*2+i, t]
                C_actual[i, k] = s
        
        if np.allclose(C_actual, C_expected):
            return {"U": U, "V": V, "W": W}

    if time.time() > deadline - 5.0:
        return schoolbook()

    return schoolbook()


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
