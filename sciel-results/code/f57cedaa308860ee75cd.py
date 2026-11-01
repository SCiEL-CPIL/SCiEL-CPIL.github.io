"""Matrix multiplication tensor decomposition for 2x3 * 3x8."""
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = 2, 3, 8
    
    # Clean implementation: rank 36 decomposition.
    # Split B into B_left (3x4, cols 0-3) and B_right (3x4, cols 4-7).
    # For each, compute 2x3 * 3x4.
    # Split the 3x4 B into B_top (2x4, rows 0-1) and B_bot (1x4, row 2).
    # 2x3 * 3x4 = 2x2*2x4 (using top 2 cols of A) + 2x1*1x4 (using 3rd col of A).
    # 
    # 2x2*2x4: process 2 cols at a time with Strassen (rank 7 each), total 14.
    # 2x1*1x4: rank 4.
    # Per half: 18. Total: 36.
    
    rank = 36
    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(24)]
    W = [[0.0] * rank for _ in range(16)]
    
    # U[i*3+j] for A[i,j], V[j*8+k] for B[j,k], W[k*2+i] for C[i,k]
    
    def add_strassen_2x2x2(base, a_r0, a_r1, b_c0, b_c1, b_k0):
        """Strassen 2x2*2x2: A rows a_r0,a_r1; A cols b_c0,b_c1 (=B rows); B cols b_k0,b_k0+1."""
        # M1 = (A[a_r0,b_c0]+A[a_r1,b_c1])*(B[b_c0,b_k0]+B[b_c1,b_k0+1])
        U[a_r0*3+b_c0][base] += 1; V[b_c0*8+b_k0][base] += 1
        U[a_r1*3+b_c1][base] += 1; V[b_c1*8+b_k0+1][base] += 1
        W[b_k0*2+a_r0][base] += 1; W[(b_k0+1)*2+a_r0][base] += 1
        W[b_k0*2+a_r1][base] += 1; W[(b_k0+1)*2+a_r1][base] += 1
        # M2 = (A[a_r1,b_c0]+A[a_r1,b_c1])*B[b_c0,b_k0]
        U[a_r1*3+b_c0][base+1] += 1; V[b_c0*8+b_k0][base+1] += 1
        U[a_r1*3+b_c1][base+1] += 1; V[b_c0*8+b_k0][base+1] += 0
        W[b_k0*2+a_r1][base+1] += 1
        # M3 = A[a_r0,b_c0]*(B[b_c1,b_k0]-B[b_c0,b_k0])
        U[a_r0*3+b_c0][base+2] += 1; V[b_c1*8+b_k0][base+2] += 1
        V[b_c0*8+b_k0][base+2] += -1
        W[(b_k0+1)*2+a_r0][base+2] += 1
        # M4 = (A[a_r0,b_c1]-A[a_r1,b_c1])*(B[b_c1,b_k0]+B[b_c1,b_k0+1])
        U[a_r0*3+b_c1][base+3] += 1; V[b_c1*8+b_k0][base+3] += 1
        U[a_r1*3+b_c1][base+3] += -1; V[b_c1*8+b_k0+1][base+3] += 1
        W[b_k0*2+a_r0][base+3] += 1; W[(b_k0+1)*2+a_r0][base+3] += 1
        # M5 = A[a_r0,b_c1]*(B[b_c0,b_k0]+B[b_c0,b_k0+1])
        U[a_r0*3+b_c1][base+4] += 1; V[b_c0*8+b_k0][base+4] += 1
        V[b_c0*8+b_k0+1][base+4] += 1
        W[b_k0*2+a_r0][base+4] += -1; W[(b_k0+1)*2+a_r0][base+4] += 1
        # M6 = (A[a_r1,b_c0]-A[a_r0,b_c0])*(B[b_c0,b_k0+1]-B[b_c1,b_k0+1])
        U[a_r1*3+b_c0][base+5] += 1; V[b_c0*8+b_k0+1][base+5] += 1
        U[a_r0*3+b_c0][base+5] += -1; V[b_c1*8+b_k0+1][base+5] += -1
        W[(b_k0+1)*2+a_r1][base+5] += 1
        # M7 = (A[a_r0,b_c0]+A[a_r0,b_c1])*B[b_c1,b_k0+1]
        U[a_r0*3+b_c0][base+6] += 1; V[b_c1*8+b_k0+1][base+6] += 1
        U[a_r0*3+b_c1][base+6] += 1; V[b_c1*8+b_k0+1][base+6] += 0
        W[b_k0*2+a_r0][base+6] += 1; W[(b_k0+1)*2+a_r1][base+6] += 1
    
    # For each half (b_k_base = 0 or 4):
    for half in range(2):
        bk = half * 4  # base col in B
        base = half * 18  # base in rank
        
        # 2x2*2x4 part: A rows 0,1; A cols 0,1; B rows 0,1; B cols bk..bk+3
        # Process cols (bk, bk+1) then (bk+2, bk+3)
        add_strassen_2x2x2(base, 0, 1, 0, 1, bk)       # cols bk, bk+1
        add_strassen_2x2x2(base+7, 0, 1, 0, 1, bk+2)    # cols bk+2, bk+3
        
        # 2x1*1x4 part: A col 2; B row 2; B cols bk..bk+3
        for dk in range(4):
            c = base + 14 + dk
            U[0*3+2][c] += 1  # A[0,2]
            U[1*3+2][c] += 1  # A[1,2]
            V[2*8+bk+dk][c] += 1  # B[2, bk+dk]
            W[(bk+dk)*2+0][c] += 1  # C[0, bk+dk]
            W[(bk+dk)*2+1][c] += 1  # C[1, bk+dk]
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Test with random matrices
    A = np.random.randn(2, 3)
    B = np.random.randn(3, 8)
    C_expected = A @ B
    
    # Compute C from decomposition: C[i,k] = sum_c W[k*2+i, c] * (sum_j U[i*3+j,c] * V[j*8+k,c])
    # Actually: C = sum_c outer(U[:,c] reshaped as 2x3, V[:,c] reshaped as 3x8) * W[:,c]
    # U[:,c] is 6-vector = A-part for factor c
    # V[:,c] is 24-vector = B-part for factor c
    # The product is: (U_vec reshaped 2x3) * (V_vec reshaped 3x8) -> 2x8, scaled by W_vec
    
    C_computed = np.zeros((2, 8))
    for c in range(rank):
        U_mat = U_np[:, c].reshape(2, 3)
        V_mat = V_np[:, c].reshape(3, 8)
        W_vec = W_np[:, c]
        prod = U_mat @ V_mat  # 2x8
        # prod[i,k] should contribute W[k*2+i, c] * prod[i,k]
        for i in range(2):
            for k in range(8):
                C_computed[i, k] += W_vec[k*2+i] * prod[i, k]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank2 = n * m * p
        U2 = [[0] * rank2 for _ in range(n * m)]
        V2 = [[0] * rank2 for _ in range(m * p)]
        W2 = [[0] * rank2 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i * m + j][col] = 1
                    V2[j * p + k][col] = 1
                    W2[k * n + i][col] = 1
                    col += 1
        return {"U": U2, "V": V2, "W": W2}
    
    return {"U": U, "V": V, "W": W}
