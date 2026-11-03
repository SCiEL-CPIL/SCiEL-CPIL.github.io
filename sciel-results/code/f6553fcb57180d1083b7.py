"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    start_time = time.time()
    deadline = start_time + 110

    # Build rank-42 Strassen decomposition
    # A: 2x4, B: 4x6, C: 2x6
    # C[i][k] = sum_j A[i][j]*B[j][k]
    # Split j into {0,1} and {2,3}, split k into {0,1},{2,3},{4,5}
    # 6 sub-blocks, each 7 Strassen mults = 42 total
    
    # Each rank-1 term: U[row_a, r] * U[row_b, r] ... 
    # Actually: C = sum_r (u_r . A) * (v_r . B) * w_r
    # where u_r is a vector of length n*m = 8 (indexed by i*m+j = i*4+j)
    # v_r is a vector of length m*p = 24 (indexed by j*p+k = j*6+k)
    # w_r is a vector of length p*n = 12 (indexed by k*n+i = k*2+i)
    
    # For each Strassen multiplication M = L_A * L_B:
    #   L_A = sum_{i,j} alpha_{i,j} * A[i][j]  => u_r[i*4+j] = alpha_{i,j}
    #   L_B = sum_{j,k} beta_{j,k} * B[j][k]   => v_r[j*6+k] = beta_{j,k}
    #   Contribution to C[i][k] is gamma_{i,k} * M => w_r[k*2+i] = gamma_{i,k}
    
    rank = 42
    U = np.zeros((n * m, rank))  # 8 x 42
    V = np.zeros((m * p, rank))  # 24 x 42
    W = np.zeros((p * n, rank))  # 12 x 42
    
    r = 0
    # j_offsets: (0,1) and (2,3); k_offsets: (0,1),(2,3),(4,5)
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            # A block: A[i][j0], A[i][j0+1] for i=0,1
            # B block: B[j0][k0], B[j0][k0+1], B[j0+1][k0], B[j0+1][k0+1]
            
            # a00=A[0][j0], a01=A[0][j0+1], a10=A[1][j0], a11=A[1][j0+1]
            # b00=B[j0][k0], b01=B[j0][k0+1], b10=B[j0+1][k0], b11=B[j0+1][k0+1]
            
            # M1 = (a00+a11)(b00+b11)
            # C contribution: C[0][k0] += M1, C[0][k0+1] += M1, C[1][k0] += M1, C[1][k0+1] += M1
            # Wait no. Let me re-derive.
            # C_block[0,0] = M1 + M4 - M5 + M7
            # C_block[0,1] = M3 + M5
            # C_block[1,0] = M2 + M4
            # C_block[1,1] = M1 - M2 + M3 + M6
            # So:
            # C[0][k0] = M1 + M4 - M5 + M7
            # C[0][k0+1] = M3 + M5
            # C[1][k0] = M2 + M4
            # C[1][k0+1] = M1 - M2 + M3 + M6
            
            # M1 = (a00+a11)(b00+b11)
            U[0*4+j0, r] = 1; U[1*4+j0+1, r] = 1  # a00 + a11
            V[j0*6+k0, r] = 1; V[(j0+1)*6+(k0+1), r] = 1  # b00 + b11
            W[k0*2+0, r] = 1; W[(k0+1)*2+0, r] = 1  # C[0][k0]+=1, C[0][k0+1]+=1
            W[k0*2+1, r] = 1; W[(k0+1)*2+1, r] = 1  # C[1][k0]+=1, C[1][k0+1]+=1
            r += 1
            
            # M2 = (a10+a11)*b00
            U[1*4+j0, r] = 1; U[1*4+j0+1, r] = 1  # a10 + a11
            V[j0*6+k0, r] = 1  # b00
            W[k0*2+1, r] = 1  # C[1][k0] += M2
            W[(k0+1)*2+1, r] = -1  # C[1][k0+1] -= M2
            r += 1
            
            # M3 = a00*(b01-b11)
            U[0*4+j0, r] = 1  # a00
            V[j0*6+(k0+1), r] = 1; V[(j0+1)*6+(k0+1), r] = -1  # b01 - b11
            W[(k0+1)*2+0, r] = 1  # C[0][k0+1] += M3
            W[(k0+1)*2+1, r] = 1  # C[1][k0+1] += M3
            r += 1
            
            # M4 = a11*(b10-b00)
            U[1*4+j0+1, r] = 1  # a11
            V[(j0+1)*6+k0, r] = 1; V[j0*6+k0, r] = -1  # b10 - b00
            W[k0*2+0, r] = 1  # C[0][k0] += M4
            W[k0*2+1, r] = 1  # C[1][k0] += M4
            r += 1
            
            # M5 = (a00+a01)*b11
            U[0*4+j0, r] = 1; U[0*4+j0+1, r] = 1  # a00 + a01
            V[(j0+1)*6+(k0+1), r] = 1  # b11
            W[k0*2+0, r] = -1  # C[0][k0] -= M5
            W[(k0+1)*2+0, r] = 1  # C[0][k0+1] += M5
            r += 1
            
            # M6 = (a10-a00)*(b00+b01)
            U[1*4+j0, r] = 1; U[0*4+j0, r] = -1  # a10 - a00
            V[j0*6+k0, r] = 1; V[j0*6+(k0+1), r] = 1  # b00 + b01
            W[(k0+1)*2+1, r] = 1  # C[1][k0+1] += M6
            r += 1
            
            # M7 = (a01-a11)*(b10+b11)
            U[0*4+j0+1, r] = 1; U[1*4+j0+1, r] = -1  # a01 - a11
            V[(j0+1)*6+k0, r] = 1; V[(j0+1)*6+(k0+1), r] = 1  # b10 + b11
            W[k0*2+0, r] = 1  # C[0][k0] += M7
            r += 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat, n, m, p, num_tests=50):
        for _ in range(num_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for r in range(U_mat.shape[1]):
                u_val = np.dot(U_mat[:, r], A.flatten())
                v_val = np.dot(V_mat[:, r], B.flatten())
                # w is indexed by k*n+i
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += u_val * v_val * W_mat[k*n+i, r]
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    if verify(U, V, W, n, m, p, num_tests=50):
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: trivial rank-48 decomposition
    rank_t = n * m * p
    U_t = [[0] * rank_t for _ in range(n * m)]
    V_t = [[0] * rank_t for _ in range(m * p)]
    W_t = [[0] * rank_t for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U_t[i * m + j][r] = 1
                V_t[j * p + k][r] = 1
                W_t[k * n + i][r] = 1
    return {"U": U_t, "V": V_t, "W": W_t}
# EVOLVE-BLOCK-END
