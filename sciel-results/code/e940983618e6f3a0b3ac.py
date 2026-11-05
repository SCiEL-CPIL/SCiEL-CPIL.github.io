import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Strategy: Block decomposition 6x6 = 2x2 blocks of 3x3 matrices
    # Outer: 2x2 Strassen (rank 7)
    # Inner: 3x3 schoolbook (rank 27)
    # Total rank: 7 * 27 = 189
    
    # 1. Build 3x3 schoolbook factors (rank 27)
    # U3, V3, W3 of size 9x27
    rank_3 = 27
    U3 = [[0]*rank_3 for _ in range(9)]
    V3 = [[0]*rank_3 for _ in range(9)]
    W3 = [[0]*rank_3 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3 + j][col] = 1
                V3[j*3 + k][col] = 1
                W3[k*3 + i][col] = 1
                col += 1
                
    # 2. Build 2x2 Strassen factors (rank 7)
    # U2, V2, W2 of size 4x7
    rank_2 = 7
    U2 = [[0]*rank_2 for _ in range(4)]
    V2 = [[0]*rank_2 for _ in range(4)]
    W2 = [[0]*rank_2 for _ in range(4)]
    
    # P1 = (A11+A22)*(B11+B22) -> C11, C22
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # P2 = (A21+A22)*B11 -> C21(+1), C22(-1)
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # P3 = A11*(B12-B22) -> C12(+1), C22(+1)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # P4 = A22*(B21-B11) -> C11(+1), C21(+1)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # P5 = (A11+A12)*B22 -> C12(+1), C11(-1)
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[2][4] = 1; W2[0][4] = -1
    
    # P6 = (A21-A11)*(B11+B12) -> C22(+1)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # P7 = (A12-A22)*(B21+B22) -> C11(+1)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # 3. Combine via tensor product
    total_rank = rank_2 * rank_3  # 189
    
    U = [[0] * total_rank for _ in range(36)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # Mapping:
    # i = i3*2 + i2, j = j3*2 + j2, k = k3*2 + k2
    # u_idx = i*6 + j
    # v_idx = j*6 + k
    # w_idx = k*6 + i
    
    for r3 in range(rank_3):
        for r2 in range(rank_2):
            r = r3 * rank_2 + r2
            
            for i3 in range(3):
                for j3 in range(3):
                    for k3 in range(3):
                        # U part
                        for i2 in range(2):
                            for j2 in range(2):
                                u_idx = (i3*2 + i2)*6 + (j3*2 + j2)
                                val_u = U3[i3*3 + j3][r3] * U2[i2*2 + j2][r2]
                                U[u_idx][r] = val_u
                                
                        # V part
                        for j2 in range(2):
                            for k2 in range(2):
                                v_idx = (j3*2 + j2)*6 + (k3*2 + k2)
                                val_v = V3[j3*3 + k3][r3] * V2[j2*2 + j2][r2]
                                V[v_idx][r] = val_v
                                
                        # W part
                        for k2 in range(2):
                            for i2 in range(2):
                                w_idx = (k3*2 + k2)*6 + (i3*2 + i2)
                                val_w = W3[k3*3 + i3][r3] * W2[k2*2 + i2][r2]
                                W[w_idx][r] = val_w

    # Verification (optional, but good for safety)
    if time.time() < deadline:
        try:
            # Quick spot check: verify a few entries
            # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] should be 1 if i,j,k form valid path
            # Actually for matrix mult tensor T[i,j,k] = 1 iff the indices are consistent
            # T[i,j,k] = delta_{i,k} * delta_{j,i}... no.
            # Matrix mult tensor: T[i,j,k] = 1 if we multiply A[i,j] by B[j,k] to get C[i,k]
            # So T[i,j,k] is 1 for all i,j,k? No.
            # The tensor is T_{i,j,k} = 1 for all i,j,k in range(n) is WRONG.
            # T_{i,j,k} = 1 iff the operation is defined. For matrix mult, all i,j,k are valid.
            # Wait, the tensor T has T_{i,j,k} = 1 for all i,j,k?
            # No, the matrix multiplication tensor T has T_{i,j,k} = 1 for all i,j,k?
            # Let's check: C_{ik} = sum_j A_{ij} B_{jk}
            # So the term A_{ij} B_{jk} contributes to C_{ik}.
            # Thus T_{i,j,k} = 1 for all i,j,k.
            # Yes, for square matrix multiplication, the tensor is all-ones.
            
            # Let's verify sum_r U[i,r]V[j,r]W[k,r] == 1 for a few (i,j,k)
            import random
            random.seed(42)
            for _ in range(100):
                i = random.randint(0, 5)
                j = random.randint(0, 5)
                k = random.randint(0, 5)
                s = 0
                for r in range(total_rank):
                    s += U[i*6+j][r] * V[j*6+k][r] * W[k*6+i][r]
                if s != 1:
                    # Fallback to schoolbook if verification fails
                    rank = 216
                    U_f = [[0] * rank for _ in range(36)]
                    V_f = [[0] * rank for _ in range(36)]
                    W_f = [[0] * rank for _ in range(36)]
                    for ii in range(6):
                        for jj in range(6):
                            for kk in range(6):
                                col = ii * 36 + jj * 6 + kk
                                U_f[ii * 6 + jj][col] = 1
                                V_f[jj * 6 + kk][col] = 1
                                W_f[kk * 6 + ii][col] = 1
                    return {"U": U_f, "V": V_f, "W": W_f}
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
