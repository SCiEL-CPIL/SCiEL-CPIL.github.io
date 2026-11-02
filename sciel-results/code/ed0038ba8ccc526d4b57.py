"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 120.0

def construct():
    # Strassen-based decomposition: apply rank-7 2x2 matmul to each 2x2 block of the 2x10 matrix.
    # Total rank = 5 * 7 = 35 (vs schoolbook 40)
    n, m, p = DIMENSIONS  # 2, 2, 10
    
    # Time guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5:
        # Return schoolbook as safe fallback
        rank_s = n * m * p
        U_s = [[0] * rank_s for _ in range(n * m)]
        V_s = [[0] * rank_s for _ in range(m * p)]
        W_s = [[0] * rank_s for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_s[i * m + j][r] = 1
                    V_s[j * p + k][r] = 1
                    W_s[k * n + i][r] = 1
        return {"U": U_s, "V": V_s, "W": W_s}
    num_blocks = p // 2  # 5 blocks of 2 columns each
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0] * rank for _ in range(p * n)]   # 20 x 35
    
    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each multiplication Mu, we need:
    # U side: coefficients on A elements (rows indexed by i*m+j)
    # V side: coefficients on B elements (rows indexed by j*p+k)  
    # W side: coefficients on C elements (rows indexed by k*n+i)
    
    # U row indices: A00=0, A01=1, A10=2, A11=3
    # V row indices for block b: B00=0*10+2b, B01=0*10+2b+1, B10=1*10+2b, B11=1*10+2b+1
    # W row indices for block b: C00=2b*2+0, C01=2b+1*2+0, C10=2b*2+1, C11=2b+1*2+1
    
    # Define the 7 Strassen terms: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict mapping (i,j) -> coeff for A[i][j]
    # V_coeffs: dict mapping (j,k_offset) -> coeff for B[j][k] in this block
    # W_coeffs: dict mapping (k_offset, i) -> coeff for C[i][k] in this block
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        (
            {(0,0): 1, (1,1): 1},   # U: A00 + A11
            {(0,0): 1, (1,1): 1},   # V: B00 + B11 (k_offset 0,1)
            {(0,0): 1, (1,0): 1, (0,1): 1, (1,1): -1}  # W: C00 + C10 - C01 + C11... 
            # Wait, let me redo. C00 = M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
            # So M1 contributes: C00 += 1, C11 += 1
            # Let me restructure.
        ),
    ]
    
    # Let me be more careful. For each multiplication Mu, the W side encodes how Mu contributes to C.
    # M1: C00 += M1, C11 += M1  => W: C00:1, C11:1
    # M2: C10 += M2, C11 -= M2  => W: C10:1, C11:-1
    # M3: C01 += M3, C11 += M3  => W: C01:1, C11:1
    # M4: C00 += M4, C10 += M4  => W: C00:1, C10:1
    # M5: C01 += M5, C00 -= M5  => W: C01:1, C00:-1
    # M6: C11 += M6              => W: C11:1
    # M7: C00 += M7              => W: C00:1
    
    # So each term is: (U_side, V_side, W_side) where each side is a list of (row_index, coeff)
    # U rows: 0=A00, 1=A01, 2=A10, 3=A11
    # V rows (within block): 0=B00, 1=B01, 2=B10, 3=B11  (actual V row = j*10 + 2b + k_offset)
    # W rows (within block): 0=C[0][2b], 1=C[0][2b+1], 2=C[1][2b], 3=C[1][2b+1]
    #   actual W row = (2b+k_offset)*2 + i
    
    strassen_decomp = [
        # M1 = (A00+A11) * (B00+B11)
        (
            [(0, 1), (3, 1)],       # U: A00 + A11
            [(0, 1), (3, 1)],       # V: B00 + B11 (k_off 0 and 1)
            [(0, 1), (3, 1)],       # W: C[0][2b]:1, C[1][2b+1]:1  => rows (2b)*2+0 and (2b+1)*2+1
        ),
        # M2 = (A10+A11) * B00
        (
            [(2, 1), (3, 1)],       # U: A10 + A11
            [(0, 1)],               # V: B00
            [(2, 1), (3, -1)],      # W: C[1][2b]:1, C[1][2b+1]:-1
        ),
        # M3 = A00 * (B01-B11)
        (
            [(0, 1)],               # U: A00
            [(1, 1), (3, -1)],      # V: B01 - B11
            [(1, 1), (3, 1)],       # W: C[0][2b+1]:1, C[1][2b+1]:1
        ),
        # M4 = A11 * (B10-B00)
        (
            [(3, 1)],               # U: A11
            [(2, 1), (0, -1)],      # V: B10 - B00
            [(0, 1), (2, 1)],       # W: C[0][2b]:1, C[1][2b]:1
        ),
        # M5 = (A00+A01) * B11
        (
            [(0, 1), (1, 1)],       # U: A00 + A01
            [(3, 1)],               # V: B11
            [(1, 1), (0, -1)],      # W: C[0][2b+1]:1, C[0][2b]:-1
        ),
        # M6 = (A10-A00) * (B00+B01)
        (
            [(2, 1), (0, -1)],      # U: A10 - A00
            [(0, 1), (1, 1)],       # V: B00 + B01
            [(3, 1)],               # W: C[1][2b+1]:1
        ),
        # M7 = (A01-A11) * (B10+B11)
        (
            [(1, 1), (3, -1)],      # U: A01 - A11
            [(2, 1), (3, 1)],       # V: B10 + B11
            [(0, 1)],               # W: C[0][2b]:1
        ),
    ]
    
    for b in range(num_blocks):
        base_col = b * 7
        for t in range(7):
            col = base_col + t
            u_side, v_side, w_side = strassen_decomp[t]
            
            for u_row, u_coeff in u_side:
                U[u_row][col] = u_coeff
            
            for v_koff, v_coeff in v_side:
                # v_koff: 0=B00, 1=B01, 2=B10, 3=B11
                # j = v_koff // 2, k = 2*b + v_koff % 2
                j = v_koff // 2
                k = 2 * b + v_koff % 2
                V[j * p + k][col] = v_coeff
            
            for w_koff, w_coeff in w_side:
                # w_koff: 0=C[0][2b], 1=C[0][2b+1], 2=C[1][2b], 3=C[1][2b+1]
                # i = w_koff // 2, k = 2*b + w_koff % 2
                i = w_koff // 2
                k = 2 * b + w_koff % 2
                W[k * n + i][col] = w_coeff
    
    # Verify correctness with numpy
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for random A, B, compute C = A*B and compare with tensor decomposition
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Reconstruct C: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    C_reconstructed = np.zeros((n, p))
    for r in range(rank):
        u_r = U_np[:, r].reshape(n, m)
        v_r = V_np[:, r].reshape(m, p)
        w_r = W_np[:, r].reshape(p, n)
        # Rank-1 contribution: for each (i,j,k), u_r[i,j]*v_r[j,k]*w_r[k,i]
        # Sum over j: C[i,k] += sum_j u_r[i,j]*v_r[j,k]*w_r[k,i]
        # = (u_r @ v_r)[i,k] * w_r[k,i]... no, w_r[k,i] depends on i and k
        # C[i,k] += w_r[k,i] * sum_j u_r[i,j] * v_r[j,k]
        # = w_r[k,i] * (u_r @ v_r)[i,k]
        uv = u_r @ v_r  # shape (n, p)
        # w_r is (p, n), so w_r[k,i] = w_r.T[i,k]
        C_reconstructed += uv * w_r.T
    
    if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
        # Fallback: return schoolbook decomposition (rank 40)
        rank_s = n * m * p
        U_s = [[0] * rank_s for _ in range(n * m)]
        V_s = [[0] * rank_s for _ in range(m * p)]
        W_s = [[0] * rank_s for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_s[i * m + j][r] = 1
                    V_s[j * p + k][r] = 1
                    W_s[k * n + i][r] = 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
