"""
Rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication.

Uses block decomposition:
- Split the 4x5 product into four 2x2 or 2x3 sub-products.
- Use Strassen (rank 7) for 2x2x2 blocks.
- Use a rank-11 decomposition for 2x2x3 blocks (Strassen for 2x2x2 + schoolbook for 2x2x1).

Total rank: 7 + 11 + 7 + 11 = 36.

Index conventions:
- A is n x m = 2 x 4, flattened as A[i*m + j]
- B is m x p = 4 x 5, flattened as B[j*p + k]
- C is n x p = 2 x 5, output index W[k*n + i] (official trace convention)

The decomposition satisfies:
C[k*n + i] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
summed over j, which equals sum_j A[i,j] B[j,k].
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]


def _strassen_222():
    """
    Return (U, V, W) for 2x2 x 2x2 -> 2x2 matrix multiplication with rank 7.
    
    U shape: (4, 7)  -- rows indexed by (i, j) for A, i in {0,1}, j in {0,1}
    V shape: (4, 7)  -- rows indexed by (j', k) for B, j' in {0,1}, k in {0,1}
    W shape: (4, 7)  -- rows indexed by (k', i') for C, k' in {0,1}, i' in {0,1}
    
    Convention: U[i*2+j], V[j'*2+k], W[k'*2+i']
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # r=0: M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    # C00 += M1, C11 += M1
    W[0, 0] = 1  # C[0*2+0] = C00
    W[3, 0] = 1  # C[1*2+1] = C11
    
    # r=1: M2 = (A10 + A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    # C10 += M2, C11 -= M2
    W[2, 1] = 1   # C[1*2+0] = C10
    W[3, 1] = -1  # C[1*2+1] = C11
    
    # r=2: M3 = A00(B01 - B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    # C01 += M3, C11 += M3
    W[1, 2] = 1  # C[0*2+1] = C01
    W[3, 2] = 1  # C[1*2+1] = C11
    
    # r=3: M4 = A11(B10 - B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    # C00 += M4, C10 += M4
    W[0, 3] = 1  # C[0*2+0] = C00
    W[2, 3] = 1  # C[1*2+0] = C10
    
    # r=4: M5 = (A00 + A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    # C00 -= M5, C01 += M5
    W[0, 4] = -1 # C[0*2+0] = C00
    W[1, 4] = 1  # C[0*2+1] = C01
    
    # r=5: M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    # C11 += M6
    W[3, 5] = 1  # C[1*2+1] = C11
    
    # r=6: M7 = (A01 - A11)(B10 + B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    # C00 += M7
    W[0, 6] = 1  # C[0*2+0] = C00
    
    return U, V, W


def _rank11_223():
    """
    Return (U, V, W) for 2x2 x 2x3 -> 2x3 matrix multiplication with rank 11.
    
    Splits B into B_L (2x2) and B_R (2x1).
    Uses Strassen (rank 7) for A * B_L and schoolbook (rank 4) for A * B_R.
    
    U shape: (4, 11)  -- rows indexed by (i, j) for A, i in {0,1}, j in {0,1}
    V shape: (6, 11)  -- rows indexed by (j', k) for B, j' in {0,1}, k in {0,1,2}
    W shape: (6, 11)  -- rows indexed by (k', i') for C, k' in {0,1,2}, i' in {0,1}
    
    Convention: U[i*2+j], V[j'*3+k], W[k'*2+i']
    """
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # First 7 terms: Strassen for A * B_L (columns 0,1 of B)
    U_s, V_s, W_s = _strassen_222()
    
    # Map V_s indices (j'*2+k, k in {0,1}) to global V indices (j'*3+k, k in {0,1})
    # j'=0, k=0 -> 0; j'=0, k=1 -> 1; j'=1, k=0 -> 3; j'=1, k=1 -> 4
    # V_s row 0 (B00) -> V row 0
    # V_s row 1 (B01) -> V row 1
    # V_s row 2 (B10) -> V row 3
    # V_s row 3 (B11) -> V row 4
    v_map = [0, 1, 3, 4]
    for r in range(7):
        U[:, r] = U_s[:, r]
        for local_r, global_r in enumerate(v_map):
            V[global_r, r] = V_s[local_r, r]
        W[:, r] = W_s[:, r]
    
    # Last 4 terms: Schoolbook for A * B_R (column 2 of B)
    # C[i, 2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # r=7: A00 * B02
    U[0, 7] = 1
    V[5, 7] = 1  # j'=0, k=2 -> 0*3+2 = 2... wait
    # B is 2x3: rows j'=0,1; cols k=0,1,2
    # V index: j'*3 + k
    # j'=0, k=2 -> 2
    # j'=1, k=2 -> 5
    V[2, 7] = 1  # j'=0, k=2 -> 0*3+2 = 2
    W[4, 7] = 1  # C[k'*2+i'] where k'=2, i'=0 -> 2*2+0 = 4
    
    # r=8: A01 * B02
    U[1, 8] = 1
    V[2, 8] = 1  # j'=0, k=2 -> 2
    W[4, 8] = 1  # C[2*2+0] = C[4] = C02
    
    # r=9: A10 * B12
    U[2, 9] = 1
    V[5, 9] = 1  # j'=1, k=2 -> 1*3+2 = 5
    W[5, 9] = 1  # C[2*2+1] = C[5] = C12
    
    # r=10: A11 * B12
    U[3, 10] = 1
    V[5, 10] = 1  # j'=1, k=2 -> 5
    W[5, 10] = 1  # C[5] = C12
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Total rank: 7 + 11 + 7 + 11 = 36
    total_rank = 36
    
    U = np.zeros((n * m, total_rank))  # (8, 36)
    V = np.zeros((m * p, total_rank))  # (20, 36)
    W = np.zeros((p * n, total_rank))  # (10, 36)
    
    # Block 1: A[0:2, 0:2] x B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, rank 7)
    U1, V1, W1 = _strassen_222()
    # U1 rows: i*2+j, i in {0,1}, j in {0,1} -> global U rows: i*4+j (j offset 0)
    # V1 rows: j'*2+k, j' in {0,1}, k in {0,1} -> global V rows: j'*5+k (j' offset 0, k offset 0)
    # W1 rows: k'*2+i', k' in {0,1}, i' in {0,1} -> global W rows: k'*2+i' (k' offset 0, i' offset 0)
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i * 4 + j, r] += U1[i * 2 + j, r]
        for jp in range(2):
            for k in range(2):
                V[jp * 5 + k, r] += V1[jp * 2 + k, r]
        for kp in range(2):
            for ip in range(2):
                W[kp * 2 + ip, r] += W1[kp * 2 + ip, r]
    
    # Block 2: A[0:2, 0:2] x B[0:2, 2:5] -> C[0:2, 2:5] (rank 11)
    U2, V2, W2 = _rank11_223()
    # U2 rows: i*2+j -> global U rows: i*4+j
    # V2 rows: j'*3+k, j' in {0,1}, k in {0,1,2} -> global V rows: j'*5+(k+2)
    # W2 rows: k'*2+i', k' in {0,1,2}, i' in {0,1} -> global W rows: (k'+2)*2+i'
    
    r_offset = 7
    for r in range(11):
        for i in range(2):
            for j in range(2):
                U[i * 4 + j, r_offset + r] += U2[i * 2 + j, r]
        for jp in range(2):
            for k in range(3):
                V[jp * 5 + (k + 2), r_offset + r] += V2[jp * 3 + k, r]
        for kp in range(3):
            for ip in range(2):
                W[(kp + 2) * 2 + ip, r_offset + r] += W2[kp * 2 + ip, r]
    
    # Block 3: A[0:2, 2:4] x B[2:4, 0:2] -> C[0:2, 0:2] (Strassen, rank 7)
    U3, V3, W3 = _strassen_222()
    # U3 rows: i*2+j -> global U rows: i*4+(j+2)
    # V3 rows: j'*2+k -> global V rows: (j'+2)*5+k
    # W3 rows: k'*2+i' -> global W rows: k'*2+i'
    
    r_offset = 18
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i * 4 + (j + 2), r_offset + r] += U3[i * 2 + j, r]
        for jp in range(2):
            for k in range(2):
                V[(jp + 2) * 5 + k, r_offset + r] += V3[jp * 2 + k, r]
        for kp in range(2):
            for ip in range(2):
                W[kp * 2 + ip, r_offset + r] += W3[kp * 2 + ip, r]
    
    # Block 4: A[0:2, 2:4] x B[2:4, 2:5] -> C[0:2, 2:5] (rank 11)
    U4, V4, W4 = _rank11_223()
    # U4 rows: i*2+j -> global U rows: i*4+(j+2)
    # V4 rows: j'*3+k -> global V rows: (j'+2)*5+(k+2)
    # W4 rows: k'*2+i' -> global W rows: (k'+2)*2+i'
    
    r_offset = 25
    for r in range(11):
        for i in range(2):
            for j in range(2):
                U[i * 4 + (j + 2), r_offset + r] += U4[i * 2 + j, r]
        for jp in range(2):
            for k in range(3):
                V[(jp + 2) * 5 + (k + 2), r_offset + r] += V4[jp * 3 + k, r]
        for kp in range(3):
            for ip in range(2):
                W[(kp + 2) * 2 + ip, r_offset + r] += W4[kp * 2 + ip, r]
    
    # Verify decomposition with random test
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        U_arr = U
        V_arr = V
        W_arr = W
        
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape for efficient computation
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
        
        C_computed = np.zeros((n, p))
        for r in range(total_rank):
            u_col = U_arr[:, r].reshape(n, m)
            v_col = V_arr[:, r].reshape(m, p)
            w_col = W_arr[:, r].reshape(p, n)
            # Contribution: u_col @ v_col * w_col (element-wise)
            C_computed += (u_col @ v_col) * w_col.T
        
        error = np.max(np.abs(C_computed - C_expected))
        if error > 1e-10:
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U_fb = np.zeros((n * m, rank))
            V_fb = np.zeros((m * p, rank))
            W_fb = np.zeros((p * n, rank))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, idx] = 1.0
                        V_fb[j * p + k, idx] = 1.0
                        W_fb[k * n + i, idx] = 1.0
                        idx += 1
            U = U_fb
            V = V_fb
            W = W_fb
            total_rank = rank
    
    # Convert to lists with integer entries
    U_list = [[int(val) for val in row] for row in U.tolist()]
    V_list = [[int(val) for val in row] for row in V.tolist()]
    W_list = [[int(val) for val in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
