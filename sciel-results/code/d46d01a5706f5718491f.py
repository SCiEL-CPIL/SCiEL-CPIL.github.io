"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    schoolbook_rank = n * m * p  # 32
    deadline = time.time() + 130  # safety margin before 150s kill

    # Build the multiplication tensor T[a,b,c] where:
    # a = i*m + j  (i in [0,n), j in [0,m))
    # b = j*p + k  (j in [0,m), k in [0,p))
    # c = k*n + i  (k in [0,p), i in [0,n))  [official trace convention W[k*n+i, r]]
    # T[a,b,c] = 1 if the indices are consistent (same j in a and b, same i in a and c, same k in b and c)
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        """Check exact decomposition."""
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        """Convert numpy arrays to list-of-lists format."""
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    # === STRASSEN-BASED CONSTRUCTION ===
    # Key insight: 2x4x4 can be decomposed by splitting into blocks.
    # A is 2x4 = two 2x2 blocks: A = [A11 | A12]
    # B is 4x4 = four 2x2 blocks: B = [[B11, B12], [B21, B22]]
    # C = A*B gives 2x4: C = [A11*B11 + A12*B21 | A11*B12 + A12*B22]
    # Each 2x2*2x2 product uses Strassen with rank 7.
    # Total: 7 (for A11*B11) + 7 (for A12*B21) + 7 (for A11*B12) + 7 (for A12*B22) = 28
    # But we can share: A11 appears in two products, A12 in two.
    # Actually, let's just do 4 separate Strassen decompositions = rank 28.

    def strassen_2x2_decomp():
        """
        Return Strassen decomposition of 2x2 * 2x2.
        U: (4, 7), V: (4, 7), W: (4, 7)
        where indices are:
          U: row = i*2+j (i,j in {0,1}), col = t (t in 0..6)
          V: row = j*2+k (j,k in {0,1}), col = t
          W: row = k*2+i (k,i in {0,1}), col = t  [trace convention: k*n+i with n=2]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # Strassen's 7 products for C = A*B where A,B,C are 2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # A indices: A00=0, A01=1, A10=2, A11=3
        # B indices: B00=0, B01=1, B10=2, B11=3
        # C indices (trace k*2+i): C00: k=0,i=0 -> 0; C01: k=1,i=0 -> 1; C10: k=0,i=1 -> 2; C11: k=1,i=1 -> 3
        # Wait: W[k*n+i] with n=2. For C[i,j] where i is row, j is col:
        # In our tensor convention, c = k*n + i where k is the col of C (which is j in standard notation) and i is row of C.
        # So C[i,j] -> index j*2 + i
        # C00 -> 0*2+0 = 0
        # C01 -> 1*2+0 = 2
        # C10 -> 0*2+1 = 1
        # C11 -> 1*2+1 = 3

        # t=0: M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0  # C00 and C11

        # t=1: M2 = (A10+A11)*B00
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[1, 1] = 1.0; W[3, 1] = -1.0  # C10 and C11 (C11 gets -M2)

        # t=2: M3 = A00*(B01-B11)
        U[0, 2] = 1.0
        V[1, 2] = 1.0; V[3, 2] = -1.0
        W[2, 2] = 1.0; W[3, 2] = 1.0  # C01 and C11

        # t=3: M4 = A11*(B10-B00)
        U[3, 3] = 1.0
        V[2, 3] = 1.0; V[0, 3] = -1.0
        W[0, 3] = 1.0; W[1, 3] = 1.0  # C00 and C10

        # t=4: M5 = (A00+A01)*B11
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[2, 4] = 1.0  # C00 gets -M5, C01 gets +M5

        # t=5: M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[1, 5] = 1.0
        W[3, 5] = 1.0  # C11 gets +M6

        # t=6: M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[2, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0  # C00 gets +M7

        return U, V, W

    # Now build the full 2x4 * 4x4 decomposition using 4 Strassen blocks.
    # A is 2x4: rows 0,1; cols 0..3
    # Split A into A_left = A[:, 0:2] (2x2) and A_right = A[:, 2:4] (2x2)
    # B is 4x4: rows 0..3, cols 0..3
    # Split B into B_top = B[0:2, :] (2x4) and B_bot = B[2:4, :] (2x4)
    # Further split B_top into B_top_left = B[0:2, 0:2] and B_top_right = B[0:2, 2:4]
    # Similarly B_bot into B_bot_left = B[2:4, 0:2] and B_bot_right = B[2:4, 2:4]
    #
    # C = A * B (2x4)
    # C[:, 0:2] = A_left * B_top_left + A_right * B_bot_left  (2x2 + 2x2)
    # C[:, 2:4] = A_left * B_top_right + A_right * B_bot_right (2x2 + 2x2)
    #
    # So we need 4 Strassen products:
    # P1 = A_left * B_top_left  -> contributes to C[:, 0:2]
    # P2 = A_right * B_bot_left -> contributes to C[:, 0:2]
    # P3 = A_left * B_top_right -> contributes to C[:, 2:4]
    # P4 = A_right * B_bot_right -> contributes to C[:, 2:4]

    # Map local 2x2 indices to global indices:
    # A_left: A[i, j] for i in {0,1}, j in {0,1} -> global A index i*4 + j
    # A_right: A[i, j] for i in {0,1}, j in {2,3} -> global A index i*4 + (j+2)
    # B_top_left: B[j, k] for j in {0,1}, k in {0,1} -> global B index j*4 + k
    # B_top_right: B[j, k] for j in {0,1}, k in {2,3} -> global B index j*4 + (k+2)
    # B_bot_left: B[j, k] for j in {2,3}, k in {0,1} -> global B index (j+2)*4 + k
    # B_bot_right: B[j, k] for j in {2,3}, k in {2,3} -> global B index (j+2)*4 + (k+2)
    # C_left: C[i, k] for i in {0,1}, k in {0,1} -> global C index k*2 + i
    # C_right: C[i, k] for i in {0,1}, k in {2,3} -> global C index (k+2)*2 + i

    U_s, V_s, W_s = strassen_2x2_decomp()

    # Build full U (8 x 28), V (16 x 28), W (8 x 28)
    full_rank = 28
    U_full = np.zeros((n * m, full_rank))  # 8 x 28
    V_full = np.zeros((m * p, full_rank))  # 16 x 28
    W_full = np.zeros((p * n, full_rank))  # 8 x 28

    # Block 1: A_left * B_top_left -> C_left
    # A_left local (i,j) -> global i*4 + j
    # B_top_left local (j,k) -> global j*4 + k
    # C_left local (i,k) -> global k*2 + i
    for t in range(7):
        col = 0 * 7 + t
        # U: A_left
        for li in range(2):
            for lj in range(2):
                local_a = li * 2 + lj
                global_a = li * 4 + lj
                U_full[global_a, col] = U_s[local_a, t]
        # V: B_top_left
        for lj in range(2):
            for lk in range(2):
                local_b = lj * 2 + lk
                global_b = lj * 4 + lk
                V_full[global_b, col] = V_s[local_b, t]
        # W: C_left
        for lk in range(2):
            for li in range(2):
                local_c = lk * 2 + li
                global_c = lk * 2 + li
                W_full[global_c, col] = W_s[local_c, t]

    # Block 2: A_right * B_bot_left -> C_left
    # A_right local (i,j) -> global i*4 + (j+2)
    # B_bot_left local (j,k) -> global (j+2)*4 + k
    # C_left local (i,k) -> global k*2 + i
    for t in range(7):
        col = 1 * 7 + t
        for li in range(2):
            for lj in range(2):
                local_a = li * 2 + lj
                global_a = li * 4 + (lj + 2)
                U_full[global_a, col] = U_s[local_a, t]
        for lj in range(2):
            for lk in range(2):
                local_b = lj * 2 + lk
                global_b = (lj + 2) * 4 + lk
                V_full[global_b, col] = V_s[local_b, t]
        for lk in range(2):
            for li in range(2):
                local_c = lk * 2 + li
                global_c = lk * 2 + li
                W_full[global_c, col] = W_s[local_c, t]

    # Block 3: A_left * B_top_right -> C_right
    # A_left local (i,j) -> global i*4 + j
    # B_top_right local (j,k) -> global j*4 + (k+2)
    # C_right local (i,k) -> global (k+2)*2 + i
    for t in range(7):
        col = 2 * 7 + t
        for li in range(2):
            for lj in range(2):
                local_a = li * 2 + lj
                global_a = li * 4 + lj
                U_full[global_a, col] = U_s[local_a, t]
        for lj in range(2):
            for lk in range(2):
                local_b = lj * 2 + lk
                global_b = lj * 4 + (lk + 2)
                V_full[global_b, col] = V_s[local_b, t]
        for lk in range(2):
            for li in range(2):
                local_c = lk * 2 + li
                global_c = (lk + 2) * 2 + li
                W_full[global_c, col] = W_s[local_c, t]

    # Block 4: A_right * B_bot_right -> C_right
    # A_right local (i,j) -> global i*4 + (j+2)
    # B_bot_right local (j,k) -> global (j+2)*4 + (k+2)
    # C_right local (i,k) -> global (k+2)*2 + i
    for t in range(7):
        col = 3 * 7 + t
        for li in range(2):
            for lj in range(2):
                local_a = li * 2 + lj
                global_a = li * 4 + (lj + 2)
                U_full[global_a, col] = U_s[local_a, t]
        for lj in range(2):
            for lk in range(2):
                local_b = lj * 2 + lk
                global_b = (lj + 2) * 4 + (lk + 2)
                V_full[global_b, col] = V_s[local_b, t]
        for lk in range(2):
            for li in range(2):
                local_c = lk * 2 + li
                global_c = (lk + 2) * 2 + li
                W_full[global_c, col] = W_s[local_c, t]

    # Verify
    if verify(U_full, V_full, W_full, tol=1e-8):
        return to_lists(U_full, V_full, W_full, full_rank)

    # If Strassen construction has a bug, fall back to schoolbook
    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    U, V, W = build_schoolbook()
    return to_lists(U, V, W, schoolbook_rank)
# EVOLVE-BLOCK-END
