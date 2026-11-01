"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen rank-7 decomposition for 2x2 matrix multiplication.
    #
    # Matrix multiplication: C = A * B where A, B, C are 2x2.
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # The tensor T has indices (i, j, k) corresponding to A[i,j] * B[j,k] = C[i,k].
    # In the standard flattening:
    #   U has shape (n*m, rank) = (4, 7), indexed by [i*m + j, r]
    #   V has shape (m*p, rank) = (4, 7), indexed by [j*p + k, r]
    #   W has shape (p*n, rank) = (4, 7), indexed by [k*n + i, r]
    #
    # Wait, let me re-read the convention: W[k*n+i, r].
    # So W is indexed by [k*n + i, r] where k is the column of C (which is the "p" index) 
    # and i is the row of C (which is the "n" index).
    # Actually, for matrix multiplication A (n x m) * B (m x p) = C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor T_{i,j,k} = 1 if this is a valid term, i.e., always for 2x2x2.
    # We want: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = C[i,k]
    # 
    # The convention says:
    # U: shape (n*m, rank), indexed by [i*m + j, r]  -- wait, let me check.
    # The task says: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # So U is indexed by (i, j) flattened as i*m + j.
    # V is indexed by (j, k) flattened as j*p + k.
    # W is indexed by (k, i) flattened as k*n + i.
    #
    # For 2x2x2: n=2, m=2, p=2.
    # U: 4 x rank, indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V: 4 x rank, indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W: 4 x rank, indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    #   where W[k*n+i] means: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    #
    # Strassen's 7 products:
    # P1 = (A11 + A22) * (B11 + B22)
    # P2 = (A21 + A22) * B11
    # P3 = A11 * (B12 - B22)
    # P4 = A22 * (B21 - B11)
    # P5 = (A11 + A12) * B22
    # P6 = (A21 - A11) * (B11 + B12)
    # P7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    #
    # Let me define the linear combinations.
    # A indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B indices: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    #
    # For each product r (0..6), we need:
    # u_r[i,j] such that sum over i,j of u_r[i,j] * A[i,j] gives the first factor
    # v_r[j,k] such that sum over j,k of v_r[j,k] * B[j,k] gives the second factor
    # w_r[i,k] such that sum over r of w_r[i,k] * P_r gives C[i,k]
    #
    # Product 0 (P1): (A11 + A22) * (B11 + B22)
    #   u_0: A11=1, A22=1 -> u_0[0,0]=1, u_0[1,1]=1, others 0
    #   v_0: B11=1, B22=1 -> v_0[0,0]=1, v_0[1,1]=1, others 0
    #   w_0: contributes to C11 (+1), C22 (+1)
    #     w_0[0,0]=1, w_0[1,1]=1, others 0
    #     In W indexing [k*n+i]: C11=(i=0,k=0) -> W[0*2+0]=W[0]=1
    #     C22=(i=1,k=1) -> W[1*2+1]=W[3]=1
    #
    # Product 1 (P2): (A21 + A22) * B11
    #   u_1: A21=1, A22=1 -> u_1[1,0]=1, u_1[1,1]=1, others 0
    #   v_1: B11=1 -> v_1[0,0]=1, others 0
    #   w_1: contributes to C21 (+1)
    #     C21=(i=1,k=0) -> W[0*2+1]=W[1]=1
    #
    # Product 2 (P3): A11 * (B12 - B22)
    #   u_2: A11=1 -> u_2[0,0]=1, others 0
    #   v_2: B12=1, B22=-1 -> v_2[0,1]=1, v_2[1,1]=-1, others 0
    #   w_2: contributes to C12 (+1), C22 (+1)
    #     C12=(i=0,k=1) -> W[1*2+0]=W[2]=1
    #     C22=(i=1,k=1) -> W[1*2+1]=W[3]=1
    #
    # Product 3 (P4): A22 * (B21 - B11)
    #   u_3: A22=1 -> u_3[1,1]=1, others 0
    #   v_3: B21=1, B11=-1 -> v_3[1,0]=1, v_3[0,0]=-1, others 0
    #   w_3: contributes to C11 (+1), C21 (+1)
    #     C11=(i=0,k=0) -> W[0*2+0]=W[0]=1
    #     C21=(i=1,k=0) -> W[0*2+1]=W[1]=1
    #
    # Product 4 (P5): (A11 + A12) * B22
    #   u_4: A11=1, A12=1 -> u_4[0,0]=1, u_4[0,1]=1, others 0
    #   v_4: B22=1 -> v_4[1,1]=1, others 0
    #   w_4: contributes to C11 (-1)
    #     C11=(i=0,k=0) -> W[0*2+0]=W[0]=-1
    #
    # Product 5 (P6): (A21 - A11) * (B11 + B12)
    #   u_5: A21=1, A11=-1 -> u_5[1,0]=1, u_5[0,0]=-1, others 0
    #   v_5: B11=1, B12=1 -> v_5[0,0]=1, v_5[0,1]=1, others 0
    #   w_5: contributes to C22 (+1)
    #     C22=(i=1,k=1) -> W[1*2+1]=W[3]=1
    #
    # Product 6 (P7): (A12 - A22) * (B21 + B22)
    #   u_6: A12=1, A22=-1 -> u_6[0,1]=1, u_6[1,1]=-1, others 0
    #   v_6: B21=1, B22=1 -> v_6[1,0]=1, v_6[1,1]=1, others 0
    #   w_6: contributes to C11 (+1)
    #     C11=(i=0,k=0) -> W[0*2+0]=W[0]=1

    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to set U[i*m+j, r]
    def set_U(i, j, r, val):
        U[i * m + j][r] = val

    # Helper to set V[j*p+k, r]
    def set_V(j, k, r, val):
        V[j * p + k][r] = val

    # Helper to set W[k*n+i, r]
    def set_W(k, i, r, val):
        W[k * n + i][r] = val

    # Product 0 (P1): (A11 + A22) * (B11 + B22)
    set_U(0, 0, 0, 1)  # A11
    set_U(1, 1, 0, 1)  # A22
    set_V(0, 0, 0, 1)  # B11
    set_V(1, 1, 0, 1)  # B22
    set_W(0, 0, 0, 1)  # C11
    set_W(1, 1, 0, 1)  # C22

    # Product 1 (P2): (A21 + A22) * B11
    set_U(1, 0, 1, 1)  # A21
    set_U(1, 1, 1, 1)  # A22
    set_V(0, 0, 1, 1)  # B11
    set_W(0, 1, 1, 1)  # C21

    # Product 2 (P3): A11 * (B12 - B22)
    set_U(0, 0, 2, 1)  # A11
    set_V(0, 1, 2, 1)  # B12
    set_V(1, 1, 2, -1) # B22
    set_W(1, 0, 2, 1)  # C12
    set_W(1, 1, 2, 1)  # C22

    # Product 3 (P4): A22 * (B21 - B11)
    set_U(1, 1, 3, 1)  # A22
    set_V(1, 0, 3, 1)  # B21
    set_V(0, 0, 3, -1) # B11
    set_W(0, 0, 3, 1)  # C11
    set_W(0, 1, 3, 1)  # C21

    # Product 4 (P5): (A11 + A12) * B22
    set_U(0, 0, 4, 1)  # A11
    set_U(0, 1, 4, 1)  # A12
    set_V(1, 1, 4, 1)  # B22
    set_W(0, 0, 4, -1) # C11

    # Product 5 (P6): (A21 - A11) * (B11 + B12)
    set_U(1, 0, 5, 1)  # A21
    set_U(0, 0, 5, -1) # A11
    set_V(0, 0, 5, 1)  # B11
    set_V(0, 1, 5, 1)  # B12
    set_W(1, 1, 5, 1)  # C22

    # Product 6 (P7): (A12 - A22) * (B21 + B22)
    set_U(0, 1, 6, 1)  # A12
    set_U(1, 1, 6, -1) # A22
    set_V(1, 0, 6, 1)  # B21
    set_V(1, 1, 6, 1)  # B22
    set_W(0, 0, 6, 1)  # C11

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
