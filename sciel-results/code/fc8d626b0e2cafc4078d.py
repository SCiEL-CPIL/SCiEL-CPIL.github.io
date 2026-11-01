import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 8
    
    # The tensor T[a,b,c] for C[i,k] = sum_j A[i,j] B[j,k]
    # a = i*m + j (size n*m = 6)
    # b = j*p + k (size m*p = 24)
    # c = k*n + i (size p*n = 16)
    
    # We want to find U (6, r), V (24, r), W (16, r) such that
    # sum_d U[a,d] V[b,d] W[c,d] = 1 if (i,j,k) consistent, else 0.
    
    # Strategy:
    # 1. Construct a valid rank-44 solution using Kronecker structure.
    #    We can view the multiplication as a sequence of operations.
    #    Specifically, we can group the k-dimension (size 8) into 4 pairs of size 2.
    #    For each pair of k's (k0, k1), we perform a 2x3 * 3x2 multiplication.
    #    The rank of 2x3 * 3x2 can be reduced.
    #    
    #    Let's look at 2x3 * 3x2.
    #    A is 2x3, B is 3x2. C is 2x2.
    #    We can split A into A_left (2x1) and A_right (2x2)? No, that's not quite right for sharing.
    #    
    #    Better: Use the known rank-11 decomposition for 2x3 * 3x2.
    #    Actually, let's derive a simple one.
    #    Schoolbook rank for 2x3*3x2 is 12.
    #    Can we do 11?
    #    C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    #    C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    #    C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    #    C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    #    
    #    Let's try to share products.
    #    Consider (A[0,0]+A[1,1]+A[0,2]) * (B[0,0]+B[1,1]+B[2,0]) ... this gets messy.
    #    
    #    Alternative: Use Strassen on the 2x2 part.
    #    Let's split the 3x2 matrix B into B0 (1x2) and B1 (2x2).
    #    A = [A0 (2x1), A1 (2x2)].
    #    C = A0 * B0 + A1 * B1.
    #    A0*B0 is rank 2 (outer product of 2x1 and 1x2).
    #    A1*B1 is 2x2 * 2x2 multiplication. Rank of 2x2 mult is 7 (Strassen).
    #    Total rank = 2 + 7 = 9?
    #    Wait, A0 is 2x1, B0 is 1x2. The product is a 2x2 matrix.
    #    The tensor for A0*B0 has rank 2?
    #    (A0[0]*B0[0,0], A0[0]*B0[0,1], A0[1]*B0[1,0], A0[1]*B0[1,1]).
    #    Yes, this is a rank-2 tensor (sum of 2 rank-1 tensors).
    #    So, rank(2x3 * 3x2) <= rank(2x1*1x2) + rank(2x2*2x2) = 2 + 7 = 9.
    #    
    #    Is 9 the minimal rank for 2x3*3x2?
    #    Lower bound: max(2*3, 3*2, 2*2) is not a lower bound.
    #    Rank(2x2*2x2) is 7.
    #    If we could do better than 9, it would be a big deal.
    #    Let's assume rank 9 is achievable and valid.
    #    
    #    If rank(2x3*3x2) = 9, then for 2x3*3x8 (which is 4 blocks of 2x3*3x2),
    #    we can just repeat the block 4 times.
    #    Total rank = 4 * 9 = 36.
    #    
    #    Let's verify if we can construct a rank-9 decomposition for 2x3*3x2.
    #    A = [[a00, a01, a02], [a10, a11, a12]]
    #    B = [[b00, b01], [b10, b11], [b20, b21]]
    #    
    #    Part 1: A0 * B0 where A0 = [[a00], [a10]], B0 = [[b00, b01]]
    #    Terms:
    #    t0 = a00 * b00
    #    t1 = a00 * b01
    #    t2 = a10 * b00
    #    t3 = a10 * b01
    #    Contribution to C:
    #    C00 += t0
    #    C01 += t1
    #    C10 += t2
    #    C11 += t3
    #    
    #    Part 2: A1 * B1 where A1 = [[a01, a02], [a11, a12]], B1 = [[b10, b11], [b20, b21]]
    #    This is 2x2 * 2x2. Use Strassen (rank 7).
    #    Let M0 = a01, M1 = a02, M2 = a11, M3 = a12
    #    Let N0 = b10, N1 = b11, N2 = b20, N3 = b21
    #    
    #    Strassen for 2x2:
    #    P0 = M0 * (N3 - N1)
    #    P1 = (M0 + M1) * N1
    #    P2 = (M2 + M3) * N0
    #    P3 = M3 * (N2 - N0)
    #    P4 = (M0 + M3) * (N0 + N1)
    #    P5 = (M1 - M2) * (N2 + N3)
    #    P6 = (M3 - M0) * (N0 + N1)  <-- Wait, standard Strassen is:
    #    
    #    Standard Strassen (C = A*B):
    #    M1 = (A11 + A22) * (B11 + B22)
    #    M2 = (A21 + A22) * B11
    #    M3 = A11 * (B12 - B22)
    #    M4 = A22 * (B21 - B11)
    #    M5 = (A11 + A12) * B22
    #    M6 = (A21 - A11) * (B11 + B12)
    #    M7 = (A12 - A22) * (B21 + B22)
    #    
    #    C11 = M1 + M4 - M5 + M7
    #    C12 = M3 + M5
    #    C21 = M2 + M4
    #    C22 = M1 - M2 + M3 + M6
    #    
    #    Mapping to our indices:
    #    A11 = a01, A12 = a02
    #    A21 = a11, A22 = a12
    #    B11 = b10, B12 = b11
    #    B21 = b20, B22 = b21
    #    
    #    M1 = (a01 + a12) * (b10 + b21)
    #    M2 = (a11 + a12) * b10
    #    M3 = a01 * (b11 - b21)
    #    M4 = a12 * (b20 - b10)
    #    M5 = (a01 + a02) * b21
    #    M6 = (a11 - a01) * (b10 + b11)
    #    M7 = (a02 - a12) * (b20 + b21)
    #    
    #    C11 (which is C[0,0] from this block) = M1 + M4 - M5 + M7
    #    C12 (which is C[0,1] from this block) = M3 + M5
    #    C21 (which is C[1,0] from this block) = M2 + M4
    #    C22 (which is C[1,1] from this block) = M1 - M2 + M3 + M6
    #    
    #    So total rank for 2x3*3x2 is 4 (from A0B0) + 7 (from A1B1) = 11.
    #    Wait, I said 9 earlier. Why?
    #    A0 is 2x1, B0 is 1x2.
    #    A0 * B0 = [[a00*b00, a00*b01], [a10*b00, a10*b01]].
    #    This requires 4 multiplications if done schoolbook.
    #    Can we do it in 2?
    #    (a00+a10)(b00+b01) = a00b00 + a00b01 + a10b00 + a10b01.
    #    This gives the sum of all 4 terms.
    #    We need individual terms.
    #    So A0*B0 is rank 4?
    #    Tensor for A0*B0:
    #    T[i, j, k] where i in {0,1}, j in {0}, k in {0,1}.
    #    Wait, the "j" index in the global problem is the intermediate dimension.
    #    In the 2x3*3x2 subproblem, the intermediate dimension is size 3.
    #    We split it into 1 and 2.
    #    The sub-tensor for the first part (size 1) is 2x1 * 1x2.
    #    The rank of the 2x1 * 1x2 multiplication tensor is 2?
    #    Let's check.
    #    X is 2x1, Y is 1x2. Z = X Y.
    #    Z[0,0] = X[0] Y[0,0]
    #    Z[0,1] = X[0] Y[0,1]
    #    Z[1,0] = X[1] Y[0,0]
    #    Z[1,1] = X[1] Y[0,1]
    #    This is the outer product of X and Y.
    #    The tensor T[i, k, l] = 1 if consistent?
    #    Actually, the multiplication tensor for $m \times n$ by $n \times p$ has rank $mn$ if $n=1$?
    #    No, the rank of the identity matrix tensor (which is what $1 \times 1 \times 1$ is) is 1.
    #    For $2 \times 1 \times 1 \times 2$, the tensor is $\sum_{i=0}^1 \sum_{k=0}^1 e_i \otimes e_0 \otimes (e_i \text{ mapped to output?})$
    #    Let's look at the linear map.
    #    $Z = X Y$.
    #    $Z_{ik} = X_i Y_{0k}$.
    #    This is a bilinear map.
    #    The tensor is $\sum_{i=0}^1 \sum_{k=0}^1 e_i \otimes e_0 \otimes e_{ik}$? No.
    #    The indices are: Input X (size 2), Input Y (size 2), Output Z (size 4).
    #    $Z_{00} = X_0 Y_0$
    #    $Z_{01} = X_0 Y_1$
    #    $Z_{10} = X_1 Y_0$
    #    $Z_{11} = X_1 Y_1$
    #    This is exactly the tensor for $2 \times 2$ matrix multiplication if we view X as a row and Y as a column?
    #    No, it's the tensor for the outer product.
    #    The rank of the outer product tensor $U \otimes V$ (where U is 2, V is 2) is 1?
    #    No, the map $(x, y) \to x \otimes y$ is rank 1?
    #    $x \otimes y = \sum_i \sum_k x_i y_k e_i \otimes e_k$.
    #    This is not a sum of rank-1 tensors in the bilinear sense?
    #    Wait, the definition of tensor rank for matrix multiplication $A_{m \times n} B_{n \times p}$ is the minimum number of multiplications.
    #    For $2 \times 1 \times 1 \times 2$, we need $X_0 Y_0, X_0 Y_1, X_1 Y_0, X_1 Y_1$.
    #    These are 4 independent products.
    #    Can we compute them with fewer?
    #    $S_1 = (X_0+X_1)(Y_0+Y_1) = X_0Y_0 + X_0Y_1 + X_1Y_0 + X_1Y_1$.
    #    $S_2 = (X_0+X_1)(Y_0-Y_1) = X_0Y_0 - X_0Y_1 + X_1Y_0 - X_1Y_1$.
    #    $S_3 = (X_0-X_1)(Y_0+Y_1) = X_0Y_0 + X_0Y_1 - X_1Y_0 - X_1Y_1$.
    #    $S_4 = (X_0-X_1)(Y_0-Y_1) = X_0Y_0 - X_0Y_1 - X_1Y_0 + X_1Y_1$.
    #    We can solve for the 4 products using 4 sums.
    #    So the rank is 4.
    #    
    #    So my previous estimate of rank 2 for $2 \times 1 \times 1 \times 2$ was wrong. It is 4.
    #    Therefore, rank(2x3*3x2) <= 4 + 7 = 11.
    #    
    #    So, using 4 blocks of 2x3*3x2, we get rank 4 * 11 = 44.
    #    
    #    This is a valid solution with rank 44.
    #    Score = 48/44 = 1.0909.
    
    # Let's construct the rank-44 solution.
    
    U = np.zeros((n*m, 44), dtype=np.int64)
    V = np.zeros((m*p, 44), dtype=np.int64)
    W = np.zeros((p*n, 44), dtype=np.int64)
    
    col_offset = 0
    
    # We have 4 blocks for k = (0,1), (2,3), (4,5), (6,7)
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Block 1: A[:, 0] * B[0, k0:k1]  (2x1 * 1x2)
        # A indices: i*m + 0
        # B indices: 0*p + k0, 0*p + k1
        # C indices: k0*n + i, k1*n + i
        
        # Terms:
        # t0 = A[0,0] * B[0,k0] -> C[0, k0] += t0
        # t1 = A[0,0] * B[0,k1] -> C[0, k1] += t1
        # t2 = A[1,0] * B[0,k0] -> C[1, k0] += t2
        # t3 = A[1,0] * B[0,k1] -> C[1, k1] += t3
        
        # U (size 6): index i*m + 0
        # V (size 24): index 0*p + k
        # W (size 16): index k*n + i
        
        # t0
        U[0*m + 0, col_offset] = 1
        V[0*p + k0, col_offset] = 1
        W[k0*n + 0, col_offset] = 1
        col_offset += 1
        
        # t1
        U[0*m + 0, col_offset] = 1
        V[0*p + k1, col_offset] = 1
        W[k1*n + 0, col_offset] = 1
        col_offset += 1
        
        # t2
        U[1*m + 0, col_offset] = 1
        V[0*p + k0, col_offset] = 1
        W[k0*n + 1, col_offset] = 1
        col_offset += 1
        
        # t3
        U[1*m + 0, col_offset] = 1
        V[0*p + k1, col_offset] = 1
        W[k1*n + 1, col_offset] = 1
        col_offset += 1
        
        # Block 2: A[:, 1:3] * B[1:3, k0:k1]  (2x2 * 2x2)
        # A indices: i*m + 1, i*m + 2
        # B indices: 1*p + k0, 1*p + k1, 2*p + k0, 2*p + k1
        # C indices: k0*n + i, k1*n + i
        
        # Strassen terms for 2x2 * 2x2
        # A_sub = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
        # B_sub = [[B[1,k0], B[1,k1]], [B[2,k0], B[2,k1]]]
        
        # M1 = (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        # U: A[0,1] is 0*m+1, A[1,2] is 1*m+2
        # V: B[1,k0] is 1*p+k0, B[2,k1] is 2*p+k1
        # W: M1 contributes to C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        # C[0,k0] = M1 + M4 - M5 + M7
        # C[0,k1] = M3 + M5
        # C[1,k0] = M2 + M4
        # C[1,k1] = M1 - M2 + M3 + M6
        
        # We need to add coefficients to W.
        
        # M1
        U[0*m + 1, col_offset] = 1
        U[1*m + 2, col_offset] = 1
        V[1*p + k0, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        # W contributions:
        W[k0*n + 0, col_offset] = 1  # C[0,k0] += M1
        W[k1*n + 1, col_offset] = 1  # C[1,k1] += M1
        col_offset += 1
        
        # M2 = (A[1,1] + A[1,2]) * B[1,k0]
        U[1*m + 1, col_offset] = 1
        U[1*m + 2, col_offset] = 1
        V[1*p + k0, col_offset] = 1
        # W contributions:
        W[k0*n + 1, col_offset] = 1  # C[1,k0] += M2
        W[k1*n + 1, col_offset] = -1 # C[1,k1] -= M2
        col_offset += 1
        
        # M3 = A[0,1] * (B[1,k1] - B[2,k1])
        U[0*m + 1, col_offset] = 1
        V[1*p + k1, col_offset] = 1
        V[2*p + k1, col_offset] = -1
        # W contributions:
        W[k1*n + 0, col_offset] = 1  # C[0,k1] += M3
        W[k1*n + 1, col_offset] = 1  # C[1,k1] += M3
        col_offset += 1
        
        # M4 = A[1,2] * (B[2,k0] - B[1,k0])
        U[1*m + 2, col_offset] = 1
        V[2*p + k0, col_offset] = 1
        V[1*p + k0, col_offset] = -1
        # W contributions:
        W[k0*n + 0, col_offset] = 1  # C[0,k0] += M4
        W[k0*n + 1, col_offset] = 1  # C[1,k0] += M4
        col_offset += 1
        
        # M5 = (A[0,1] + A[0,2]) * B[2,k1]
        U[0*m + 1, col_offset] = 1
        U[0*m + 2, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        # W contributions:
        W[k0*n + 0, col_offset] = -1 # C[0,k0] -= M5
        W[k1*n + 0, col_offset] = 1  # C[0,k1] += M5
        col_offset += 1
        
        # M6 = (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1*m + 1, col_offset] = 1
        U[0*m + 1, col_offset] = -1
        V[1*p + k0, col_offset] = 1
        V[1*p + k1, col_offset] = 1
        # W contributions:
        W[k1*n + 1, col_offset] = 1  # C[1,k1] += M6
        col_offset += 1
        
        # M7 = (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0*m + 2, col_offset] = 1
        U[1*m + 2, col_offset] = -1
        V[2*p + k0, col_offset] = 1
        V[2*p + k1, col_offset] = 1
        # W contributions:
        W[k0*n + 0, col_offset] = 1  # C[0,k0] += M7
        col_offset += 1

    # Verify the solution
    def verify(U, V, W):
        r = U.shape[1]
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        # Build expected tensor
        T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T_exp[a_idx, b_idx, c_idx] = 1
        return np.array_equal(T_check, T_exp)

    if verify(U, V, W):
        # We have a valid rank-44 solution.
        # Try to see if we can improve?
        # Given time constraints, return this.
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if construction failed (should not happen)
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.int64)
        V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.int64)
        W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_sb[a_idx, col] = 1
                    V_sb[b_idx, col] = 1
                    W_sb[c_idx, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
