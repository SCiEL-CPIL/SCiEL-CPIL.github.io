import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Verified Strassen terms for 2x2x2.
    # For each term: (a_vec, b_vec, c_vec)
    # U[i, j, r] -> row i in A (2 rows), col j (2 cols) -> flattened i*2+j -> 4 elements
    # V[j, k, r] -> row j in B (2 rows), col k (2 cols) -> flattened j*2+k -> 4 elements
    # W[k, i, r] -> row k in C (2 cols), col i (2 rows) -> flattened k*2+i -> 4 elements
    #
    # The 4 elements of a_vec are ordered [A00, A01, A10, A11]
    # The 4 elements of b_vec are ordered [B00, B01, B10, B11]
    # The 4 elements of c_vec are ordered [C00, C10, C01, C11]
    #   where C00 is C[0,0], C10 is C[1,0], C01 is C[0,1], C11 is C[1,1]
    #   Index: C row i, col k. Flattened: k*2 + i.
    #   C00: k=0, i=0 -> 0*2+0 = 0
    #   C10: k=0, i=1 -> 0*2+1 = 1
    #   C01: k=1, i=0 -> 1*2+0 = 2
    #   C11: k=1, i=1 -> 1*2+1 = 3

    # Strassen terms for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = (A00+A01)B11
    # M5 = (A01+A11)(B10+B11)
    # M6 = (A00-A10)(B01+B11)
    # M7 = (A01-A11)(B00+B10)

    # C = A*B
    # C00 = A00B00 + A01B10
    # C10 = A10B00 + A11B10
    # C01 = A00B01 + A01B11
    # C11 = A10B01 + A11B11

    # Let's express each C_ij as a linear combination of M1..M7
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M2 = A10B00 + A11B00
    # M3 = A00B10 - A00B11
    # M4 = A00B11 + A01B11
    # M5 = A01B10 + A01B11 + A11B10 + A11B11
    # M6 = A00B01 + A00B11 - A10B01 - A10B11
    # M7 = A01B00 + A01B10 - A11B00 - A11B10

    # C00 = A00B00 + A01B10
    # C10 = A10B00 + A11B10
    # C01 = A00B01 + A01B11
    # C11 = A10B01 + A11B11

    # Let's check standard Strassen reconstruction:
    # C00 = M1 - M4 + M6 + M7
    # C10 = M3 + M4 - M5 + M7  (Wait, let's derive properly)
    # C01 = M2 + M5 - M6 - M7
    # C11 = M1 + M2 - M3 + M5

    # Let's verify C10 = A10B00 + A11B10
    # M3 + M4 - M5 + M7 = (A00B10 - A00B11) + (A00B11 + A01B11) - (A01B10 + A01B11 + A11B10 + A11B11) + (A01B00 + A01B10 - A11B00 - A11B10)
    # = A00B10 + A01B11 - A01B10 - A01B11 - A11B10 - A11B11 + A01B00 + A01B10 - A11B00 - A11B10
    # = A00B10 - A11B10 - A11B11 + A01B00 - A11B00 - A11B10
    # This does not look like A10B00 + A11B10. My term assignments might be wrong.

    # Let's just use the known correct mapping from a standard reference or derive carefully.
    # Standard Strassen:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B11)
    # P4 = A11*(B01-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10+A11)*(B01-B00)  -> Wait, different variants exist.

    # Let's use the exact program's terms which were verified to work.
    # The previous program had:
    # terms = [
    #     ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
    #     ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),   # M2
    #     ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),   # M3
    #     ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),   # M4
    #     ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),   # M5
    #     ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
    #     ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])    # M7
    # ]
    # This yielded rank 35 and score 1.1429.
    # The "c_vec" order was [C00, C10, C01, C11] -> indices 0,1,2,3 -> k*2+i.
    # This is consistent with my index mapping above.

    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]

    num_blocks = p // 2  # 5 blocks of 2x2
    R = 7 * num_blocks  # rank 35

    U = np.zeros((n*m, R), dtype=float)
    V = np.zeros((m*p, R), dtype=float)
    W = np.zeros((p*n, R), dtype=float)

    col = 0
    for q in range(num_blocks):
        for a_vec, b_vec, c_vec in terms:
            # U indices: i*n + j? No, A is n x m. Flattened i*m + j.
            # n=2, m=2. So U[i*2+j, col]
            # a_vec is [A00, A01, A10, A11] -> [0,0, 0,1, 1,0, 1,1]
            for idx in range(4):
                i = idx // 2
                j = idx % 2
                U[i*m + j, col] = a_vec[idx]

            # V indices: j*p + k. B is m x p.
            # j is row (0,1), k is col (0..9).
            # For block q, the local 2x2 B block has cols 2q, 2q+1.
            # b_vec is [B00, B01, B10, B11] -> [B[0,2q], B[0,2q+1], B[1,2q], B[1,2q+1]]
            # idx 0: B[0,2q] -> j=0, k=2q
            # idx 1: B[0,2q+1] -> j=0, k=2q+1
            # idx 2: B[1,2q] -> j=1, k=2q
            # idx 3: B[1,2q+1] -> j=1, k=2q+1
            V[0*p + 2*q, col] = b_vec[0]
            V[0*p + 2*q+1, col] = b_vec[1]
            V[1*p + 2*q, col] = b_vec[2]
            V[1*p + 2*q+1, col] = b_vec[3]

            # W indices: k*n + i. C is p x n.
            # k is col of C (0..9), i is row of C (0,1).
            # c_vec is [C00, C10, C01, C11]
            # C00: C[0,2q] -> i=0, k=2q -> idx 2q*2+0 = 4q
            # C10: C[1,2q] -> i=1, k=2q -> idx 2q*2+1 = 4q+1
            # C01: C[0,2q+1] -> i=0, k=2q+1 -> idx (2q+1)*2+0 = 4q+2
            # C11: C[1,2q+1] -> i=1, k=2q+1 -> idx (2q+1)*2+1 = 4q+3
            W[4*q + 0, col] = c_vec[0]
            W[4*q + 1, col] = c_vec[1]
            W[4*q + 2, col] = c_vec[2]
            W[4*q + 3, col] = c_vec[3]

            col += 1

    # Verification
    np.random.seed(12345)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B

    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for r in range(R):
                # u_val = sum_{i',j} U[i*m+j', r] A[i', j] ?
                # No, the tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Here:
                # U is (n*m, R) -> U[i*m+j, r]
                # V is (m*p, R) -> V[j*p+k, r]
                # W is (p*n, R) -> W[k*n+i, r]
                #
                # C[i,k] = sum_{j} A[i,j] B[j,k]
                # = sum_{j} A[i,j] B[j,k] * 1
                # = sum_r (sum_{j} U[i*m+j, r] A[i,j]) * (sum_{j'} V[j'*p+k, r] B[j',k]) * W[k*n+i, r]
                # Wait, the indices must match for the product to contract correctly.
                # The standard form is T[i,j,k] = sum_r u_r[i,j] v_r[j,k] w_r[k,i]
                # Here U depends on i,j. V depends on j,k. W depends on k,i.
                #
                # Let's check the contraction:
                # u_val = sum_{j} U[i*m+j, r] A[i,j]  -> This sums over j, which is correct for the U part.
                # v_val = sum_{j'} V[j'*p+k, r] B[j',k] -> This sums over j', which is correct for the V part.
                # But u_val and v_val are scalars. Their product is a scalar.
                # The j index in U and j' in V are independent in this scalar multiplication.
                # This is WRONG. The tensor decomposition requires that the free indices of U, V, W are contracted properly.
                #
                # Actually, the identity is:
                # C[i,k] = sum_{j} A[i,j] B[j,k]
                # If we have T[i,j,k] = sum_r u_{ij}^r v_{jk}^r w_{ki}^r, then
                # sum_{j} A[i,j] B[j,k] T[i,j,k] ... no, that's not right.
                #
                # The correct identity for tensor decomposition of matrix multiplication is:
                # C[i,k] = sum_{r} ( sum_{j} U[i,j,r] A[i,j] ) ( sum_{j'} V[j',k,r] B[j',k] ) W[k,i,r]
                # This is NOT generally true unless U, V, W are structured such that the sums over j and j' produce the same "j" value, or the contraction is different.
                #
                # Wait, the standard formulation is:
                # The tensor M_{ijk} = delta_{ik} (for identity?) No, it's the multiplication tensor.
                # M[i,j,k] = 1 if i,j,k correspond to a term A[i,j]B[j,k] contributing to C[i,k].
                # So M[i,j,k] = 1 if j is the intermediate index.
                # Decomposition: M[i,j,k] = sum_r u^r_{ij} v^r_{jk} w^r_{ki}
                # Then C[i,k] = sum_{j} A[i,j] B[j,k] M[i,j,k] = sum_{j} A[i,j] B[j,k] (sum_r u^r_{ij} v^r_{jk} w^r_{ki})
                # = sum_r w^r_{ki} (sum_j u^r_{ij} A[i,j]) (sum_j v^r_{jk} B[j,k])
                # This matches my formula above! The key is that j is summed over in BOTH the U and V sums, and they are the SAME j.
                # So my verification code should work.

                u_val = 0.0
                for j in range(m):
                    u_val += U[i*m + j, r] * A[i, j]

                v_val = 0.0
                for jp in range(m):
                    v_val += V[jp*p + k, r] * B[jp, k]

                s += u_val * v_val * W[k*n + i, r]
            C_decomp[i, k] = s

    if not np.allclose(C_true, C_decomp, atol=1e-10):
        # Fallback to schoolbook
        R_school = n * m * p  # 40
        U = np.zeros((n*m, R_school), dtype=float)
        V = np.zeros((m*p, R_school), dtype=float)
        W = np.zeros((p*n, R_school), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        if time.time() > deadline - 10:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
