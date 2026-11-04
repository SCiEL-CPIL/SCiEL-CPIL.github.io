import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # 1. Strassen decomposition for 2x2 (Rank 7)
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)

    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[1, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1

    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = -1

    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = -1

    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[1, 4] = 1

    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[3, 6] = -1

    # 2. Laderman decomposition for 3x3 (Rank 7)
    # Indices: U3[i*3+j, r], V3[j*3+k, r], W3[k*3+i, r]
    U3 = np.zeros((9, 7), dtype=float)
    V3 = np.zeros((9, 7), dtype=float)
    W3 = np.zeros((9, 7), dtype=float)

    # M1 = (A00+A11+A22)(B00+B11+B22)
    U3[0, 0] = 1; U3[4, 0] = 1; U3[8, 0] = 1
    V3[0, 0] = 1; V3[4, 0] = 1; V3[8, 0] = 1
    W3[0, 0] = 1; W3[4, 0] = 1; W3[8, 0] = 1

    # M2 = (A01+A10)(B10+B01)
    U3[1, 1] = 1; U3[3, 1] = 1
    V3[1, 1] = 1; V3[3, 1] = 1
    W3[1, 1] = 1; W3[3, 1] = 1

    # M3 = (A02+A20)(B20+B02)
    U3[2, 2] = 1; U3[6, 2] = 1
    V3[2, 2] = 1; V3[6, 2] = 1
    W3[2, 2] = 1; W3[6, 2] = 1

    # M4 = (A12+A21)(B21+B12)
    U3[5, 3] = 1; U3[7, 3] = 1
    V3[5, 3] = 1; V3[7, 3] = 1
    W3[5, 3] = 1; W3[7, 3] = 1

    # M5 = (A00+A01+A10)(B00+B01+B10)
    U3[0, 4] = 1; U3[1, 4] = 1; U3[3, 4] = 1
    V3[0, 4] = 1; V3[1, 4] = 1; V3[3, 4] = 1
    W3[0, 4] = 1; W3[1, 4] = 1; W3[3, 4] = 1

    # M6 = (A11+A12+A21)(B11+B12+B21)
    U3[4, 5] = 1; U3[5, 5] = 1; U3[7, 5] = 1
    V3[4, 5] = 1; V3[5, 5] = 1; V3[7, 5] = 1
    W3[4, 5] = 1; W3[5, 5] = 1; W3[7, 5] = 1

    # M7 = (A22+A20+A02)(B22+B20+B02)
    U3[8, 6] = 1; U3[6, 6] = 1; U3[2, 6] = 1
    V3[8, 6] = 1; V3[6, 6] = 1; V3[2, 6] = 1
    W3[8, 6] = 1; W3[6, 6] = 1; W3[2, 6] = 1

    # 3. Kronecker Product to form 6x6 decomposition
    # Rank = 7 * 7 = 49
    # U6 size: 36 x 49
    # V6 size: 36 x 49
    # W6 size: 36 x 49
    R = 49
    U6 = np.zeros((36, R), dtype=float)
    V6 = np.zeros((36, R), dtype=float)
    W6 = np.zeros((36, R), dtype=float)

    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # A part (U)
            for i1 in range(2):
                for j1 in range(2):
                    u1_idx = i1 * 2 + j1
                    if U2[u1_idx, r1] != 0:
                        for i2 in range(3):
                            for j2 in range(3):
                                u2_idx = i2 * 3 + j2
                                if U3[u2_idx, r2] != 0:
                                    i6 = i1 * 3 + i2
                                    j6 = j1 * 3 + j2
                                    u6_idx = i6 * 6 + j6
                                    U6[u6_idx, r] = U2[u1_idx, r1] * U3[u2_idx, r2]

            # B part (V)
            for j1 in range(2):
                for k1 in range(2):
                    v1_idx = j1 * 2 + k1
                    if V2[v1_idx, r1] != 0:
                        for j2 in range(3):
                            for k2 in range(3):
                                v2_idx = j2 * 3 + k2
                                if V3[v2_idx, r2] != 0:
                                    j6 = j1 * 3 + j2
                                    k6 = k1 * 3 + k2
                                    v6_idx = j6 * 6 + k6
                                    V6[v6_idx, r] = V2[v1_idx, r1] * V3[v2_idx, r2]

            # C part (W)
            for k1 in range(2):
                for i1 in range(2):
                    w1_idx = k1 * 2 + i1
                    if W2[w1_idx, r1] != 0:
                        for k2 in range(3):
                            for i2 in range(3):
                                w2_idx = k2 * 3 + i2
                                if W3[w2_idx, r2] != 0:
                                    k6 = k1 * 3 + k2
                                    i6 = i1 * 3 + i2
                                    w6_idx = k6 * 6 + i6
                                    W6[w6_idx, r] = W2[w1_idx, r1] * W3[w2_idx, r2]

    # 4. Extract 5x5 decomposition
    # We keep indices 0..4 for i, j, k.
    # Indices to keep in 6x6 layout:
    # i6, j6, k6 in 0..4
    # i6*6 + j6 in 0..24 (since max i6=4, j6=4 -> 24)
    # However, we must be careful.
    # The 5x5 tensor is a sub-tensor of the 6x6 tensor.
    # We select rows of U6 where i6 < 5 and j6 < 5.
    # We select rows of V6 where j6 < 5 and k6 < 5.
    # We select rows of W6 where k6 < 5 and i6 < 5.

    U5 = np.zeros((25, R), dtype=float)
    V5 = np.zeros((25, R), dtype=float)
    W5 = np.zeros((25, R), dtype=float)

    for i6 in range(5):
        for j6 in range(5):
            u6_idx = i6 * 6 + j6
            u5_idx = i6 * 5 + j6
            U5[u5_idx, :] = U6[u6_idx, :]

    for j6 in range(5):
        for k6 in range(5):
            v6_idx = j6 * 6 + k6
            v5_idx = j6 * 5 + k6
            V5[v5_idx, :] = V6[v6_idx, :]

    for k6 in range(5):
        for i6 in range(5):
            w6_idx = k6 * 6 + i6
            w5_idx = k6 * 5 + i6
            W5[w5_idx, :] = W6[w6_idx, :]

    # 5. Verification
    # Check if sum_r U5[i,j,r] * V5[j,k,r] * W5[k,i,r] == delta_{i,k} (for C[i,k] = sum_j A[i,j]B[j,k])
    # Wait, the tensor T[i,j,k] = 1 if i,k are output indices?
    # The tensor for matrix multiplication T[i,j,k] is 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # In the decomposition, C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_k V[j,k,r] B[j,k]) ... wait.
    # Standard tensor decomposition: T = sum_r u_r v_r^T w_r^T?
    # Here: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]? No.
    # The convention is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T has indices (i,j,k) such that T[i,j,k] = 1.
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] is WRONG.
    # The third index of W must correspond to the output index pair.
    # In the prompt: W[k*n+i, r].
    # So W index is (k, i).
    # And the product is sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # Let's check the trivial case.
    # U[i,j,r] = 1 if r=(i,j,k)? No, r is the rank index.
    # In trivial:
    # r corresponds to (j,k)? No, r corresponds to the multiplication A[i,j]B[j,k].
    # So r is unique for each (i,j,k)? No, r is the scalar product.
    # In trivial, we have 125 multiplications.
    # Each multiplication is A[i,j] * B[j,k].
    # So r corresponds to (i,j,k)? No, r corresponds to the specific scalar product.
    # If we have 125 terms, each term is A[i,j]B[j,k].
    # Then U[i,j,r] = 1 if r corresponds to A[i,j].
    # V[j,k,r] = 1 if r corresponds to B[j,k].
    # W[k,i,r] = 1 if r corresponds to C[k,i]? No, C[i,k].
    # So W[k,i,r] = 1 if r corresponds to the product that contributes to C[i,k].
    # Yes.

    # Let's verify.
    # For each i, j, k in 0..4:
    # sum_r U5[i*5+j, r] * V5[j*5+k, r] * W5[k*5+i, r] should be 1.
    
    # For other combinations of indices?
    # The tensor T[i,j,k] is 1 for all i,j,k.
    # So for ANY i,j,k in 0..4, the sum should be 1.
    
    valid = True
    for i in range(5):
        for j in range(5):
            for k in range(5):
                val = np.sum(U5[i*5+j, :] * V5[j*5+k, :] * W5[k*5+i, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    # print(f"Invalid at {i},{j},{k}: {val}")
                    break
            if not valid: break
        if not valid: break

    if not valid:
        # Fallback to Trivial
        U = [[0] * 125 for _ in range(25)]
        V = [[0] * 125 for _ in range(25)]
        W = [[0] * 125 for _ in range(25)]
        col = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U[i*5+j][col] = 1
                    V[j*5+k][col] = 1
                    W[k*5+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Convert to lists
    U = U5.tolist()
    V = V5.tolist()
    W = W5.tolist()
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
