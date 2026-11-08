import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Rank calculation:
    # 4x4x4 (Strassen via tensor product of 2x2x2): 49
    # 4x4x1 (Trivial): 16
    # 4x1x4 (Trivial): 16
    # 4x1x1 (Trivial): 4
    # Total: 85
    rank = 85

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # 2x2x2 Strassen factors
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices for 2x2 matrices: 0:00, 1:01, 2:10, 3:11
    u2 = np.zeros((4, 7))
    v2 = np.zeros((4, 7))
    w2 = np.zeros((4, 7))

    # Term 0: (a00+a11)(b00+b11) -> c00, c11
    u2[0, 0] = 1; u2[3, 0] = 1
    v2[0, 0] = 1; v2[3, 0] = 1
    w2[0, 0] = 1; w2[3, 0] = 1

    # Term 1: (a10+a11)b00 -> c10, c11(-1)
    u2[2, 1] = 1; u2[3, 1] = 1
    v2[0, 1] = 1
    w2[2, 1] = 1; w2[3, 1] = -1

    # Term 2: a00(b01-b11) -> c01, c11
    u2[0, 2] = 1
    v2[1, 2] = 1; v2[3, 2] = -1
    w2[1, 2] = 1; w2[3, 2] = 1

    # Term 3: a11(b10-b00) -> c00, c10
    u2[3, 3] = 1
    v2[2, 3] = 1; v2[0, 3] = -1
    w2[0, 3] = 1; w2[2, 3] = 1

    # Term 4: (a00+a01)b11 -> c00(-1), c01
    u2[0, 4] = 1; u2[1, 4] = 1
    v2[3, 4] = 1
    w2[0, 4] = -1; w2[1, 4] = 1

    # Term 5: (a10-a00)(b00+b01) -> c11
    u2[0, 5] = -1; u2[2, 5] = 1
    v2[0, 5] = 1; v2[1, 5] = 1
    w2[3, 5] = 1

    # Term 6: (a01-a11)(b10+b11) -> c00
    u2[1, 6] = 1; u2[3, 6] = -1
    v2[2, 6] = 1; v2[3, 6] = 1
    w2[0, 6] = 1

    # Construct 4x4x4 factors using tensor product logic
    # U_4x4 has shape (16, 49). Index i = (br, bc, lr, lc) -> (br*2+lr, bc*2+lc)
    # The tensor product M_4,4,4 = M_2,2,2 (x) M_2,2,2
    # For each outer term t and inner term k:
    # The factor for U is u2[:, t] (x) u2[:, k]
    # But we must be careful with the mapping of indices.
    # Let U4[:, t*7+k] = (u2[:, t] (x) u2[:, k])
    # The Kronecker product of vectors x (size 4) and y (size 4) is size 16.
    # x = [x0, x1, x2, x3], y = [y0, y1, y2, y3]
    # x (x) y = [x0y0, x0y1, x0y2, x0y3, x1y0, x1y1, x1y2, x1y3, x2y0, ...]
    # Our global index for A is (i, j) where i = br*2+lr, j = bc*2+lc.
    # Flattened index is i*4 + j.
    # i = br*2+lr, j = bc*2+lc.
    # Index = (br*2+lr)*4 + (bc*2+lc) = br*8 + lr*4 + bc*2 + lc.
    # The Kronecker product order x (x) y usually maps:
    # If x corresponds to the "block" indices (br, bc) and y to "inner" (lr, lc)?
    # Let's check standard Kronecker definition.
    # If we view A as a 2x2 matrix of 2x2 blocks.
    # A = [A11 A12; A21 A22]
    # vec(A) in row-major: [A11(0,0), A11(0,1), A11(1,0), A11(1,1), A12(0,0), ...]
    # This matches i*4+j.
    # The tensor product of the decomposition of A and B for the 2x2 level and 2x2 level.
    # Let's just explicitly construct the 4x4 factors by iterating t, k and filling the vector.
    
    col = 0
    for t in range(7):
        for k in range(7):
            # u2[:, t] and u2[:, k] are length 4 vectors.
            # We need to form the length 16 vector for U.
            # The component at (br, bc, lr, lc) is u2[br*2+lr? No.
            # In the 2x2 Strassen, the index 0,1,2,3 corresponds to (0,0), (0,1), (1,0), (1,1).
            # Let's call the indices of u2 as (i2, j2) where i2, j2 in {0,1}.
            # u2[i2*2+j2, t] is the coefficient for A[i2, j2] in term t.
            # For the 4x4 case, A[br*2+lr, bc*2+lc].
            # The term t acts on the block indices (br, bc).
            # The term k acts on the inner indices (lr, lc).
            # So U4[i, j, col] = u2[br*2+lr? No, u2 indices are 0..3.
            # u2[br*2+bc, t] ? No.
            # In 2x2 Strassen, u2[0] is a00, u2[1] is a01, u2[2] is a10, u2[3] is a11.
            # So u2[idx, t] where idx = i2*2 + j2.
            # In 4x4, the "block" indices are br, bc. The "inner" indices are lr, lc.
            # The coefficient for A[br*2+lr, bc*2+lc] in the term (t,k) is:
            # u2[br*2+bc, t] * u2[lr*2+lc, k]
            # Let's verify this.
            # If we pick a specific global index (i, j).
            # br = i // 2, lr = i % 2.
            # bc = j // 2, lc = j % 2.
            # coeff = u2[br*2+bc, t] * u2[lr*2+lc, k]
            # This looks correct.
            
            for i in range(4):
                for j in range(4):
                    br = i // 2
                    lr = i % 2
                    bc = j // 2
                    lc = j % 2
                    
                    val = u2[br*2+bc, t] * u2[lr*2+lc, k]
                    if val != 0:
                        U[i*5 + j, col] = val
                        
                    val_v = v2[bc*2+lc, t] * v2[lr*2+lc, k]
                    # Wait, B indices are (j, k).
                    # B[j, k] where j is row, k is col.
                    # In 4x4, B[br_B*2+lr_B, bc_B*2+lc_B].
                    # The term t acts on block indices (br_B, bc_B).
                    # The term k acts on inner indices (lr_B, lc_B).
                    # So V4[j, k, col] = v2[br_B*2+bc_B, t] * v2[lr_B*2+lc_B, k]
                    # Here j = br_B*2+lr_B, k_global = bc_B*2+lc_B.
                    # br_B = j // 2, lr_B = j % 2.
                    # bc_B = k_global // 2, lc_B = k_global % 2.
                    # So V4[j, k_global, col] = v2[(j//2)*2 + (k_global//2), t] * v2[(j%2)*2 + (k_global%2), k]
                    
            # Let's just fill the whole column for V and W too.
            for j in range(4):
                for k in range(4):
                    br_B = j // 2
                    lr_B = j % 2
                    bc_B = k // 2
                    lc_B = k % 2
                    val_v = v2[br_B*2+bc_B, t] * v2[lr_B*2+lc_B, k]
                    if val_v != 0:
                        V[j*5 + k, col] = val_v
                        
            for k in range(4):
                for i in range(4):
                    # C[k, i]
                    br_C = k // 2
                    lr_C = k % 2
                    bc_C = i // 2
                    lc_C = i % 2
                    val_w = w2[br_C*2+bc_C, t] * w2[lr_C*2+lc_C, k]
                    if val_w != 0:
                        W[k*4 + i, col] = val_w
            col += 1

    # 4x4x1: A[0:4, 0:4] * B[0:4, 4] -> C[0:4, 4]
    # B is 5x5. Column 4 is the 5th column.
    # A is 4x5. Rows 0-3, Cols 0-3.
    # C is 5x4. Rows 0-4, Cols 0-3.
    # C[i, 4] is not possible, C is p x n = 5 x 4.
    # Wait, C index is W[k*n+i, r] -> W[k*4+i, r].
    # C[k, i].
    # We want C[i, 4]? No, n=4, p=5.
    # A is n x m = 4 x 5.
    # B is m x p = 5 x 5.
    # C is p x n = 5 x 4.
    # C[k, i] = sum_j A[i, j] B[j, k].
    # The trivial part for the "new" dimensions.
    # A[0:4, 4] is the 5th column of A.
    # B[4, 0:5] is the 5th row of B.
    #
    # Block 1: A[0:4, 0:4] * B[0:4, 4] -> C[4, 0:4]
    # C[4, i] = sum_{j=0}^3 A[i, j] B[j, 4]
    # This requires 16 terms.
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            col += 1

    # Block 2: A[0:4, 4] * B[4, 0:4] -> C[0:4, 0:4]
    # C[k, i] = A[i, 4] B[4, k]
    for i in range(4):
        for k in range(4):
            U[i * 5 + 4, col] = 1
            V[4 * 5 + k, col] = 1
            W[k * 4 + i, col] = 1
            col += 1

    # Block 3: A[0:4, 4] * B[4, 4] -> C[4, 0:4]? No, C[4, i]?
    # C[k, i] = A[i, 4] B[4, k].
    # For k=4, C[4, i] = A[i, 4] B[4, 4].
    # This adds to C[4, i] which was already computed in Block 1?
    # Block 1 computed C[4, i] = sum_{j=0}^3 A[i, j] B[j, 4].
    # So we need to add the j=4 term: A[i, 4] B[4, 4].
    for i in range(4):
        U[i * 5 + 4, col] = 1
        V[4 * 5 + 4, col] = 1
        W[4 * 4 + i, col] = 1
        col += 1

    # Check time
    if time.time() - start_time > 110:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
