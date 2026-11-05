import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 4, 4, 7
    total_rank = 93

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    current_rank = 0

    # 7-term 2x2x2 Strassen: each term is (x_pattern, y_pattern, c_pattern)
    # x_pattern, y_pattern, c_pattern are 2x2 arrays
    strassen_2x2 = [
        # Term 0: M0 = (a11+a22)(b11+b22)
        (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]),
         np.array([[1, 0], [0, 1]])),
        # Term 1: M1 = (a21+a22)b11
        (np.array([[0, 0], [1, 1]]), np.array([[1, 0], [0, 0]]),
         np.array([[0, 0], [1, -1]])),
        # Term 2: M2 = a11(b21-b11)
        (np.array([[1, 0], [0, 0]]), np.array([[0, 1], [0, -1]]),
         np.array([[0, 1], [0, 1]])),
        # Term 3: M3 = a22(b12-b22)
        (np.array([[0, 0], [0, 1]]), np.array([[1, 0], [-1, 0]]),
         np.array([[1, 0], [1, 0]])),
        # Term 4: M4 = (a11+a12)b22
        (np.array([[1, 1], [0, 0]]), np.array([[0, 0], [0, 1]]),
         np.array([[-1, 1], [0, 0]])),
        # Term 5: M5 = (a21-a11)(b11+b12)
        (np.array([[-1, 0], [1, 0]]), np.array([[1, 1], [0, 0]]),
         np.array([[0, 0], [0, 1]])),
        # Term 6: M6 = (a12-a22)(b21+b22)
        (np.array([[0, 1], [0, -1]]), np.array([[0, 0], [1, 1]]),
         np.array([[1, 0], [0, 0]])),
    ]

    # 7-term 4x4x4 Strassen: (x_pattern, y_pattern) are 4x4, c_pattern is 4x4
    # Using the standard decomposition:
    # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    # M1 = (A10+A11)B00 -> C10 += M1, C11 -= M1
    # M2 = A00(B10-B00) -> C01 += M2, C11 += M2  (wait, let me be more careful)
    # Standard Strassen 4x4:
    # C = A*B where A, B, C are 4x4, split into 2x2 blocks of 2x2
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B00)
    # M3 = A11(B01-B11)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0 - M4 + M5
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 - M3 + M6

    # Define patterns for 4x4 (using 2x2 block indices 0,1,2,3 for TL,TR,BL,BR)
    strassen_4x4 = [
        # M0 = (A00+A11)(B00+B11)
        (np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
         np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
         np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
        # M1 = (A10+A11)B00
        (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]]),
         np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
         np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1]])),
        # M2 = A00(B10-B00)
        (np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
         np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, -1, 0], [0, 1, 0, -1]]),
         np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]])),
        # M3 = A11(B01-B11)
        (np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
         np.array([[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, -1]]),
         np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])),
        # M4 = (A00+A01)B11
        (np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
         np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
         np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])),
        # M5 = (A10-A00)(B00+B01)
        (np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]),
         np.array([[1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
         np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]])),
        # M6 = (A01-A11)(B10+B11)
        (np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
         np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]]),
         np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])),
    ]

    # Part 1: 4x4x4 Strassen (rank 49 = 7*7)
    for m_idx in range(7):
        x_4x4, y_4x4, c_4x4 = strassen_4x4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_2x2, y_2x2, c_2x2 = strassen_2x2[t_idx]

            # Build U contributions: for each 2x2 block (bi,bj) in 4x4
            for bi in range(2):
                for bj in range(2):
                    block_val_x = x_4x4[bi*2:(bi+1)*2, bj*2:(bj+1)*2]
                    # x_4x4 is structured so blocks are either 0 or scalar*identity
                    # Actually let's just directly use the patterns
                    pass
            
            # More direct approach: iterate over all positions
            for i in range(4):
                for j in range(4):
                    U_val = 0.0
                    for t2 in range(2):
                        for s2 in range(2):
                            # The 4x4 pattern is a Kronecker product of 2x2 block patterns
                            # x_4x4[i,j] = sum over block contributions
                            pass
                    # Just read directly from the 4x4 array
                    U[i * m + j, r] = x_4x4[i, j] * 1.0  # placeholder
            # This approach is getting complicated. Let me use the original method.
            U[:, r] = 0.0
            V[:, r] = 0.0
            W[:, r] = 0.0
            
            # For each 2x2 sub-block position in the 4x4
            for bi in range(2):
                for bj in range(2):
                    x_block = x_4x4[bi*2:(bi+1)*2, bj*2:(bj+1)*2]
                    y_block = y_4x4[bi*2:(bi+1)*2, bj*2:(bj+1)*2]
                    c_block = c_4x4[bi*2:(bi+1)*2, bj*2:(bj+1)*2]
                    
                    # Check if this block is non-zero and use 2x2 Strassen term
                    if np.any(x_block != 0) or np.any(y_block != 0):
                        # For simplicity, handle the structured case
                        pass

    # Let me restart with the original working approach but restructured
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    current_rank = 0

    # Original Strassen 2x2 patterns as dicts
    strassen_2x2_dicts = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]

    # Strassen 4x4x4: 7 outer terms as (x_block_pattern, y_block_pattern, c_block_pattern)
    # Each is a dict from (block_row, block_col) to scalar
    strassen_4x4_dicts = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    # Part 1: 4x4x4 Strassen (rank 49)
    for m_idx in range(7):
        x_def, y_def, c_def = strassen_4x4_dicts[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2_dicts[t_idx]

            for (li, lj), cx in x_sub.items():
                for (br, bc), cb in x_def.items():
                    i = 2 * br + li
                    j = 2 * bc + lj
                    U[i * m + j, r] += cx * cb

            for (li, lj), cy in y_sub.items():
                for (br, bc), cb in y_def.items():
                    i = 2 * br + li
                    j = 2 * bc + lj
                    V[i * p + j, r] += cy * cb

            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_def.items():
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    W[k_final * n + i_final, r] += cc * cb_c

    # Part 2: 4x4x3 as 4 * (2x2x3), each of rank 11
    def add_2x2x3(ar, ac, br, cr):
        nonlocal current_rank

        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2_dicts[t_idx]

            for (li, lj), cx in x_sub.items():
                i = 2 * ar + li
                j = 2 * ac + lj
                U[i * m + j, r] += cx

            for (li, lj), cy in y_sub.items():
                i = 2 * br + li
                j = 4 + lj
                V[i * p + j, r] += cy

            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = 2 * cr + li
                k_final = 4 + lj
                W[k_final * n + i_final, r] += cc

        # Schoolbook for column 6
        for di in range(2):
            for dj in range(2):
                r = current_rank
                current_rank += 1
                U[(2 * ar + di) * m + (2 * ac + dj), r] = 1.0
                V[(2 * br + dj) * p + 6, r] = 1.0
                W[6 * n + (2 * cr + di), r] = 1.0

    add_2x2x3(0, 0, 0, 0)
    add_2x2x3(0, 1, 1, 0)
    add_2x2x3(1, 0, 0, 1)
    add_2x2x3(1, 1, 1, 1)

    assert current_rank == 93, f"Expected 93, got {current_rank}"

    U = U[:, :current_rank].tolist()
    V = V[:, :current_rank].tolist()
    W = W[:, :current_rank].tolist()

    if not _verify(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")

    return {"U": U, "V": V, "W": W}
