"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0

    # Total standard terms
    total_terms = n * m * p
    # We will use rank 59
    rank = total_terms - 1

    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Define the specific 2x2x2 block to replace with Strassen
    strassen_block = {(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)}

    col = 0

    # Add all standard terms except those in the Strassen block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_block:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    # Add Strassen terms for the block i in {0,1}, j in {0,1}, k in {0,1}
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for u_idx, coeff in u_coeffs:
            U[u_idx][col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx][col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx][col] += coeff
        col += 1

    # Pre-calculate indices
    u00, u01 = 0, 1
    u10, u11 = 5, 6
    v00, v01 = 0, 1
    v10, v11 = 6, 7
    w00, w01 = 0, 2
    w10, w11 = 1, 3

    # M1 = (A00 + A11)(B00 + B11) -> C00 +1, C11 +1
    add_term([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)])
    # M2 = (A10 + A11)B00 -> C10 +1, C11 -1
    add_term([(u10, 1), (u11, 1)], [(v00, 1)], [(w10, 1), (w11, -1)])
    # M3 = A00(B01 - B11) -> C01 +1, C11 +1
    add_term([(u00, 1)], [(v01, 1), (v11, -1)], [(w01, 1), (w11, 1)])
    # M4 = A11(B10 - B00) -> C00 +1, C10 +1
    add_term([(u11, 1)], [(v10, 1), (v00, -1)], [(w00, 1), (w10, 1)])
    # M5 = (A00 + A01)B11 -> C00 -1, C01 +1
    add_term([(u00, 1), (u01, 1)], [(v11, 1)], [(w00, -1), (w01, 1)])
    # M6 = (A10 - A00)(B00 + B01) -> C11 +1
    add_term([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], [(w11, 1)])
    # M7 = (A01 - A11)(B10 + B11) -> C00 +1
    add_term([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], [(w00, 1)])

    # Now try to reduce rank further by combining terms
    # We have 60 - 8 + 7 = 59 terms
    # Try to find a linear combination that reduces rank
    # This is a search problem. We'll use a simple greedy approach.
    # But first, let's check if we can do better with a different block split.
    # For 2x5x6, we can split into 2x2x6 and 2x3x6
    # 2x2x6: rank 2*2*6 = 24, with Strassen on 2x2x2: 24 - 8 + 7 = 23
    # 2x3x6: rank 2*3*6 = 36, with Strassen on 2x2x2: 36 - 8 + 7 = 35
    # Total: 23 + 35 = 58

    # Let's implement this split
    U2 = [[0] * 58 for _ in range(n * m)]
    V2 = [[0] * 58 for _ in range(m * p)]
    W2 = [[0] * 58 for _ in range(p * n)]
    col2 = 0

    # First part: 2x2x6 (i in {0,1}, j in {0,1}, k in {0,1,2,3,4,5})
    # Use Strassen for j in {0,1}, k in {0,1}
    strassen_block2 = {(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)}

    for i in range(n):
        for j in range(2):
            for k in range(p):
                if (i, j, k) in strassen_block2:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U2[u_idx][col2] = 1
                V2[v_idx][col2] = 1
                W2[w_idx][col2] = 1
                col2 += 1

    # Add Strassen terms for the block i in {0,1}, j in {0,1}, k in {0,1}
    def add_term2(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col2
        for u_idx, coeff in u_coeffs:
            U2[u_idx][col2] += coeff
        for v_idx, coeff in v_coeffs:
            V2[v_idx][col2] += coeff
        for w_idx, coeff in w_coeffs:
            W2[w_idx][col2] += coeff
        col2 += 1

    # M1
    add_term2([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)])
    # M2
    add_term2([(u10, 1), (u11, 1)], [(v00, 1)], [(w10, 1), (w11, -1)])
    # M3
    add_term2([(u00, 1)], [(v01, 1), (v11, -1)], [(w01, 1), (w11, 1)])
    # M4
    add_term2([(u11, 1)], [(v10, 1), (v00, -1)], [(w00, 1), (w10, 1)])
    # M5
    add_term2([(u00, 1), (u01, 1)], [(v11, 1)], [(w00, -1), (w01, 1)])
    # M6
    add_term2([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], [(w11, 1)])
    # M7
    add_term2([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], [(w00, 1)])

    # Second part: 2x3x6 (i in {0,1}, j in {2,3,4}, k in {0,1,2,3,4,5})
    # Use Strassen for i in {0,1}, j in {2,3}, k in {0,1}
    strassen_block3 = {(0,2,0), (0,2,1), (0,3,0), (0,3,1), (1,2,0), (1,2,1), (1,3,0), (1,3,1)}

    for i in range(n):
        for j in range(2, m):
            for k in range(p):
                if (i, j, k) in strassen_block3:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U2[u_idx][col2] = 1
                V2[v_idx][col2] = 1
                W2[w_idx][col2] = 1
                col2 += 1

    # Add Strassen terms for the block i in {0,1}, j in {2,3}, k in {0,1}
    u20, u21 = 0*5+2, 0*5+3
    u30, u31 = 1*5+2, 1*5+3
    v20, v21 = 2*6+0, 2*6+1
    v30, v31 = 3*6+0, 3*6+1
    w20, w21 = 0*2+0, 1*2+0
    w30, w31 = 0*2+1, 1*2+1

    # M1
    add_term2([(u20, 1), (u31, 1)], [(v20, 1), (v31, 1)], [(w20, 1), (w31, 1)])
    # M2
    add_term2([(u30, 1), (u31, 1)], [(v20, 1)], [(w30, 1), (w31, -1)])
    # M3
    add_term2([(u20, 1)], [(v21, 1), (v31, -1)], [(w21, 1), (w31, 1)])
    # M4
    add_term2([(u31, 1)], [(v30, 1), (v20, -1)], [(w20, 1), (w30, 1)])
    # M5
    add_term2([(u20, 1), (u21, 1)], [(v31, 1)], [(w20, -1), (w21, 1)])
    # M6
    add_term2([(u30, 1), (u20, -1)], [(v20, 1), (v21, 1)], [(w31, 1)])
    # M7
    add_term2([(u21, 1), (u31, -1)], [(v30, 1), (v31, 1)], [(w20, 1)])

    # Check if col2 is 58
    assert col2 == 58, f"Expected 58, got {col2}"

    # Verify the decomposition
    def verify(U, V, W, n, m, p, rank):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)

        # For each (i, j, k), sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i*m+j, :]
                    v_row = V_np[j*p+k, :]
                    w_row = W_np[k*n+i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-10:
                        return False
        return True

    if verify(U2, V2, W2, n, m, p, 58):
        return {"U": U2, "V": V2, "W": W2}
    else:
        # Fall back to rank 59
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
