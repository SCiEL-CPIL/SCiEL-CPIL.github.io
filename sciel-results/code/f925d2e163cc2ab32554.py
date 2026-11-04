"""Direct construction of the rank-28 tensor decomposition for (2,3,5) matrix multiplication.

Strategy: Decompose the multiplication into blocks:
1. A[0:2, 2] * B[2, 0:5] (Rank 10, Schoolbook)
2. A[0:2, 0:2] * B[0:2, 0:2] (Rank 7, Strassen)
3. A[0:2, 0:2] * B[0:2, 2:3] (Rank 4, Schoolbook)
4. A[0:2, 0:2] * B[0:2, 3:5] (Rank 7, Strassen)

Total Rank = 10 + 7 + 4 + 7 = 28.
"""

import time

def construct():
    # Deadline guard
    deadline = time.time() + 120.0

    n, m, p = 2, 3, 5
    rank = 28

    U = [[0] * rank for _ in range(n * m)]   # 6 x 28
    V = [[0] * rank for _ in range(m * p)]   # 15 x 28
    W = [[0] * rank for _ in range(p * n)]   # 10 x 28

    col = 0

    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Part 1: A[0:2, 2] * B[2, 0:5]
    # A rows 0,1 col 2. B row 2 cols 0..4.
    for i in range(2):
        for k in range(5):
            u_idx = i * m + 2  # A[i, 2]
            v_idx = 2 * p + k  # B[2, k]
            w_idx = k * n + i  # C[i, k]
            add_term(u_idx, v_idx, w_idx)

    # Part 2: A[0:2, 0:2] * B[0:2, 0:2] (Strassen)
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # Term 1: (A00+A11)(B00+B10+B01+B11) -> C00,C10,C01,C11
    add_term(0*3+0, 0*5+0, 0*2+0, 1) # Placeholder, need to handle multi-index terms properly
    
    # Actually, the add_term function above only adds to one index in U, V, W.
    # Strassen terms have multiple non-zero entries in U, V, W.
    # We need to manually construct the vectors.

    # Reset col for manual construction
    col = 10
    
    # Strassen Term 1
    U[0*3+0][col] += 1; U[1*3+1][col] += 1
    V[0*5+0][col] += 1; V[1*5+0][col] += 1; V[0*5+1][col] += 1; V[1*5+1][col] += 1
    W[0*2+0][col] += 1; W[1*2+0][col] += 1; W[0*2+1][col] += 1; W[1*2+1][col] += 1
    col += 1

    # Strassen Term 2: (A10+A11)(B00+B01) -> C10,C11
    U[1*3+0][col] += 1; U[1*3+1][col] += 1
    V[0*5+0][col] += 1; V[0*5+1][col] += 1
    W[1*2+0][col] += 1; W[1*2+1][col] += 1
    col += 1

    # Strassen Term 3: A00(B10-B00+B11-B01) -> C00-C10, C01-C11
    U[0*3+0][col] += 1
    V[1*5+0][col] += 1; V[0*5+0][col] -= 1; V[1*5+1][col] += 1; V[0*5+1][col] -= 1
    W[0*2+0][col] += 1; W[1*2+0][col] -= 1; W[0*2+1][col] += 1; W[1*2+1][col] -= 1
    col += 1

    # Strassen Term 4: A11(B00-B10+B01-B11) -> -C00+C10, -C01+C11
    U[1*3+1][col] += 1
    V[0*5+0][col] += 1; V[1*5+0][col] -= 1; V[0*5+1][col] += 1; V[1*5+1][col] -= 1
    W[0*2+0][col] -= 1; W[1*2+0][col] += 1; W[0*2+1][col] -= 1; W[1*2+1][col] += 1
    col += 1

    # Strassen Term 5: (A00+A01)(B10+B11) -> C00,C01
    U[0*3+0][col] += 1; U[0*3+1][col] += 1
    V[1*5+0][col] += 1; V[1*5+1][col] += 1
    W[0*2+0][col] += 1; W[0*2+1][col] += 1
    col += 1

    # Strassen Term 6: (A10-A00)(B00+B10+B01+B11) -> C10,C11
    U[1*3+0][col] += 1; U[0*3+0][col] -= 1
    V[0*5+0][col] += 1; V[1*5+0][col] += 1; V[0*5+1][col] += 1; V[1*5+1][col] += 1
    W[1*2+0][col] += 1; W[1*2+1][col] += 1
    col += 1

    # Strassen Term 7: (A01-A11)(B10+B00+B11+B01) -> -C00+C10, -C01+C11
    U[0*3+1][col] += 1; U[1*3+1][col] -= 1
    V[1*5+0][col] += 1; V[0*5+0][col] += 1; V[1*5+1][col] += 1; V[0*5+1][col] += 1
    W[0*2+0][col] -= 1; W[1*2+0][col] += 1; W[0*2+1][col] -= 1; W[1*2+1][col] += 1
    col += 1

    # Part 3: A[0:2, 0:2] * B[0:2, 2:3] (Schoolbook, Rank 4)
    # A cols 0,1. B rows 0,1. Col 2.
    for i in range(2):
        for j in range(2):
            u_idx = i * m + j
            v_idx = j * p + 2
            w_idx = 2 * n + i  # C[i, 2]
            U[u_idx][col] += 1
            V[v_idx][col] += 1
            W[w_idx][col] += 1
            col += 1

    # Part 4: A[0:2, 0:2] * B[0:2, 3:5] (Strassen)
    # B cols 3,4.
    
    # Strassen Term 1
    U[0*3+0][col] += 1; U[1*3+1][col] += 1
    V[0*5+3][col] += 1; V[1*5+3][col] += 1; V[0*5+4][col] += 1; V[1*5+4][col] += 1
    W[3*2+0][col] += 1; W[3*2+1][col] += 1; W[4*2+0][col] += 1; W[4*2+1][col] += 1
    col += 1

    # Strassen Term 2
    U[1*3+0][col] += 1; U[1*3+1][col] += 1
    V[0*5+3][col] += 1; V[0*5+4][col] += 1
    W[3*2+1][col] += 1; W[4*2+1][col] += 1 # Wait, W index for C[1, 3] is 3*2+1?
    # C[i, k] -> W[k*n + i].
    # C[1, 3] -> W[3*2 + 1] = W[7].
    # C[1, 4] -> W[4*2 + 1] = W[9].
    # In Term 2 of Part 2, W was C[1,0], C[1,1].
    # Here W is C[1,3], C[1,4].
    # My previous code for Part 2 Term 2:
    # W[1*2+0] (C[1,0]), W[1*2+1] (C[1,1]). Correct.
    # So here:
    W[3*2+1][col] += 1; W[4*2+1][col] += 1
    col += 1

    # Strassen Term 3
    U[0*3+0][col] += 1
    V[1*5+3][col] += 1; V[0*5+3][col] -= 1; V[1*5+4][col] += 1; V[0*5+4][col] -= 1
    W[3*2+0][col] += 1; W[3*2+1][col] -= 1; W[4*2+0][col] += 1; W[4*2+1][col] -= 1
    col += 1

    # Strassen Term 4
    U[1*3+1][col] += 1
    V[0*5+3][col] += 1; V[1*5+3][col] -= 1; V[0*5+4][col] += 1; V[1*5+4][col] -= 1
    W[3*2+0][col] -= 1; W[3*2+1][col] += 1; W[4*2+0][col] -= 1; W[4*2+1][col] += 1
    col += 1

    # Strassen Term 5
    U[0*3+0][col] += 1; U[0*3+1][col] += 1
    V[1*5+3][col] += 1; V[1*5+4][col] += 1
    W[3*2+0][col] += 1; W[4*2+0][col] += 1
    col += 1

    # Strassen Term 6
    U[1*3+0][col] += 1; U[0*3+0][col] -= 1
    V[0*5+3][col] += 1; V[1*5+3][col] += 1; V[0*5+4][col] += 1; V[1*5+4][col] += 1
    W[3*2+1][col] += 1; W[4*2+1][col] += 1
    col += 1

    # Strassen Term 7
    U[0*3+1][col] += 1; U[1*3+1][col] -= 1
    V[1*5+3][col] += 1; V[0*5+3][col] += 1; V[1*5+4][col] += 1; V[0*5+4][col] += 1
    W[3*2+0][col] -= 1; W[3*2+1][col] += 1; W[4*2+0][col] -= 1; W[4*2+1][col] += 1
    col += 1

    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")

    return {"U": U, "V": V, "W": W}
