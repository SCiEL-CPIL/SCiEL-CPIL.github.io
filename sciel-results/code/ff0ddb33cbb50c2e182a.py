"""Combined: Strassen block for 3x3x3 + schoolbook for trailing 4 columns."""
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 3, 7

    # Strassen 2x2x2 decomposition embedded in 3x3x3 (k=0,1,2)
    # Splits A into 4 blocks: A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # B into 4 blocks: B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    # 7 products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # Strassen rank-7 entries for 2x2x2 (indices i,j,k in {0,1})
    # U[i*2+j], V[j*2+k], W[k*2+i]
    strassen_U = [
        [1, 0, 0, 0, 1, 0, 0],  # A[0,0]
        [0, 0, 1, 0, 0, -1, 0], # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],  # A[1,0]
        [1, 1, 0, 1, 0, 0, -1], # A[1,1]
    ]
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1]
        [1, 1, 0, -1, 1, 0, 1],  # B[1,0]
        [1, 0, -1, 0, 1, 0, 1],  # B[1,1]
    ]
    strassen_W = [
        [1, 0, 1, 0, -1, 0, 1],  # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]
        [1, 1, 0, 1, 0, 0, 0],   # C[1,0]
        [1, -1, 1, 0, 0, 1, 0],  # C[1,1]
    ]

    # Verify Strassen correctness quickly
    def _verify_strassen():
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for r in range(7):
                        s += strassen_U[i * 2 + j][r] * strassen_V[j * 2 + k][r] * strassen_W[k * 2 + i][r]
                    if s != 1:
                        return False
        return True

    strassen_ok = _verify_strassen()

    # Build full decomposition
    # For k=0,1,2 (first 3 columns): use Strassen for the 2x2 sub-block (i in {0,1}, j in {0,1})
    # For i=2 or j=2 with k in {0,1,2}: use schoolbook (3 terms each for the 3x3 part)
    # For k=3,4,5,6: use schoolbook for all i,j

    rank = 0
    U = []
    V = []
    W = []

    def add_term(u_row, u_val, v_row, v_val, w_row, w_val):
        nonlocal rank
        while len(U) <= u_row:
            U.append([0] * rank)
        while len(V) <= v_row:
            V.append([0] * rank)
        while len(W) <= w_row:
            W.append([0] * rank)
        U[u_row].append(u_val)
        V[v_row].append(v_val)
        W[w_row].append(w_val)
        rank += 1

    # Pre-size
    max_U = n * m  # 9
    max_V = m * p  # 21
    max_W = p * n  # 21
    U = [[0] * 0 for _ in range(max_U)]
    V = [[0] * 0 for _ in range(max_V)]
    W = [[0] * 0 for _ in range(max_W)]

    def add_term2(u_row, u_val, v_row, v_val, w_row, w_val):
        U[u_row].append(u_val)
        V[v_row].append(v_val)
        W[w_row].append(w_val)

    # Case 1: k in {0,1,2}, i in {0,1}, j in {0,1} -> Strassen
    if strassen_ok:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for r in range(7):
                        add_term2(i * 3 + j, strassen_U[i * 2 + j][r],
                                  j * 7 + k, strassen_V[j * 2 + k][r],
                                  k * 3 + i, strassen_W[k * 2 + i][r])
        # That's 2*2*2*7 = 56 terms for the 2x2x2 sub-block
        # Wait, that's wrong. Strassen is rank 7 for 2x2x2, meaning 7 terms total.
        # Each term r contributes to all (i,j,k) in the 2x2x2 block.
        # So we add 7 terms, each with entries for all (i,j,k) in {0,1}^3.
        pass
    # Redo: Strassen gives 7 terms for the entire 2x2x2 block
    # Each term r has U entries for (i,j) in {0,1}^2, V for (j,k) in {0,1}^2, W for (k,i) in {0,1}^2

    # Clear and restart
    U = [[0] * 0 for _ in range(max_U)]
    V = [[0] * 0 for _ in range(max_V)]
    W = [[0] * 0 for _ in range(max_W)]

    # Strassen: 7 terms covering i in {0,1}, j in {0,1}, k in {0,1}
    if strassen_ok:
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_val = strassen_U[i * 2 + j][r]
                    if u_val != 0:
                        U[i * 3 + j].append(u_val)
                    else:
                        U[i * 3 + j].append(0)
            for j in range(2):
                for k in range(2):
                    v_val = strassen_V[j * 2 + k][r]
                    if v_val != 0:
                        V[j * 7 + k].append(v_val)
                    else:
                        V[j * 7 + k].append(0)
            for k in range(2):
                for i in range(2):
                    w_val = strassen_W[k * 2 + i][r]
                    if w_val != 0:
                        W[k * 3 + i].append(w_val)
                    else:
                        W[k * 3 + i].append(0)

    # Now handle the remaining cases with schoolbook
    # For each (i,j,k) not covered by Strassen, add a schoolbook term
    # Strassen covered: i in {0,1}, j in {0,1}, k in {0,1}
    # Need schoolbook for:
    #   - i=2, j in {0,1,2}, k in {0,1,2}  (3*3 = 9 terms)
    #   - i in {0,1}, j=2, k in {0,1,2}    (2*3 = 6 terms)
    #   - i in {0,1,2}, j in {0,1,2}, k in {3,4,5,6}  (3*3*4 = 36 terms)
    # Total schoolbook: 9 + 6 + 36 = 51 terms

    # But wait, we need to be careful: Strassen only covers the 2x2x2 sub-block.
    # The full 3x3x3 block needs:
    #   - Strassen for i,j,k in {0,1} (covered above)
    #   - Schoolbook for i=2 or j=2 or k=2 within the 3x3x3 block
    #   - Schoolbook for k in {3,4,5,6} for all i,j

    # Let me enumerate which (i,j,k) are NOT covered by Strassen:
    # Strassen covers (i,j,k) where i in {0,1}, j in {0,1}, k in {0,1}
    # Not covered: all other (i,j,k) in [0,3)x[0,3)x[0,7)

    for i in range(3):
        for j in range(3):
            for k in range(7):
                if i < 2 and j < 2 and k < 2:
                    continue  # Covered by Strassen
                # Schoolbook term
                U[i * 3 + j].append(1)
                V[j * 7 + k].append(1)
                W[k * 3 + i].append(1)

    # Now all rows must have the same length (rank)
    rank = len(U[0])
    # Pad any short rows (shouldn't happen but safety)
    for row in U:
        while len(row) < rank:
            row.append(0)
    for row in V:
        while len(row) < rank:
            row.append(0)
    for row in W:
        while len(row) < rank:
            row.append(0)

    # Verify correctness
    def verify():
        for i in range(3):
            for j in range(3):
                for k in range(7):
                    s = 0
                    for r in range(rank):
                        s += U[i * 3 + j][r] * V[j * 7 + k][r] * W[k * 3 + i][r]
                    if s != 1:
                        return False
        return True

    if not verify():
        # Fallback to pure schoolbook
        U = [[0] * 0 for _ in range(max_U)]
        V = [[0] * 0 for _ in range(max_V)]
        W = [[0] * 0 for _ in range(max_W)]
        for i in range(3):
            for j in range(3):
                for k in range(7):
                    U[i * 3 + j].append(1)
                    V[j * 7 + k].append(1)
                    W[k * 3 + i].append(1)
        rank = 63

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
