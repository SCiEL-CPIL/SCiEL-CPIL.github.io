import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, each 4x7 or 4x7."""
    # Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # M2: (A10+A11)B00 -> C10, C11
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1               # B00
    W[2, 1] = 1; W[3, 1] = -1 # C10, -C11
    
    # M3: A00(B01-B11) -> C01, C11
    U[0, 2] = 1               # A00
    V[1, 2] = 1; V[3, 2] = -1 # B01, -B11
    W[1, 2] = 1; W[3, 2] = 1  # C01, C11
    
    # M4: A11(B10-B00) -> C00, C10
    U[3, 3] = 1               # A11
    V[2, 3] = 1; V[0, 3] = -1 # B10, -B00
    W[0, 3] = 1; W[2, 3] = 1  # C00, C10
    
    # M5: (A00+A10)B11 -> C00, C01
    U[0, 4] = 1; U[2, 4] = 1  # A00, A10
    V[3, 4] = 1               # B11
    W[0, 4] = -1; W[1, 4] = 1 # -C00, C01
    
    # M6: (A10-A00)(B00+B10) -> C11
    U[2, 5] = 1; U[0, 5] = -1 # A10, -A00
    V[0, 5] = 1; V[2, 5] = 1  # B00, B10
    W[3, 5] = 1               # C11
    
    # M7: (A01-A11)(B01+B11) -> C00
    U[1, 6] = 1; U[3, 6] = -1 # A01, -A11
    V[1, 6] = 1; V[3, 6] = 1  # B01, B11
    W[0, 6] = 1               # C00
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    DEADLINE = 118.0  # safety margin
    
    # Strategy: Use 2x2x2 Strassen blocks where possible, schoolbook for rest.
    # For 3x5x5, we can place 2x2x2 blocks at:
    # (0,0,0): covers i={0,1}, j={0,1}, k={0,1}
    # (0,2,0): covers i={0,1}, j={2,3}, k={0,1}
    # (0,0,2): covers i={0,1}, j={0,1}, k={2,3}
    # (0,2,2): covers i={0,1}, j={2,3}, k={2,3}
    # That's 4 blocks covering i={0,1}, j={0,1,2,3}, k={0,1,2,3}
    # Remaining: i=2 (all), or j=4, or k=4 for i in {0,1}
    
    # Let's count: 4 Strassen blocks cover 4*8=32 terms with rank 4*7=28
    # Total terms = 3*5*5 = 75
    # Remaining terms = 75 - 32 = 43
    # Total rank = 28 + 43 = 71
    
    # But can we do better? Let's also try to use Strassen for i=2 row
    # For i=2, we have a 1x5x5 sub-problem. We can't use 2x2x2 there.
    # But we could potentially use a different decomposition for the
    # remaining terms.
    
    # Actually, let's try a different arrangement:
    # Block at (0,0,0): i={0,1}, j={0,1}, k={0,1}
    # Block at (1,2,0): i={1,2}, j={2,3}, k={0,1}
    # Block at (0,2,2): i={0,1}, j={2,3}, k={2,3}
    # Block at (1,0,2): i={1,2}, j={0,1}, k={2,3}
    
    # Check overlaps:
    # (0,0,0): i=0,1; j=0,1; k=0,1
    # (1,2,0): i=1,2; j=2,3; k=0,1
    # (0,2,2): i=0,1; j=2,3; k=2,3
    # (1,0,2): i=1,2; j=0,1; k=2,3
    
    # Overlap check:
    # (0,0,0) and (1,2,0): i overlap at 1, j: {0,1} vs {2,3} - no j overlap. OK
    # (0,0,0) and (0,2,2): i overlap at 0,1, j: {0,1} vs {2,3} - no j overlap. OK
    # (0,0,0) and (1,0,2): i overlap at 1, j: {0,1} vs {0,1} - j overlap! k: {0,1} vs {2,3} - no k overlap. OK
    # (1,2,0) and (0,2,2): i overlap at 1, j: {2,3} vs {2,3} - j overlap! k: {0,1} vs {2,3} - no k overlap. OK
    # (1,2,0) and (1,0,2): i overlap at 1,2, j: {2,3} vs {0,1} - no j overlap. OK
    # (0,2,2) and (1,0,2): i overlap at 1, j: {2,3} vs {0,1} - no j overlap. OK
    
    # So this arrangement works! Let's use it.
    blocks = [
        (0, 0, 0),
        (1, 2, 0),
        (0, 2, 2),
        (1, 0, 2),
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    total_terms = n * m * p
    strassen_terms_count = num_strassen_blocks * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    # Build the decomposition
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Get Strassen block
    sU, sV, sW = _strassen_2x2x2()
    
    col = 0
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk
                    if i < n and j < m and k < p:
                        for s in range(7):
                            U[i*m+j][col+s] += sU[di*2+dj, s]
                            V[j*p+k][col+s] += sV[dj*2+dk, s]
                            W[k*n+i][col+s] += sW[dk*2+di, s]
        col += 7
    
    # Mark covered positions
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk
                    if i < n and j < m and k < p:
                        covered[i][j][k] = True
    
    # Add schoolbook for uncovered positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
    
    # Verify and check deadline
    if time.time() - start_time > DEADLINE:
        # Return what we have (should be correct)
        return {"U": U, "V": V, "W": W}
    
    # Quick verification
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(total_rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fall back to pure schoolbook
        total_rank_fb = n * m * p
        U_fb = [[0.0] * total_rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * total_rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * total_rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1.0
                    V_fb[j*p+k][col_fb] = 1.0
                    W_fb[k*n+i][col_fb] = 1.0
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
