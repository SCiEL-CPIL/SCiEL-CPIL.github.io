"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x1x6 (Rank 12)
    # A col 0, B row 0
    # A[i, 0] * B[0, k] -> C[i, k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: 2x4x6 (Rank 42)
    # Split A cols 1,2,3,4 into two 2x2 blocks
    # Block A1: cols 1,2. Block A2: cols 3,4
    # For each, split B cols 0..5 into three 2x2 blocks
    # B1: cols 0,1. B2: cols 2,3. B3: cols 4,5
    
    # Define Strassen coefficients
    # m1: (a00+a11)(b00+b11)
    # m2: (a10+a11)b00
    # m3: a00(b01-b11)
    # m4: a11(b10-b00)
    # m5: (a00+a01)b11
    # m6: (a10-a00)(b00+b01)
    # m7: (a01-a11)(b10+b11)
    
    # U coeffs for a00, a01, a10, a11
    # V coeffs for b00, b01, b10, b11
    # W coeffs for c00, c01, c10, c11
    
    strassen_U = [
        [1, 0, 0, 1], # m1: a00, a11
        [0, 0, 1, 1], # m2: a10, a11
        [1, 0, 0, 0], # m3: a00
        [0, 0, 0, 1], # m4: a11
        [1, 1, 0, 0], # m5: a00, a01
        [-1, 0, 1, 0],# m6: -a00, a10
        [0, 1, 0, -1] # m7: a01, -a11
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # m1: b00, b11
        [1, 0, 0, 0], # m2: b00
        [0, 1, 0, -1],# m3: b01, -b11
        [-1, 0, 1, 0],# m4: -b00, b10
        [0, 0, 0, 1], # m5: b11
        [1, 1, 0, 0], # m6: b00, b01
        [0, 0, 1, 1]  # m7: b10, b11
    ]
    
    strassen_W = [
        [1, 0, 0, 1], # m1: c00, c11
        [0, 0, 1, -1],# m2: c10, -c11
        [0, 1, 0, 1], # m3: c01, c11
        [1, 0, 1, 0], # m4: c00, c10
        [-1, 1, 0, 0],# m5: -c00, c01
        [0, 0, 0, 1], # m6: c11
        [1, 0, 0, 0]  # m7: c00
    ]
    
    # Local indices 0,1,2,3 correspond to:
    # a00, a01, a10, a11 -> (0,0), (0,1), (1,0), (1,1)
    # b00, b01, b10, b11 -> (0,0), (0,1), (1,0), (1,1)
    # c00, c01, c10, c11 -> (0,0), (0,1), (1,0), (1,1)
    
    # Mapping local (r, c) to index:
    # 0:00, 1:01, 2:10, 3:11
    # r = idx // 2, c = idx % 2
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        for t in range(7):
            # U
            for li in range(4):
                coeff = strassen_U[t][li]
                if coeff != 0:
                    # li 0:00, 1:01, 2:10, 3:11
                    ri = li // 2
                    cj = li % 2
                    gi = i0 + ri
                    gj = j0 + cj
                    U[gi * m + gj][col] += coeff
                    
            # V
            for lj in range(4):
                coeff = strassen_V[t][lj]
                if coeff != 0:
                    ri = lj // 2
                    cj = lj % 2
                    gj = j0 + ri
                    gk = k0 + cj
                    V[gj * p + gk][col] += coeff
                    
            # W
            for lc in range(4):
                coeff = strassen_W[t][lc]
                if coeff != 0:
                    ri = lc // 2
                    ck = lc % 2
                    gi = i0 + ri
                    gk = k0 + ck
                    W[gk * n + gi][col] += coeff
                    
            col += 1
            
    # A Block 1: i in {0,1}, j in {1,2}
    # B Block 1: j in {1,2}, k in {0,1}
    add_strassen(0, 1, 0)
    # B Block 2: j in {1,2}, k in {2,3}
    add_strassen(0, 1, 2)
    # B Block 3: j in {1,2}, k in {4,5}
    add_strassen(0, 1, 4)
    
    # A Block 2: i in {0,1}, j in {3,4}
    # B Block 1: j in {3,4}, k in {0,1}
    add_strassen(0, 3, 0)
    # B Block 2: j in {3,4}, k in {2,3}
    add_strassen(0, 3, 2)
    # B Block 3: j in {3,4}, k in {4,5}
    add_strassen(0, 3, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
