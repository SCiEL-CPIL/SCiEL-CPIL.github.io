"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U shape: (2*4, R) = (8, R)
    # V shape: (4*4, R) = (16, R)
    # W shape: (4*2, R) = (8, R)
    
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 matrix multiplication
    # Inputs: A (2x2), B (2x2)
    # Outputs: C (2x2)
    # Indices for local 2x2 A: 0,1,2,3 (row-major)
    # Indices for local 2x2 B: 0,1,2,3 (row-major)
    # Indices for local 2x2 C: 0,1,2,3 (row-major)
    
    # The 7 terms of Strassen's algorithm
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: length 2 (for A's rows? No, U is n*m. Here n=2, m=2 local)
    # In local 2x2, U corresponds to A indices.
    # U is (n*m, rank). Here local n=2, m=2. So U is (4, 7).
    # V is (m*p, rank). Local m=2, p=2. So V is (4, 7).
    # W is (p*n, rank). Local p=2, n=2. So W is (4, 7).
    
    # Strassen Terms:
    # M1 = (a+e)(f+h)
    # M2 = (c+d)(f+h)
    # M3 = (a+b)h
    # M4 = c(e+h)
    # M5 = (d-f)(a+b)
    # M6 = (e+g)f
    # M7 = g(d-f)
    
    # Mapping of local indices to global indices:
    # Global A (2x4): 
    # A1 (cols 0,1) -> Global indices 0,1,4,5
    # A2 (cols 2,3) -> Global indices 2,3,6,7
    # Global B (4x4):
    # B11 (rows 0,1, cols 0,1) -> Global indices 0,1,4,5
    # B12 (rows 0,1, cols 2,3) -> Global indices 2,3,6,7
    # B21 (rows 2,3, cols 0,1) -> Global indices 8,9,12,13
    # B22 (rows 2,3, cols 2,3) -> Global indices 10,11,14,15
    # Global C (2x4):
    # C1 (cols 0,1) -> Global indices 0,1,4,5
    # C2 (cols 2,3) -> Global indices 2,3,6,7
    
    # We perform 4 independent 2x2 multiplications:
    # P1 = A1 * B11 -> contributes to C1
    # P2 = A2 * B21 -> contributes to C1
    # P3 = A1 * B12 -> contributes to C2
    # P4 = A2 * B22 -> contributes to C2
    
    # Local Strassen Vectors (for 2x2)
    # u: [a, b, c, d] indices 0,1,2,3
    # v: [e, f, g, h] indices 0,1,2,3
    # w: [p00, p01, p10, p11] indices 0,1,2,3
    
    # M1: (a+e)(f+h)
    # a: u[0]=1, e: v[0]=1
    # f: v[1]=1, h: v[3]=1
    # Product terms: a*f, a*h, e*f, e*h
    # This is not a single rank-1 term in the tensor sense of (sum u)(sum v).
    # The tensor decomposition requires U, V, W to be such that 
    # C = U * V^T * W^T? No.
    # C_{ik} = sum_t U_{it} V_{jt} W_{...}
    # For 2x2x2 tensor, C_{r,c} = sum_t U_{r,t} V_{c,t} W_{?,t}?
    # Standard form: T_{ijk} = sum U_{ik} V_{jk} W_{...}
    # Here indices are:
    # A is n x m. U is (n*m) x R.
    # B is m x p. V is (m*p) x R.
    # C is p x n (or n x p). W is (p*n) x R.
    
    # For local 2x2:
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen Rank-7 Decomposition (Integer Coeffs):
    # Term 0: U=[1,0,0,0], V=[1,0,0,0], W=[1,0,0,0] (ae)
    # Term 1: U=[0,1,0,0], V=[1,0,0,0], W=[0,0,1,0] (ce) -- Wait, C10 is c*e.
    # This is schoolbook.
    
    # Correct Strassen 7 terms for 2x2:
    # 1. U=[1,1,0,0], V=[1,0,1,0], W=[0,1,0,0]?
    
    # I will use the verified decomposition from a known source:
    # https://en.wikipedia.org/wiki/Strassen_algorithm
    # The 7 products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Reconstruction:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's map these to U, V, W vectors.
    # Local A: 0:A11, 1:A12, 2:A21, 3:A22
    # Local B: 0:B11, 1:B12, 2:B21, 3:B22
    # Local C: 0:C11, 1:C12, 2:C21, 3:C22
    
    terms = [
        # M1: (A11+A22)(B11+B22)
        # U: [1, 0, 0, 1] (A11, A22)
        # V: [1, 0, 0, 1] (B11, B22)
        # W: [1, 0, 0, 1] (C11, C22) -- Wait, C11 and C22 both get +1.
        # But W is a vector. The contribution to C is outer product?
        # No, W is a vector of length 4 (p*n).
        # The term adds U*V*W to the tensor?
        # Yes.
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A21+A22)B11
        # U: [0, 0, 1, 1]
        # V: [1, 0, 0, 0]
        # W: [0, 0, 1, 0] (C21)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3: A11(B21-B22)
        # U: [1, 0, 0, 0]
        # V: [0, 0, 1, -1]
        # W: [0, 1, 0, 0] (C12)
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        # M4: A22(B12-B11)
        # U: [0, 0, 0, 1]
        # V: [-1, 1, 0, 0]
        # W: [1, 0, 1, 0] (C11, C21)
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]),
        # M5: (A11+A12)B22
        # U: [1, 1, 0, 0]
        # V: [0, 0, 0, 1]
        # W: [-1, 1, 0, 0] (C11: -1, C12: 1)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6: (A21-A11)(B11+B12)
        # U: [-1, 0, 1, 0]
        # V: [1, 1, 0, 0]
        # W: [0, 0, 0, 1] (C22)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (A12-A22)(B21+B22)
        # U: [0, 1, 0, -1]
        # V: [0, 0, 1, 1]
        # W: [1, 0, 0, 1] (C11, C22)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])
    ]
    
    # Mapping functions
    # A1 local -> Global
    def map_A1(local_idx):
        # local 0,1,2,3 -> global 0,1,4,5
        return [0, 1, 4, 5][local_idx]
        
    def map_A2(local_idx):
        # local 0,1,2,3 -> global 2,3,6,7
        return [2, 3, 6, 7][local_idx]

    def map_B11(local_idx):
        return [0, 1, 4, 5][local_idx]

    def map_B12(local_idx):
        return [2, 3, 6, 7][local_idx]

    def map_B21(local_idx):
        return [8, 9, 12, 13][local_idx]

    def map_B22(local_idx):
        return [10, 11, 14, 15][local_idx]

    def map_C1(local_idx):
        # C1 is cols 0,1. Local C is 2x2.
        # Local 0,1,2,3 -> Global 0,1,4,5
        return [0, 1, 4, 5][local_idx]

    def map_C2(local_idx):
        # C2 is cols 2,3. Local C is 2x2.
        # Local 0,1,2,3 -> Global 2,3,6,7
        return [2, 3, 6, 7][local_idx]

    col = 0
    # P1 = A1 * B11 -> C1
    for u, v, w in terms:
        for i in range(4):
            if u[i] != 0:
                U[map_A1(i)][col] += u[i]
        for j in range(4):
            if v[j] != 0:
                V[map_B11(j)][col] += v[j]
        for k in range(4):
            if w[k] != 0:
                W[map_C1(k)][col] += w[k]
        col += 1

    # P2 = A2 * B21 -> C1
    for u, v, w in terms:
        for i in range(4):
            if u[i] != 0:
                U[map_A2(i)][col] += u[i]
        for j in range(4):
            if v[j] != 0:
                V[map_B21(j)][col] += v[j]
        for k in range(4):
            if w[k] != 0:
                W[map_C1(k)][col] += w[k]
        col += 1

    # P3 = A1 * B12 -> C2
    for u, v, w in terms:
        for i in range(4):
            if u[i] != 0:
                U[map_A1(i)][col] += u[i]
        for j in range(4):
            if v[j] != 0:
                V[map_B12(j)][col] += v[j]
        for k in range(4):
            if w[k] != 0:
                W[map_C2(k)][col] += w[k]
        col += 1

    # P4 = A2 * B22 -> C2
    for u, v, w in terms:
        for i in range(4):
            if u[i] != 0:
                U[map_A2(i)][col] += u[i]
        for j in range(4):
            if v[j] != 0:
                V[map_B22(j)][col] += v[j]
        for k in range(4):
            if w[k] != 0:
                W[map_C2(k)][col] += w[k]
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
