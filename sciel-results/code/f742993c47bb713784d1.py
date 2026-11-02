"""Program for matrix multiplication 2x3x7 tensor decomposition."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Hybrid Strassen/Schoolbook decomposition for rank 39
    # Split A (2x3) into A1 (2x1) and A2 (2x2)
    # Split B (3x7) into B1 (1x7) and B2 (2x7)
    # Split B1 into B1a (1x2), B1b (1x2), B1c (1x3)
    # Split B2 into B2a (2x2), B2b (2x2), B2c (2x3)
    #
    # Blocks:
    # A1*B1a: 2x1x2 -> rank 4 (schoolbook)
    # A1*B1b: 2x1x2 -> rank 4 (schoolbook)
    # A1*B1c: 2x1x3 -> rank 6 (schoolbook)
    # A2*B2a: 2x2x2 -> rank 7 (Strassen)
    # A2*B2b: 2x2x2 -> rank 7 (Strassen)
    # A2*B2c: 2x2x3 -> rank 11 (schoolbook for 2x2x3)
    # Total: 4+4+6+7+7+11 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # (6, 39)
    V = [[0] * rank for _ in range(m * p)]  # (21, 39)
    W = [[0] * rank for _ in range(p * n)]  # (14, 39)
    
    col = 0
    
    # Helper: add a term with given U, V, W indices and coefficient
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] = coeff
        V[v_idx][col] = coeff
        W[w_idx][col] = coeff
        col += 1
    
    # Block 1: A1 (rows 0,1 of A) * B1a (cols 0,1 of B, row 0 of B)
    # A1 has shape 2x1, B1a has shape 1x2
    # Result C1a has shape 2x2
    # Schoolbook: 2*1*2 = 4 terms
    # A[i,0] * B[0,k] -> C[i,k] for i in {0,1}, k in {0,1}
    # A index: i*m + j = i*3 + 0 = 3i
    # B index: j*p + k = 0*7 + k = k
    # W index: k*n + i = 2k + i
    for i in range(2):
        for k in range(2):
            u_idx = i * 3 + 0  # A[i, 0]
            v_idx = 0 * 7 + k  # B[0, k]
            w_idx = k * 2 + i  # W[k*n + i]
            add_term(u_idx, v_idx, w_idx, 1)
    
    # Block 2: A1 * B1b (cols 2,3 of B, row 0 of B)
    # A[i,0] * B[0,k] -> C[i,k] for i in {0,1}, k in {2,3}
    # A index: i*3 + 0
    # B index: 0*7 + k = k
    # W index: k*2 + i
    for i in range(2):
        for k in range(2, 4):
            u_idx = i * 3 + 0
            v_idx = 0 * 7 + k
            w_idx = k * 2 + i
            add_term(u_idx, v_idx, w_idx, 1)
    
    # Block 3: A1 * B1c (cols 4,5,6 of B, row 0 of B)
    # A[i,0] * B[0,k] -> C[i,k] for i in {0,1}, k in {4,5,6}
    for i in range(2):
        for k in range(4, 7):
            u_idx = i * 3 + 0
            v_idx = 0 * 7 + k
            w_idx = k * 2 + i
            add_term(u_idx, v_idx, w_idx, 1)
    
    # Block 4: A2 (rows 0,1 of A, cols 1,2) * B2a (rows 1,2 of B, cols 0,1)
    # A2 has shape 2x2, B2a has shape 2x2
    # Result C2a has shape 2x2
    # Strassen for 2x2x2: rank 7
    # A2[i,j] * B2a[j,k] -> C2a[i,k]
    # A index: i*3 + (1+j) = 3i + 1 + j
    # B index: (1+j)*7 + k = 7 + 7j + k
    # W index: k*2 + i
    
    # Strassen's algorithm for 2x2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B00)
    # M4 = A11(B01 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M7
    
    # Let's use the standard Strassen formulation
    # We need to carefully map indices
    
    # For block 4 (A2*B2a), local indices:
    # A2[i,j] where i,j in {0,1}: global A index = i*3 + 1 + j
    # B2a[j,k] where j,k in {0,1}: global B index = (1+j)*7 + k = 7 + 7j + k
    # W local: k*2 + i
    
    # Strassen terms for this block
    # M1 = (A2[0,0] + A2[1,1])(B2a[0,0] + B2a[1,1])
    u1 = 0*3 + 1 + 0  # A2[0,0] = A[0,1]
    u2 = 1*3 + 1 + 1  # A2[1,1] = A[1,2]
    v1 = 7 + 7*0 + 0  # B2a[0,0] = B[1,0]
    v2 = 7 + 7*1 + 1  # B2a[1,1] = B[2,1]
    
    # We need to add products of sums, which requires expanding
    # (u1 + u2)(v1 + v2) = u1*v1 + u1*v2 + u2*v1 + u2*v2
    # But Strassen uses 7 multiplications total, not 8
    # The trick is that we share intermediate products
    
    # Let me just do schoolbook for now to be safe, then optimize
    # Actually, let me implement Strassen properly
    
    # For 2x2x2, the 7 Strassen multiplications are:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B00)
    # P4 = A11*(B01-B10)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    # C00 = P1 - P4 + P6 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P5 - P7
    
    # Each P involves products of linear combinations
    # P1: (A00+A11)*(B00+B11)
    # This is one "multiplication" in the tensor sense, but in our factorization
    # we need to represent it as a single rank-1 term? No, that's not right.
    
    # Actually, in tensor decomposition, each "multiplication" corresponds to
    # a rank-1 tensor. But Strassen's algorithm uses 7 such multiplications.
    # However, each multiplication is of linear combinations, so in the
    # factor matrices, we have:
    # U column: coefficients of A variables
    # V column: coefficients of B variables
    # W column: how the product contributes to C
    
    # For P1 = (A00+A11)*(B00+B11):
    # U: has 1 at A00 position and 1 at A11 position
    # V: has 1 at B00 position and 1 at B11 position
    # W: has coefficients for C00, C11 (from P1's contribution)
    # C00 gets +P1, C11 gets +P1
    
    # Let me implement this properly
    
    # Block 4: A2*B2a, Strassen
    # Local A indices: (0,0)->global 0*3+1=1, (0,1)->0*3+2=2, (1,0)->1*3+1=4, (1,1)->1*3+2=5
    # Local B indices: (0,0)->7+0=7, (0,1)->7+1=8, (1,0)->7+7=14, (1,1)->7+7+1=15
    # W: local (k,i) -> global k*2+i
    
    a00, a01, a10, a11 = 1, 2, 4, 5  # Global A indices
    b00, b01, b10, b11 = 7, 8, 14, 15  # Global B indices
    
    # P1 = (A00+A11)*(B00+B11)
    U[a00][col] = 1; U[a11][col] = 1
    V[b00][col] = 1; V[b11][col] = 1
    W[0*2+0][col] = 1   # C00 += P1
    W[1*2+1][col] = 1   # C11 += P1
    col += 1
    
    # P2 = (A10+A11)*B00
    U[a10][col] = 1; U[a11][col] = 1
    V[b00][col] = 1
    W[0*2+1][col] = 1   # C10 += P2
    col += 1
    
    # P3 = A00*(B10-B00)
    U[a00][col] = 1
    V[b10][col] = 1; V[b00][col] = -1
    W[0*2+0][col] = 1   # C01 += P3
    col += 1
    
    # P4 = A11*(B01-B10)
    U[a11][col] = 1
    V[b01][col] = 1; V[b10][col] = -1
    W[1*2+1][col] = -1  # C11 -= P4 (wait, C11 = P1 - P3 + P5 - P7, and C00 = P1 - P4 + P6 + P7)
    # Let me recheck: C00 = P1 - P4 + P6 + P7, C11 = P1 - P3 + P5 - P7
    W[0*2+0][col] = -1  # C00 -= P4
    col += 1
    
    # P5 = (A00+A01)*B11
    U[a00][col] = 1; U[a01][col] = 1
    V[b11][col] = 1
    W[0*2+1][col] = 1   # C01 += P5
    W[1*2+1][col] = 1   # C11 += P5
    col += 1
    
    # P6 = (A10-A00)*(B00+B01)
    U[a10][col] = 1; U[a00][col] = -1
    V[b00][col] = 1; V[b01][col] = 1
    W[0*2+0][col] = 1   # C00 += P6
    col += 1
    
    # P7 = (A01-A11)*(B10+B11)
    U[a01][col] = 1; U[a11][col] = -1
    V[b10][col] = 1; V[b11][col] = 1
    W[0*2+0][col] = 1   # C00 += P7
    W[1*2+1][col] = -1  # C11 -= P7
    col += 1
    
    # Block 5: A2*B2b, Strassen
    # B2b: rows 1,2 of B, cols 2,3
    # B indices: (1,2)->7+2=9, (1,3)->7+3=10, (2,2)->14+2=16, (2,3)->14+3=17
    b00_2, b01_2, b10_2, b11_2 = 9, 10, 16, 17
    
    # P1 = (A00+A11)*(B00+B11)
    U[a00][col] = 1; U[a11][col] = 1
    V[b00_2][col] = 1; V[b11_2][col] = 1
    W[2*2+0][col] = 1   # C02 += P1
    W[3*2+1][col] = 1   # C13 += P1
    col += 1
    
    # P2 = (A10+A11)*B00
    U[a10][col] = 1; U[a11][col] = 1
    V[b00_2][col] = 1
    W[2*2+1][col] = 1   # C12 += P2
    col += 1
    
    # P3 = A00*(B10-B00)
    U[a00][col] = 1
    V[b10_2][col] = 1; V[b00_2][col] = -1
    W[2*2+0][col] = 1   # C02 += P3
    col += 1
    
    # P4 = A11*(B01-B10)
    U[a11][col] = 1
    V[b01_2][col] = 1; V[b10_2][col] = -1
    W[2*2+0][col] = -1  # C02 -= P4
    col += 1
    
    # P5 = (A00+A01)*B11
    U[a00][col] = 1; U[a01][col] = 1
    V[b11_2][col] = 1
    W[2*2+1][col] = 1   # C03 += P5
    W[3*2+1][col] = 1   # C13 += P5
    col += 1
    
    # P6 = (A10-A00)*(B00+B01)
    U[a10][col] = 1; U[a00][col] = -1
    V[b00_2][col] = 1; V[b01_2][col] = 1
    W[2*2+0][col] = 1   # C02 += P6
    col += 1
    
    # P7 = (A01-A11)*(B10+B11)
    U[a01][col] = 1; U[a11][col] = -1
    V[b10_2][col] = 1; V[b11_2][col] = 1
    W[2*2+0][col] = 1   # C02 += P7
    W[3*2+1][col] = -1  # C13 -= P7
    col += 1
    
    # Block 6: A2*B2c, schoolbook for 2x2x3
    # A2 has shape 2x2, B2c has shape 2x3
    # A[i,j] * B[j,k] -> C[i,k] for i in {0,1}, j in {0,1}, k in {4,5,6}
    # A index: i*3 + 1 + j
    # B index: (1+j)*7 + k = 7 + 7j + k
    # W index: k*2 + i
    for i in range(2):
        for j in range(2):
            for k in range(4, 7):
                u_idx = i * 3 + 1 + j
                v_idx = 7 + 7 * j + k
                w_idx = k * 2 + i
                add_term(u_idx, v_idx, w_idx, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
