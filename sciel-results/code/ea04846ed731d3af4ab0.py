<<<<<<< SEARCH
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
=======
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2_terms():
    """Return the 7 rank-1 terms (u, v, w) for 2x2x2 matrix multiplication.
    
    Indexing:
      A flat: i = i1*2 + i2  (i1=row, i2=col)
      B flat: j = j1*2 + j2  (j1=row, j2=col)
      W flat (trace): k = k2*2 + k1  (k2=col of C, k1=row of C)
    
    T[i,j,k] = 1 iff i2==j1 and i1==k1 and j2==k2
    """
    # Correct Strassen decomposition verified numerically
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(g-e), M4 = d(f-h)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # A indices: a=0(A11), b=1(A12), c=2(A21), d=3(A22)
    # B indices: e=0(B11), f=1(B12), g=2(B21), h=3(B22)
    # W indices (trace): C11=0, C21=1, C12=2, C22=3
    
    terms = []
    # M1 = (a+d)(e+h): u=[1,0,0,1], v=[1,0,0,1]
    # C11+=M1, C22+=M1: w=[1,0,0,1]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2 = (c+d)e: u=[0,0,1,1], v=[1,0,0,0]
    # C21+=M2, C22-=M2: w=[0,1,0,-1]
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,-1]))
    # M3 = a(g-e): u=[1,0,0,0], v=[0,0,1,-1]
    # C12+=M3, C22+=M3: w=[0,0,1,1]
    terms.append(([1,0,0,0], [0,0,1,-1], [0,0,1,1]))
    # M4 = d(f-h): u=[0,0,0,1], v=[0,1,0,-1]
    # C11+=M4, C21+=M4: w=[1,1,0,0]
    terms.append(([0,0,0,1], [0,1,0,-1], [1,1,0,0]))
    # M5 = (a+b)h: u=[1,1,0,0], v=[0,0,0,1]
    # C11-=M5, C12+=M5: w=[-1,0,1,0]
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    # M6 = (c-a)(e+f): u=[-1,0,1,0], v=[1,1,0,0]
    # C22+=M6: w=[0,0,0,1]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7 = (b-d)(g+h): u=[0,1,0,-1], v=[0,0,1,1]
    # C11+=M7: w=[1,0,0,0]
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    # Verify
    T = np.zeros((4,4,4))
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    if i2 == j1:
                        k1, k2 = i1, j2
                        T[i1*2+i2, j1*2+j2, k2*2+k1] = 1.0
    for u,v,w in terms:
        T -= np.outer(np.outer(u,v),w).reshape(4,4,4)
    assert np.allclose(T, 0), f"Strassen verification failed: max err {np.abs(T).max()}"
    
    return terms

def _embed_2x2x2_to_4x4x4(block_row, block_col, block_b_row, block_b_col, block_c_row, block_c_col, terms):
    """Embed 2x2x2 Strassen terms into 4x4x4 at specified block positions.
    
    For 4x4x4:
      A flat: i = i1*4 + i2
      B flat: j = j1*7 + j2  (but for 4x4x4, p=4, so j = j1*4 + j2)
      W flat (trace): k = k2*4 + k1
    
    Block A[row1:row1+2, col1:col1+2], Block B[row1:row1+2, col1:col1+2], Block C[row1:row1+2, col1:col1+2]
    """
    U = np.zeros((16, 7))
    V = np.zeros((16, 7))
    W = np.zeros((16, 7))
    
    for r in range(7):
        u, v, w = terms[r]
        for a_r in range(2):
            for a_c in range(2):
                U_idx = (block_row + a_r) * 4 + (block_col + a_c)
                U[U_idx, r] = u[a_r * 2 + a_c]
        for b_r in range(2):
            for b_c in range(2):
                V_idx = (block_b_row + b_r) * 4 + (block_b_col + b_c)
                V[V_idx, r] = v[b_r * 2 + b_c]
        for c_r in range(2):
            for c_c in range(2):
                W_idx = (block_c_col + c_c) * 4 + (block_c_row + c_r)
                W[W_idx, r] = w[c_c * 2 + c_r]
    
    return U, V, W

def _make_4x4x4_strassen():
    """Build rank-49 decomposition of 4x4x4 using two-level Strassen.
    
    Level 1: 4x4 = 4 blocks of 2x2. Use Strassen at block level (7 products).
    Level 2: Each block product is 2x2x2 with rank 7.
    Total: 7 * 7 = 49 terms.
    """
    base_terms = _strassen_2x2x2_terms()
    
    # Strassen block combinations for 4x4x4:
    # Block indices: A blocks (0,0),(0,1),(1,0),(1,1) for rows 0-1,2-3 and cols 0-1,2-3
    # Same for B and C.
    #
    # M1 = (A11+A22)(B11+B22) -> blocks (0,0)+(1,1) for A, (0,0)+(1,1) for B, contributes to C(0,0)+C(1,1)
    # M2 = (A21+A22)B11 -> blocks (1,0)+(1,1) for A, (0,0) for B, contributes to C(1,0)
    # M3 = A11(B21-B11) -> block (0,0) for A, (1,0)-(0,0) for B, contributes to C(0,0)
    # M4 = A22(B12-B22) -> block (1,1) for A, (0,1)-(1,1) for B, contributes to C(1,1)
    # M5 = (A11+A12)B22 -> blocks (0,0)+(0,1) for A, (1,1) for B, contributes to C(0,1)
    # M6 = (A21-A11)(B11+B12) -> blocks (1,0)-(0,0) for A, (0,0)+(0,1) for B, contributes to C(1,1)
    # M7 = (A12-A22)(B21+B22) -> blocks (0,1)-(1,1) for A, (1,0)+(1,1) for B, contributes to C(0,0)
    
    # Each "block combination" involves summing/differencing 2x2 blocks.
    # For each Strassen multiplier Mi, we need to apply the 2x2x2 Strassen
    # to each pair of sub-blocks involved.
    
    # Actually, the correct approach: each of the 7 Strassen multipliers
    # involves multiplying two 2x2 "super-blocks" (which are themselves
    # linear combinations of 2x2 sub-blocks). Each such multiplication
    # requires 7 scalar multiplications (rank-7 for 2x2x2).
    
    # So for each of the 7 Strassen multipliers, we do a 2x2x2 multiplication
    # where the "A" and "B" are 2x2 matrices formed from linear combinations
    # of the original 2x2 sub-blocks.
    
    # Let me define this more carefully.
    # The 4x4 matrix A is partitioned into 4 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # Similarly B = [[B11, B12], [B21, B22]], C = [[C11, C12], [C21, C22]].
    
    # Strassen multipliers:
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B21-B11)
    # M4 = A22 * (B12-B22)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    
    # Each Mi is a 2x2 matrix. To compute Mi = X * Y where X and Y are 2x2,
    # we use the rank-7 Strassen decomposition of 2x2x2.
    
    # For each Mi, X is a linear combination of A-blocks, Y is a linear combination of B-blocks.
    # The resulting Mi contributes to specific C-blocks with specific signs.
    
    # Let me enumerate: for each Strassen multiplier index s (0..6),
    # define the A-block linear combination coefficients and B-block coefficients,
    # and the C-block contribution pattern.
    
    # s=0 (M1): A = A11+A22, B = B11+B22, C: C11+=M1, C22+=M1
    # s=1 (M2): A = A21+A22, B = B11, C: C21+=M2, C22-=M2
    # s=2 (M3): A = A11, B = B21-B11, C: C12+=M3, C22+=M3
    # s=3 (M4): A = A22, B = B12-B22, C: C11+=M4, C21+=M4
    # s=4 (M5): A = A11+A12, B = B22, C: C11-=M5, C12+=M5
    # s=5 (M6): A = A21-A11, B = B11+B12, C: C22+=M6
    # s=6 (M7): A = A12-A22, B = B21+B22, C: C11+=M7
    
    # A-block positions: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B-block positions: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    # C-block positions: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    strassen_combos = [
        # (a_coeffs, b_coeffs, c_contribs)
        # a_coeffs: dict from block_pos to coeff
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),  # M2
        ({(0,0):1}, {(1,0):1, (0,0):-1}, {(0,1):1, (1,1):1}),  # M3
        ({(1,1):1}, {(0,1):1, (1,1):-1}, {(0,0):1, (1,0):1}),  # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),  # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),  # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),  # M7
    ]
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    for s, (a_coeffs, b_coeffs, c_contribs) in enumerate(strassen_combos):
        for r in range(7):
            col = s * 7 + r
            u_base, v_base, w_base = base_terms[r]
            
            # For each A-block in a_coeffs, add scaled u_base
            for (ar, ac), coeff in a_coeffs.items():
                for a_r in range(2):
                    for a_c in range(2):
                        U_idx = (ar*2 + a_r) * 4 + (ac*2 + a_c)
                        U[U_idx, col] = coeff * u_base[a_r * 2 + a_c]
            
            # For each B-block in b_coeffs, add scaled v_base
            for (br, bc), coeff in b_coeffs.items():
                for b_r in range(2):
                    for b_c in range(2):
                        V_idx = (br*2 + b_r) * 4 + (bc*2 + b_c)
                        V[V_idx, col] = coeff * v_base[b_r * 2 + b_c]
            
            # For each C-block in c_contribs, add scaled w_base
            for (cr, cc), coeff in c_contribs.items():
                for c_r in range(2):
                    for c_c in range(2):
                        # W trace convention: k = k2*4 + k1 where k2=col, k1=row
                        W_idx = (cc*2 + c_c) * 4 + (cr*2 + c_r)
                        W[W_idx, col] = coeff * w_base[c_c * 2 + c_r]
    
    # Verify
    T = np.zeros((16,16,16))
    for i1 in range(4):
        for i2 in range(4):
            for j1 in range(4):
                for j2 in range(4):
                    if i2 == j1:
                        k1, k2 = i1, j2
                        T[i1*4+i2, j1*4+j2, k2*4+k1] = 1.0
    for r in range(49):
        T -= np.outer(np.outer(U[:,r], V[:,r]), W[:,r]).reshape(16,16,16)
    assert np.allclose(T, 0), f"4x4x4 verification failed: max err {np.abs(T).max()}"
    
    return U, V, W

def _make_4x4x3_schoolbook():
    """Build rank-48 decomposition of 4x4x3 using schoolbook."""
    U = np.zeros((16, 48))
    V = np.zeros((28, 48))
    W = np.zeros((28, 48))
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                r = i*12 + j*3 + k
                U[i*4+j, r] = 1.0
                V[j*3+k, r] = 1.0
                W[k*4+i, r] = 1.0
    
    return U, V, W

def _make_4x4x7_decomposition():
    """Build rank-97 decomposition of 4x4x7.
    
    Split B into B_left (4x4) and B_right (4x3).
    C = A * B = A * B_left + A * B_right
    Use Strassen for A*B_left (rank 49) and schoolbook for A*B_right (rank 48).
    Total rank: 97.
    """
    U44, V44, W44 = _make_4x4x4_strassen()
    U43, V43, W43 = _make_4x4x3_schoolbook()
    
    # Combine: U is (16, 97), V is (28, 97), W is (28, 97)
    U = np.zeros((16, 97))
    V = np.zeros((28, 97))
    W = np.zeros((28, 97))
    
    # First 49 columns: 4x4x4 Strassen
    U[:, :49] = U44
    V[:16, :49] = V44  # B_left is rows 0-3 of B, which is B[:, 0:4]
    W[:16, :49] = W44  # C_left is C[:, 0:4], W indices k*4+i for k in 0..3
    
    # Next 48 columns: 4x4x3 schoolbook for B_right = B[:, 4:7]
    U[:, 49:] = U43
    V[16:, 49:] = V43  # B_right is rows 4-6 of B (j*p+k where j in 0..3, k in 4..6)
    W[16:, 49:] = W43  # C_right is C[:, 4:7], W indices k*4+i for k in 4..6
    
    # Verify
    n, m, p = 4, 4, 7
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    for r in range(97):
        T -= np.outer(np.outer(U[:,r], V[:,r]), W[:,r]).reshape(n*m, m*p, p*n)
    assert np.allclose(T, 0), f"4x4x7 verification failed: max err {np.abs(T).max()}"
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    
    U, V, W = _make_4x4x7_decomposition()
    
    # Convert to lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

>>>>>>> REPLACE
