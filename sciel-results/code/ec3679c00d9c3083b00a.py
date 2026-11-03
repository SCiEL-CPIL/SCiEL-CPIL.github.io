"""Tensor decomposition for 4x4 * 4x7 matrix multiplication."""
DIMENSIONS = [4, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return rank-7 decomposition for 2x2 * 2x2 -> 2x2.
    U: (4,7), V: (4,7), W: (4,7)
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # Index: row = i*2+j for U, j*2+k for V, k*2+i for W
    
    # m1 = (a11+a22)(b11+b22)
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # m2 = (a21+a22)b11
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[1*2+0, 1] = 1; W[1*2+1, 1] = 1
    
    # m3 = a11(b21-b22)
    U[0*2+0, 2] = 1
    V[1*2+0, 2] = 1; V[1*2+1, 2] = -1
    W[0*2+0, 2] = 1; W[1*2+0, 2] = 1
    
    # m4 = a22(b12-b11)
    U[1*2+1, 3] = 1
    V[0*2+1, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+1, 3] = 1; W[1*2+1, 3] = 1
    
    # m5 = (a11+a12)b22
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+1, 4] = 1; W[1*2+1, 4] = 1
    
    # m6 = (a21-a11)(b11+b12)
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1; W[0*2+1, 5] = 1
    
    # m7 = (a12-a22)(b21+b22)
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[1*2+0, 6] = 1; W[0*2+0, 6] = 1
    
    return U, V, W


def _trivial_decomp(n, m, p):
    """Return schoolbook decomposition for n x m * m x p -> n x p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions.
    If T1 = sum_r U1[:,r] ⊗ V1[:,r] ⊗ W1[:,r]  (for dims n1,m1,p1)
    and T2 = sum_s U2[:,s] ⊗ V2[:,s] ⊗ W2[:,s]  (for dims n2,m2,p2)
    Then T1 ⊗ T2 = sum_{r,s} (U1[:,r]⊗U2[:,s]) ⊗ (V1[:,r]⊗V2[:,s]) ⊗ (W1[:,r]⊗W2[:,s])
    
    The Kronecker product here is in the "block" sense:
    U_full[i1*m1 + i2, r*S2 + s] = U1[i1, r] * U2[i2, s]
    where i1 in [0,n1), i2 in [0,n2), so full n = n1*n2, m = m1*m2, etc.
    
    Actually, the standard Kronecker for matmul:
    (A1⊗A2)(B1⊗B2) = (A1B1)⊗(A2B2)
    where A1 is n1×m1, A2 is n2×m2, B1 is m1×p1, B2 is m2×p2.
    Result C is (n1*n2) × (p1*p2).
    
    U indices: (i1*m1 + i2) for A, but wait - A1 is n1×m1, A2 is n2×m2.
    A = A1 ⊗ A2 is (n1*n2) × (m1*m2).
    Element A[i1*n2 + i2, j1*m2 + j2] = A1[i1,j1]*A2[i2,j2]
    
    Hmm, let me think about the index convention more carefully.
    For matmul n×m * m×p:
    U has shape (n*m, rank), indexed by (i,j) -> row i*m+j
    V has shape (m*p, rank), indexed by (j,k) -> row j*p+k
    W has shape (p*n, rank), indexed by (k,i) -> row k*n+i
    
    For Kronecker: A is (n1*n2)×(m1*m2), B is (m1*m2)×(p1*p2), C is (n1*n2)×(p1*p2)
    A[i1*n2+i2, j1*m2+j2] = A1[i1,j1] * A2[i2,j2]
    B[j1*m2+j2, k1*p2+k2] = B1[j1,k1] * B2[j2,k2]
    C[i1*n2+i2, k1*p2+k2] = C1[i1,k1] * C2[i2,k2]
    
    So for U: row index = i*m + j where n=n1*n2, m=m1*m2
    i = i1*n2 + i2, j = j1*m2 + j2
    row = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
    
    U[row, r*S2+s] = U1[i1*m1+j1, r] * U2[i2*m2+j2, s]
    
    For V: row = j*p + k where m=m1*m2, p=p1*p2
    j = j1*m2+j2, k = k1*p2+k2
    row = (j1*m2+j2)*(p1*p2) + (k1*p2+k2)
    V[row, r*S2+s] = V1[j1*p1+k1, r] * V2[j2*p2+k2, s]
    
    For W: row = k*n + i where p=p1*p2, n=n1*n2
    k = k1*p2+k2, i = i1*n2+i2
    row = (k1*p2+k2)*(n1*n2) + (i1*n2+i2)
    W[row, r*S2+s] = W1[k1*n1+i1, r] * W2[k2*n2+i2, s]
    """
    # Determine dimensions from matrix shapes
    n1, S1 = U1.shape
    m1, _ = V1.shape
    p1, _ = W1.shape
    # n1 = n1*m1, so actual n1 = n1 // m1
    n1_actual = n1 // m1
    p1_actual = p1 // n1_actual
    
    n2, S2 = U2.shape
    m2, _ = V2.shape
    p2, _ = W2.shape
    n2_actual = n2 // m2
    p2_actual = p2 // n2_actual
    
    n = n1_actual * n2_actual
    m = m1 * m2
    p = p1_actual * p2_actual
    rank = S1 * S2
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(S1):
        for s in range(S2):
            col = r * S2 + s
            for i1 in range(n1_actual):
                for j1 in range(m1):
                    for i2 in range(n2_actual):
                        for j2 in range(m2):
                            row_u = (i1 * n2_actual + i2) * m + (j1 * m2 + j2)
                            U[row_u, col] = U1[i1 * m1 + j1, r] * U2[i2 * m2 + j2, s]
            
            for j1 in range(m1):
                for k1 in range(p1_actual):
                    for j2 in range(m2):
                        for k2 in range(p2_actual):
                            row_v = (j1 * m2 + j2) * p + (k1 * p2_actual + k2)
                            V[row_v, col] = V1[j1 * p1_actual + k1, r] * V2[j2 * p2_actual + k2, s]
            
            for k1 in range(p1_actual):
                for i1 in range(n1_actual):
                    for k2 in range(p2_actual):
                        for i2 in range(n2_actual):
                            row_w = (k1 * p2_actual + k2) * n + (i1 * n2_actual + i2)
                            W[row_w, col] = W1[k1 * n1_actual + i1, r] * W2[k2 * n2_actual + i2, s]
    
    return U, V, W


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Uses Kronecker product of Strassen (2x2x2, rank 7) with trivial decompositions.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    time_limit = 120
    
    # Strategy: 4x4x7 = (2x2x2) ⊗ (2x2x1) for the first 4 columns,
    # plus 4x4x3 for the remaining 3 columns.
    # Actually, let's think differently.
    # 4x4x7: We can split as 4x4x4 + 4x4x3.
    # For 4x4x4: use (2x2x2) ⊗ (2x2x2) = rank 7*7 = 49
    # For 4x4x3: use (2x2x1) ⊗ (2x2x3) ... hmm, that's not clean.
    # Alternative: 4x4x3 = (2x2x3) ⊗ (2x1x1) = rank(2x2x3) * rank(2x1x1)
    # rank(2x2x3) via Strassen-like? Actually 2x2x3 can be done in rank 6 (schoolbook).
    # Or we can do 4x4x3 directly.
    
    # Let me try: 4x4x7 = (2x2x7) ⊗ (2x2x1)
    # rank(2x2x7): we need to compute 2x2 * 2x7 -> 2x7
    # This is like 7 independent 2x2*2x1 multiplications... no.
    # Actually 2x2 * 2x7: for each of the 7 output columns, we do a 2x2 * 2x1.
    # Using Strassen for each: 7 * 7 = 49? No, Strassen gives 7 mults for 2x2*2x2.
    # For 2x2*2x1, we need 4 mults (schoolbook) or can we do better?
    # 2x2 * 2x1 -> 2x1: c1 = a11*b1 + a12*b2, c2 = a21*b1 + a22*b2. That's 4 mults.
    # Can't really do better for 2x1.
    # So 2x2x7 via schoolbook: 2*2*7 = 28.
    # Then (2x2x7) ⊗ (2x2x1): rank = 28 * (2*2*1) = 28*4 = 112. That's just schoolbook.
    
    # Better approach: 4x4x4 via Kronecker of two Strassen: rank 49.
    # 4x4x3: Let's just use schoolbook: 4*4*3 = 48.
    # Total: 49 + 48 = 97.
    
    # Or: 4x4x7 directly. Let's try (2x2x2)⊗(2x2x4) + ... no.
    
    # Let's go with: 4x4x4 (rank 49 via Kron Strassen) + 4x4x3 (rank 48 schoolbook) = 97.
    
    # Build 4x4x4 via Kronecker of two 2x2x2 Strassen
    U1, V1, W1 = _strassen_2x2()
    U2, V2, W2 = _strassen_2x2()
    U_444, V_444, W_444 = _kron_decomp(U1, V1, W1, U2, V2, W2)
    # U_444: (16, 49), V_444: (16, 49), W_444: (16, 49)
    
    # Build 4x4x3 via schoolbook
    U_443, V_443, W_443 = _trivial_decomp(4, 4, 3)
    # U_443: (16, 48), V_443: (12, 48), W_443: (12, 48)
    
    # Now embed into full 4x4x7 space.
    # Full: n=4, m=4, p=7
    # U: (16, 97), V: (28, 97), W: (28, 97)
    
    # The 4x4x4 part handles B columns 0..3 (j=0..3, k=0..3)
    # The 4x4x3 part handles B columns 4..6 (j=0..3, k=4..6)
    
    # U: same for both (A is 4x4 in both cases)
    U_full = np.zeros((16, 49 + 48))
    U_full[:, 0:49] = U_444
    U_full[:, 49:97] = U_443
    
    # V: B is 4x7, so V has 28 rows.
    # V_444 corresponds to B[0:4, 0:4] -> rows 0..15 (j*p+k where j in 0..3, k in 0..3, p=4)
    #   row = j*4 + k, j in 0..3, k in 0..3 -> rows 0..15
    # V_443 corresponds to B[0:4, 4:7] -> rows 16..27 (j in 0..3, k in 4..6, p=7)
    #   row = j*7 + k, j in 0..3, k in 4..6 -> rows 4,5,6, 11,12,13, 18,19,20, 25,26,27
    # Wait, that's not contiguous. Let me re-examine.
    # V_443 has shape (12, 48) where 12 = 4*3. The indexing is j*3+k where j in 0..3, k in 0..2.
    # But in the full problem, V has shape (28, rank) where 28 = 4*7, indexed by j*7+k.
    # So for the 4x4x3 part, the k values are 4,5,6 (the last 3 columns of B).
    # V_443[j*3+k, r] should map to V_full[j*7+(k+4), r]
    
    V_full = np.zeros((28, 97))
    # Place V_444: rows j*4+k for j in 0..3, k in 0..3 -> but in full, it's j*7+k
    for j in range(4):
        for k in range(4):
            V_full[j * 7 + k, 0:49] = V_444[j * 4 + k, 0:49]
    # Place V_443: rows j*3+k for j in 0..3, k in 0..2 -> maps to j*7+(k+4)
    for j in range(4):
        for k in range(3):
            V_full[j * 7 + (k + 4), 49:97] = V_443[j * 3 + k, 0:48]
    
    # W: C is 4x7, so W has 28 rows indexed by k*4+i.
    # W_444: shape (16, 49), indexed by k*4+i for k in 0..3, i in 0..3.
    #   In full: k*7+i for k in 0..3, i in 0..3.
    # W_443: shape (12, 48), indexed by k*3+i for k in 0..2, i in 0..3.
    #   In full: (k+4)*7+i for k in 0..2, i in 0..3.
    
    W_full = np.zeros((28, 97))
    for k in range(4):
        for i in range(4):
            W_full[k * 7 + i, 0:49] = W_444[k * 4 + i, 0:49]
    for k in range(3):
        for i in range(4):
            W_full[(k + 4) * 7 + i, 49:97] = W_443[k * 3 + i, 0:48]
    
    if time.time() - start_time > time_limit:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

# EVOLVE-BLOCK-END
