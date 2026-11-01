"""FTW-inspired decomposition for 2x4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

def _frac(x):
    """Represent half-integer as float."""
    return float(x)

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 4, 6
    
    # FTW Decomposition:
    # T_{2x4x6} = T_{2x2x2} \otimes T_{1x2x3}
    
    # Core 1: 2x2x2 Strassen Tensor
    # Rank 7
    # U1 (2x2 x 7), V1 (2x2 x 7), W1 (2x2 x 7)
    # Indices: U1[i, j, k], V1[j, l, k], W1[i, l, k]
    # Strassen formulas:
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    U1 = np.zeros((4, 7)) # Flattened 2x2
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # Helper to add to flattened matrix
    def add_U1(i, j, k, val):
        U1[i*2+j, k] += val
    def add_V1(j, l, k, val):
        V1[j*2+l, k] += val
    def add_W1(i, l, k, val):
        W1[i*2+l, k] += val
        
    # M0
    add_U1(0,0,0,1); add_U1(1,1,0,1)
    add_V1(0,0,0,1); add_V1(1,1,0,1)
    add_W1(0,0,0,1); add_W1(0,1,0,1); add_W1(1,0,0,1); add_W1(1,1,0,1) # Wait, W is output coeff.
    
    # Let's define the tensor entries directly for the Kronecker product.
    # It is safer to construct the full U, V, W matrices of size (8, 42), (24, 42), (12, 42)
    # by iterating over the rank-1 terms of the Kronecker product.
    
    # The Kronecker product of two tensors T_A (rank R_A) and T_B (rank R_B) has rank R_A * R_B.
    # The rank-1 terms are pairs (u_a, v_b), (u'_a, v'_b), (w_a, w'_b).
    
    # Define Core 1 (2x2x2) Rank-1 terms
    # Each term is (U_vec, V_vec, W_vec) where U_vec is in R^4, V_vec in R^4, W_vec in R^4.
    core1_terms = []
    
    # M0
    u = np.zeros(4); u[0]=1; u[3]=1
    v = np.zeros(4); v[0]=1; v[3]=1
    w = np.zeros(4); w[0]=1; w[1]=1; w[2]=1; w[3]=1 # C00, C01, C10, C11 all get +1?
    # Strassen:
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # Let's list the 7 terms for 2x2x2
    # Term 0: (A00+A11)(B00+B11)
    # U: [1, 0, 0, 1]
    # V: [1, 0, 0, 1]
    # W: [1, 1, 1, 1] ? No.
    # C00 += 1, C01 += 0, C10 += 0, C11 += 1?
    # Standard Strassen:
    # M0 contributes to C00 (+1), C11 (+1).
    # Wait, C00 = M0 + M3 - M4 + M6.
    # C01 = M2 + M4.
    # C10 = M1 + M3.
    # C11 = M0 - M1 + M2 + M5.
    
    # So for M0:
    # W00=1, W01=0, W10=0, W11=1.
    
    terms_2x2x2 = [
        # M0
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M1: (A10+A11)B00
        # U: [0,0,1,1]
        # V: [1,0,0,0]
        # W: C10 += 1, C11 -= 1.
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        # M2: A00(B01-B11)
        # U: [1,0,0,0]
        # V: [0,1,-1,0]
        # W: C01 += 1, C11 += 1.
        ([1,0,0,0], [0,1,-1,0], [0,1,0,1]),
        # M3: A11(B10-B00)
        # U: [0,0,0,1]
        # V: [0,0,1,-1] # B10 - B00. B indices: 00, 01, 10, 11.
        # V[2]=1, V[0]=-1.
        # W: C00 += 1, C10 += 1.
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        # M4: (A00+A01)B11
        # U: [1,1,0,0]
        # V: [0,0,0,1]
        # W: C00 -= 1, C01 += 1.
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # M5: (A10-A00)(B00+B01)
        # U: [-1,0,1,0]
        # V: [1,1,0,0]
        # W: C11 += 1.
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        # M6: (A01-A11)(B10+B11)
        # U: [0,1,-1,0]
        # V: [0,0,1,1]
        # W: C00 += 1.
        ([0,1,-1,0], [0,0,1,1], [1,0,0,0]),
    ]
    
    # Core 2: 1x2x3 Tensor
    # Rank 6.
    # Terms:
    # 0: A0 B00
    # 1: A1 B10
    # 2: A0 B01
    # 3: A1 B11
    # 4: A0 B02
    # 5: A1 B12
    
    # U2 (1x2 x 6): Flattened size 2.
    # V2 (2x3 x 6): Flattened size 6.
    # W2 (1x3 x 6): Flattened size 3.
    
    terms_1x2x3 = [
        # A0 B00
        ([1,0], [1,0,0,0,0,0], [1,0,0]),
        # A1 B10
        ([0,1], [0,0,1,0,0,0], [1,0,0]),
        # A0 B01
        ([1,0], [0,1,0,0,0,0], [0,1,0]),
        # A1 B11
        ([0,1], [0,0,0,1,0,0], [0,1,0]),
        # A0 B02
        ([1,0], [0,0,0,0,1,0], [0,0,1]),
        # A1 B12
        ([0,1], [0,0,0,0,0,1], [0,0,1]),
    ]
    
    # Construct full U, V, W
    # Total Rank = 7 * 6 = 42
    rank = 42
    U = np.zeros((8, rank))
    V = np.zeros((24, rank))
    W = np.zeros((12, rank))
    
    # Mapping of indices:
    # Global A index (i, a) where i in 0..1, a in 0..3.
    # a = a1 * 2 + a0. (a1 from core1, a0 from core2? No.)
    # Core1 is 2x2x2. Core2 is 1x2x3.
    # m dimension: 2 (core1) * 2 (core2) = 4.
    # Let m = m1 * 2 + m0.
    # m1 is index in Core1 (0,1). m0 is index in Core2 (0,1).
    # n dimension: 2 (core1) * 1 (core2) = 2.
    # n = n1 * 1 + n0. n1 is Core1 (0,1). n0 is Core2 (0).
    # p dimension: 2 (core1) * 3 (core2) = 6.
    # p = p1 * 3 + p0.
    # p1 is Core1 (0,1). p0 is Core2 (0,1,2).
    
    # U row index: n * m + i? No, U is (n*m, rank).
    # Input A is n x m. Index (i, j).
    # Row in U = i * m + j = i * 4 + j.
    # j = j1 * 2 + j0.
    # i = i1 * 1 + i0.
    
    # V row index: m * p + j? No, V is (m*p, rank).
    # Input B is m x p. Index (j, k).
    # Row in V = j * p + k = j * 6 + k.
    # j = j1 * 2 + j0.
    # k = k1 * 3 + k0.
    
    # W row index: p * n + k? No, W is (p*n, rank).
    # Output C is n x p. Index (i, k).
    # Trace convention W[k*n + i, r].
    # Row in W = k * n + i = k * 2 + i.
    # k = k1 * 3 + k0.
    # i = i1 * 1 + i0.
    
    for idx1, (u1, v1, w1) in enumerate(terms_2x2x2):
        for idx2, (u2, v2, w2) in enumerate(terms_1x2x3):
            col = idx1 * 6 + idx2
            
            # U contributions
            # u1 has 4 entries (i1, j1). u2 has 2 entries (j0).
            # i = i1. j = j1*2 + j0.
            for i1 in range(2):
                for j1 in range(2):
                    coeff1 = u1[i1*2 + j1]
                    if coeff1 != 0:
                        for j0 in range(2):
                            coeff2 = u2[j0]
                            if coeff2 != 0:
                                i = i1
                                j = j1 * 2 + j0
                                U[i*4 + j, col] += coeff1 * coeff2
            
            # V contributions
            # v1 has 4 entries (j1, k1). v2 has 6 entries (j0, k0).
            # j = j1*2 + j0. k = k1*3 + k0.
            for j1 in range(2):
                for k1 in range(2):
                    coeff1 = v1[j1*2 + k1]
                    if coeff1 != 0:
                        for j0 in range(2):
                            for k0 in range(3):
                                coeff2 = v2[j0*3 + k0]
                                if coeff2 != 0:
                                    j = j1 * 2 + j0
                                    k = k1 * 3 + k0
                                    V[j*6 + k, col] += coeff1 * coeff2
            
            # W contributions
            # w1 has 4 entries (i1, k1). w2 has 3 entries (k0).
            # i = i1. k = k1*3 + k0.
            for i1 in range(2):
                for k1 in range(2):
                    coeff1 = w1[i1*2 + k1]
                    if coeff1 != 0:
                        for k0 in range(3):
                            coeff2 = w2[k0]
                            if coeff2 != 0:
                                i = i1
                                k = k1 * 3 + k0
                                W[k*2 + i, col] += coeff1 * coeff2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
