"""Matrix multiplication 2x2x14: rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x14 matrix using Strassen's algorithm.
    
    Strategy: Group the 14 columns of B into 7 pairs. For each pair,
    apply Strassen's 2x2x2 matrix multiplication algorithm. Since the
    2x2 matrix A is shared across all pairs, we can use the same
    Strassen decomposition for each pair.
    
    Strassen's algorithm for 2x2x2 uses 7 rank-1 terms instead of 8
    (schoolbook). For 7 pairs, total rank = 7 * 7 = 49.
    
    The Strassen decomposition for 2x2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    
    For each pair of columns (2k, 2k+1) in B, we apply this decomposition.
    Each M_i is a scalar (from A) times a 1x2 vector (from B's pair of columns).
    The result is a 1x2 vector that contributes to the corresponding pair of
    output columns.
    
    In the tensor decomposition:
    - u is in R^4 (A-space: indices 00,01,10,11)
    - v is in R^28 (B-space: indices (j,k) for j in {0,1}, k in {0..13})
    - w is in R^28 (C-space: indices (i',k') for i' in {0,1}, k' in {0..13})
    
    For pair k (columns 2k and 2k+1):
    - v has non-zeros only in positions (0,2k), (0,2k+1), (1,2k), (1,2k+1)
    - w has non-zeros only in positions (0,2k), (0,2k+1), (1,2k), (1,2k+1)
    """
    n, m, p = DIMENSIONS  # 2, 2, 14
    
    # Strassen decomposition coefficients
    # Each entry is (u_coeffs, v_coeffs, w_coeffs) for one rank-1 term
    # u_coeffs: 4 values for A-space (indices 00,01,10,11)
    # v_coeffs: 4 values for B-space (indices 00,01,10,11 within a pair)
    # w_coeffs: 4 values for C-space (indices 00,01,10,11 within a pair)
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11): u=[1,0,0,1], v=[1,0,1,0], w=[1,0,1,0]
    # M2 = (A10+A11)B00:       u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    # M3 = A00(B10-B11):       u=[1,0,0,0], v=[0,0,1,-1], w=[0,1,0,0]
    # M4 = A11(B01-B10):       u=[0,0,0,1], v=[0,1,-1,0], w=[1,0,1,0]
    # M5 = (A00+A01)B11:       u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
    # M6 = (A10-A00)(B00+B10): u=[-1,0,1,0], v=[1,0,1,0], w=[0,0,0,1]
    # M7 = (A01-A11)(B01+B11): u=[0,1,0,-1], v=[0,1,0,1], w=[1,0,0,0]
    
    strassen_terms = [
        ([1,0,0,1], [1,0,1,0], [1,0,1,0]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # M2
        ([1,0,0,0], [0,0,1,-1], [0,1,0,0]),  # M3
        ([0,0,0,1], [0,1,-1,0], [1,0,1,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5
        ([-1,0,1,0], [1,0,1,0], [0,0,0,1]),  # M6
        ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),  # M7
    ]
    
    rank = 7 * 7  # 49
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    col = 0
    for pair_idx in range(7):
        for term_idx in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[term_idx]
            
            # u: A-space (4 elements: 00,01,10,11)
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = u_coeffs[i * 2 + j]
            
            # v: B-space (28 elements: (j,k) for j in {0,1}, k in {0..13})
            # For pair pair_idx, columns are 2*pair_idx and 2*pair_idx+1
            # v_coeffs has 4 elements: [B00, B01, B10, B11] for the pair
            # B00 = (j=0, k=2*pair_idx), B01 = (j=0, k=2*pair_idx+1)
            # B10 = (j=1, k=2*pair_idx), B11 = (j=1, k=2*pair_idx+1)
            k0 = 2 * pair_idx
            k1 = 2 * pair_idx + 1
            V[0 * p + k0][col] = v_coeffs[0]  # B00
            V[0 * p + k1][col] = v_coeffs[1]  # B01
            V[1 * p + k0][col] = v_coeffs[2]  # B10
            V[1 * p + k1][col] = v_coeffs[3]  # B11
            
            # w: C-space (28 elements: (i',k') for i' in {0,1}, k' in {0..13})
            # w_coeffs has 4 elements: [C00, C01, C10, C11] for the pair
            # C00 = (i'=0, k'=2*pair_idx), C01 = (i'=0, k'=2*pair_idx+1)
            # C10 = (i'=1, k'=2*pair_idx), C11 = (i'=1, k'=2*pair_idx+1)
            W[k0 * n + 0][col] = w_coeffs[0]  # C00
            W[k1 * n + 0][col] = w_coeffs[1]  # C01
            W[k0 * n + 1][col] = w_coeffs[2]  # C10
            W[k1 * n + 1][col] = w_coeffs[3]  # C11
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
