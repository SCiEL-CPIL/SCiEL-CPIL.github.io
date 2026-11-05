"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_tensor(n, m, p):
    """Build the tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    # Use einsum for efficient computation
    # U: (n_a, rank), V: (n_b, rank), W: (n_c, rank)
    # Result: (n_a, n_b, n_c)
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(result, T, atol=tol)

def _get_rank17_terms():
    """
    Returns the 17-term decomposition for 2x3x3 matrix multiplication.
    
    Decomposition strategy:
    1. Strassen for the 2x2x2 block (7 terms)
    2. Direct multiplication for the remaining parts (10 terms)
    
    Total: 17 terms.
    """
    terms = []
    
    # Part 1: Strassen for 2x2 x 2x2
    # Indices for A (2x2): 0,1,2,3 (row-major: 00,01,10,11)
    # Indices for B (2x2): 0,1,2,3
    # Indices for C (2x2): 0,1,2,3
    
    # M1 = (A00 + A11) * (B00 + B11)
    terms.append(([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]))
    
    # M2 = (A10 + A11) * B00
    terms.append(([2, 3], [1, 1], [0], [1], [1, 3], [1, -1])) # Note: W indices adjusted for C10, C11
    
    # M3 = A00 * (B10 - B00)
    terms.append(([0], [1], [2, 0], [1, -1], [2, 3], [1, 1]))
    
    # M4 = A11 * (B01 - B11)
    terms.append(([3], [1], [1, 3], [1, -1], [0, 1], [1, 1]))
    
    # M5 = (A00 + A01) * B11
    terms.append(([0, 1], [1, 1], [3], [1], [0, 2], [-1, 1])) # Wait, Strassen M5 is (A00+A01)*B11? No.
    # Standard Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B00)
    # M4 = A11*(B01-B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B11+B21) -> (A01-A11)*(B11+B01) for 2x2
    
    # Let's re-verify the specific indices used in the seed program which was valid.
    # The seed program had:
    # 1. ([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]) -> This looks like it's mixing indices from a larger block?
    # The seed program used indices 0-5 for U, 0-8 for V, 0-5 for W.
    # U indices: 0,1,2,3,4,5 correspond to A (2x3).
    # 0: A00, 1: A01, 2: A02, 3: A10, 4: A11, 5: A12.
    # V indices: 0-8 correspond to B (3x3).
    # 0: B00, 1: B01, 2: B02, 3: B10, 4: B11, 5: B12, 6: B20, 7: B21, 8: B22.
    # W indices: 0-5 correspond to C (2x3).
    # 0: C00, 1: C01, 2: C02, 3: C10, 4: C11, 5: C12.
    
    # The seed's "Part 1" used indices 0,4 for U. This is A00 and A11.
    # It used indices 0,4 for V. This is B00 and B11.
    # This suggests it was treating the top-left 2x2 block of B?
    # B00 is idx 0. B11 is idx 4.
    # So it is multiplying A[0,1;0,1] (indices 0,1,3,4? No, A00=0, A01=1, A10=3, A11=4)
    # by B[0,1;0,1] (indices 0,1,3,4).
    # Result C[0,1;0,1] (indices 0,1,3,4).
    
    # Let's just copy the valid terms from the seed program to ensure correctness,
    # as deriving Strassen indices manually is error-prone.
    
    terms = []
    # Part 1: Strassen-like for the 2x2 block of A (rows 0,1; cols 0,1) and 2x2 block of B (rows 0,1; cols 0,1)
    # A submatrix indices: 0,1,3,4
    # B submatrix indices: 0,1,3,4
    # C submatrix indices: 0,1,3,4
    
    # M1 = (A00 + A11) * (B00 + B11)
    terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1])) # C00, C10? No, C indices 0,1,3,4.
    # Seed had: [0, 3] for W. W0=C00, W3=C10.
    # Wait, Strassen M1 contributes to C00, C01, C10, C11.
    # The seed's term 1 only had W indices 0,3. This is suspicious.
    
    # Let's look at the seed's terms again.
    # Term 1: U:0,4 (A00,A11). V:0,4 (B00,B11). W:0,3 (C00,C10).
    # This term calculates (A00+A11)*(B00+B11) and adds to C00 and C10?
    # C00 = A00B00 + A01B10 + A02B20.
    # C10 = A10B00 + A11B10 + A12B20.
    # This term seems to be part of a specific decomposition.
    
    # Since the seed program's terms were verified to work (score 1.0588), 
    # and I cannot easily derive a better one, I will reuse the exact terms 
    # but structure the code more cleanly.
    
    terms = []
    # Part 1: 7 terms
    terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]))
    terms.append(([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]))
    terms.append(([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]))
    terms.append(([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]))
    terms.append(([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]))
    terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    terms.append(([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]))
    
    # Part 2: 4 terms
    terms.append(([0], [1], [2], [1], [4], [1]))
    terms.append(([1], [1], [5], [1], [4], [1]))
    terms.append(([3], [1], [2], [1], [5], [1]))
    terms.append(([4], [1], [5], [1], [5], [1]))
    
    # Part 3: 6 terms
    terms.append(([2], [1], [6], [1], [0], [1]))
    terms.append(([2], [1], [7], [1], [2], [1]))
    terms.append(([2], [1], [8], [1], [4], [1]))
    terms.append(([5], [1], [6], [1], [1], [1]))
    terms.append(([5], [1], [7], [1], [3], [1]))
    terms.append(([5], [1], [8], [1], [5], [1]))
    
    return terms

def _terms_to_matrices(terms, n_a, n_b, n_c):
    """Convert list of terms to U, V, W matrices."""
    rank = len(terms)
    U = np.zeros((n_a, rank))
    V = np.zeros((n_b, rank))
    W = np.zeros((n_c, rank))
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    T = _build_tensor(n, m, p)
    
    terms = _get_rank17_terms()
    U, V, W = _terms_to_matrices(terms, n_a, n_b, n_c)
    
    if not _verify_decomposition(U, V, W, T):
        # Fallback to schoolbook (rank 18)
        rank_fb = n * m * p
        U_fb = np.zeros((n_a, rank_fb))
        V_fb = np.zeros((n_b, rank_fb))
        W_fb = np.zeros((n_c, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
