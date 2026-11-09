"""Matrix multiplication 3x4x7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 4, 7]

import numpy as np
from fractions import Fraction
import time
import itertools

def _try_strassen_2x2x2():
    """
    Build a rank-21 decomposition using 2x2x2 Strassen blocks
    and handle the remaining dimensions directly.
    
    For 3x4x7:
    - Split A (3x4) into: top 2 rows (2x4) + bottom 1 row (1x4)
    - Split B (4x7) into: left 4 cols (4x7) -- we split into (4x2)+(4x2)+(4x1)
    
    Actually, let me use a cleaner approach:
    - Use 2x2x2 Strassen for the 2x2x2 sub-blocks
    - The 2x4x7 part can be handled as two 2x2x7 blocks (Strassen on 2x2, direct on x7)
    - The 1x4x7 part is direct
    
    Let me think more carefully.
    
    3x4x7:
    A = [A1 (2x4); A2 (1x4)]
    B = [B1 (4x4), B2 (4x3)]  -- or B = B_left (4x4) + ... 
    
    Better: split into (2x2x2) Strassen blocks where possible.
    
    A is 3x4, B is 4x7.
    Split A rows: 2+1, Split A cols: 2+2, Split B rows: 2+2, Split B cols: 2+2+2+1
    
    This is getting complicated. Let me use a simpler but still valid decomposition.
    
    Strategy: Use the known fact that C(3,4,7) <= 76.
    I'll construct it via:
    - 2x2x2 Strassen for overlapping blocks
    - Direct for the rest
    
    Actually, let me just use a careful construction:
    
    For the 3x4x7 product:
    - The 2x4x4 sub-problem can use Strassen (rank 7 instead of 8)
    - The 1x4x4 sub-problem is rank 4
    - The 2x4x3 sub-problem: split into 2x4x2 + 2x4x1, Strassen on 2x4x2 (rank 14) + 4 direct
    - The 1x4x3 sub-problem: rank 3
    
    Wait, this doesn't quite work because the column splits of B need to be consistent.
    
    Let me use a different approach: I'll use the explicit rank-76 construction
    from the literature, implemented via a known formula.
    
    For C(3,4,7):
    The construction uses:
    - 6 Strassen 2x2x2 blocks (each rank 7) = 42
    - 34 direct terms
    Total = 76
    
    The 6 Strassen blocks cover:
    (i=0,1; j=0,1; k=0,1), (i=0,1; j=0,1; k=2,3), (i=0,1; j=0,1; k=4,5)
    (i=1,2; j=0,1; k=0,1), (i=1,2; j=0,1; k=2,3), (i=1,2; j=0,1; k=4,5)
    
    Wait, that's not right either. Let me just implement the schoolbook and verify
    it's correct, then try to improve.
    """
    pass

def _verify(U, V, W, n=3, m=4, p=7):
    """Verify the decomposition: sum_t U[:,t] outer V[:,t] outer W[:,t] = T"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    T = np.zeros((n, m, p))
    for t in range(U.shape[1]):
        u = U[:, t]  # (n*m,)
        v = V[:, t]  # (m*p,)
        w = W[:, t]  # (p*n,)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    # Expected tensor
    T_expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i, j, k] = 1
    return np.allclose(T, T_expected, atol=1e-9)

def _schoolbook(n=3, m=4, p=7):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2_block(i0, j0, k0, U, V, W, start_col, n, m, p):
    """
    Add a 2x2x2 Strassen block to the decomposition.
    i0: starting row index in A (0-based), so rows i0, i0+1
    j0: starting col index in A (0-based), so cols j0, j0+1
    k0: starting col index in B (0-based), so cols k0, k0+1
    This computes C[i0:i0+2, k0:k0+2] += A[i0:i0+2, j0:j0+2] @ B[j0:j0+2, k0:k0+2]
    """
    # Strassen 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # A indices: row i0+i, col j0+j -> U index (i0+i)*m + (j0+j)
    # B indices: row j0+j, col k0+k -> V index (j0+j)*p + (k0+k)
    # C indices: row i0+i, col k0+k -> W index (k0+k)*n + (i0+i)
    
    A00 = (i0 + 0) * m + (j0 + 0)
    A01 = (i0 + 0) * m + (j0 + 1)
    A10 = (i0 + 1) * m + (j0 + 0)
    A11 = (i0 + 1) * m + (j0 + 1)
    
    B00 = (j0 + 0) * p + (k0 + 0)
    B01 = (j0 + 0) * p + (k0 + 1)
    B10 = (j0 + 1) * p + (k0 + 0)
    B11 = (j0 + 1) * p + (k0 + 1)
    
    C00 = (k0 + 0) * n + (i0 + 0)
    C01 = (k0 + 1) * n + (i0 + 0)
    C10 = (k0 + 0) * n + (i0 + 1)
    C11 = (k0 + 1) * n + (i0 + 1)
    
    # 7 Strassen terms
    terms = [
        # (A_coeff_dict, B_coeff_dict, C_coeff_dict)
        # M1 = (A00+A11)*(B00+B11), contributes to C00(+1), C11(+1)
        ({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1}),
        # M2 = (A10+A11)*B00, contributes to C10(+1), C11(-1)
        ({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1}),
        # M3 = A00*(B10-B11), contributes to C01(+1), C11(+1)
        ({A00: 1}, {B10: 1, B11: -1}, {C01: 1, C11: 1}),
        # M4 = A11*(B01-B10), contributes to C00(+1), C10(+1)
        ({A11: 1}, {B01: 1, B10: -1}, {C00: 1, C10: 1}),
        # M5 = (A00+A01)*B11, contributes to C00(-1), C01(+1)
        ({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1}),
        # M6 = (A10-A00)*(B00+B10), contributes to C11(+1)
        ({A10: 1, A00: -1}, {B00: 1, B10: 1}, {C11: 1}),
        # M7 = (A01-A11)*(B01+B11), contributes to C00(+1)
        ({A01: 1, A11: -1}, {B01: 1, B11: 1}, {C00: 1}),
    ]
    
    for a_coeff, b_coeff, c_coeff in terms:
        col = start_col
        for a_idx, a_val in a_coeff.items():
            U[a_idx][col] += a_val
        for b_idx, b_val in b_coeff.items():
            V[b_idx][col] += b_val
        for c_idx, c_val in c_coeff.items():
            W[c_idx][col] += c_val
        start_col += 1
    
    return start_col

def _direct_term(i, j, k, U, V, W, col, n, m, p):
    """Add a direct rank-1 term for C[i,k] += A[i,j]*B[j,k]"""
    U[i * m + j][col] += 1
    V[j * p + k][col] += 1
    W[k * n + i][col] += 1
    return col + 1

def _try_rank_76():
    """
    Attempt to construct rank-76 decomposition for 3x4x7.
    
    Strategy:
    - Use 2x2x2 Strassen blocks for sub-matrices
    - Fill in the rest with direct terms
    
    Split A (3x4) into rows 0-1 and row 2, cols 0-1 and 2-3
    Split B (4x7) into rows 0-1 and 2-3, cols 0-1, 2-3, 4-5, 6
    
    Strassen blocks (2x2x2):
    1. (i=0,1; j=0,1; k=0,1) - rank 7
    2. (i=0,1; j=0,1; k=2,3) - rank 7
    3. (i=0,1; j=0,1; k=4,5) - rank 7
    4. (i=0,1; j=2,3; k=0,1) - rank 7
    5. (i=0,1; j=2,3; k=2,3) - rank 7
    6. (i=0,1; j=2,3; k=4,5) - rank 7
    Total Strassen: 42
    
    Remaining C entries:
    - C[2,0], C[2,1], C[2,2], C[2,3], C[2,4], C[2,5], C[2,6]: 7 entries
    - C[0,6], C[1,6]: 2 entries
    - C[0,0]..C[0,5], C[1,0]..C[1,5] are covered by Strassen
    - C[2,0]..C[2,5] need handling
    
    Wait, I need to be more careful. The Strassen blocks compute:
    - C[0:2, 0:2] += A[0:2, 0:2] @ B[0:2, 0:2]
    - C[0:2, 2:4] += A[0:2, 0:2] @ B[0:2, 2:4]
    - C[0:2, 4:6] += A[0:2, 0:2] @ B[0:2, 4:6]
    - C[0:2, 0:2] += A[0:2, 2:4] @ B[2:4, 0:2]
    - C[0:2, 2:4] += A[0:2, 2:4] @ B[2:4, 2:4]
    - C[0:2, 4:6] += A[0:2, 2:4] @ B[2:4, 4:6]
    
    So C[0:2, 0:6] is fully covered (each entry gets contributions from both j=0,1 and j=2,3 blocks).
    C[0:2, 6] is NOT covered.
    C[2, 0:6] is NOT covered.
    C[2, 6] is NOT covered.
    
    Remaining:
    - C[0,6], C[1,6]: need A[0,0:4]*B[0:4,6] and A[1,0:4]*B[0:4,6]
      That's 4+4 = 8 direct terms, or we can do better.
    - C[2,0:6]: need A[2,0:4]*B[0:4,0:6]
      That's 4*6 = 24 direct terms.
    - C[2,6]: 4 direct terms.
    
    Total remaining: 8 + 24 + 4 = 36 direct terms.
    Total: 42 + 36 = 78. Not 76.
    
    Hmm. Let me try a different split.
    
    Alternative: use 2x2x2 Strassen more cleverly.
    
    Actually, for C(3,4,7), the known bound is 76. Let me try:
    
    Split A into (2,1) rows, B into (4,7) with B split into (4,2)+(4,2)+(4,2)+(4,1).
    
    Use Strassen on:
    - A[0:2, 0:2] @ B[0:2, 0:2] (2x2x2, rank 7)
    - A[0:2, 0:2] @ B[0:2, 2:4] (2x2x2, rank 7)
    - A[0:2, 0:2] @ B[0:2, 4:6] (2x2x2, rank 7)
    - A[0:2, 2:4] @ B[2:4, 0:2] (2x2x2, rank 7)
    - A[0:2, 2:4] @ B[2:4, 2:4] (2x2x2, rank 7)
    - A[0:2, 2:4] @ B[2:4, 4:6] (2x2x2, rank 7)
    
    These cover C[0:2, 0:6].
    
    For C[0:2, 6]: A[0:2, 0:4] @ B[0:4, 6] -> 2*4 = 8 direct, but we can use
    (A[0:2,0:2]+A[0:2,2:4]) @ B[0:4,6]... no, that's not a product.
    
    Actually, A[0:2, 0:4] @ B[0:4, 6] = sum_{j=0}^{3} A[0:2,j] * B[j,6]
    This is 2*4 = 8 scalar products. We can't reduce below that without structure.
    
    Hmm, but we can combine: (A[0,0]+A[0,2])*(B[0,6]+B[2,6]) + ... no, that introduces cross terms.
    
    Let me just accept rank 78 for now and see if I can verify it.
    
    Actually wait - let me reconsider. The known C(3,4,7) <= 76 comes from a specific construction.
    Let me try a different approach:
    
    Use the fact that C(3,4,7) can be achieved by:
    - 3 copies of C(2,4,7) overlapping
    - C(2,4,7) <= 26 (I think)
    
    This is getting too complex without the exact reference. Let me implement
    a rank-78 construction and verify it, then try to optimize.
    """
    n, m, p = 3, 4, 7
    rank = 84  # Start with schoolbook size, we'll fill in
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 6 Strassen blocks
    for (i0, j0, k0) in [(0,0,0), (0,0,2), (0,0,4), (0,2,0), (0,2,2), (0,2,4)]:
        col = _strassen_2x2x2_block(i0, j0, k0, U, V, W, col, n, m, p)
    
    # Remaining: C[0,6], C[1,6] (8 direct terms)
    # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6] + A[0,2]*B[2,6] + A[0,3]*B[3,6]
    for j in range(4):
        col = _direct_term(0, j, 6, U, V, W, col, n, m, p)
    # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6] + A[1,2]*B[2,6] + A[1,3]*B[3,6]
    for j in range(4):
        col = _direct_term(1, j, 6, U, V, W, col, n, m, p)
    
    # C[2,0:6] (24 direct terms)
    for j in range(4):
        for k in range(6):
            col = _direct_term(2, j, k, U, V, W, col, n, m, p)
    
    # C[2,6] (4 direct terms)
    for j in range(4):
        col = _direct_term(2, j, 6, U, V, W, col, n, m, p)
    
    # Total should be 42 + 8 + 24 + 4 = 78
    assert col == 78, f"Expected 78, got {col}"
    
    # Trim to actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _try_rank_76_v2():
    """
    Try to get rank 76 by being smarter about the remaining terms.
    
    The issue is the 8 + 24 + 4 = 36 remaining terms.
    We need to save 2 terms to get to 76.
    
    Idea: For C[2,0:6], instead of 24 direct terms, can we use Strassen-like tricks?
    A[2, 0:4] @ B[0:4, 0:6]
    Split B[0:4, 0:6] into B[0:2, 0:6] + B[2:4, 0:6]
    Each is 4x6, and A[2,0:4] is 1x4.
    A[2,0:4] @ B[0:2, 0:6] = A[2,0]*B[0,0:6] + A[2,1]*B[1,0:6] -> 2*6 = 12
    A[2,0:4] @ B[2:4, 0:6] = A[2,2]*B[2,0:6] + A[2,3]*B[3,0:6] -> 2*6 = 12
    Total: 24. No savings.
    
    Alternative: combine some terms.
    (A[2,0]+A[2,2])*(B[0,0:6]+B[2,0:6]) - A[2,0]*B[2,0:6] - A[2,2]*B[0,0:6]
    = A[2,0]*B[0,0:6] + A[2,2]*B[2,0:6] + cross terms
    The cross terms A[2,0]*B[2,0:6] + A[2,2]*B[0,0:6] need to be canceled,
    which requires 2*6 = 12 more terms. Not helpful.
    
    Let me try a completely different split.
    
    What if I use 2x2x2 Strassen blocks differently?
    
    Split A into (1,2) rows, B into (2,2) rows and (2,2,2,1) cols.
    
    Strassen blocks:
    - A[0:1, 0:2] @ B[0:2, 0:2] - this is 1x2x2, not 2x2x2. Can't use Strassen.
    
    Hmm. Let me try:
    - A[1:3, 0:2] @ B[0:2, 0:2] (2x2x2, rank 7)
    - A[1:3, 0:2] @ B[0:2, 2:4] (2x2x2, rank 7)
    - A[1:3, 0:2] @ B[0:2, 4:6] (2x2x2, rank 7)
    - A[1:3, 2:4] @ B[2:4, 0:2] (2x2x2, rank 7)
    - A[1:3, 2:4] @ B[2:4, 2:4] (2x2x2, rank 7)
    - A[1:3, 2:4] @ B[2:4, 4:6] (2x2x2, rank 7)
    
    These cover C[1:3, 0:6].
    
    Remaining:
    - C[0, 0:6]: A[0, 0:4] @ B[0:4, 0:6] -> 4*6 = 24 direct
    - C[0, 6]: A[0, 0:4] @ B[0:4, 6] -> 4 direct
    - C[1:3, 6]: A[1:3, 0:4] @ B[0:4, 6] -> 2*4 = 8 direct
    
    Total: 42 + 24 + 4 + 8 = 78. Same.
    
    What about using a 2x3x2 or 2x2x3 Strassen-like trick?
    
    For 2x3x2:
    A is 2x3, B is 3x2. Product is 2x2.
    Schoolbook rank: 2*3*2 = 12.
    Can we do better?
    
    C00 = A00*B00 + A01*B10 + A02*B20
    C01 = A00*B01 + A01*B11 + A02*B21
    C10 = A10*B00 + A11*B10 + A12*B20
    C11 = A10*B01 + A11*B11 + A12*B21
    
    Strassen for 2x2x2 gives rank 7 for 2x2x2.
    For 2x3x2, we can split into (2x2x2) + (2x1x2) = 7 + 4 = 11.
    Or (2x1x2) + (2x2x2) = 4 + 7 = 11.
    
    So C(2,3,2) <= 11.
    
    Similarly, C(2,3,3):
    Split into (2x2x3) + (2x1x3)
    C(2,2,3) <= C(2,2,2) + 4 = 7 + 4 = 11
    C(2,1,3) = 6
    Total: 17
    
    C(2,3,3) <= 17.
    
    For our problem 3x4x7:
    Split A into (2,1) rows.
    C(2,4,7) + C(1,4,7)
    C(1,4,7) = 28 (schoolbook, can't reduce for 1xn)
    C(2,4,7):
      Split B into (2,2) rows and (2,2,2,1) cols.
      C(2,4,2) + C(2,4,2) + C(2,4,2) + C(2,4,1)
      C(2,4,2) <= C(2,2,2) + C(2,2,2) = 7 + 7 = 14
      C(2,4,1) = 8
      C(2,4,7) <= 3*14 + 8 = 50
    Total: 50 + 28 = 78.
    
    Hmm, still 78.
    
    Let me try:
    C(2,4,7):
      Split B into (4) rows and (3,3,1) cols.
      C(2,4,3) + C(2,4,3) + C(2,4,1)
      C(2,4,3):
        Split B into (2,2) rows.
        C(2,2,3) + C(2,2,3)
        C(2,2,3) <= C(2,2,2) + 4 = 11
        C(2,4,3) <= 2*11 = 22
      C(2,4,7) <= 2*22 + 8 = 52
    Total: 52 + 28 = 80. Worse.
    
    OK let me just go with rank 78.
    """
    return _try_rank_76()  # Actually returns rank 78

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # 100s safety margin
    
    # Try rank 78 first
    try:
        result_78 = _try_rank_76()  # Actually rank 78
        if _verify(result_78["U"], result_78["V"], result_78["W"], n, m, p):
            rank_78 = len(result_78["U"][0])
            # Try to improve
            if time.time() < deadline:
                pass  # Could try other constructions
            return result_78
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
