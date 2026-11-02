"""Matrix multiplication 4x4x7 tensor decomposition with rank 63."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_entries():
    """Return (U, V, W) entries for Strassen 2x2x2 rank-7 decomposition.
    
    Indexing:
    - U[row, col] where row = i*2+j for A[i,j], col = step index 0..6
    - V[row, col] where row = j*2+k for B[j,k], col = step index 0..6
    - W[row, col] where row = k*2+i for C[i,k], col = step index 0..6
    
    Strassen's 7 products:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)(B11)
    M3 = (A11)(B12-B22)
    M4 = (A22)(B21-B11)
    M5 = (A11+A12)(B22)
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    U = {}
    V = {}
    W = {}
    
    # M1: (A11+A22)(B11+B22) -> A00=1, A11=1; B00=1, B11=1; C00=1, C11=1
    U[(0, 0)] = 1; U[(3, 0)] = 1
    V[(0, 0)] = 1; V[(3, 0)] = 1
    W[(0, 0)] = 1; W[(3, 0)] = 1
    
    # M2: (A21+A22)(B11) -> A10=1, A11=1; B00=1; C10=1, C11=-1
    U[(2, 1)] = 1; U[(3, 1)] = 1
    V[(0, 1)] = 1
    W[(2, 1)] = 1; W[(3, 1)] = -1
    
    # M3: (A11)(B12-B22) -> A00=1; B01=1, B11=-1; C01=1, C11=1
    U[(0, 2)] = 1
    V[(1, 2)] = 1; V[(3, 2)] = -1
    W[(1, 2)] = 1; W[(3, 2)] = 1
    
    # M4: (A22)(B21-B11) -> A11=1; B10=1, B00=-1; C00=1, C10=1
    U[(3, 3)] = 1
    V[(2, 3)] = 1; V[(0, 3)] = -1
    W[(0, 3)] = 1; W[(2, 3)] = 1
    
    # M5: (A11+A12)(B22) -> A00=1, A01=1; B11=1; C00=-1, C01=1
    U[(0, 4)] = 1; U[(1, 4)] = 1
    V[(3, 4)] = 1
    W[(0, 4)] = -1; W[(1, 4)] = 1
    
    # M6: (A21-A11)(B11+B12) -> A10=1, A00=-1; B00=1, B01=1; C11=1
    U[(2, 5)] = 1; U[(0, 5)] = -1
    V[(0, 5)] = 1; V[(1, 5)] = 1
    W[(3, 5)] = 1
    
    # M7: (A12-A22)(B21+B22) -> A01=1, A11=-1; B10=1, B11=1; C00=1
    U[(1, 6)] = 1; U[(3, 6)] = -1
    V[(2, 6)] = 1; V[(3, 6)] = 1
    W[(0, 6)] = 1
    
    return U, V, W


def _build_strassen_4x4x4(n, m, p):
    """Build Strassen-based rank-15 decomposition for 4x4x4.
    
    Uses 2x2x2 Strassen (rank 7) applied to 2x2 blocks.
    The 4x4x4 tensor is decomposed as:
    - 2 independent 2x2x2 Strassen calls (for top-left and bottom-right blocks)
    - Plus correction terms for cross-block interactions
    
    Actually, for 4x4x4 we can use a recursive Strassen approach:
    Split A, B, C into 2x2 blocks of 2x2 matrices.
    Each 2x2x2 multiplication uses 7 scalar multiplications.
    Total: 7 * 4 = 28? No, that's not right either.
    
    Let me use a different approach: the 4x4x4 tensor has known rank 49,
    but we can do better by exploiting the structure.
    
    For simplicity and correctness, let me use a block decomposition:
    A = [[A11, A12], [A21, A22]] where each block is 2x2
    B = [[B11, B12], [B21, B22]] where each block is 2x2
    C = [[C11, C12], [C21, C22]] where each block is 2x2
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each term like A11*B11 is a 2x2x2 multiplication, rank 7 with Strassen.
    We have 8 such terms, so naive rank = 8*7 = 56.
    But we can share some computations.
    
    For now, let me just implement schoolbook for 4x4x4 and focus on
    getting a valid decomposition. The key improvement is using Strassen
    for the 4x4x3 part if possible.
    
    Actually, let me reconsider. The simplest correct approach:
    Use schoolbook for the full 4x4x7 (rank 112).
    Then try to improve by using Strassen for sub-blocks.
    """
    # For 4x4x4, use schoolbook: rank 64
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W, n*m*p


def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Target: T[i,j,k] = 1 for all valid indices
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=1e-10)


def _schoolbook_4x4x7(n, m, p):
    """Schoolbook decomposition for 4x4x7, rank 112."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W, rank


def _strassen_based_4x4x7(n, m, p):
    """
    Build a Strassen-based decomposition for 4x4x7.
    
    Strategy: Split B into B1 (4x4) and B2 (4x3).
    A*B = [A*B1 | A*B2]
    
    For A*B1 (4x4x4): Use a block Strassen approach.
    For A*B2 (4x4x3): Use schoolbook.
    
    For the 4x4x4 part, we use the following approach:
    - Apply 2x2x2 Strassen to each 2x2 block pair
    - This gives a structured decomposition
    
    Actually, let me use a cleaner approach:
    The 4x4x4 tensor can be decomposed using 4 independent 2x2x2 tensors
    (one for each 2x2 block of C), each with rank 7 (Strassen).
    But they share A and B entries, so we need to be careful.
    
    Let me use a different strategy: recursive Strassen.
    4x4x4 = 2x(2x2x2) applied at two levels.
    
    For correctness and simplicity, let me implement the following:
    Use Strassen's 2x2x2 decomposition for the 4x4x4 subproblem by
    treating it as 4 independent 2x2x2 multiplications with shared
    input entries.
    """
    # For 4x4x4: rank 49 is optimal, but let me use a structured approach.
    # I'll use the fact that 4x4x4 can be done with 49 multiplications
    # via a known algorithm, but for simplicity let me use schoolbook (64)
    # and focus on the overall structure.
    
    # Actually, let me implement a proper Strassen-based 4x4x4 decomposition.
    # The key insight: 4x4x4 tensor = sum of 4 shifted 2x2x2 tensors + cross terms.
    
    # For now, use schoolbook for 4x4x4 (rank 64) and schoolbook for 4x4x3 (rank 48).
    # Total: 112 (same as full schoolbook). This isn't an improvement.
    
    # To get an improvement, I need to use Strassen for at least one part.
    # Let me use Strassen for the 4x4x4 part.
    
    # Here's a concrete approach for 4x4x4 using Strassen:
    # A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    # B = [[B11, B12], [B21, B22]] (2x2 blocks of 2x2)
    # C = A*B, where C[i,k] = sum_j A[i,j]*B[j,k]
    
    # C11 = A11*B11 + A12*B21  (2x2 result)
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each 2x2x2 multiplication uses 7 Strassen products.
    # We have 8 such multiplications, but some can be shared.
    
    # For simplicity, let me just use 8 independent Strassen calls:
    # Total rank = 8 * 7 = 56 for the 4x4x4 part.
    # Plus 48 for 4x4x3 schoolbook.
    # Total: 104. Still better than 112.
    
    # Let me implement this:
    U_list = []
    V_list = []
    W_list = []
    
    sU, sV, sW = _strassen_2x2x2_entries()
    
    # Helper to add a Strassen block with offsets
    def add_strassen_block(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, start_col):
        """Add a 2x2x2 Strassen block with given offsets."""
        for key, val in sU.items():
            ai, aj = key
            r = a_row_off + ai
            c = a_col_off + aj
            col = start_col + key[1]
            U_list.append((r, c, col, val))
        
        for key, val in sV.items():
            bi, bj = key
            r = b_row_off + bi
            c = b_col_off + bj
            col = start_col + key[1]
            V_list.append((r, c, col, val))
        
        for key, val in sW.items():
            ci, cj = key
            r = c_row_off + ci
            c = c_col_off + cj
            col = start_col + key[1]
            W_list.append((r, c, col, val))
        
        return start_col + 7
    
    # For 4x4x4:
    # A11 is at rows 0-1, cols 0-1 (in 4x4 A, row-major: A[i,j] -> index i*4+j)
    # A12 is at rows 0-1, cols 2-3
    # A21 is at rows 2-3, cols 0-1
    # A22 is at rows 2-3, cols 2-3
    
    # B11 is at rows 0-1, cols 0-1
    # B12 is at rows 0-1, cols 2-3
    # B21 is at rows 2-3, cols 0-1
    # B22 is at rows 2-3, cols 2-3
    
    # C11 is at rows 0-1, cols 0-1 (C[i,k] -> index k*4+i)
    # C12 is at rows 0-1, cols 2-3
    # C21 is at rows 2-3, cols 0-1
    # C22 is at rows 2-3, cols 2-3
    
    # C11 = A11*B11 + A12*B21
    # This is sum of two 2x2x2 multiplications.
    # In tensor decomposition, C11[i,k] = (A11*B11)[i,k] + (A12*B21)[i,k]
    # Each term is a separate rank-7 contribution.
    
    col_offset = 0
    
    # A11*B11 -> C11
    col_offset = add_strassen_block(
        a_row_off=0, a_col_off=0,   # A11: rows 0-1, cols 0-1 in 4x4 A
        b_row_off=0, b_col_off=0,   # B11: rows 0-1, cols 0-1 in 4x4 B
        c_row_off=0, c_col_off=0,   # C11: rows 0-1, cols 0-1 in 4x4 C
        start_col=col_offset
    )
    
    # A12*B21 -> C11
    col_offset = add_strassen_block(
        a_row_off=0, a_col_off=2,   # A12: rows 0-1, cols 2-3
        b_row_off=2, b_col_off=0,   # B21: rows 2-3, cols 0-1
        c_row_off=0, c_col_off=0,   # C11
        start_col=col_offset
    )
    
    # A11*B12 -> C12
    col_offset = add_strassen_block(
        a_row_off=0, a_col_off=0,
        b_row_off=0, b_col_off=2,   # B12: rows 0-1, cols 2-3
        c_row_off=0, c_col_off=2,   # C12
        start_col=col_offset
    )
    
    # A12*B22 -> C12
    col_offset = add_strassen_block(
        a_row_off=0, a_col_off=2,
        b_row_off=2, b_col_off=2,   # B22: rows 2-3, cols 2-3
        c_row_off=0, c_col_off=2,
        start_col=col_offset
    )
    
    # A21*B11 -> C21
    col_offset = add_strassen_block(
        a_row_off=2, a_col_off=0,   # A21: rows 2-3, cols 0-1
        b_row_off=0, b_col_off=0,
        c_row_off=2, c_col_off=0,   # C21
        start_col=col_offset
    )
    
    # A22*B21 -> C21
    col_offset = add_strassen_block(
        a_row_off=2, a_col_off=2,
        b_row_off=2, b_col_off=0,
        c_row_off=2, c_col_off=0,
        start_col=col_offset
    )
    
    # A21*B12 -> C22
    col_offset = add_strassen_block(
        a_row_off=2, a_col_off=0,
        b_row_off=0, b_col_off=2,
        c_row_off=2, c_col_off=2,   # C22
        start_col=col_offset
    )
    
    # A22*B22 -> C22
    col_offset = add_strassen_block(
        a_row_off=2, a_col_off=2,
        b_row_off=2, b_col_off=2,
        c_row_off=2, c_col_off=2,
        start_col=col_offset
    )
    
    # Now add schoolbook for 4x4x3 (the last 3 columns of B)
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):  # k = 4, 5, 6
                U_list.append((i*4 + j, col_offset, 1))
                V_list.append((j*7 + k, col_offset, 1))
                W_list.append((k*4 + i, col_offset, 1))
                col_offset += 1
    
    # Convert to matrices
    total_rank = col_offset
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for r, c, col, val in U_list:
        U[r, col] += val
    for r, c, col, val in V_list:
        V[r, col] += val
    for r, c, col, val in W_list:
        W[r, col] += val
    
    return U, V, W, total_rank


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=1e-10)


def _to_fraction_matrices(U, V, W):
    """Convert numpy float matrices to Fraction matrices."""
    U_frac = [[Fraction(int(U[i,j]), 1) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_frac = [[Fraction(int(V[i,j]), 1) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_frac = [[Fraction(int(W[i,j]), 1) for j in range(W.shape[1])] for i in range(W.shape[0])]
    return U_frac, V_frac, W_frac


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Try Strassen-based decomposition for 4x4x7
    U, V, W, rank = _strassen_based_4x4x7(n, m, p)
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        # Convert to fractions for exactness
        U_frac, V_frac, W_frac = _to_fraction_matrices(U, V, W)
        return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Fallback to schoolbook if verification fails
    U, V, W, rank = _schoolbook_4x4x7(n, m, p)
    U_frac, V_frac, W_frac = _to_fraction_matrices(U, V, W)
    return {"U": U_frac, "V": V_frac, "W": W_frac}

# EVOLVE-BLOCK-END
