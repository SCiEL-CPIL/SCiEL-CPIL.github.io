import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 7]
n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
T_START = time.time()
DEADLINE = 150.0
SAFETY = 8.0

def _schoolbook():
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_blockwise():
    """
    Decompose 2x4 * 4x7 by splitting into 2x2 blocks.
    
    A (2x4) = [A00 A01] where A00, A01 are 2x2
    B (4x7) = [B00 B01; B10 B11] where each is 2x2
    
    C = A * B = [A00*B00 + A01*B10,  A00*B01 + A01*B11]
    
    Use Strassen for each 2x2*2x2 product (7 mults each instead of 8).
    Total: 2 blocks * 7 = 14 mults per output column group... wait, let me think more carefully.
    
    Actually, C is 2x7. Split C into two 2x2 blocks and one 2x3 block? No, 7 is odd.
    
    Let me split differently:
    A is 2x4: split into A = [A1 | A2] where A1, A2 are 2x2
    B is 4x7: split into B = [B1; B2] where B1 is 2x7, B2 is 2x7
    
    C = A1*B1 + A2*B2 (2x7 result)
    
    Now A1*B1 is 2x2 * 2x7. Split B1 = [B1a | B1b] where B1a is 2x3, B1b is 2x4.
    This gets complicated. Let me use a cleaner approach.
    
    Better: Think of it as computing 2x7 = (2x4)*(4x7).
    Split the 4 middle dimension into two 2s:
    A = [A1 | A2], A1,A2 ∈ R^{2x2}
    B = [B1; B2], B1,B2 ∈ R^{2x7}
    
    C = A1*B1 + A2*B2
    
    For A1*B1 (2x2 * 2x7): split B1 = [B1a | B1b] with B1a 2x3, B1b 2x4.
    Hmm, uneven splits. Let me just use Strassen on the 2x2*2x2 subblocks.
    
    Actually, the cleanest way: split all three dimensions where possible.
    n=2, m=4, p=7.
    
    Split m=4 into m1=2, m2=2.
    A = [A1 | A2], each 2x2
    B = [B1; B2], each 2x7
    
    C = A1*B1 + A2*B2  (2x7)
    
    Now A1*B1: 2x2 * 2x7. Split p=7 into p1=3, p2=4? Or use Strassen-like.
    For 2x2 * 2x7, we can write B1 = [B1a | B1b] where B1a is 2x3, B1b is 2x4.
    A1*B1 = [A1*B1a | A1*B1b]
    A1*B1a is 2x2*2x3: this is like 2x2*2x2 + 2x2*2x1. 
    A1*B1b is 2x2*2x4: use Strassen on 2x2*2x2 blocks? 4/2=2 blocks of 2x2.
    
    This is getting complex. Let me just enumerate a known good decomposition.
    
    Alternative: Use the fact that we can compute each 2x2 * 2x2 block with Strassen (7 mults).
    
    Let's split:
    A (2x4) = [A00 A01]  (two 2x2 blocks)
    B (4x7) = [B00 B01 B02; B10 B11 B12] where each Bij is 2x2 or 2x1
    
    Hmm, 7 = 2+2+2+1. Let's do 7 = 3+4 or just handle it as 2x7.
    
    Simplest correct approach: Use Strassen for the 2x2 * 2x2 subproblems.
    
    Split m=4 as 2+2, p=7 as 3+4.
    A = [A1 | A2], A1,A2 ∈ R^{2x2}
    B = [B1; B2], B1 ∈ R^{2x3}, B2 ∈ R^{2x4}
    
    C = A1*B1 + A2*B2 ∈ R^{2x7}
    
    For A1*B1 (2x2 * 2x3): 
      B1 = [B1a | B1b], B1a 2x2, B1b 2x1
      A1*B1a: use Strassen (7 mults) → 2x2
      A1*B1b: 2x2 * 2x1 = 2x1, needs 4 mults (or 2 if we're clever, but let's be safe)
      Actually 2x2 * 2x1: result[i] = sum_j A[i,j]*B[j,0]. That's 4 scalar mults.
      Total for A1*B1: 7 + 4 = 11 mults
      
    For A2*B2 (2x2 * 2x4):
      B2 = [B2a | B2b], B2a 2x2, B2b 2x2
      A2*B2a: Strassen, 7 mults
      A2*B2b: Strassen, 7 mults
      Total: 14 mults
      
    Grand total: 11 + 14 = 25 mults? That seems too good. Let me verify.
    
    Wait, I need to be more careful. The "mults" here are scalar multiplications in the tensor decomposition sense.
    
    Let me just directly construct the Strassen-based decomposition.
    """
    
    # Split: A (2x4) = [A1 | A2], each 2x2
    #         B (4x7) = [B1; B2], B1 is 2x7, B2 is 2x7
    # C = A1*B1 + A2*B2
    #
    # For each 2x2 * 2x7 product, split the 7 into 3+4:
    #   B1 = [B1a | B1b], B1a 2x3, B1b 2x4
    #   A1*B1a: 2x2 * 2x3
    #   A1*B1b: 2x2 * 2x4
    #
    # 2x2 * 2x3: split 3 into 2+1
    #   B1a = [B1a1 | B1a2], B1a1 2x2, B1a2 2x1
    #   A1*B1a1: Strassen 2x2*2x2 = 7 mults
    #   A1*B1a2: 2x2*2x1 = 4 mults (each output element needs 2 mults, 2 outputs)
    #   Total: 11
    #
    # 2x2 * 2x4: split 4 into 2+2
    #   B1b = [B1b1 | B1b2], each 2x2
    #   A1*B1b1: Strassen = 7 mults
    #   A1*B1b2: Strassen = 7 mults
    #   Total: 14
    #
    # So A1*B1 = 11 + 14 = 25 mults
    # Similarly A2*B2 = 25 mults
    # Total: 50 mults
    #
    # This gives rank 50!
    
    rank = 50
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    col = 0
    
    # Strassen 2x2 * 2x2 decomposition
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def strassen_2x2_2x2(u_base, v_base, w_base, col_start, 
                         a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Add Strassen decomposition for 2x2 * 2x2.
        a_rows: indices of the 2 rows in A (for U)
        a_cols: indices of the 2 cols in A (for U) 
        b_rows: indices of the 2 rows in B (for V)
        b_cols: indices of the 2 cols in B (for V)
        c_rows: indices of the 2 rows in C (for W)
        c_cols: indices of the 2 cols in C (for W)
        Returns new col.
        """
        c = col_start
        # M1 = (A11+A22)*(B11+B22)
        for (ar, ac) in [(a_rows[0], a_cols[0]), (a_rows[1], a_cols[1])]:
            U[ar * m + ac][c] = 1
        for (br, bc) in [(b_rows[0], b_cols[0]), (b_rows[1], b_cols[1])]:
            V[br * p + bc][c] = 1
        for (cr, cc) in [(c_rows[0], c_cols[0]), (c_rows[1], c_cols[1])]:
            W[cr * n + cc][c] = 1  # will be adjusted below
        # We need to track contributions carefully. Let me use a different approach.
        # Instead of directly filling W, let me accumulate contributions.
        
        # Actually, let me just directly compute the full decomposition by tracking
        # each intermediate product and its contribution to C.
        return c + 1
    
    # This direct approach is getting tangled. Let me use a cleaner method:
    # Build the decomposition by explicitly listing each scalar multiplication
    # and its contribution to U, V, W.
    
    # Reset
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    
    # Helper: add one scalar mult to the decomposition
    def add_mult(u_idx, v_idx, w_contribs):
        """
        u_idx: (i, j) index into A (row i, col j) -> U index i*m+j
        v_idx: (j2, k) index into B (row j2, col k) -> V index j2*p+k
        w_contribs: list of ((i2, k2), coeff) for contributions to C
        """
        nonlocal col
        U[u_idx[0] * m + u_idx[1]][col] = 1
        V[v_idx[0] * p + v_idx[1]][col] = 1
        for (ci, ck), coeff in w_contribs:
            W[ci * n + ck][col] += coeff
        col += 1
    
    # Helper: add Strassen for 2x2 * 2x2
    # A block at rows ar0, ar0+1 and cols ac0, ac0+1 in A
    # B block at rows br0, br0+1 and cols bc0, bc0+1 in B
    # C block at rows cr0, cr0+1 and cols cc0, cc0+1 in C
    def add_strassen(ar0, ac0, br0, bc0, cr0, cc0):
        """
        A submatrix: A[ar0:ar0+2, ac0:ac0+2]
        B submatrix: B[br0:br0+2, bc0:bc0+2]
        C submatrix: C[cr0:cr0+2, cc0:cc0+2]
        
        Strassen:
        M1 = (A11+A22)*(B11+B22)
        M2 = (A21+A22)*B11
        M3 = A11*(B12-B22)
        M4 = A22*(B21-B11)
        M5 = (A11+A12)*B22
        M6 = (A21-A11)*(B11+B12)
        M7 = (A12-A22)*(B21+B22)
        
        C11 = M1+M4-M5+M7
        C12 = M3+M5
        C21 = M2+M4
        C22 = M1-M2+M3+M6
        
        A11 = A[ar0,ac0], A12 = A[ar0,ac0+1]
        A21 = A[ar0+1,ac0], A22 = A[ar0+1,ac0+1]
        B11 = B[br0,bc0], B12 = B[br0,bc0+1]
        B21 = B[br0+1,bc0], B22 = B[br0+1,bc0+1]
        C11 = C[cr0,cc0], C12 = C[cr0,cc0+1]
        C21 = C[cr0+1,cc0], C22 = C[cr0+1,cc0+1]
        """
        nonlocal col
        
        # M1 = (A11+A22)*(B11+B22)
        # U factors: A11 and A22
        # V factors: B11 and B22
        # This is actually TWO scalar mults: A11*B11 + A22*B22
        # Wait no, Strassen combines them. The point is we compute (A11+A22) and (B11+B22)
        # as intermediate values, then multiply. But in tensor decomposition, each scalar
        # multiplication is a rank-1 term. So M1 = (A11+A22)*(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # That's 4 scalar mults, not 1! 
        #
        # I was confused. Strassen reduces the NUMBER of scalar multiplications from 8 to 7
        # for 2x2*2x2, but each "multiplication" in Strassen is still a product of two linear
        # combinations, which in tensor decomposition terms is still multiple rank-1 terms.
        #
        # Wait, no. In Strassen, M1 = (A11+A22)*(B11+B22). This is ONE scalar multiplication
        # where the two operands are linear combinations. In the tensor decomposition, this
        # corresponds to:
        #   U has 1 in positions for A11 and A22
        #   V has 1 in positions for B11 and B22
        #   W has 1 in positions for C11, C12, C21, C22 with appropriate signs
        #
        # So M1 contributes to:
        #   C11 += M1 (coeff +1)
        #   C12 += M1 (coeff +1) -- wait no
        #
        # Let me re-derive. C11 = M1+M4-M5+M7, so M1 contributes +1 to C11.
        # C12 = M3+M5, so M1 contributes 0 to C12.
        # C21 = M2+M4, so M1 contributes 0 to C21.
        # C22 = M1-M2+M3+M6, so M1 contributes +1 to C22.
        
        # M1: (A11+A22)*(B11+B22)
        U[ar0 * m + ac0][col] = 1       # A11
        U[(ar0+1) * m + (ac0+1)][col] = 1  # A22
        V[br0 * p + bc0][col] = 1       # B11
        V[(br0+1) * p + (bc0+1)][col] = 1  # B22
        W[cr0 * n + cc0][col] = 1       # C11 += M1
        W[(cr0+1) * n + (cc0+1)][col] = 1  # C22 += M1
        col += 1
        
        # M2 = (A21+A22)*B11
        U[(ar0+1) * m + ac0][col] = 1   # A21
        U[(ar0+1) * m + (ac0+1)][col] = 1  # A22
        V[br0 * p + bc0][col] = 1       # B11
        W[(cr0+1) * n + cc0][col] = 1   # C21 += M2
        col += 1
        
        # M3 = A11*(B12-B22)
        U[ar0 * m + ac0][col] = 1       # A11
        V[br0 * p + (bc0+1)][col] = 1   # B12
        V[(br0+1) * p + (bc0+1)][col] = -1  # -B22
        W[cr0 * n + (cc0+1)][col] = 1   # C12 += M3
        W[(cr0+1) * n + (cc0+1)][col] = 1  # C22 += M3
        col += 1
        
        # M4 = A22*(B21-B11)
        U[(ar0+1) * m + (ac0+1)][col] = 1  # A22
        V[(br0+1) * p + bc0][col] = 1     # B21
        V[br0 * p + bc0][col] = -1        # -B11
        W[cr0 * n + cc0][col] = 1         # C11 += M4
        W[(cr0+1) * n + cc0][col] = 1     # C21 += M4
        col += 1
        
        # M5 = (A11+A12)*B22
        U[ar0 * m + ac0][col] = 1       # A11
        U[ar0 * m + (ac0+1)][col] = 1   # A12
        V[(br0+1) * p + (bc0+1)][col] = 1  # B22
        W[cr0 * n + cc0][col] = -1      # C11 -= M5
        W[cr0 * n + (cc0+1)][col] = 1   # C12 += M5
        col += 1
        
        # M6 = (A21-A11)*(B11+B12)
        U[(ar0+1) * m + ac0][col] = 1   # A21
        U[ar0 * m + ac0][col] = -1      # -A11
        V[br0 * p + bc0][col] = 1       # B11
        V[br0 * p + (bc0+1)][col] = 1   # B12
        W[(cr0+1) * n + (cc0+1)][col] = 1  # C22 += M6
        col += 1
        
        # M7 = (A12-A22)*(B21+B22)
        U[ar0 * m + (ac0+1)][col] = 1   # A12
        U[(ar0+1) * m + (ac0+1)][col] = -1  # -A22
        V[(br0+1) * p + bc0][col] = 1   # B21
        V[(br0+1) * p + (bc0+1)][col] = 1  # B22
        W[cr0 * n + cc0][col] = 1       # C11 += M7
        col += 1
    
    # Now handle the full 2x4 * 4x7 problem.
    # A (2x4): rows 0,1; cols 0,1,2,3
    # B (4x7): rows 0,1,2,3; cols 0,1,2,3,4,5,6
    # C (2x7): rows 0,1; cols 0,1,2,3,4,5,6
    #
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #
    # Let A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
    #     B1 = B[0:2, :] (2x7), B2 = B[2:4, :] (2x7)
    # C = A1*B1 + A2*B2
    #
    # A1*B1: 2x2 * 2x7. Split B1 into B1a = B1[:, 0:2] (2x2), B1b = B1[:, 2:4] (2x2), B1c = B1[:, 4:7] (2x3)
    # A1*B1a: Strassen 2x2*2x2 -> 7 mults
    # A1*B1b: Strassen 2x2*2x2 -> 7 mults
    # A1*B1c: 2x2*2x3. Split B1c = B1c_a (2x2) + B1c_b (2x1)
    #   A1*B1c_a: Strassen -> 7 mults
    #   A1*B1c_b: 2x2*2x1 -> 4 mults (each of 2 output elements needs 2 mults)
    #   Total A1*B1c: 11 mults
    # Total A1*B1: 7+7+11 = 25 mults
    #
    # Similarly A2*B2: 25 mults
    # Grand total: 50 mults
    
    # A1*B1a: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    add_strassen(0, 0, 0, 0, 0, 0)  # ar0=0, ac0=0, br0=0, bc0=0, cr0=0, cc0=0
    
    # A1*B1b: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    add_strassen(0, 0, 0, 2, 0, 2)  # ar0=0, ac0=0, br0=0, bc0=2, cr0=0, cc0=2
    
    # A1*B1c_a: A[:, 0:2] * B[0:2, 4:6] -> C[:, 4:6]
    add_strassen(0, 0, 0, 4, 0, 4)  # ar0=0, ac0=0, br0=0, bc0=4, cr0=0, cc0=4
    
    # A1*B1c_b: A[:, 0:2] * B[0:2, 6:7] -> C[:, 6:7]
    # 2x2 * 2x1: C[i,6] = A[i,0]*B[0,6] + A[i,1]*B[1,6]
    # 4 scalar mults:
    # C[0,6] += A[0,0]*B[0,6]
    add_mult((0, 0), (0, 6), [((0, 6), 1)])
    # C[0,6] += A[0,1]*B[1,6]
    add_mult((0, 1), (1, 6), [((0, 6), 1)])
    # C[1,6] += A[1,0]*B[0,6]
    add_mult((1, 0), (0, 6), [((1, 6), 1)])
    # C[1,6] += A[1,1]*B[1,6]
    add_mult((1, 1), (1, 6), [((1, 6), 1)])
    
    # A2*B2a: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    add_strassen(0, 2, 2, 0, 0, 0)  # ar0=0, ac0=2, br0=2, bc0=0, cr0=0, cc0=0
    
    # A2*B2b: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    add_strassen(0, 2, 2, 2, 0, 2)  # ar0=0, ac0=2, br0=2, bc0=2, cr0=0, cc0=2
    
    # A2*B2c_a: A[:, 2:4] * B[2:4, 4:6] -> C[:, 4:6]
    add_strassen(0, 2, 2, 4, 0, 4)  # ar0=0, ac0=2, br0=2, bc0=4, cr0=0, cc0=4
    
    # A2*B2c_b: A[:, 2:4] * B[2:4, 6:7] -> C[:, 6:7]
    # 2x2 * 2x1: C[i,6] += A[i,2]*B[2,6] + A[i,3]*B[3,6]
    add_mult((0, 2), (2, 6), [((0, 6), 1)])
    add_mult((0, 3), (3, 6), [((0, 6), 1)])
    add_mult((1, 2), (2, 6), [((1, 6), 1)])
    add_mult((1, 3), (3, 6), [((1, 6), 1)])
    
    # Total cols used:
    # Strassen blocks: 6 * 7 = 42
    # 2x2*2x1 blocks: 2 * 4 = 8
    # Total: 50
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = Ua[i * m + j] @ Va[j * p + k]
                if lhs != Wa[k * n + i]:
                    # Fall back to schoolbook
                    return _schoolbook()
    
    return {"U": U, "V": V, "W": W}

def _rows_of(mat):
    return [tuple(r) for r in mat]

def _validate(U, V, W):
    r = len(U[0])
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = Ua[i * m + j] @ Va[j * p + k]
                if lhs != Wa[k * n + i]:
                    return False
    return True

def construct():
    # Try Strassen blockwise first
    result = _strassen_blockwise()
    
    # Verify the result
    if _validate(result["U"], result["V"], result["W"]):
        return result
    
    # Fall back to schoolbook
    return _schoolbook()
