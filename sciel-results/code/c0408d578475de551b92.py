"""Rewritten program: 2x2 block Strassen with 3x3 schoolbook sub-blocks -> rank 189."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _build_3x3_schoolbook():
    """Return U3, V3, W3 (lists of lists) for 3x3 matrix multiplication, rank 27.
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=3."""
    n = m = p = 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _build_6x6_strassen():
    """Build rank-189 decomposition for 6x6 matrix multiplication using
    2x2 block Strassen with 3x3 schoolbook sub-block multiplications.
    
    Block layout (2x2 blocks of size 3x3):
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    C = [[C11, C12], [C21, C22]]
    
    Strassen formulas:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22) B11
    M3 = A11 (B12 - B22)
    M4 = A22 (B21 - B11)
    M5 = (A11 + A12) B22
    M6 = (A21 - A11) (B11 + B12)
    M7 = (A12 - A22) (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n = m = p = 6
    block = 3  # each block is 3x3
    rank = 7 * block * block * block  # 7 Strassen multiplies * 27 schoolbook = 189
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x rank
    V = [[0] * rank for _ in range(m * p)]  # 36 x rank
    W = [[0] * rank for _ in range(p * n)]  # 36 x rank
    
    # 3x3 schoolbook sub-decomposition
    U3, V3, W3 = _build_3x3_schoolbook()
    
    # For each Strassen multiplier t (0..6), we need to express:
    #   (linear combo of A blocks) * (linear combo of B blocks)
    # using the 3x3 schoolbook on each pair.
    
    # Strassen multiplier definitions:
    # Each multiplier is (A_coeff, B_coeff) where A_coeff is a 2x2 matrix of coeffs
    # applied to A blocks, B_coeff similarly for B blocks.
    
    # M1: (A11 + A22) * (B11 + B22)
    # M2: (A21 + A22) * B11
    # M3: A11 * (B12 - B22)
    # M4: A22 * (B21 - B11)
    # M5: (A11 + A12) * B22
    # M6: (A21 - A11) * (B11 + B12)
    # M7: (A12 - A22) * (B21 + B22)
    
    # A block positions: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B block positions: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    
    # For each multiplier, define which A blocks and B blocks are used with what signs
    strassen_defs = [
        # (list of (a_row, a_col, sign), list of (b_row, b_col, sign))
        ([((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))]),   # M1
        ([((1,0,1),(1,1,1))], [((0,0,1))]),              # M2
        ([((0,0,1))], [((0,1,1),(1,1,-1))]),             # M3
        ([((1,1,1))], [((1,0,1),(0,0,-1))]),             # M4
        ([((0,0,1),(0,1,1))], [((1,1,1))]),              # M5
        ([((1,0,1),(0,0,-1))], [((0,0,1),(0,1,1))]),     # M6
        ([((0,1,1),(1,1,-1))], [((1,0,1),(1,1,1))]),     # M7
    ]
    
    # C block contributions:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    c_contribs = {
        (0,0): [(0,1),(3,1),(4,-1),(6,1)],    # C11
        (0,1): [(2,1),(4,1)],                  # C12
        (1,0): [(1,1),(3,1)],                  # C21
        (1,1): [(0,1),(1,-1),(2,1),(5,1)],     # C22
    }
    
    col = 0
    for t in range(7):
        a_blocks, b_blocks = strassen_defs[t]
        # For this Strassen multiplier, do a 3x3 schoolbook multiplication
        for i3 in range(block):
            for j3 in range(block):
                for k3 in range(block):
                    # This corresponds to one rank-1 term in the 3x3 schoolbook
                    # U3[i3*3+j3][c3] = 1, V3[j3*3+k3][c3] = 1, W3[k3*3+i3][c3] = 1
                    # where c3 = i3*9 + j3*3 + k3
                    c3 = i3 * 9 + j3 * 3 + k3
                    u_val = U3[i3 * 3 + j3][c3]
                    v_val = V3[j3 * 3 + k3][c3]
                    w_val = W3[k3 * 3 + i3][c3]
                    
                    # U: for each A block (ar, ac) with sign sa
                    for (ar, ac, sa) in a_blocks:
                        # A block (ar, ac) occupies rows ar*3..ar*3+2, cols ac*3..ac*3+2
                        # within that block, row i3, col j3
                        row_idx = ar * block + i3
                        col_idx = ac * block + j3
                        U[row_idx][col] += sa * u_val
                    
                    # V: for each B block (br, bc) with sign sb
                    for (br, bc, sb) in b_blocks:
                        row_idx = br * block + j3
                        col_idx = bc * block + k3
                        V[row_idx][col] += sb * v_val
                    
                    # W: for each C block (cr, cc) that receives contribution from multiplier t
                    for (cr, cc), sign in c_contribs.get((cr, cc), []):
                        pass  # handled below
                    
                    # Actually, W needs to accumulate from all c_contribs
                    for (cr, cc) in c_contribs:
                        for (mt, sgn) in c_contribs[(cr, cc)]:
                            if mt == t:
                                row_idx = cr * block + k3
                                col_idx = cc * block + i3
                                W[row_idx][col] += sgn * w_val
                    
                    col += 1
    
    return U, V, W


def construct():
    deadline = time.time() + 145.0  # safety margin before 150s limit
    
    start = time.time()
    
    # Build the Strassen-based rank-189 decomposition
    U, V, W = _build_6x6_strassen()
    
    elapsed = time.time() - start
    if elapsed > deadline:
        # Should not happen, but return anyway
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
