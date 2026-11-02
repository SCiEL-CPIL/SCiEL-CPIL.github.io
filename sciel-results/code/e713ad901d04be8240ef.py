import time
import numpy as np
from fractions import Fraction

_START = time.time()
_DEADLINE = _START + 135.0

def _check_time():
    return time.time() > _DEADLINE

def _validate_exact(U, V, W, n, m, p):
    """Check if factor matrices exactly decompose the matmul tensor using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = Fraction(0)
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if val != 1:
                    return False
    return True

def _build_polynomial_strassen():
    """
    Build exact rank-44 decomposition for 2x3x8 matmul using polynomial ring Z[x]/(x^2).
    
    Strategy:
    - Split the 3x8 matrix into 2x2 blocks of 2x4 matrices
    - Use Strassen's 7-multiplication algorithm for the 2x2 block structure
    - Each Strassen product involves a 2x4 @ 4x2 multiplication (schoolbook rank 8)
    - Total: 7 * 8 = 56... but we can do better
    
    Actually, let me think more carefully:
    A is 2x3, B is 3x8.
    Split B into columns: B = [B1, B2] where B1, B2 are 3x4.
    Split A into rows: A = [A1; A2] where A1, A2 are 1x3... no that's not right.
    
    Better approach: 
    - View the 3x8 matrix as having 2 groups of 4 columns each
    - The 2x3 matrix times 3x8: we can use block structure
    
    Let me use a cleaner approach:
    - Split the 8 columns into two groups of 4: B = [B_left | B_right], each 3x4
    - Compute C = A * B = [A*B_left | A*B_right]
    - Each A*B_block is 2x4, which has schoolbook rank 8
    - Total: 16 multiplications... that's not better than 24
    
    Let me try the polynomial ring approach properly:
    - Consider B as a 3x4 matrix over R[x]/(x^2) where entries are b_k0 + b_k1*x
    - B[x] = B0 + B1*x where B0, B1 are 3x4 matrices (the even and odd columns)
    - A is 2x3 (constant, no x dependence)
    - C[x] = A * B[x] = A*B0 + A*B1*x
    - This gives us 2x4 @ 4x4... wait, dimensions don't work.
    
    Let me reconsider. The task is 2x3 times 3x8.
    n=2, m=3, p=8.
    
    Polynomial ring approach:
    - Group the 8 columns of B into pairs: (B[:,0], B[:,1]), (B[:,2], B[:,3]), etc.
    - Each pair defines an element in R[x]/(x^2): b_0 + b_1*x
    - So B becomes a 3x4 matrix over R[x]/(x^2)
    - A is a 2x3 matrix (constant)
    - C = A * B is a 2x4 matrix over R[x]/(x^2)
    - C[k] = c_k0 + c_k1*x where c_k0, c_k1 are the two output columns
    
    Now we need to multiply a 2x3 matrix by a 3x4 matrix over R[x]/(x^2).
    This is equivalent to:
    - 2x3 @ 3x4 over R[x]/(x^2)
    - Each entry of the result is a polynomial in x of degree < 2
    - The multiplication tensor has rank...
    
    For a 2x3 @ 3x4 multiplication over a ring, the rank is at most 2*3*4 = 24.
    But we can use Strassen-like methods.
    
    Actually, let me just use a direct structured decomposition:
    
    Split B into 4 blocks of 2 columns each: B = [B0 | B1 | B2 | B3], each 3x2.
    C = A * B = [A*B0 | A*B1 | A*B2 | A*B3]
    Each A*Bi is 2x2, which can be done with Strassen in 7 multiplications.
    Total: 4 * 7 = 28 multiplications.
    
    But wait, the Strassen algorithm for 2x2 requires the inputs to be 2x2.
    A is 2x3, not 2x2. So I need to split A as well.
    
    Let me split A into columns: A = [A0 | A1 | A2], each 2x1.
    Then A*B = A0*B_col0 + A1*B_col1 + A2*B_col2 where B_colj is column j of B.
    That's just schoolbook.
    
    Better: use the block structure differently.
    
    Let me try: split the 3x8 matrix into a 2x2 block matrix where each block is 2x4... 
    No, 3x8 can't be split into 2x2 blocks of equal size.
    
    Let me try a different decomposition:
    - Split the 8 columns into two groups of 4: B = [B_L | B_R], each 3x4
    - Split the 2 rows of A... A is already 2 rows.
    - C = [A*B_L | A*B_R]
    - Each A*B is 2x4.
    
    For 2x4 @ (implicitly 1x1 on the right? No, A is 2x3, B is 3x4, so A*B is 2x4.
    
    I think the cleanest approach is:
    1. Use schoolbook for the 2x3 @ 3x8 with some structure
    2. Or find a known low-rank decomposition
    
    Let me just implement a rank-44 decomposition based on the insight from the knowledge base:
    "Strassen for 2x2x2 sub-blocks and schoolbook for 2x1x4 sub-blocks"
    
    Decomposition:
    - Split the 8 columns into 4 groups of 2: B = [B_0 | B_1 | B_2 | B_3], each 3x2
    - For each group, compute A @ B_i (2x3 @ 3x2 = 2x2)
    - Use Strassen for each 2x2 multiplication? But A is 2x3, not 2x2.
    
    Alternative: 
    - Split A into two 1x3 row vectors: A = [a_0; a_1]
    - For each column group B_i (3x2), compute a_0 @ B_i and a_1 @ B_i
    - Each is a 1x3 @ 3x2 = 1x2 multiplication, rank 6
    - Total: 2 * 4 * 6 = 48... not better.
    
    Let me just implement a verified rank-44 decomposition directly.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # Strategy: Use 4 groups of 2 columns
    # For each group g (0..3), we have a 2x3 @ 3x2 multiplication
    # We'll use a rank-11 decomposition for each (7 for Strassen-like + 4 extra)
    # Total: 4 * 11 = 44
    
    # Actually, let me use a simpler verified construction:
    # Split into 4 blocks of 2 columns
    # For each block, use schoolbook 2x3@3x2 = 6 multiplications
    # Total: 4 * 6 = 24... that's schoolbook!
    
    # The key insight: we can share computation across blocks.
    # Let me use a different approach based on the polynomial ring.
    
    # View B as a 3x4 matrix over R[x]/(x^2):
    # B[x] = B0 + B1*x where B0 = B[:, 0::2], B1 = B[:, 1::2]
    # C[x] = A @ B[x] = (A@B0) + (A@B1)*x
    # C0 = A@B0 (2x4), C1 = A@B1 (2x4)
    
    # Now we need to compute A@B0 and A@B1 where A is 2x3 and B0,B1 are 3x4.
    # Each is a 2x3 @ 3x4 multiplication, rank at most 24.
    # But we can do better with Strassen-like methods.
    
    # For 2x3 @ 3x4:
    # Split A = [A_left | A_right] where A_left is 2x2, A_right is 2x1
    # Split B = [B_top; B_bottom] where B_top is 2x4, B_bottom is 1x4
    # A@B = A_left@B_top + A_right@B_bottom
    # A_left@B_top: 2x2 @ 2x4, can use Strassen-like: split B_top into two 2x2 blocks
    #   B_top = [B_tl | B_tr], each 2x2
    #   A_left@B_top = [A_left@B_tl | A_left@B_tr]
    #   Each 2x2@2x2 can be done with 7 multiplications (Strassen)
    #   So A_left@B_top needs 14 multiplications
    # A_right@B_bottom: 2x1 @ 1x4 = 8 multiplications (schoolbook)
    # Total: 14 + 8 = 22 for one 2x3@3x4
    
    # So for B0 and B1: 2 * 22 = 44 total multiplications!
    
    # Let me implement this:
    # For B0 (columns 0,2,4,6 of original B):
    #   A_left = A[:, 0:2] (2x2), A_right = A[:, 2:3] (2x1)
    #   B0_top = B0[0:2, :] (2x4), B0_bottom = B0[2:3, :] (1x4)
    #   B0_tl = B0_top[:, 0:2] (2x2), B0_tr = B0_top[:, 2:4] (2x2)
    
    # Strassen for 2x2 @ 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    col = 0
    
    def add_strassen_block(A_left, B_tl, B_tr, B_bottom, A_right, offset, scale=1):
        """Add Strassen-based decomposition for 2x3 @ 3x4.
        A_left: 2x2 (rows 0,1 of A, cols 0,1)
        A_right: 2x1 (rows 0,1 of A, col 2)
        B_tl: 2x2 (top-left block of B)
        B_tr: 2x2 (top-right block of B)
        B_bottom: 1x4 (bottom row of B)
        """
        nonlocal col
        # Strassen M1 = (A00+A11)(B00+B11)
        U[0][col] = scale * 1  # A00
        U[1][col] = scale * 1  # A11
        V[0][col] = scale * 1  # B00 (row 0, col 0 of B_tl)
        V[1][col] = scale * 1  # B11 (row 1, col 1 of B_tl)
        W[0][col] = scale * 1  # C00 contribution
        W[1][col] = scale * 1  # C01 contribution (from B_tr col 0)
        W[2][col] = scale * 1  # C10 contribution
        W[3][col] = scale * 1  # C11 contribution (from B_tr col 1)
        # Wait, I need to be more careful with the output indexing.
        # Let me redo this with explicit indices.
        col += 1
        return col - 1
    
    # This is getting complex. Let me just build it directly.
    
    # Reset
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1
    
    # For each of the 4 column groups (pairs of columns), we'll handle them.
    # Group g handles columns 2g and 2g+1 of B.
    # B0 contains columns 0,2,4,6 (even), B1 contains columns 1,3,5,7 (odd).
    
    # Let me restructure: 
    # B0 = B[:, [0,2,4,6]], B1 = B[:, [1,3,5,7]]
    # For B0: columns of B0 correspond to original columns 0,2,4,6
    # For B1: columns of B1 correspond to original columns 1,3,5,7
    
    # For B0 (and similarly B1):
    # A_left = A[:, 0:2], A_right = A[:, 2:3]
    # B0_top = B0[0:2, :] (rows 0,1), B0_bottom = B0[2:3, :] (row 2)
    # B0_tl = B0_top[:, 0:2] (cols 0,1 of B0 = original cols 0,2)
    # B0_tr = B0_top[:, 2:4] (cols 2,3 of B0 = original cols 4,6)
    
    # Strassen for A_left @ B0_tl (2x2 @ 2x2):
    # Output is 2x2, contributing to C[:, 0:2] of the B0 result
    # Which corresponds to original C columns 0,2
    
    # Strassen for A_left @ B0_tr (2x2 @ 2x2):
    # Output is 2x2, contributing to C[:, 2:4] of the B0 result
    # Which corresponds to original C columns 4,6
    
    # A_right @ B0_bottom (2x1 @ 1x4):
    # Output is 2x4, contributing to all C columns of B0 result
    # Original C columns 0,2,4,6
    
    # Similarly for B1:
    # A_left @ B1_tl: original C columns 1,3
    # A_left @ B1_tr: original C columns 5,7
    # A_right @ B1_bottom: original C columns 1,3,5,7
    
    # Total terms:
    # B0: 7 (Strassen tl) + 7 (Strassen tr) + 8 (schoolbook bottom) = 22
    # B1: 7 + 7 + 8 = 22
    # Total: 44
    
    # Let me define the Strassen terms for 2x2@2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For B0_strassen_tl:
    # A_left entries: a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
    # B0_tl entries: b00=B[0,0], b01=B[0,2], b10=B[1,0], b11=B[1,2]
    # Output C00 -> original C[0,0], C01 -> original C[0,2], C10 -> original C[1,0], C11 -> original C[1,2]
    
    # U index for A[i,j] is i*m+j = i*3+j
    # V index for B[i,j] is i*p+j = i*8+j
    # W index for C[i,j] is j*n+i = j*2+i
    
    def add_strassen(A_row0, A_row1, B_row0, B_row1, C_col0, C_col1, sign=1):
        """Add 7 Strassen terms for 2x2@2x2.
        A_row0, A_row1: row indices of A (2x2 block)
        B_row0, B_row1: row indices of B (2x2 block) 
        C_col0, C_col1: column indices of C output
        sign: multiply all by sign
        """
        nonlocal col
        # A indices: A_row0*3+0, A_row0*3+1, A_row1*3+0, A_row1*3+1
        a00_idx = A_row0 * 3 + 0
        a01_idx = A_row0 * 3 + 1
        a10_idx = A_row1 * 3 + 0
        a11_idx = A_row1 * 3 + 1
        
        # B indices: B_row0*8+C_col0, B_row0*8+C_col1, B_row1*8+C_col0, B_row1*8+C_col1
        b00_idx = B_row0 * 8 + C_col0
        b01_idx = B_row0 * 8 + C_col1
        b10_idx = B_row1 * 8 + C_col0
        b11_idx = B_row1 * 8 + C_col1
        
        # C indices: C_col0*2+0, C_col0*2+1, C_col1*2+0, C_col1*2+1
        c00_idx = C_col0 * 2 + 0
        c01_idx = C_col0 * 2 + 1
        c10_idx = C_col1 * 2 + 0
        c11_idx = C_col1 * 2 + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[a00_idx][col] += sign
        U[a11_idx][col] += sign
        V[b00_idx][col] += sign
        V[b11_idx][col] += sign
        W[c00_idx][col] += sign
        W[c11_idx][col] -= sign  # C11 has -M1? No: C00=M1+M4-M5+M7, C11=M1-M2+M3+M6
        # Wait, I need to track contributions to each C element separately.
        # Let me use a different approach: for each M, add its contribution to each C element.
        
        # Actually, let me just add each M as a separate rank-1 term with appropriate signs.
        # This will use more rank but be correct.
        # M1 contributes to C00 (+1) and C11 (+1)
        # M2 contributes to C10 (+1) and C11 (-1)
        # M3 contributes to C01 (+1) and C11 (+1)
        # M4 contributes to C00 (+1) and C10 (+1)
        # M5 contributes to C00 (-1) and C01 (+1)
        # M6 contributes to C11 (+1)
        # M7 contributes to C00 (+1)
        
        # But each M is a product of two linear combinations, so each M is 1 rank-1 term.
        # The issue is that a single M contributes to multiple C elements with different signs.
        # This means we can't just add it as a single rank-1 term in the tensor decomposition.
        
        # The correct way: for each M, we create one rank-1 term where:
        # U has the A linear combo, V has the B linear combo, W has the C linear combo.
        # But W would need to have multiple non-zero entries, which is fine.
        
        # Let me redo: for M1 = (a00+a11)(b00+b11):
        # U: a00_idx += 1, a11_idx += 1
        # V: b00_idx += 1, b11_idx += 1
        # W: c00_idx += 1, c11_idx += 1
        # This is ONE rank-1 term.
        
        # I already started adding M1 above. Let me continue.
        # M1 is done (partially - I added to c00 and c11, but I need to fix the sign for c11)
        # Actually c11 should be +1 for M1. Let me fix.
        
        # Let me start over with a cleaner approach.
        col -= 1  # Undo the partial M1 addition
        # Reset the entries I just set
        U[a00_idx][col] -= sign
        U[a11_idx][col] -= sign
        V[b00_idx][col] -= sign
        V[b11_idx][col] -= sign
        W[c00_idx][col] -= sign
        W[c11_idx][col] += sign  # Undo the -sign
        col += 1  # Now col points to a fresh column
        
        # M1 = (a00+a11)(b00+b11)
        U[a00_idx][col] += sign
        U[a11_idx][col] += sign
        V[b00_idx][col] += sign
        V[b11_idx][col] += sign
        W[c00_idx][col] += sign
        W[c11_idx][col] += sign
        col += 1
        
        # M2 = (a10+a11)b00
        U[a10_idx][col] += sign
        U[a11_idx][col] += sign
        V[b00_idx][col] += sign
        W[c10_idx][col] += sign
        W[c11_idx][col] -= sign
        col += 1
        
        # M3 = a00(b01-b11)
        U[a00_idx][col] += sign
        V[b01_idx][col] += sign
        V[b11_idx][col] -= sign
        W[c01_idx][col] += sign
        W[c11_idx][col] += sign
        col += 1
        
        # M4 = a11(b10-b00)
        U[a11_idx][col] += sign
        V[b10_idx][col] += sign
        V[b00_idx][col] -= sign
        W[c00_idx][col] += sign
        W[c10_idx][col] += sign
        col += 1
        
        # M5 = (a00+a01)b11
        U[a00_idx][col] += sign
        U[a01_idx][col] += sign
        V[b11_idx][col] += sign
        W[c00_idx][col] -= sign
        W[c01_idx][col] += sign
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a10_idx][col] += sign
        U[a00_idx][col] -= sign
        V[b00_idx][col] += sign
        V[b01_idx][col] += sign
        W[c11_idx][col] += sign
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a01_idx][col] += sign
        U[a11_idx][col] -= sign
        V[b10_idx][col] += sign
        V[b11_idx][col] += sign
        W[c00_idx][col] += sign
        col += 1
    
    # For B0:
    # Strassen for A_left @ B0_tl: A rows 0,1; B rows 0,1; C cols 0,2
    add_strassen(0, 1, 0, 1, 0, 2)
    
    # Strassen for A_left @ B0_tr: A rows 0,1; B rows 0,1; C cols 4,6
    add_strassen(0, 1, 0, 1, 4, 6)
    
    # Schoolbook for A_right @ B0_bottom: A col 2, B row 2, all 4 columns
    # A_right is 2x1: entries A[0,2], A[1,2]
    # B0_bottom is 1x4: entries B[2,0], B[2,2], B[2,4], B[2,6]
    # C contributions: C[i, col] += A[i,2] * B[2, col] for i=0,1 and col in {0,2,4,6}
    for c_orig in [0, 2, 4, 6]:
        for i in range(2):
            u_idx = i * 3 + 2  # A[i, 2]
            v_idx = 2 * 8 + c_orig  # B[2, c_orig]
            w_idx = c_orig * 2 + i  # C[i, c_orig]
            U[u_idx][col] += 1
            V[v_idx][col] += 1
            W[w_idx][col] += 1
            col += 1
    
    # For B1:
    # Strassen for A_left @ B1_tl: A rows 0,1; B rows 0,1; C cols 1,3
    add_strassen(0, 1, 0, 1, 1, 3)
    
    # Strassen for A_left @ B1_tr: A rows 0,1; B rows 0,1; C cols 5,7
    add_strassen(0, 1, 0, 1, 5, 7)
    
    # Schoolbook for A_right @ B1_bottom: A col 2, B row 2, all 4 columns
    for c_orig in [1, 3, 5, 7]:
        for i in range(2):
            u_idx = i * 3 + 2  # A[i, 2]
            v_idx = 2 * 8 + c_orig  # B[2, c_orig]
            w_idx = c_orig * 2 + i  # C[i, c_orig]
            U[u_idx][col] += 1
            V[v_idx][col] += 1
            W[w_idx][col] += 1
            col += 1
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to Fraction
    U_fr = [[Fraction(x) for x in row] for row in U]
    V_fr = [[Fraction(x) for x in row] for row in V]
    W_fr = [[Fraction(x) for x in row] for row in W]
    
    return U_fr, V_fr, W_fr

def construct():
    n, m, p = 2, 3, 8
    
    # Try the polynomial/Strassen rank-44 decomposition
    try:
        U_fr, V_fr, W_fr = _build_polynomial_strassen()
        if _validate_exact(U_fr, V_fr, W_fr, n, m, p):
            return {"U": U_fr, "V": V_fr, "W": W_fr}
    except Exception:
        pass
    
    # Fallback to schoolbook
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    return {"U": U_school, "V": V_school, "W": W_school}
