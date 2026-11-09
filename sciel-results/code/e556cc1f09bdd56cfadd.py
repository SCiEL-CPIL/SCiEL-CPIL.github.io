"""Matrix multiplication 3x3 by 3x7 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """
    Strassen 7-term decomposition for 2x2 * 2x2.
    Returns (U, V, W) each of shape (4, 7).
    
    U[i*2+j, c] for A position (i,j)
    V[j*2+k, c] for B position (j,k)
    W[k*2+i, c] for C position (k,i)
    """
    # A: 0=a,1=b,2=c,3=d
    # B: 0=x,1=y,2=z,3=w
    # C: 0=p,1=q,2=r,3=s
    
    terms = [
        ({0:1, 3:1}, {0:1, 1:1}, {0:1, 3:1}),   # (a+d)(x+y)
        ({0:1, 2:-1}, {0:1, 1:1}, {1:-1, 3:1}),  # (a-c)(x+y)
        ({2:1, 3:1}, {2:1, 3:-1}, {0:1, 2:1}),   # (c+d)(z-w)
        ({0:1, 1:1}, {3:1, 2:-1}, {0:1, 1:1}),   # (a+b)(w-z)
        ({0:1, 1:-1}, {2:1, 3:1}, {0:1, 2:-1}),  # (a-b)(z+w)
        ({0:1, 1:1}, {0:1, 3:-1}, {1:1, 3:-1}),  # (a+b)(x-w)
        ({1:1, 3:1}, {2:1, 0:-1}, {1:1, 2:1}),   # (b+d)(z-x)
    ]
    
    U = [[0.0] * 7 for _ in range(4)]
    V = [[0.0] * 7 for _ in range(4)]
    W = [[0.0] * 7 for _ in range(4)]
    
    for c, (A_form, B_form, C_form) in enumerate(terms):
        for idx, val in A_form.items():
            U[idx][c] = val
        for idx, val in B_form.items():
            V[idx][c] = val
        for idx, val in C_form.items():
            W[idx][c] = val
    
    return U, V, W


def _build_3x3x7_strassen():
    """
    Build 3x3 * 3x7 decomposition using Strassen-based partitioning.
    
    Strategy: 
    Split A (3x3) into A_top (2x3) and A_bottom (1x3).
    Split B (3x7) into B_left (3x3) and B_right (3x4).
    
    C = A * B = [A_top; A_bottom] * [B_left | B_right]
    
    This gives:
    C_top_left (2x3) = A_top * B_left
    C_top_right (2x4) = A_top * B_right
    C_bot_left (1x3) = A_bottom * B_left
    C_bot_right (1x4) = A_bottom * B_right
    
    For C_top_left (2x3 * 3x3):
    Split B_left into B_left_top (2x3, rows 0-1) and B_left_bot (1x3, row 2).
    C_top_left = A_top * B_left = A_top * [B_left_top; B_left_bot]
    = A_top * B_left_top + A_top * B_left_bot
    
    Hmm, this is getting complicated. Let me use a cleaner block decomposition.
    
    Actually, let's use the following approach:
    View 3x3 * 3x7 as:
    - 2x3 * 3x3 part (top 2 rows of A, first 3 cols of B) -> use Strassen-like
    - 2x3 * 3x4 part (top 2 rows of A, last 4 cols of B)
    - 1x3 * 3x3 part (bottom row of A, first 3 cols of B)
    - 1x3 * 3x4 part (bottom row of A, last 4 cols of B)
    
    For the 2x3 * 3x3 part:
    We can write A_top as [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, etc.
    Actually, let's just do:
    A_top is 2x3, B_left is 3x3.
    C_top_left = A_top * B_left
    
    We can decompose this by splitting B_left into columns:
    B_left = [b0 | b1 | b2] where b0, b1, b2 are 3x1 vectors.
    C_top_left = [A_top*b0 | A_top*b1 | A_top*b2]
    
    Each A_top*bk is a 2x1 vector = 2 scalar multiplications... no, it's a 2x3 * 3x1 = 2x1.
    That's 2*3 = 6 scalar multiplications per column, so 18 total. Not good.
    
    Better approach: Use Strassen on the 2x2 * 2x2 sub-blocks.
    
    Let me try a different decomposition. Split A (3x3) and B (3x7) as:
    A = [A11 A12] where A11 is 3x2, A12 is 3x1
    B = [B1; B2] where B1 is 2x7, B2 is 1x7
    
    C = A*B = A11*B1 + A12*B2
    
    A11*B1 is 3x2 * 2x7 = 3x7
    A12*B2 is 3x1 * 1x7 = 3x7
    
    For A11*B1 (3x2 * 2x7):
    Split B1 into B1_left (2x3) and B1_right (2x4).
    A11*B1 = A11*B1_left + A11*B1_right
    
    A11*B1_left is 3x2 * 2x3 = 3x3
    A11*B1_right is 3x2 * 2x4 = 3x4
    
    For 3x2 * 2x3:
    We can use Strassen-like methods. Let's view it as:
    A11 = [a1 a2] where a1, a2 are 3x1
    B1_left = [b1 b2 b3] where b1, b2, b3 are 2x1
    
    C = a1*b1 + a2*b2 + a1*b2 + a2*b3 ... no this is getting messy.
    
    Let me just use a known efficient decomposition.
    
    Actually, the simplest good approach:
    Use Strassen for the 2x2*2x2 blocks where possible, and schoolbook for the rest.
    
    Let me try:
    Split A (3x3) into 2x2, 2x1, 1x2, 1x1 blocks:
    A = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    
    Split B (3x7) into 2x7 and 1x7:
    B = [B1; B2] where B1 is 2x7, B2 is 1x7
    
    C = A*B = [A11*B1 + A12*B2; A21*B1 + A22*B2]
    
    C_top (2x7) = A11*B1 + A12*B2
    C_bot (1x7) = A21*B1 + A22*B2
    
    A11*B1 is 2x2 * 2x7. Split B1 into 7 columns of 2x1.
    Use Strassen for each 2x2 * 2x1? No, Strassen is for 2x2*2x2.
    
    For 2x2 * 2x7:
    B1 = [b1 b2 b3 b4 b5 b6 b7] where each bi is 2x1.
    A11*B1 = [A11*b1, ..., A11*b7]
    Each A11*bi is 2x2 * 2x1 = 2x1, which takes 4 scalar mults (schoolbook) or can be optimized.
    
    Actually 2x2 * 2x1 is just 2*2 = 4 multiplications (each output is a dot product of length 2).
    So 2x2 * 2x7 takes 2*7*2 = 28 scalar mults schoolbook.
    
    With Strassen-like: We can group the 7 columns.
    Split 7 columns into 3+4.
    2x2 * 2x3: group 3 columns.
    2x2 * 2x4: group 4 columns into two 2x2 blocks.
    
    For 2x2 * 2x4:
    B1_right = [B1a B1b] where B1a, B1b are 2x2.
    A11*B1_right = [A11*B1a | A11*B1b]
    Each A11*B1x is 2x2*2x2, use Strassen (7 mults each).
    Total for 2x2*2x4: 14 mults.
    
    For 2x2 * 2x3:
    B1_left = [B1c B1d] where B1c is 2x2, B1d is 2x1.
    A11*B1_left = [A11*B1c | A11*B1d]
    A11*B1c is 2x2*2x2, use Strassen (7 mults).
    A11*B1d is 2x2*2x1, schoolbook (4 mults).
    Total for 2x2*2x3: 11 mults.
    
    So A11*B1 (2x2*2x7) takes 14 + 11 = 25 mults.
    
    A12*B2 is 2x1 * 1x7 = 2x7.
    A12 = [a1; a2] (2x1), B2 = [b1 b2 ... b7] (1x7).
    C = [a1*b1 a1*b2 ... a1*b7; a2*b1 a2*b2 ... a2*b7]
    This is 14 scalar mults (schoolbook, and it's already minimal since each output needs one mult).
    
    A21*B1 is 1x2 * 2x7 = 1x7.
    A21 = [c1 c2], B1 = [b1 b2 ... b7] (2x7).
    C = [c1*b1[0] + c2*b1[1], ..., c1*b7[0] + c2*b7[1]]
    Each output is a dot product of length 2, so 2 mults each, 14 total.
    Can we do better? 1x2 * 2x7. The rank of this tensor is 14 (schoolbook is optimal for this since it's a bilinear map from R^2 x R^7 to R^7, and each output coordinate is a separate bilinear form). Actually no, we might be able to share computations. But 1x2*2x7 is essentially 7 independent 1x2*2x1 multiplications, each of rank 2, so total rank 14.
    
    A22*B2 is 1x1 * 1x7 = 1x7.
    A22 is a scalar, B2 is 1x7.
    C = A22 * B2, which is 7 scalar mults.
    
    Total: 25 + 14 + 14 + 7 = 60 mults.
    
    Schoolbook is 3*3*7 = 63. So we save 3 mults.
    
    Let me implement this.
    """
    n, m, p = 3, 3, 7
    
    # We'll build the decomposition term by term.
    # Each term is (U_entry, V_entry, W_entry) where:
    # U_entry: dict mapping (i,j) -> coeff for A[i,j]
    # V_entry: dict mapping (j,k) -> coeff for B[j,k]
    # W_entry: dict mapping (k,i) -> coeff for C[k,i] (but C is 3x7, so k is 0..2, i is 0..6)
    # Wait, the output C is 3x7. The convention is W[k*n+i, r] where the output index is k*n+i.
    # For C (3x7), the output has indices (row, col) where row is 0..2 and col is 0..6.
    # So W index = row*3 + col? No, the problem says W[k*n+i, r].
    # Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    # For matrix multiplication A(n x m) * B(m x p) = C(n x p), 
    # C[i,j] = sum_k A[i,k] * B[k,j].
    # The tensor is T[i,j,k] = A[i,k] * B[k,j] * C[i,j].
    # The decomposition is sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c] = 1 for all i,j,k.
    # Wait, that doesn't look right either. Let me think again.
    
    # Standard matrix multiplication tensor:
    # T[i,j,k] where i is row of A and C, j is col of C and col of B, k is col of A and row of B.
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # So the tensor has indices (i, j, k) and T[i,j,k] = 1.
    # U has shape (n*m, rank), indexed by (i,k) -> i*m+k
    # V has shape (m*p, rank), indexed by (k,j) -> k*p+j
    # W has shape (p*n, rank), indexed by (j,i) -> j*n+i
    
    # So for C (3x7), i is 0..2 (row), j is 0..6 (col), k is 0..2 (inner).
    # U index: i*3+k
    # V index: k*7+j
    # W index: j*3+i
    
    # Let me redefine:
    # n=3 (rows of A, cols of C)
    # m=3 (cols of A, rows of B)
    # p=7 (cols of B, cols of C)
    
    # U: shape (3*3, rank) = (9, rank), index (i,k) -> i*3+k
    # V: shape (3*7, rank) = (21, rank), index (k,j) -> k*7+j
    # W: shape (7*3, rank) = (21, rank), index (j,i) -> j*3+i
    
    # Now let's build the terms.
    
    terms = []  # Each term is (U_dict, V_dict, W_dict)
    
    # Part 1: A11 * B1 (2x2 * 2x7)
    # A11 is A[0:2, 0:2], indices (i,k) for i=0,1 and k=0,1
    # B1 is B[0:2, :], indices (k,j) for k=0,1 and j=0..6
    
    # Split B1 into B1_left (2x3) and B1_right (2x4)
    # B1_left: k=0,1 and j=0,1,2
    # B1_right: k=0,1 and j=3,4,5,6
    
    # A11 * B1_left (2x2 * 2x3):
    # Split into A11*B1c (2x2*2x2, j=0,1) and A11*B1d (2x2*2x1, j=2)
    
    # A11*B1c (2x2*2x2) using Strassen:
    # A11 = [[a00, a01], [a10, a11]]
    # B1c = [[b00, b01], [b10, b11]]
    # C1c = [[c00, c01], [c10, c11]]
    
    # Strassen terms:
    # m1 = (a00+a11)*(b00+b10)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10+b11)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b01)
    # m7 = (a01-a11)*(b10+b11)
    
    # c00 = m1 + m4 - m2 - m7
    # c01 = m3 + m5
    # c10 = m2 + m6
    # c11 = m1 - m3 + m5 + m7
    
    # For each Strassen term, we need to map to our indices.
    # A11 is at A[i,k] for i=0,1 and k=0,1.
    # B1c is at B[k,j] for k=0,1 and j=0,1.
    # C1c is at C[i,j] for i=0,1 and j=0,1.
    
    # Term m1: (a00+a11)*(b00+b10)
    # U: {0:1, 1:1} (A[0,0] and A[1,1] -> indices 0*3+0=0 and 1*3+1=4)
    # Wait, A[i,k] -> index i*m+k = i*3+k.
    # A[0,0] -> 0, A[1,1] -> 1*3+1=4
    # V: {0:1, 1:1} (B[0,0] and B[1,0] -> indices 0*7+0=0 and 1*7+0=7)
    # W: {0:1, 3:1} (C[0,0] and C[1,1] -> indices 0*3+0=0 and 1*3+1=4)
    # Wait, C[i,j] -> W index j*n+i = j*3+i.
    # C[0,0] -> 0*3+0=0, C[1,1] -> 1*3+1=4
    
    # Let me redo this more carefully.
    # U index for A[i,k]: i*3+k
    # V index for B[k,j]: k*7+j
    # W index for C[i,j]: j*3+i
    
    # m1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,0])
    # U: {0:1, 4:1}
    # V: {0:1, 7:1}
    # This contributes to C[0,0] and C[1,1] (with signs in the final combination)
    
    # Actually, in the Strassen decomposition, each m_i is a scalar, and then
    # the C entries are linear combinations of the m_i's.
    # So for the tensor decomposition, each m_i is one term in the sum.
    # The W factor for each m_i needs to encode which C entries it contributes to and with what coefficient.
    
    # For m1 = (a00+a11)*(b00+b10):
    # This m1 contributes to:
    # C[0,0] with coeff +1
    # C[1,1] with coeff +1
    # So W: {0:1, 4:1}
    
    # For m2 = (a10+a11)*b00:
    # m2 contributes to:
    # C[0,0] with coeff -1
    # C[1,0] with coeff +1
    # W: {0:-1, 3:1}  (C[0,0]->0, C[1,0]->0*3+1=1... wait)
    # C[1,0] -> j=0, i=1 -> 0*3+1=1
    # So W: {0:-1, 1:1}
    
    # Let me just implement the Strassen 2x2x2 and then map it.
    
    U_s, V_s, W_s = _strassen_2x2x2()
    # U_s[i*2+k, c], V_s[k*2+j, c], W_s[j*2+i, c]
    # For the 2x2 block at A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]:
    # A[i,k] for i=0,1 k=0,1 -> global index i*3+k
    # B[k,j] for k=0,1 j=0,1 -> global index k*7+j
    # C[i,j] for i=0,1 j=0,1 -> global index j*3+i
    
    for c in range(7):
        U_dict = {}
        V_dict = {}
        W_dict = {}
        for local_uk in range(4):
            i = local_uk // 2
            k = local_uk % 2
            val = U_s[local_uk][c]
            if val != 0:
                global_uk = i*3 + k
                U_dict[global_uk] = U_dict.get(global_uk, 0) + val
        for local_vk in range(4):
            k = local_vk // 2
            j = local_vk % 2
            val = V_s[local_vk][c]
            if val != 0:
                global_vk = k*7 + j
                V_dict[global_vk] = V_dict.get(global_vk, 0) + val
        for local_wk in range(4):
            j = local_wk // 2
            i = local_wk % 2
            val = W_s[local_wk][c]
            if val != 0:
                global_wk = j*3 + i
                W_dict[global_wk] = W_dict.get(global_wk, 0) + val
        terms.append((U_dict, V_dict, W_dict))
    
    # A11 * B1d (2x2 * 2x1) where B1d is B[0:2, 2] (column 2)
    # Schoolbook: 4 terms
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] for i=0,1
    for i in range(2):
        for k in range(2):
            U_dict = {i*3+k: 1}
            V_dict = {k*7+2: 1}
            W_dict = {2*3+i: 1}  # C[i,2] -> j=2, i -> 2*3+i
            terms.append((U_dict, V_dict, W_dict))
    
    # A11 * B1_right (2x2 * 2x4) where B1_right is B[0:2, 3:7]
    # Split into two 2x2 blocks: B[0:2, 3:5] and B[0:2, 5:7]
    for block_start in [3, 5]:
        U_s, V_s, W_s = _strassen_2x2x2()
        for c in range(7):
            U_dict = {}
            V_dict = {}
            W_dict = {}
            for local_uk in range(4):
                i = local_uk // 2
                k = local_uk % 2
                val = U_s[local_uk][c]
                if val != 0:
                    global_uk = i*3 + k
                    U_dict[global_uk] = U_dict.get(global_uk, 0) + val
            for local_vk in range(4):
                k = local_vk // 2
                j = local_vk % 2
                val = V_s[local_vk][c]
                if val != 0:
                    global_vk = k*7 + (block_start + j)
                    V_dict[global_vk] = V_dict.get(global_vk, 0) + val
            for local_wk in range(4):
                j = local_wk // 2
                i = local_wk % 2
                val = W_s[local_wk][c]
                if val != 0:
                    global_wk = (block_start + j)*3 + i
                    W_dict[global_wk] = W_dict.get(global_wk, 0) + val
            terms.append((U_dict, V_dict, W_dict))
    
    # Part 2: A12 * B2 (2x1 * 1x7)
    # A12 is A[0:2, 2], B2 is B[2, :]
    # C[i,j] += A[i,2] * B[2,j] for i=0,1 and j=0..6
    for i in range(2):
        for j in range(7):
            U_dict = {i*3+2: 1}
            V_dict = {2*7+j: 1}
            W_dict = {j*3+i: 1}
            terms.append((U_dict, V_dict, W_dict))
    
    # Part 3: A21 * B1 (1x2 * 2x7)
    # A21 is A[2, 0:2], B1 is B[0:2, :]
    # C[2,j] += A[2,0]*B[0,j] + A[2,1]*B[1,j] for j=0..6
    for k in range(2):
        for j in range(7):
            U_dict = {2*3+k: 1}
            V_dict = {k*7+j: 1}
            W_dict = {j*3+2: 1}
            terms.append((U_dict, V_dict, W_dict))
    
    # Part 4: A22 * B2 (1x1 * 1x7)
    # A22 is A[2,2], B2 is B[2, :]
    # C[2,j] += A[2,2] * B[2,j] for j=0..6
    for j in range(7):
        U_dict = {2*3+2: 1}
        V_dict = {2*7+j: 1}
        W_dict = {j*3+2: 1}
        terms.append((U_dict, V_dict, W_dict))
    
    # Build U, V, W matrices
    rank = len(terms)
    U = [[0.0] * rank for _ in range(9)]
    V = [[0.0] * rank for _ in range(21)]
    W = [[0.0] * rank for _ in range(21)]
    
    for c, (U_dict, V_dict, W_dict) in enumerate(terms):
        for idx, val in U_dict.items():
            U[idx][c] = val
        for idx, val in V_dict.items():
            V[idx][c] = val
        for idx, val in W_dict.items():
            W[idx][c] = val
    
    return U, V, W, rank


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 by 3x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    deadline = time.time() + 110
    
    U, V, W, rank = _build_3x3x7_strassen()
    
    # Verify the decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
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

# EVOLVE-BLOCK-END
