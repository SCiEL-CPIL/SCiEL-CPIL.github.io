"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x3 matrix by a 3x4 matrix.
    
    The schoolbook rank is 2*3*4 = 24.
    We use Strassen's algorithm for the 2x2x2 sub-blocks to reduce the rank.
    
    The 2x3 x 3x4 multiplication can be split as:
    A (2x3) * B (3x4) = [A_1 A_2] * [B_1; B_2] where
    A_1, A_2 are 2x2 (wait, 3 columns... let me re-examine)
    
    Actually: A is 2x3, B is 3x4.
    Split A into A_left (2x2) and A_right (2x1).
    Split B into B_top (2x4) and B_bottom (1x4).
    
    So: C = A_left * B_top + A_right * B_bottom
    
    A_left * B_top: 2x2 * 2x4. This is two independent 2x2*2x2 multiplications.
    Each 2x2*2x2 has Strassen rank 7. So 2*7 = 14.
    
    A_right * B_bottom: 2x1 * 1x4. This is outer product style.
    C_ij = A_right_i * B_bottom_j.
    This can be done with rank... let's think.
    C has 2*4 = 8 entries. Each entry C_ij = a_i * b_j.
    This is a rank-1 tensor in the (i,j) space? No, it's a sum of...
    Actually, C_ij = a_i * b_j is an outer product. The tensor T_{i,j,k} where k is the shared index (size 1).
    T_{i,j,0} = a_i * b_j. This is rank 1 if we consider it as a 2x4x1 tensor.
    Wait, the tensor for 2x1*1x4 multiplication: 
    The tensor has indices i (0..1), j (0..3), and the shared index k (0..0).
    T_{i,k,j} = delta_{k,0} * ... no.
    
    Let me use the standard tensor definition. For matrix multiplication of (n x m) * (m x p):
    T_{i,k,j} = 1 if the middle index matches, else 0.
    Specifically, for the term A_{i,m} * B_{m,j}, the tensor entry is T_{i,m,j} = 1.
    
    For 2x1*1x4: i in {0,1}, m in {0}, j in {0,1,2,3}.
    T_{i,0,j} = 1 for all i, j.
    This tensor has rank 1! Because T_{i,0,j} = u_i * v_0 * w_j where u_i=1, v_0=1, w_j=1.
    Wait, that would mean C_ij = sum_r U_{i,r} V_{0,r} W_{j,r} = U_{i,r} * V_{0,r} * W_{j,r}.
    If rank 1: C_ij = U_i * V_0 * W_j.
    We need C_ij = a_i * b_j. So U_i = a_i, V_0 = 1, W_j = b_j. Yes! Rank 1.
    
    So total rank = 14 + 1 = 15?
    
    But wait, the knowledge says rank 22 was achieved. Let me re-examine.
    
    Hmm, the issue is that the two parts A_left*B_top and A_right*B_bottom share the same output matrix C.
    In a tensor decomposition, we need a single set of rank terms that cover ALL entries of C.
    
    For A_left * B_top (2x2 * 2x4):
    This computes C_left = A_left * B_top, which is 2x4.
    C_left_{i,j} = sum_{m=0}^{1} A_{i,m} * B_{m,j}
    
    For A_right * B_bottom (2x1 * 1x4):
    This computes C_right = A_right * B_bottom, which is 2x4.
    C_right_{i,j} = A_{i,2} * B_{2,j}
    
    Total C_{i,j} = C_left_{i,j} + C_right_{i,j}
    
    The full tensor T_{i,m,j} for 2x3*3x4:
    T_{i,m,j} = 1 for all i in {0,1}, m in {0,1,2}, j in {0,1,2,3}.
    
    We need to decompose this tensor into rank-1 terms.
    
    Let's try to get rank 15.
    For the 2x2*2x4 part, we need 14 rank-1 terms (2 copies of Strassen 2x2x2).
    For the 2x1*1x4 part, we need 1 rank-1 term.
    Total 15.
    
    But can we overlap? The 15th term (for A_right*B_bottom) uses m=2.
    The 14 terms for A_left*B_top use m=0 and m=1.
    So they don't overlap in the middle index m. They are independent.
    So rank 15 should work.
    
    But the knowledge says rank 22. Maybe I'm making an error.
    
    Let me double-check the Strassen decomposition for 2x2*2x2.
    Strassen: C = A*B where A,B are 2x2.
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B01 - B11)
    M4 = A11(B10 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each M is a product of two linear combinations. So each M corresponds to one rank-1 term in the tensor decomposition.
    7 terms for 2x2*2x2. Correct.
    
    For 2x2*2x4: split the 4 columns into two groups of 2.
    B = [B_left | B_right] where B_left, B_right are 2x2.
    C = A * B = [A*B_left | A*B_right]
    This is two independent 2x2*2x2 multiplications.
    So 7 + 7 = 14 terms. Correct.
    
    For 2x1*1x4: rank 1. Correct.
    
    So total rank 15. Let me verify this more carefully.
    
    Actually, I think the issue is that the tensor T_{i,m,j} must satisfy:
    T_{i,m,j} = sum_r U_{i,r} V_{m,r} W_{j,r}
    
    For the 2x2*2x4 part (m=0,1), we have 14 terms where V_{m,r} is non-zero for m=0,1 and zero for m=2.
    For the 2x1*1x4 part (m=2), we have 1 term where V_{m,r} is non-zero for m=2 and zero for m=0,1.
    
    These are compatible because the V matrices have disjoint support in the m dimension.
    So the total rank is 14 + 1 = 15.
    
    But wait, I need to verify that rank 15 is actually achievable. Let me construct it.
    
    Hmm, but the prompt says rank 22 was measured. Let me be safe and go with 22 first, then try to improve.
    
    Actually, let me try rank 15 and verify. If it fails, I'll fall back to 22.
    """
    n, m, p = 2, 3, 4
    # We'll try rank 15 first.
    # 14 terms for 2x2*2x4 (m=0,1), 1 term for 2x1*1x4 (m=2).
    
    rank = 15
    U = [[0] * rank for _ in range(n)]  # U[i, r] for i in 0..1
    V = [[0] * rank for _ in range(m)]  # V[mm, r] for mm in 0..2
    W = [[0] * rank for _ in range(p)]  # W[j, r] for j in 0..3
    
    # Strassen for 2x2*2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    def strassen_2x2x2(offset_u, offset_v, offset_w, base_r):
        """Add 7 Strassen terms starting at rank index base_r."""
        r = base_r
        # M1: U0 += 1, U1 += 1, V0 += 1, V1 += 1, W0 += 1, W1 += 1
        U[0][r] += 1; U[1][r] += 1
        V[offset_v + 0][r] += 1; V[offset_v + 1][r] += 1
        W[offset_w + 0][r] += 1; W[offset_w + 1][r] += 1
        r += 1
        # M2: U1 += 1, V0 += 1, W0 += 1
        U[1][r] += 1
        V[offset_v + 0][r] += 1
        W[offset_w + 0][r] += 1
        r += 1
        # M3: U0 += 1, V0 += 1, W1 += 1, W0 += -1
        U[0][r] += 1
        V[offset_v + 0][r] += 1
        W[offset_w + 1][r] += 1; W[offset_w + 0][r] += -1
        r += 1
        # M4: U1 += 1, V1 += 1, W0 += 1, W1 += -1
        U[1][r] += 1
        V[offset_v + 1][r] += 1
        W[offset_w + 0][r] += 1; W[offset_w + 1][r] += -1
        r += 1
        # M5: U0 += 1, U1 += 1, V1 += 1, W0 += 1
        U[0][r] += 1; U[1][r] += 1
        V[offset_v + 1][r] += 1
        W[offset_w + 0][r] += 1
        r += 1
        # M6: U1 += 1, U0 += -1, V0 += 1, W0 += 1, W1 += 1
        U[1][r] += 1; U[0][r] += -1
        V[offset_v + 0][r] += 1
        W[offset_w + 0][r] += 1; W[offset_w + 1][r] += 1
        r += 1
        # M7: U0 += 1, U1 += -1, V1 += 1, W0 += 1, W1 += 1
        U[0][r] += 1; U[1][r] += -1
        V[offset_v + 1][r] += 1
        W[offset_w + 0][r] += 1; W[offset_w + 1][r] += 1
        r += 1
        return r
    
    # First 2x2*2x2: A (2x2, rows 0-1, cols 0-1) * B_left (2x2, rows 0-1, cols 0-1)
    # This contributes to C[0:2, 0:2]
    r = strassen_2x2x2(0, 0, 0, 0)  # 7 terms, r = 7
    
    # Second 2x2*2x2: A (2x2, rows 0-1, cols 0-1) * B_right (2x2, rows 0-1, cols 2-3)
    # This contributes to C[0:2, 2:4]
    r = strassen_2x2x2(0, 0, 2, 7)  # 7 terms, r = 14
    
    # Third: 2x1*1x4: A_right (row 2) * B_bottom (row 2)
    # C[i,j] += A[i,2] * B[2,j]
    # Rank 1: U[i,14] = 1, V[2,14] = 1, W[j,14] = 1
    for i in range(2):
        U[i][14] = 1
    V[2][14] = 1
    for j in range(4):
        W[j][14] = 1
    
    # Verify: check that T_{i,mm,j} = sum_r U[i,r] V[mm,r] W[j,r] equals 1 for all i,mm,j
    def verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for mm in range(m):
                for j in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i][r] * V[mm][r] * W[j][r]
                    if s != 1:
                        return False, (i, mm, j, s)
        return True, None
    
    ok, err = verify(U, V, W, n, m, p, rank)
    if not ok:
        # Fall back to rank 22 (schoolbook-like with some Strassen)
        # Use full schoolbook for safety
        rank = 24
        U = [[0] * rank for _ in range(n)]
        V = [[0] * rank for _ in range(m)]
        W = [[0] * rank for _ in range(p)]
        col = 0
        for i in range(n):
            for mm in range(m):
                for j in range(p):
                    U[i][col] = 1
                    V[mm][col] = 1
                    W[j][col] = 1
                    col += 1
    else:
        rank = 15
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
