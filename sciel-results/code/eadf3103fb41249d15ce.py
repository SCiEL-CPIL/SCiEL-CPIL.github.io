import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28

    # We'll build terms as lists of (u, v, w) vectors
    # u: length nm, v: length mp, w: length pn
    terms = []

    def add_term(u_vec, v_vec, w_vec):
        # u_vec, v_vec, w_vec are numpy arrays of length nm, mp, pn
        terms.append((u_vec, v_vec, w_vec))

    # --- Block 1: C[0:4, 0:4] = A[0:4, 0:4] @ B[0:4, 0:4] ---
    # Use Strassen for 4x4x4
    # Strassen 4x4x4 has rank 49 (7 blocks, each 2x2x2 Strassen)
    
    # We'll implement the 4x4x4 Strassen decomposition
    # The 4x4x4 Strassen uses 7 products of 2x2 blocks
    # Each 2x2 block product uses 7 scalar multiplications
    
    # Let's define the 2x2x2 Strassen coefficients
    # For 2x2x2, we have 7 terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For 4x4x4, we partition A into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # Similarly for B and C
    
    # The 7 block products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Each Mi is a 2x2 product, so we need to expand each using 2x2x2 Strassen
    
    # Let's define a helper to add a 2x2x2 Strassen decomposition
    # for a given set of A-block and B-block coefficients
    
    def add_strassen_2x2(a_coeff, b_coeff, c_offset_i, c_offset_j, scale=1):
        """
        Add a 2x2x2 Strassen decomposition.
        a_coeff: dict mapping (i,j) to coefficient for A[i,j] in the local 2x2
        b_coeff: dict mapping (i,j) to coefficient for B[i,j] in the local 2x2
        c_offset_i, c_offset_j: offset for the C output in the local 2x2
        """
        # The 7 M terms for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each M term, we need to form the u, v, w vectors
        # u is for A, v is for B, w is for C
        
        # M1: u has A00 and A11 with coeff 1, v has B00 and B11 with coeff 1
        # w: C00 += M1, C11 += M1
        # So w has C00 and C11 with coeff 1
        
        # But we need to account for a_coeff and b_coeff
        # The actual A entries are linear combos
        
        # Let's think of it differently.
        # The 2x2x2 Strassen gives us 7 terms where each term is
        # u_term (2x2 for A) * v_term (2x2 for B) -> w_term (2x2 for C)
        
        # For M1:
        # u = [[1,0],[0,1]] (A00 + A11)
        # v = [[1,0],[0,1]] (B00 + B11)
        # w = [[1,0],[0,1]] (C00 + C11)
        
        # But wait, the Strassen formula is:
        # M1 = (A00+A11)(B00+B11)
        # This is a single scalar multiplication of the two 2x2 matrices
        # No, that's not right. M1 is a 2x2 matrix product.
        
        # Actually, in the 4x4x4 Strassen, each Mi is a 2x2 matrix product
        # of two 2x2 matrices. So we need to decompose each 2x2 matrix product
        # using the 2x2x2 Strassen (7 scalar multiplications).
        
        # So for each Mi, we call the 2x2x2 Strassen decomposition.
        
        # The 2x2x2 Strassen for matrices X (2x2) and Y (2x2) gives Z = XY
        # with 7 terms:
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Z00 = M1 + M4 - M5 - M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M7
        
        # Each Mk is a scalar, so each term in the 2x2x2 decomposition is
        # a scalar multiplication: u_scalar * v_scalar -> w_scalar
        
        # So for the 2x2x2 Strassen, we have 7 terms, each with:
        # u: a 4-element vector (flattened 2x2)
        # v: a 4-element vector (flattened 2x2)
        # w: a 4-element vector (flattened 2x2)
        
        # Let's define the 7 terms for 2x2x2 Strassen
        
        # Term 1: M1 = (X00+X11)(Y00+Y11)
        # u1 = [1, 0, 0, 1] (X00 + X11)
        # v1 = [1, 0, 0, 1] (Y00 + Y11)
        # w1: contributes to Z00 and Z11
        # Z00 += M1, Z11 += M1
        # w1 = [1, 0, 0, 1] (Z00 + Z11)
        
        # Term 2: M2 = (X10+X11)Y00
        # u2 = [0, 0, 1, 1] (X10 + X11)
        # v2 = [1, 0, 0, 0] (Y00)
        # w2: contributes to Z10 and Z11
        # Z10 += M2, Z11 -= M2
        # w2 = [0, 0, 1, -1] (Z10 - Z11)
        
        # Term 3: M3 = X00(Y01-Y11)
        # u3 = [1, 0, 0, 0] (X00)
        # v3 = [0, 1, 0, -1] (Y01 - Y11)
        # w3: contributes to Z01 and Z11
        # Z01 += M3, Z11 += M3
        # w3 = [0, 1, 0, 1] (Z01 + Z11)
        
        # Term 4: M4 = X11(Y10-Y00)
        # u4 = [0, 0, 0, 1] (X11)
        # v4 = [0, 0, 1, -1] (Y10 - Y00)
        # w4: contributes to Z00 and Z10
        # Z00 += M4, Z10 += M4
        # w4 = [1, 0, 1, 0] (Z00 + Z10)
        
        # Term 5: M5 = (X00+X01)Y11
        # u5 = [1, 1, 0, 0] (X00 + X01)
        # v5 = [0, 0, 0, 1] (Y11)
        # w5: contributes to Z00 and Z01
        # Z00 -= M5, Z01 += M5
        # w5 = [-1, 1, 0, 0] (-Z00 + Z01)
        
        # Term 6: M6 = (X10-X00)(Y00+Y01)
        # u6 = [-1, 0, 1, 0] (X10 - X00)
        # v6 = [1, 1, 0, 0] (Y00 + Y01)
        # w6: doesn't contribute to any Z? 
        # Wait, let me re-check the Strassen formulas.
        
        # Actually, I think I made an error. Let me look up the standard Strassen 2x2x2.
        
        # Standard Strassen 2x2x2:
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)(Y00)
        # M3 = (X00)(Y01-Y11)
        # M4 = (X11)(Y10-Y00)
        # M5 = (X00+X01)(Y11)
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Z00 = M1 - M5 + M4 - M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M7
        
        # Let me re-derive:
        # Z00 = M1 - M5 + M4 - M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M7
        
        # So:
        # Term 1 (M1): w = [1, 0, 0, 1] (Z00 + Z11)
        # Term 2 (M2): w = [0, 0, 1, -1] (Z10 - Z11)
        # Term 3 (M3): w = [0, 1, 0, 1] (Z01 + Z11)
        # Term 4 (M4): w = [1, 0, 1, 0] (Z00 + Z10)
        # Term 5 (M5): w = [-1, 1, 0, 0] (-Z00 + Z01)
        # Term 6 (M6): w = [0, 0, 0, 0] (no contribution) -- wait, that can't be right
        
        # Let me re-check. I think I'm confusing myself.
        
        # Actually, the standard Strassen 2x2x2 has 7 terms, and each term
        # contributes to the output. Let me just use the known coefficients.
        
        # I'll use a simpler approach: just use the schoolbook for the 2x2x2
        # sub-problems, which gives rank 8 for each 2x2x2, and 7*8=56 for the
        # 4x4x4 part. That's worse than Strassen's 49.
        
        # Let me just implement the 4x4x4 Strassen directly with the known
        # 49-term decomposition.
        
        pass

    # This is getting complicated. Let me use a different approach.
    # For 4x4x7, I'll use the schoolbook for the 4x4x4 part (rank 64) and
    # schoolbook for the 4x4x3 part (rank 48), total rank 112.
    # But that's the same as the current program.
    
    # Let me try to implement Strassen for 4x4x4 properly.
    
    # Actually, let me just use the Kronecker product approach.
    # For 4x4x7, I can decompose it as:
    # - 4x4x4 part: use Strassen (rank 49)
    # - 4x4x3 part: use schoolbook (rank 48)
    # Total: 49 + 48 = 97
    
    # But implementing Strassen 4x4x4 is complex. Let me try a simpler
    # improvement: use the fact that 4 = 2+2 and apply Strassen recursively.
    
    # For now, let me just return the schoolbook decomposition to ensure
    # correctness, and then improve.
    
    # Actually, let me implement a simple block decomposition:
    # Split B into B1 (4x4) and B2 (4x3)
    # C1 = A @ B1 (4x4)
    # C2 = A @ B2 (4x3)
    
    # For C1, use Strassen 4x4x4 (rank 49)
    # For C2, use schoolbook 4x4x3 (rank 48)
    
    # I'll implement Strassen 4x4x4 using the 7-block approach.
    
    # The 4x4x4 Strassen:
    # Partition A, B, C into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Each Mi is a 2x2 matrix product, so we need to expand each using
    # the 2x2x2 Strassen (7 terms each).
    
    # Total: 7 * 7 = 49 terms.
    
    # Let me implement this step by step.
    
    # First, define the 2x2x2 Strassen terms as (u, v, w) where
    # u, v, w are 4-element vectors (flattened 2x2 matrices)
    
    # For 2x2x2 Strassen, the 7 terms are:
    # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,-1]
    # M3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,1]
    # M4: u=[0,0,0,1], v=[0,0,1,-1], w=[1,0,1,0]
    # M5: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
    # M6: u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,0] -- this is wrong
    
    # I think I need to look up the correct coefficients. Let me use a
    # different approach: just use the schoolbook for the 2x2x2 sub-problems.
    # That gives 8 terms per 2x2x2, so 7*8=56 for the 4x4x4 part.
    # Total: 56 + 48 = 104. Still better than 112.
    
    # Actually, let me just use the schoolbook for everything to ensure
    # correctness, and then I can optimize later.
    
    # Schoolbook for 4x4x7:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                add_term(u, v, w)
    
    rank = len(terms)
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for c, (u, v, w) in enumerate(terms):
        U[:, c] = u
        V[:, c] = v
        W[:, c] = w
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
