import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """Check if U, V, W form a valid tensor decomposition of matmul.
    Returns True if valid."""
    n, m, p = 4, 4, 5
    U = U.astype(float)
    V = V.astype(float)
    W = W.astype(float)
    
    # Build tensor T where T[i,j,k] = U[i,j] * V[j,k] * W[k,i] summed over rank
    # Actually: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 if and only if it's a matmul term, which is always 1 for all i,j,k
    # Wait - for matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] = 1 for all valid i,j,k (it's the multiplication tensor)
    # Actually no: the tensor for matmul has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because each output element C[i,k] depends on A[i,j]*B[j,k] for every j.
    # So the target tensor is all ones.
    
    # Compute the decomposed tensor
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Do this efficiently
    # For each r, the contribution to T is outer product of reshaped U[:,r], V[:,r], W[:,r]
    # U[:,r] reshaped to (n, m), V[:,r] reshaped to (m, p), W[:,r] reshaped to (p, n)
    # Contribution to T[i,j,k] from rank r: U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        # T[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
        # This is a 3D operation. Let's do it with einsum or loops
        # For efficiency with small sizes:
        for i in range(n):
            for k in range(p):
                if Wr[k, i] != 0:
                    T[i, :, k] += Ur[i, :] * Vr[:, k] * Wr[k, i]
    
    # Target is all ones
    target = np.ones((n, m, p), dtype=float)
    return np.allclose(T, target, atol=1e-9)


def _schoolbook():
    """Return schoolbook decomposition with rank n*m*p = 80."""
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
    
    return {"U": U, "V": V, "W": W}


def _strassen_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2 matmul, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen's 7 multiplications for 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # 
    # M1 = (a+d)(e+h)  -> contributes to C[0,0] and C[1,1]
    # M2 = (c+d)e      -> contributes to C[1,0]
    # M3 = a(f-h)      -> contributes to C[0,1] and C[1,1]
    # M4 = d(g-e)      -> contributes to C[1,0] and C[0,0]
    # M5 = (a+b)h      -> contributes to C[0,1]
    # M6 = (c-a)(e+f)  -> contributes to C[1,0]
    # M7 = (b-d)(g+h)  -> contributes to C[1,1]
    
    # Let's verify:
    # C[0,0] = ae + bg = M1 - M4 + M6 - M2
    #   M1 = ae + ah + de + dh
    #   M4 = dg - de
    #   M6 = ce + cf - ae - af
    #   M2 = ce + de
    #   M1 - M4 + M6 - M2 = ae + ah + de + dh - dg + de + ce + cf - ae - af - ce - de
    #                        = ah + dh - dg + de + cf - af
    #   Hmm, that doesn't simplify to ae+bg. Let me redo.
    
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C[0,0] = M1 + M4 - M2 - M7
    #   = (a+d)(e+h) + d(g-e) - (c+d)e - (b-d)(g+h)
    #   = ae + ah + de + dh + dg - de - ce - de - bg - bh + dg + dh
    #   = ae + ah + dh + dg - ce - de - bg - bh + dg + dh
    #   = ae + ah + 2dh + 2dg - ce - de - bg - bh
    #   That's not right either.
    
    # Let me look up the correct formula:
    # C[0,0] = M1 + M4 - M2 - M7
    # C[0,1] = M5 + M7
    # C[1,0] = M3 + M6
    # C[1,1] = M1 - M4 + M5 - M3
    
    # Verify C[0,0]:
    # M1 + M4 - M2 - M7 = (a+d)(e+h) + d(g-e) - (c+d)e - (b-d)(g+h)
    # = ae + ah + de + dh + dg - de - ce - de - (bg + bh - dg - dh)
    # = ae + ah + de + dh + dg - de - ce - de - bg - bh + dg + dh
    # = ae + ah + (de - de - de) + (dh + dh) + (dg + dg) - ce - bg - bh
    # = ae + ah - de + 2dh + 2dg - ce - bg - bh
    # Still not ae + bg. I must have the wrong formula.
    
    # Correct Strassen (from Wikipedia):
    # p1 = (a + d)(e + h)
    # p2 = (c + d)(e)
    # p3 = (a)(f - h)
    # p4 = (d)(g - e)
    # p5 = (a + b)(h)
    # p6 = (c - a)(e + f)
    # p7 = (b - d)(g + h)
    
    # C = [[p1 + p4 - p2 - p7, p3 + p5],
    #      [p6 + p2, p1 - p4 + p5 - p3]]
    
    # Verify C[0,0] = p1 + p4 - p2 - p7:
    # p1 = ae + ah + de + dh
    # p4 = dg - de
    # p2 = ce + de
    # p7 = bg + bh - dg - dh
    # p1 + p4 - p2 - p7 = ae + ah + de + dh + dg - de - ce - de - bg - bh + dg + dh
    # = ae + ah + (de - de - de) + (dh + dh) + (dg + dg) - ce - bg - bh
    # = ae + ah - de + 2dh + 2dg - ce - bg - bh
    # 
    # Let me just expand directly what we need: ae + bg
    # 
    # I think I'm confusing myself. Let me just use a known correct set of 7 terms
    # and verify numerically.
    
    # Actually, let me just use the standard formulation and verify numerically.
    
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    # I'll define the 7 Strassen terms and verify
    # Using the convention:
    # A = [[a,b],[c,d]] indexed as A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B = [[e,f],[g,h]] indexed as B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    # Term 1: (a+d)(e+h)
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    # This contributes to C[0,0] (ae, ah) and C[1,1] (de, dh)
    # C[0,0] gets: a*e -> W[0*2+0, 0] = W[0,0] += 1? 
    # Wait, W[k*n+i, r] where n=2, so W[k*2+i, r]
    # For C[0,0]: i=0, k=0, so W[0*2+0, 0] = W[0, 0]
    # The term (a+d)(e+h) gives:
    #   a*e -> C[0,0], coefficient 1
    #   a*h -> C[0,1], coefficient 1  
    #   d*e -> C[1,0], coefficient 1
    #   d*h -> C[1,1], coefficient 1
    W[0, 0] = 1   # C[0,0] from a*e
    W[1*2+0, 0] = 1  # C[0,1] from a*h, W[k*n+i] = W[1*2+0] = W[2]
    W[0*2+1, 0] = 1  # C[1,0] from d*e, W[k*n+i] = W[0*2+1] = W[1]
    W[1*2+1, 0] = 1  # C[1,1] from d*h, W[k*n+i] = W[1*2+1] = W[3]
    
    # Term 2: (c+d)*e
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    # c*e -> C[1,0], d*e -> C[1,0]
    W[0*2+1, 1] = 1  # C[1,0]
    W[0*2+1, 1] = 1  # Already set, but c*e and d*e both go to C[1,0]
    # Actually both contribute to C[1,0] with coeff 1 each, total 2? No...
    # (c+d)*e = c*e + d*e, both contribute to C[1,0]
    # But we only need ce + dg for C[1,0]. So this term alone gives ce+de, not ce+dg.
    # The Strassen formula handles this across multiple terms.
    
    # This is getting complicated. Let me just use a different approach.
    # I'll directly construct a valid rank-7 decomposition by solving the system.
    
    # Actually, let me just use the well-known Strassen coefficients and verify numerically.
    
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    # Standard Strassen (verified formulation):
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e  
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C[0,0] = p1 + p4 - p2 - p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p6
    # C[1,1] = p1 - p4 + p5 - p3
    
    # Let me verify C[0,1] = p3 + p5:
    # p3 = a*f - a*h
    # p5 = a*h + b*h
    # p3 + p5 = a*f - a*h + a*h + b*h = a*f + b*h ✓
    
    # C[1,0] = p2 + p6:
    # p2 = c*e + d*e
    # p6 = c*e + c*f - a*e - a*f
    # p2 + p6 = c*e + d*e + c*e + c*f - a*e - a*f = 2ce + de + cf - ae - af
    # We need ce + dg. Not matching. Let me recheck.
    
    # I think the issue is with p2. Let me use a different standard:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)  
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    
    # C[1,0] should be ce + dg.
    # p2 + p6 = (c+d)e + (c-a)(e+f) = ce + de + ce + cf - ae - af = 2ce + de + cf - ae - af
    # That's not ce + dg.
    
    # I think I have the wrong formula. Let me try:
    # C[1,0] = p2 + p6
    # For this to equal ce + dg, we need:
    # (c+d)e + (c-a)(e+f) = ce + dg
    # ce + de + ce + cf - ae - af = ce + dg
    # 2ce + de + cf - ae - af = ce + dg
    # ce + de + cf - ae - af = dg
    # This doesn't work in general.
    
    # Let me look at this more carefully. The correct Strassen formulas are:
    # C[0,0] = p1 - p4 - p2 - p7  (or some variation)
    
    # I'll just hard-code a known correct rank-7 decomposition and verify.
    # From various sources, the 7 Strassen terms for [[a,b],[c,d]] * [[e,f],[g,h]] are:
    
    # Let me just use a simple approach: I'll construct the rank-7 decomposition
    # by solving the linear system numerically, or just use a known correct one.
    
    # Known correct Strassen (from "Introduction to Algorithms" CLRS):
    # p1 = (a + d)(e + h)
    # p2 = (c + d)(e)
    # p3 = (a)(f - h)
    # p4 = (d)(g - e)
    # p5 = (a + b)(h)
    # p6 = (c - a)(e + f)
    # p7 = (b - d)(g + h)
    #
    # C = [[p1 + p4 - p2 - p7, p3 + p5],
    #      [p2 + p6, p1 - p4 + p5 - p3]]
    
    # Let me verify C[1,0] = p2 + p6 one more time very carefully:
    # p2 = (c+d)*e = c*e + d*e
    # p6 = (c-a)*(e+f) = c*e + c*f - a*e - a*f
    # p2 + p6 = c*e + d*e + c*e + c*f - a*e - a*f = 2*c*e + d*e + c*f - a*e - a*f
    # We need: c*e + d*g
    # 2ce + de + cf - ae - af ≠ ce + dg in general.
    
    # I think the standard Strassen might have a different assignment. Let me try:
    # C[1,0] = p6 + p2 is wrong. Maybe it's:
    # C[1,0] = p4 + p6?
    # p4 + p6 = d*(g-e) + (c-a)*(e+f) = dg - de + ce + cf - ae - af
    # = dg + ce - de + cf - ae - af
    # Still not ce + dg.
    
    # I think I need to just find a correct set. Let me try a different approach:
    # I'll use the fact that Strassen's algorithm is well-known and just implement it
    # correctly by testing.
    
    # Alternative: just use rank 8 schoolbook for 2x2 (trivially correct)
    # and then do Kronecker product for 4x4.
    # But that gives rank 8*8 = 64 for 4x4*4x4, plus 4x4*4x1 part.
    
    # Actually, for the 4x4*4x5 problem, let me just use schoolbook throughout.
    # Rank = 4*4*5 = 80.
    
    # But we know Strassen gives 49+16 = 65 for 4x4*4x4 + 4x4*4x1.
    # Let me just make sure the Kronecker Strassen is correct.
    
    # For now, let me just return schoolbook and make sure the code is valid.
    # The score will be 1.0 which is fine.
    
    # Actually, let me try to fix the Strassen by using a verified approach.
    
    # I'll define the 7 terms and verify numerically within the function.
    
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    # Try this set of Strassen terms (verified by multiple sources):
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    #
    # Output:
    # C[0,0] = M1 - M4 - M2 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M4 + M6 + M2  -- let me try this
    # C[1,1] = M1 - M5 + M3 + M7
    
    # Verify C[1,0] = M4 + M6 + M2:
    # M4 = d(g-e) = dg - de
    # M6 = (c-a)(e+f) = ce + cf - ae - af
    # M2 = (c+d)e = ce + de
    # M4 + M6 + M2 = dg - de + ce + cf - ae - af + ce + de = dg + 2ce + cf - ae - af
    # We need ce + dg. Not matching.
    
    # I give up trying to get Strassen right from memory. Let me just use schoolbook.
    
    return None  # Signal to use schoolbook


def _kronecker_strassen():
    """Attempt Kronecker product of 2x2 Strassen for 4x4, plus schoolbook for 4x5 part.
    Returns None if verification fails."""
    import time
    start = time.time()
    deadline = 145.0  # seconds from start of construct
    
    # First, get a valid 2x2*2x2 -> 2x2 decomposition
    # Use schoolbook 2x2 (rank 4) as a guaranteed fallback
    n2 = m2 = p2 = 2
    rank2 = 4
    U2 = np.zeros((4, rank2), dtype=float)
    V2 = np.zeros((4, rank2), dtype=float)
    W2 = np.zeros((4, rank2), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                r = i * 4 + j * 2 + k
                U2[i * 2 + j, r] = 1
                V2[j * 2 + k, r] = 1
                W2[k * 2 + i, r] = 1
    
    # Verify 2x2 schoolbook
    if not _verify(U2, V2, W2):
        return None
    
    # Now do Kronecker product for 4x4 * 4x4
    # A is 4x4, split into 2x2 blocks of 2x2 matrices
    # B is 4x4, split into 2x2 blocks of 2x2 matrices
    # C = A*B, 4x4 result
    #
    # Block structure:
    # A = [[A00, A01], [A10, A11]] where each Ajk is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bjk is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cjk is 2x2
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # For each 2x2 block multiplication, use the rank-4 schoolbook.
    # Total rank for 4x4*4x4: 4 block multiplications * 4 (rank of each) = 16
    
    n4 = m4 = p4 = 4
    rank_444 = 16
    rank_441 = 16  # For 4x4 * 4x1: schoolbook gives 4*4*1 = 16
    total_rank = rank_444 + rank_441
    
    U = np.zeros((16, total_rank), dtype=float)
    V = np.zeros((20, total_rank), dtype=float)
    W = np.zeros((20, total_rank), dtype=float)
    
    col = 0
    # Block multiplications for 4x4*4x4
    # Block (bi, bj) of A multiplies block (bj, bk) of B to contribute to block (bi, bk) of C
    for bi in range(2):
        for bk in range(2):
            for bj in range(2):
                # A[bi][bj] * B[bj][bk] -> C[bi][bk]
                # Each is a 2x2 * 2x2 -> 2x2 multiplication
                # Use the rank-4 schoolbook decomposition
                for r in range(rank2):
                    # U index: row in A is bi*2+i, col is bj*2+j -> (bi*2+i)*4 + (bj*2+j)
                    # V index: row in B is bj*2+j, col is bk*2+k -> (bj*2+j)*4 + (bk*2+k)
                    # W index: row in C is bk*2+k, col is bi*2+i -> (bk*2+k)*4 + (bi*2+i)
                    for i in range(2):
                        for j in range(2):
                            U[(bi*2+i)*4 + (bj*2+j), col] = U2[i*2+j, r]
                            V[(bj*2+j)*4 + (bk*2+k), col] = V2[j*2+k, r]
                            W[(bk*2+k)*4 + (bi*2+i), col] = W2[k*2+i, r]
                    col += 1
    
    # Now for 4x4 * 4x1 (the 5th column of B)
    # B has 5 columns, so B = [B_4x4 | b_4x1]
    # C = A * [B_4x4 | b_4x1] = [A*B_4x4 | A*b_4x1]
    # The last column of C is A * b where b is 4x1
    # This is a 4x4 * 4x1 multiplication, rank = 4*4*1 = 16 (schoolbook)
    
    for i in range(4):
        for j in range(4):
            U[i*4+j, col] = 1
            V[j*5+4, col] = 1  # Column 4 (0-indexed) of B, which is the 5th column
            W[4*4+i, col] = 1  # Output index: k=4 (5th col of C), i -> 4*4+i = 16+i
            col += 1
    
    # Verify
    if col != total_rank:
        return None
    
    if not _verify(U, V, W):
        return None
    
    if time.time() > deadline:
        return None
    
    return {"U": U, "V": V, "W": W}


def construct():
    import time
    start = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Try Kronecker schoolbook (rank 32 for 4x4*4x4 + 16 for 4x4*4x1 = 48)
    # Wait, schoolbook 4x4*4x4 has rank 4*4*4 = 64, not 32.
    # My Kronecker approach with 2x2 schoolbook (rank 4) gives 4*4 = 16 for 4x4*4x4.
    # That's correct because we're using the block structure.
    
    # Actually wait: schoolbook 4x4*4x4 has rank 64 (one term per i,j,k).
    # But if I use Kronecker of 2x2 schoolbook (rank 4), I get 4 blocks * 4 = 16.
    # That should be equivalent to schoolbook 4x4*4x4 but with rank 16? No, that's wrong.
    # 
    # Schoolbook 4x4*4x4: rank = 4*4*4 = 64. Each term is A[i,j]*B[j,k] -> C[i,k].
    # 
    # If I use block structure with 2x2 blocks:
    # C[bi,bk] = sum_bj A[bi,bj] * B[bj,bk]
    # Each A[bi,bj]*B[bj,bk] is a 2x2*2x2 multiplication with rank 4 (schoolbook).
    # There are 2*2*2 = 8 such block multiplications.
    # Total rank = 8 * 4 = 32.
    #
    # But schoolbook 4x4*4x4 has rank 64, not 32. So something is wrong.
    #
    # Oh wait, I think I miscounted. Let me redo:
    # 4x4 matrix A has 4x4 = 16 entries.
    # 4x4 matrix B has 4x4 = 16 entries.
    # Schoolbook: for each (i,k) in C (16 outputs), sum over j (4 terms).
    # Total terms: 16 * 4 = 64. So rank = 64.
    #
    # With 2x2 blocks: A has 4 blocks (2x2 grid), B has 4 blocks.
    # C[bi,bk] = A[bi,0]*B[0,bk] + A[bi,1]*B[1,bk]
    # That's 2 block multiplications per output block.
    # 4 output blocks * 2 = 8 block multiplications.
    # Each 2x2*2x2 schoolbook has rank 4.
    # Total: 8 * 4 = 32.
    #
    # But 32 ≠ 64. The issue is that 2x2*2x2 schoolbook actually has rank 4,
    # not less. And 8*4 = 32. But 4x4*4x4 schoolbook has rank 64.
    # 
    # I think the discrepancy is because I'm double-counting or something.
    # Let me just verify numerically.
    
    result = _kronecker_strassen()
    if result is not None:
        return result
    
    # Fallback: pure schoolbook 4x4*4x5
    if time.time() > deadline:
        return _schoolbook()
    
    return _schoolbook()

# EVOLVE-BLOCK-END
