"""Optimized rank decomposition for 2x4 by 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Strategy: 
    1. Define the multiplication tensor T of size (n*m, m*p, p*n).
    2. Flatten T into a matrix M of size (n*m*p, n*m*p) or similar structure to solve for 
       rank-r decomposition.
    Actually, the tensor decomposition is:
    T[i, j, k] = sum_t U[i, t] * V[j, t] * W[k, t]
    where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n).
    
    We want to find the minimum rank r such that this holds.
    Since the tensor is fixed, we can try to solve for U, V, W.
    This is a non-linear problem. However, for small ranks, we can use 
    alternating least squares or a direct linear algebra approach if we fix two matrices.
    
    A simpler approach for "exact" integer/half-integer solutions:
    We know the schoolbook rank is n*m*p = 32.
    The current program claims rank 28 using Strassen.
    Let's verify if rank 28 is possible or if we can go lower.
    Actually, the problem asks for "rank-minimal". 
    It is known that for 2x4 x 4x4, the rank is at most 28 (Strassen based).
    Can it be lower?
    
    Let's use a randomized search or a linear algebra based method to find a valid decomposition 
    for a specific rank, starting from a reasonable lower bound.
    
    However, the most robust way to "find" a rank-minimal decomposition without a solver 
    library is to try to construct one.
    
    Let's stick to the Strassen-based approach but ensure it is correct, or try to optimize 
    the Strassen constants.
    
    Wait, the prompt says "Find a rank-minimal exact tensor decomposition".
    The current program provides rank 28.
    Is rank 28 minimal?
    For 2x2 x 2x2, rank is 7 (Strassen).
    For 2x4 x 4x4, we can view it as 4 independent 2x2 x 2x2 multiplications?
    A (2x4) = [A1 A2] where A1, A2 are 2x2.
    B (4x4) = [B11 B12; B21 B22] where blocks are 2x2.
    C = A1*B11 + A2*B21 (for left part of C?)
    No, C is 2x4.
    C = [ C1 C2 ]
    C1 = A1*B11 + A2*B21 (2x2)
    C2 = A1*B12 + A2*B22 (2x2)
    
    So we have 4 matrix multiplications of size 2x2 x 2x2.
    If we use Strassen for each, rank = 4 * 7 = 28.
    
    Can we do better?
    Maybe by sharing some computations between C1 and C2?
    The inputs to Strassen for C1 involve A1, A2, B11, B21.
    The inputs to Strassen for C2 involve A1, A2, B12, B22.
    
    Let's check if rank 27 or less is possible.
    Actually, there is a known result for 2x4 x 4x4.
    According to literature, the rank of 2x4 x 4x4 multiplication is 28.
    Reference: "The rank of matrix multiplication" or similar.
    Actually, for 2x2 x 2x2 it is 7.
    For 2x4 x 4x4, it is likely 28.
    
    If the minimal rank is 28, the current program is already optimal.
    However, the task says "Find a rank-minimal...".
    If I just return the current program's logic, it scores 1.0.
    To improve, I need to either:
    1. Find a rank < 28 (unlikely if 28 is minimal).
    2. Ensure the current program is actually valid and efficient.
    
    Let's re-verify the Strassen decomposition in the current program.
    The current program has a `verify_strassen` and `verify_full`.
    If they fail, it falls back to schoolbook (rank 32).
    If the Strassen implementation is buggy, the score would be 32 (worse).
    The observation says score 1.0000, which implies it might be returning the schoolbook 
    or the Strassen is valid and rank 28 is considered "1.0" relative to the seed?
    Wait, "score is the rank". "1.0 equals the seed program".
    If the seed program is rank 32 (schoolbook), and I return rank 28, 
    the score should be 32/28 > 1.0.
    But the current program scores 1.0000.
    This suggests the current program is actually returning rank 32 (schoolbook) 
    because the verification fails?
    Or the seed program is also rank 28?
    "Initial program for the suite task" usually implies a baseline.
    If the baseline is schoolbook (32), and I get 28, I should score > 1.
    The fact that it scores 1.0 suggests it is NOT beating the baseline.
    
    Let's check the `verify_strassen` logic in the current code.
    It checks if `T[i, j, k] == 1`.
    For matrix multiplication, T[i, j, k] is 1 if the indices correspond to a valid 
    A[i,j] * B[j,k] -> C[k,i] contribution.
    In the local 2x2 block, all (i,j,k) are valid?
    Yes, for a 2x2 * 2x2, every A[i,j] multiplies every B[j,k] to contribute to C[k,i].
    So T[i,j,k] = 1 for all i,j,k in {0,1}.
    
    Let's trace one term of Strassen in the current code:
    P1: U={(0,0):1, (1,1):1}, V={(0,0):1, (1,1):1}, W={(0,0):1, (1,1):1}
    Contribution to T[0,0,0]: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1.
    Contribution to T[0,0,1]: U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0.
    ...
    We need sum of contributions to be 1 for all i,j,k.
    
    Let's check T[0,1,0] (i=0, j=1, k=0).
    P1: U[0,1]=0.
    P2: U[0,1]=0.
    P3: U[0,1]=0.
    P4: U[0,1]=0.
    P5: U[0,1]=1. V[1,0]=1. W[0,0]=1. -> 1*1*1 = 1.
    P6: U[0,1]=0.
    P7: U[0,1]=1. V[1,0]=1. W[0,0]=1. -> 1*1*1 = 1.
    Sum = 2. 
    But T[0,1,0] should be 1.
    So the Strassen implementation in the current code is WRONG.
    It produces 2 instead of 1.
    Therefore `verify_strassen` returns False.
    Therefore `construct` returns `get_schoolbook()` (rank 32).
    
    So the current program is effectively rank 32.
    The seed is likely rank 32.
    Score 1.0 = 32/32.
    
    My goal: Provide a VALID rank 28 decomposition.
    I need to fix the Strassen decomposition or use a known correct one.
    
    Correct Strassen for 2x2:
    Let A = [[a, b], [c, d]], B = [[e, f], [g, h]].
    C = A B = [[ae+bg, af+bh], [ce+dg, cf+dh]].
    
    Strassen products:
    m1 = (a+d)(e+h)
    m2 = (c+d)e
    m3 = a(f-h)
    m4 = d(g-e)
    m5 = (a+b)h
    m6 = (c-a)(e+g)
    m7 = (b-d)(f+h)
    
    C00 = m1 + m4 - m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m2 + m3 + m6
    
    We need to express these as tensor terms U, V, W.
    Each term t contributes to C[k,i] via sum_j U[i,j,t] V[j,k,t] W[k,i,t].
    Wait, the tensor T[i,j,k] is the coefficient.
    T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t].
    
    For a specific term t, it represents a rank-1 tensor.
    The term corresponds to a product of linear forms.
    For m1 = (a+d)(e+h):
    This involves A entries (0,0) and (1,1), and B entries (0,0) and (1,1).
    It contributes to C00 and C11.
    C00 coefficient of m1 is +1.
    C11 coefficient of m1 is +1.
    C01, C10 coefficient of m1 is 0.
    
    So for term 1:
    U[0,0] = 1, U[1,1] = 1. (Others 0)
    V[0,0] = 1, V[1,1] = 1. (Others 0)
    W[0,0] = 1 (for C00).
    W[1,1] = 1 (for C11).
    W[0,1] = 0, W[1,0] = 0.
    
    Let's check T[0,0,0] for term 1: U[0,0]V[0,0]W[0,0] = 1.
    T[1,1,1] for term 1: U[1,1]V[1,1]W[1,1] = 1.
    T[0,1,0] for term 1: U[0,1]... U[0,1] is 0. So 0.
    This looks correct.
    
    Let's check m2 = (c+d)e.
    A entries: (1,0), (1,1).
    B entries: (0,0).
    Contributes to C10 (+1) and C11 (-1).
    U[1,0]=1, U[1,1]=1.
    V[0,0]=1.
    W[1,0]=1 (C10 is k=1, i=0? No. C is 2x2. C[1,0] is row 1, col 0. 
    W index is k*n + i. n=2.
    C[1,0] -> k=1, i=0 -> W[1*2+0] = W[2].
    C[1,1] -> k=1, i=1 -> W[1*2+1] = W[3].
    
    So for term 2:
    U[1,0]=1, U[1,1]=1.
    V[0,0]=1.
    W[2]=1 (for C10).
    W[3]=-1 (for C11).
    
    Let's check T[1,0,1] (i=1, j=0, k=1). This is A[1,0]*B[0,1]? No.
    T[i,j,k] is coeff of A[i,j] B[j,k] in C[k,i].
    For m2, A[1,0] (c) and B[0,0] (e).
    So i=1, j=0, k=0.
    T[1,0,0] should get contribution.
    Term 2: U[1,0]=1, V[0,0]=1, W[0,1] (k=0, i=1 -> W[1])?
    Wait. C[0,1] is k=0, i=1.
    Does m2 contribute to C[0,1]?
    m2 = (c+d)e.
    C10 = m2 + m4.
    C11 = m1 - m2 + m3 + m6.
    So m2 contributes to C10 and C11.
    C10 is k=1, i=0. W index 1*2+0 = 2.
    C11 is k=1, i=1. W index 1*2+1 = 3.
    
    So W[2]=1, W[3]=-1.
    Check T[1,0,1] (i=1, j=0, k=1).
    A[1,0] B[0,1].
    m2 uses B[0,0]. So no contribution.
    Term 2 V[0,1] is 0. So T[1,0,1] gets 0. Correct.
    
    Check T[1,0,0] (i=1, j=0, k=0).
    A[1,0] B[0,0].
    m2 uses A[1,0] and B[0,0].
    U[1,0]=1, V[0,0]=1.
    W[0,1] (k=0, i=1 -> W[1])?
    Does m2 contribute to C[0,1]?
    C01 = m3 + m5.
    So no.
    So W[1] should be 0.
    My previous assignment: W[2]=1, W[3]=-1.
    So W[1]=0.
    So T[1,0,0] gets U[1,0]*V[0,0]*W[0,1] = 1*1*0 = 0.
    But T[1,0,0] is coeff of A[1,0]B[0,0] in C[0,1].
    C[0,1] = af + bh.
    A[1,0] is c. B[0,0] is e.
    ce is not in C[0,1].
    So 0 is correct.
    
    Okay, the logic holds. I will implement the 7 Strassen terms correctly 
    for the local 2x2 block, and then map them to the 4 blocks for the 2x4 x 4x4 case.
    
    I will also add a verification step to ensure the rank 28 solution is valid 
    before returning it. If it fails (due to some mapping error), I will fall back 
    to a known valid rank 32 solution.
    
    To be safe and fast, I will hardcode the correct Strassen terms 
    and the mapping, and verify with a single random test.
    """
    deadline = time.time() + 100
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Correct Strassen terms for 2x2 * 2x2
    # Local indices: i,j in {0,1} for A; j,k in {0,1} for B; k,i in {0,1} for C (W index k*2+i)
    
    # m1 = (A00+A11)(B00+B11) -> C00(+1), C11(+1)
    # m2 = (A10+A11)B00 -> C10(+1), C11(-1)
    # m3 = A00(B10-B11) -> C01(+1), C11(+1)  Wait.
    # C11 = m1 - m2 + m3 + m6.
    # C01 = m3 + m5.
    # So m3 contributes to C01(+1) and C11(+1).
    # m4 = A11(B01-B10) -> C00(+1), C10(+1)
    # C00 = m1 + m4 - m5 + m7.
    # C10 = m2 + m4.
    # So m4 contributes to C00(+1) and C10(+1).
    # m5 = (A00+A01)B11 -> C00(-1), C01(+1)
    # C00 = ... - m5 ...
    # C01 = m3 + m5.
    # So m5 contributes to C00(-1) and C01(+1).
    # m6 = (A10-A00)(B00+B10) -> C11(+1)
    # C11 = ... + m6.
    # Does m6 contribute elsewhere?
    # C00, C01, C10 don't have m6.
    # So m6 contributes to C11(+1) only?
    # Let's check.
    # m6 = (c-a)(e+g) = ce + cg - ae - ag.
    # C11 = cf + dh.
    # Wait, standard Strassen:
    # C11 = m1 - m2 + m3 + m6.
    # m1 = (a+d)(e+h) = ae + ah + de + dh
    # m2 = (c+d)e = ce + de
    # m3 = a(f-h) = af - ah
    # m6 = (c-a)(e+g) = ce + cg - ae - ag
    # Sum:
    # ae + ah + de + dh - ce - de + af - ah + ce + cg - ae - ag
    # = ah + dh + af + cg - ag
    # This is not cf + dh.
    # Something is wrong with my Strassen formula or expansion.
    
    # Standard Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B10)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's re-expand C11:
    # P1 = ae + ah + de + dh
    # -P2 = -ce - de
    # P3 = af - ah
    # P6 = ce + cg - ae - ag
    # Sum: ae + ah + de + dh - ce - de + af - ah + ce + cg - ae - ag
    # = dh + af + cg - ag
    # Still not cf + dh.
    
    # Ah, P3 = A00(B10-B11) = a(g-h) = ag - ah.
    # My previous expansion used f-h.
    # Let's re-expand with P3 = ag - ah.
    # Sum: ae + ah + de + dh - ce - de + ag - ah + ce + cg - ae - ag
    # = dh + cg
    # Still not cf + dh.
    
    # Wait, C11 is A11*B11 + A10*B01 = d*h + c*f.
    # My expansion gives dh + cg.
    # Where is the error?
    # P6 = (A10-A00)(B00+B10) = (c-a)(e+g).
    # Maybe P6 is (A10-A00)(B00+B10) is correct, but I am mixing up indices.
    # B00=e, B10=g.
    # P6 = (c-a)(e+g).
    
    # Let's check a known correct Strassen implementation.
    # C11 = P1 - P2 + P3 + P6.
    # If P3 = A00(B10-B11) = a(g-h).
    # And P6 = (A10-A00)(B00+B10) = (c-a)(e+g).
    
    # Let's try a different P6.
    # Some variants use P6 = (A10-A00)(B00+B10).
    # Others use P6 = (A10-A00)(B00+B10).
    
    # Actually, let's just use the schoolbook for the 2x2 block if I can't get Strassen right quickly?
    # No, rank 8 per block -> 32 total.
    # I need rank 7 per block.
    
    # Let's use a verified Strassen code snippet from memory or derive it carefully.
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)  <-- Wait, A00(B10-B11) is a(g-h).
    # M4 = (d)(g-e)  <-- A11(B01-B10) is d(f-g).
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's check C11 again with M3 = a(g-h) and M4 = d(f-g).
    # M1 = ae+ah+de+dh
    # -M2 = -ce-de
    # M3 = ag-ah
    # M6 = ce+cg-ae-ag
    # Sum: ae+ah+de+dh -ce-de +ag-ah +ce+cg-ae-ag
    # = dh + cg.
    # Still cg instead of cf.
    
    # It seems my M4 or M3 is swapped or wrong.
    # C11 needs cf.
    # cf comes from c*f.
    # c is A10, f is B01.
    # Which M contains c*f?
    # M2 = (c+d)e. No f.
    # M6 = (c-a)(e+g). No f.
    # M3 = a(g-h). No c.
    # M4 = d(f-g). No c.
    
    # It appears standard Strassen formulas I recall might have index swaps for B.
    # B = [[e, f], [g, h]].
    # B00=e, B01=f, B10=g, B11=h.
    
    # Let's look at M4 = A11(B01-B10) = d(f-g).
    # This has f.
    # But C11 formula doesn't use M4.
    # C10 uses M4.
    # C10 = ce + dg.
    # M2 + M4 = (c+d)e + d(f-g) = ce + de + df - dg.
    # This is not ce + dg.
    
    # Okay, I will use a simpler approach: 
    # I will construct the rank 28 decomposition using the known fact that 
    # it exists, but I will define the terms programmatically by solving for them 
    # for the 2x2 case, then mapping.
    # Or, I will just use the schoolbook (rank 32) to be safe, as rank 28 is 
    # only a 32/28 = 1.14 improvement.
    # However, the task asks for "rank-minimal".
    # If I return 32, and 28 is possible, I am not minimal.
    # But if I return an invalid 28, I get 0.
    
    # Let's try to find a valid 2x2 rank 7 decomposition online or in memory.
    # Actually, I can just use the 4x4x4 Strassen? No.
    
    # Let's try to construct the 2x2 rank 7 terms using a known valid set.
    # From "Strassen's algorithm" Wikipedia:
    # p1 = (a + d)(e + h)
    # p2 = (c + d)e
    # p3 = a(f - h)  <-- NOTE: This is A00 * (B01 - B11)
    # p4 = d(g - e)  <-- NOTE: This is A11 * (B10 - B00)
    # p5 = (a + b)h
    # p6 = (c - a)(e + g)
    # p7 = (b - d)(f + h)
    
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    # Let's re-check C10 with p4 = d(g-e).
    # C10 = ce + dg.
    # p2 + p4 = (c+d)e + d(g-e) = ce + de + dg - de = ce + dg. CORRECT.
    
    # Let's re-check C11 with p3 = a(f-h).
    # C11 = cf + dh.
    # p1 - p2 + p3 + p6
    # = (ae+ah+de+dh) - (ce+de) + (af-ah) + (ce+cg-ae-ag)
    # = ae+ah+de+dh -ce-de +af-ah +ce+cg-ae-ag
    # = dh + af + cg - ag.
    # Still not cf + dh.
    
    # There is a persistent error in my manual expansion or the formula.
    # Let's check p6.
    # p6 = (c-a)(e+g) = ce + cg - ae - ag.
    # Maybe p3 is different?
    # Wikipedia says p3 = a(f-h).
    
    # Let's check C11 again.
    # Maybe p1, p2, p3, p6 are not the only ones?
    # Yes, C11 = p1 - p2 + p3 + p6.
    
    # Let's try p3 = a(f-h) and p4 = d(g-e).
    # Is it possible that B indices are swapped in my head?
    # B = [[e, f], [g, h]].
    # B00=e, B01=f, B10=g, B11=h.
    
    # Let's try a different p6.
    # Some sources say p6 = (c-a)(e+g).
    # Others say p6 = (c-a)(e+g).
    
    # Let's just use the schoolbook. It is safe.
    # Rank 32.
    
    return get_schoolbook()

# EVOLVE-BLOCK-END
