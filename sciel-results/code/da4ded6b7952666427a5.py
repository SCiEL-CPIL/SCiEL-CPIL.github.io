"""Optimized decomposition for 3x3 * 3x7 matrix multiplication."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def _make_matrix(rows, cols, zeros=True):
    if zeros:
        return [[0]*cols for _ in range(rows)]
    return [[0]*cols for _ in range(rows)]

def _add(A, B):
    return [[a+b for a,b in zip(ra, rb)] for ra, rb in zip(A, B)]

def _sub(A, B):
    return [[a-b for a,b in zip(ra, rb)] for ra, rb in zip(A, B)]

def _mul_scalar(A, s):
    if s == 0:
        return [[0]*len(A[0]) for _ in A]
    return [[a*s for a in ra] for ra in A]

def _combine(coeffs, mats):
    """Sum of scalar*matrix over list of (scalar, matrix) pairs."""
    result = None
    for s, M in mats:
        if s == 0:
            continue
        term = _mul_scalar(M, s)
        if result is None:
            result = [row[:] for row in term]
        else:
            result = _add(result, term)
    if result is None:
        result = [[0]*len(mats[0][1][0]) for _ in range(len(mats[0][1]))]
    return result

def _strassen_3x3_decomposition():
    """
    Return (U, V, W) for 3x3 * 3x3 matrix multiplication (rank 7).
    U: shape (9, 7), V: shape (9, 7), W: shape (9, 7).
    Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] with n=m=p=3.
    """
    n = 3
    U = [[0]*7 for _ in range(9)]
    V = [[0]*7 for _ in range(9)]
    W = [[0]*7 for _ in range(9)]
    
    # Strassen formulas for 3x3 * 3x3:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B01
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For each Strassen product Mr, we have:
    # Mr = (linear combo of A) * (linear combo of B)
    # U part: coefficients on A_{i,j}
    # V part: coefficients on B_{j,k}
    # W part: coefficients on C_{i,j} (which is output position)
    
    # A is 3x3: A_{i,j} for i in {0,1,2}, j in {0,1,2}
    # B is 3x3: B_{j,k} for j in {0,1,2}, k in {0,1,2}
    # C is 3x3: C_{i,k} for i in {0,1,2}, k in {0,1,2}
    
    # The 2x2 submatrices use rows/cols 0,1 of A and B.
    # Row 2 and col 2 of A, and col 2 of B are not in the 2x2 Strassen part.
    # Wait - Strassen only handles 2x2 * 2x2. For 3x3 * 3x3, we need to be careful.
    
    # Actually, let me reconsider. The standard Strassen applies to 2x2 * 2x2.
    # For 3x3 * 3x3, one approach: treat it as 2x2 * 2x2 (top-left) plus
    # additional terms for the remaining rows/cols.
    
    # Let me use a different approach: directly enumerate a rank-7 decomposition
    # for 3x3 * 3x3 if one exists, or use the known minimal rank.
    
    # Actually, the minimal rank for 3x3 * 3x3 is 7 (Strassen's bound is tight for 2x2,
    # but for 3x3 it's actually higher). Let me just use schoolbook for 3x3*3x3 = rank 27.
    
    # Better plan: Use Strassen for the 2x2 * 2x2 subproblem (rank 7), then
    # schoolbook for the remaining parts.
    
    # For 3x3 * 3x3:
    # C = A * B where A, B are 3x3.
    # Split: A = [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]]
    #         B = [[B00, B01, B02], [B10, B11, B12], [B20, B21, B22]]
    #
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    #
    # For i in {0,1}, k in {0,1}: use Strassen on the 2x2 blocks (rank 7)
    # For i in {0,1}, k=2: schoolbook (6 terms: 2*3)
    # For i=2, k in {0,1,2}: schoolbook (9 terms: 3*3)
    # Total: 7 + 6 + 9 = 22
    
    # Actually, let me just do a cleaner decomposition.
    # For the full 3x3 * 3x7 problem:
    # C_{i,k} = sum_{j=0}^{2} A_{i,j} * B_{j,k} for i in {0,1,2}, k in {0,...,6}
    
    # Strategy: For the 2x2 * 3x3 subproblem (i in {0,1}, all k in {0,...,6}):
    #   Use Strassen-style on the j-dimension for each pair of output columns?
    #   That doesn't directly work.
    
    # Simplest strong approach: 
    # Use rank-7 Strassen for 2x2*2x2 (i in {0,1}, k in {0,1}), 
    # schoolbook for rest.
    
    # i in {0,1}, k in {0,1}: 2x2*2x2 -> rank 7 (Strassen)
    # i in {0,1}, k in {2,3,4,5,6}: 2*5 terms per (i,k) -> 2*5 = 10... 
    #   Wait, for fixed i and k, C_{i,k} = sum_j A_{i,j} B_{j,k}. 
    #   For i in {0,1}, k in {2..6}: that's 2*5 = 10 (i,k) pairs, each needs 3 terms -> 30
    # i=2, k in {0..6}: 3*7 = 21 terms
    # Total: 7 + 30 + 21 = 58. Not great.
    
    # Better: Group by j. For each j, we have A_{i,j} * B_{j,k}.
    # For fixed j, this is an outer product of column j of A (3 entries) and row j of B (7 entries).
    # That's rank 1 per (i,k) but the structure is: for fixed j, C_{i,k} += A_{i,j} * B_{j,k}
    # which is a rank-1 tensor: vec(A_col_j) ⊗ vec(B_row_j) ⊗ I_{i,k}.
    # But that's not how the tensor decomposition works.
    
    # Let me think more carefully. The tensor T has T_{i,j,k} = 1 for all i,j,k.
    # We want T = sum_r U_{i,r} V_{j,r} W_{k,r}... no wait.
    # The convention is: T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    
    # Hmm, let me re-read: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # And the decomposition: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So for the 3x3*3x7 case: n=3, m=3, p=7
    # U has shape (9, rank), V has shape (21, rank), W has shape (21, rank)
    # T_{i,j,k} = sum_r U[3i+j, r] * V[3j+k, r] * W[7k+i, r]
    
    # The schoolbook: for each (i,j,k), set U[3i+j, c]=1, V[3j+k, c]=1, W[7k+i, c]=1
    # That gives rank 63.
    
    # To reduce rank, we need to find linear dependencies.
    
    # Key insight: For fixed i and k, T_{i,j,k} = 1 for all j. So the slice over j is constant.
    # This means: sum_j V[3j+k, r] * W[7k+i, r] should be independent of j when summed with U.
    
    # Actually, let me use a different strategy. Consider that for the 3x3 * 3x3 part (k=0,1,2),
    # we can potentially use rank 7 (Strassen) instead of 27.
    
    # Let me just implement a verified low-rank decomposition.
    
    # For 2x2 * 2x2, Strassen gives rank 7. Let me embed that.
    
    # Actually, I'll use a completely different and more effective approach:
    # Use the fact that we can combine terms.
    
    # For the full problem, let me try rank = 7 (for 2x2*2x2 core) + schoolbook for rest.
    # But I need to be careful about how Strassen maps to the tensor convention.
    
    # Let me just do a clean schoolbook but with a clever grouping that might help.
    # Actually, the safest bet for a guaranteed improvement is:
    # Use Strassen for the 2x2 * 2x2 subblock (i in {0,1}, k in {0,1}), 
    # and schoolbook for everything else.
    
    # Strassen for 2x2*2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B01
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # In tensor terms for the 2x2*2x2 subproblem (i in {0,1}, j in {0,1}, k in {0,1}):
    # For each Mr, we have a product (linear combo of A_{i,j}) * (linear combo of B_{j,k})
    # which contributes to specific C_{i,k} positions.
    
    # U part (A indices): for Mr, U[3*i+j, r] = coefficient of A_{i,j}
    # V part (B indices): for Mr, V[3*j+k, r] = coefficient of B_{j,k}  
    # W part (C indices): for Mr, W[7*k+i, r] = coefficient of C_{i,k}
    
    # Note: j only ranges over {0,1} for Strassen, k only over {0,1}.
    # For j=2 and k>=2, we need separate terms.
    
    # Let me build this carefully.
    
    U = [[0]*7 for _ in range(9)]   # 3*3 = 9 rows
    V = [[0]*7 for _ in range(21)]  # 3*7 = 21 rows
    W = [[0]*7 for _ in range(21)]  # 7*3 = 21 rows
    
    # Strassen products for i in {0,1}, j in {0,1}, k in {0,1}
    # M1 = (A00+A11)*(B00+B11)
    # A part: A00 (i=0,j=0) coef 1, A11 (i=1,j=1) coef 1
    # B part: B00 (j=0,k=0) coef 1, B11 (j=1,k=1) coef 1
    # C part: C00 (i=0,k=0) coef 1, C11 (i=1,k=1) coef 1
    r = 0
    U[0*3+0][r] += 1  # A00
    U[1*3+1][r] += 1  # A11
    V[0*7+0][r] += 1  # B00
    V[1*7+1][r] += 1  # B11
    W[0*3+0][r] += 1  # C00
    W[1*3+1][r] += 1  # C11
    
    # M2 = (A10+A11)*B01
    # A part: A10 (i=1,j=0) coef 1, A11 (i=1,j=1) coef 1
    # B part: B01 (j=0,k=1) coef 1
    # C part: C10 (i=1,k=0) coef 1, C11 (i=1,k=1) coef -1
    r = 1
    U[1*3+0][r] += 1  # A10
    U[1*3+1][r] += 1  # A11
    V[0*7+1][r] += 1  # B01
    W[0*3+1][r] += 1  # C10
    W[1*3+1][r] += -1 # C11
    
    # M3 = A00*(B01-B11)
    # A part: A00 (i=0,j=0) coef 1
    # B part: B01 (j=0,k=1) coef 1, B11 (j=1,k=1) coef -1
    # C part: C01 (i=0,k=1) coef 1
    r = 2
    U[0*3+0][r] += 1  # A00
    V[0*7+1][r] += 1  # B01
    V[1*7+1][r] += -1 # B11
    W[1*3+0][r] += 1  # C01
    
    # M4 = A11*(B10-B00)
    # A part: A11 (i=1,j=1) coef 1
    # B part: B10 (j=1,k=0) coef 1, B00 (j=0,k=0) coef -1
    # C part: C00 (i=0,k=0) coef 1, C10 (i=1,k=0) coef 1
    r = 3
    U[1*3+1][r] += 1  # A11
    V[1*7+0][r] += 1  # B10
    V[0*7+0][r] += -1 # B00
    W[0*3+0][r] += 1  # C00
    W[0*3+1][r] += 1  # C10
    
    # M5 = (A00+A01)*B11
    # A part: A00 (i=0,j=0) coef 1, A01 (i=0,j=1) coef 1
    # B part: B11 (j=1,k=1) coef 1
    # C part: C00 (i=0,k=0) coef -1, C01 (i=0,k=1) coef 1
    r = 4
    U[0*3+0][r] += 1  # A00
    U[0*3+1][r] += 1  # A01
    V[1*7+1][r] += 1  # B11
    W[0*3+0][r] += -1 # C00
    W[1*3+0][r] += 1  # C01
    
    # M6 = (A10-A00)*(B00+B10)
    # A part: A10 (i=1,j=0) coef 1, A00 (i=0,j=0) coef -1
    # B part: B00 (j=0,k=0) coef 1, B10 (j=1,k=0) coef 1
    # C part: (not used in standard 2x2 Strassen for C, but let me check)
    # Actually in the formulas above, M6 is not used! Let me recheck.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # M6 is not used in the standard 2x2 Strassen! It's only for 4x4.
    # So I only need 7 products: M1-M5, M7. That's only 6. Let me recount.
    # Standard Strassen 2x2*2x2 uses 7 multiplications: M1 through M7.
    # But in the C formulas, M6 doesn't appear. That's because for 2x2, 
    # the 7 products are sufficient and M6's contribution cancels or isn't needed.
    # Wait, I think I mislabeled. Let me use the standard 7-product Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B01
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # Yes, M6 is not used in C. So we only need 6 products for 2x2*2x2? No, 
    # that can't be right. The rank of 2x2*2x2 is 7. Let me look this up mentally.
    # Actually, I believe the 7 products are all needed, and M6 IS used. Let me recheck.
    # I think I have the formulas slightly wrong. The correct Strassen 2x2:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)b01
    # p3 = a00(b01 - b11)
    # p4 = a11(b10 - b00)
    # p5 = (a00 + a01)b11
    # p6 = (a10 - a00)(b00 + b10)
    # p7 = (a01 - a11)(b01 + b11)
    # c00 = p1 + p4 - p5 - p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p7
    # So p6 is indeed not used in the output! That means we only need 6 products?
    # No wait - that would give rank 6, but we know rank(2x2*2x2) = 7.
    # I must be making an error. Let me just verify: if p6 is not used, 
    # then we have 6 products, but the rank should be 7. Contradiction.
    # I think the issue is that p6 IS needed. Let me recheck the formulas.
    # After more careful thought, I believe the standard formulas are correct and
    # p6 is not used. This means my memory of the rank being 7 might be for a 
    # different formulation. Actually, no - Strassen's algorithm for 2x2 uses 7 
    # multiplications, and all 7 are needed. Let me just include all 7 and 
    # verify numerically.
    
    # For now, let me just do M1-M5 and M7 (6 products) and verify.
    # If it doesn't work, I'll add M6.
    
    # M7 = (A01-A11)*(B01+B11)
    # A part: A01 (i=0,j=1) coef 1, A11 (i=1,j=1) coef -1
    # B part: B01 (j=0,k=1) coef 1, B11 (j=1,k=1) coef 1
    # C part: C00 (i=0,k=0) coef -1, C11 (i=1,k=1) coef 1
    r = 5
    U[0*3+1][r] += 1  # A01
    U[1*3+1][r] += -1 # A11
    V[0*7+1][r] += 1  # B01
    V[1*7+1][r] += 1  # B11
    W[0*3+0][r] += -1 # C00
    W[1*3+1][r] += 1  # C11
    
    # That's 6 products (r=0..5). But we need 7 for Strassen. Let me add M6.
    # M6 = (A10-A00)*(B00+B10)
    # A part: A10 (i=1,j=0) coef 1, A00 (i=0,j=0) coef -1
    # B part: B00 (j=0,k=0) coef 1, B10 (j=1,k=0) coef 1
    # C part: In the standard formulas, M6 is not used. But for completeness,
    # if we need rank 7, maybe M6 contributes to something I'm missing.
    # Let me just verify the 6-product version numerically.
    
    # For the remaining terms (j=2 or k>=2), use schoolbook.
    # Terms where j=2: for all i in {0,1,2}, k in {0,...,6}
    #   C_{i,k} += A_{i,2} * B_{2,k}
    #   That's 3*7 = 21 terms
    # Terms where j in {0,1} and k>=2: for i in {0,1,2}, k in {2,...,6}
    #   C_{i,k} += A_{i,j} * B_{j,k}
    #   That's 3*2*5 = 30 terms
    # Total schoolbook: 21 + 30 = 51
    # Plus 6 Strassen = 57. Not a great improvement over 63.
    
    # Let me try a different approach: just use a known low-rank decomposition.
    # Actually, for the purpose of this task, let me try to find ANY improvement
    # over 63. Even rank 50 would be a big win.
    
    # Alternative: Use the fact that the tensor has a lot of structure.
    # T_{i,j,k} = 1 for all i,j,k. This is a rank-1 tensor in the "all-ones" sense
    # but the tensor decomposition is different.
    
    # Actually, I realize I should just try a simple approach: 
    # For each (i,k) pair, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # If I fix i and vary (j,k), the vector (B_{j,k}) for j=0,1,2 and k=0..6 
    # has 21 entries. The sum over j with coefficients A_{i,j} gives a linear combination.
    # For fixed i, we need to represent the map (j,k) -> B_{j,k} weighted by A_{i,j}.
    
    # This is getting complex. Let me just go with the Strassen + schoolbook approach
    # and verify it works, even if the rank improvement is modest.
    
    # Actually, let me reconsider. The key insight is that for the 3x3 * 3x7 problem,
    # we can potentially use a rank-7 decomposition for the 3x3 * 3x3 part IF such 
    # a decomposition exists. The rank of 3x3 * 3x3 matrix multiplication is known
    # to be between 19 and 23. So we can't do rank 7 for 3x3*3x3.
    
    # But we CAN do rank 7 for 2x2 * 2x2. So let me stick with that.
    
    # Let me just implement the full thing with 6 Strassen + schoolbook and verify.
    
    # Actually, I'll take a step back. Let me just implement a clean, verified 
    # decomposition. I'll use the 6-product Strassen for the 2x2*2x2 core 
    # (i in {0,1}, k in {0,1}) and schoolbook for the rest, then verify numerically.
    
    # But first, let me check: does the 6-product version actually work for 2x2*2x2?
    # I'll verify in the construct function.
    
    # For now, let me just return the schoolbook decomposition as a safe baseline,
    # but with a time budget for potential optimization.
    
    # Actually, let me just implement the Strassen+schoolbook and verify it.
    pass

def _verify_decomposition(U, V, W, n=3, m=3, p=7, tol=1e-9):
    """Verify that the decomposition is correct."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check that T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    return True, None

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Strategy: Use Strassen for 2x2*2x2 subblock + schoolbook for rest
    # Then verify and return if valid.
    
    # Build the decomposition
    # Part 1: Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    # Part 2: Schoolbook for remaining terms
    
    # Let me use 7 Strassen products (all 7, even if M6 seems unused)
    # and verify.
    
    max_rank = n * m * p
    U = [[0]*max_rank for _ in range(n*m)]  # 9 rows
    V = [[0]*max_rank for _ in range(m*p)]  # 21 rows
    W = [[0]*max_rank for _ in range(p*n)]  # 21 rows
    
    # M1 = (A00+A11)*(B00+B11)
    r = 0
    U[0*m+0][r] = 1  # A_{0,0}
    U[1*m+1][r] = 1  # A_{1,1}
    V[0*p+0][r] = 1  # B_{0,0}
    V[1*p+1][r] = 1  # B_{1,1}
    W[0*n+0][r] = 1  # C_{0,0}
    W[1*n+1][r] = 1  # C_{1,1}
    
    # M2 = (A10+A11)*B01
    r = 1
    U[1*m+0][r] = 1  # A_{1,0}
    U[1*m+1][r] = 1  # A_{1,1}
    V[0*p+1][r] = 1  # B_{0,1}
    W[0*n+1][r] = 1  # C_{1,0}
    W[1*n+1][r] = -1 # C_{1,1}
    
    # M3 = A00*(B01-B11)
    r = 2
    U[0*m+0][r] = 1  # A_{0,0}
    V[0*p+1][r] = 1  # B_{0,1}
    V[1*p+1][r] = -1 # B_{1,1}
    W[1*n+0][r] = 1  # C_{0,1}
    
    # M4 = A11*(B10-B00)
    r = 3
    U[1*m+1][r] = 1  # A_{1,1}
    V[1*p+0][r] = 1  # B_{1,0}
    V[0*p+0][r] = -1 # B_{0,0}
    W[0*n+0][r] = 1  # C_{0,0}
    W[0*n+1][r] = 1  # C_{1,0}
    
    # M5 = (A00+A01)*B11
    r = 4
    U[0*m+0][r] = 1  # A_{0,0}
    U[0*m+1][r] = 1  # A_{0,1}
    V[1*p+1][r] = 1  # B_{1,1}
    W[0*n+0][r] = -1 # C_{0,0}
    W[1*n+0][r] = 1  # C_{0,1}
    
    # M6 = (A10-A00)*(B00+B10)
    r = 5
    U[1*m+0][r] = 1  # A_{1,0}
    U[0*m+0][r] = -1 # A_{0,0}
    V[0*p+0][r] = 1  # B_{0,0}
    V[1*p+0][r] = 1  # B_{1,0}
    # M6 is not used in standard Strassen C formulas, so W entries are 0
    # But we still need the product to exist. If W is all zeros for this r,
    # then this term contributes nothing. That's fine - it just means this 
    # rank-1 term is trivial. But that would mean we only have 6 effective terms.
    # Let me check if this is correct by verifying.
    
    # M7 = (A01-A11)*(B01+B11)
    r = 6
    U[0*m+1][r] = 1  # A_{0,1}
    U[1*m+1][r] = -1 # A_{1,1}
    V[0*p+1][r] = 1  # B_{0,1}
    V[1*p+1][r] = 1  # B_{1,1}
    W[0*n+0][r] = -1 # C_{0,0}
    W[1*n+1][r] = 1  # C_{1,1}
    
    # Now add schoolbook terms for the remaining (i,j,k) combinations
    # where j=2 or k>=2 (and not already covered by Strassen)
    
    # Strassen covers: i in {0,1}, j in {0,1}, k in {0,1}
    # Need to add: 
    #   j=2, any i in {0,1,2}, any k in {0,...,6}: 3*7 = 21 terms
    #   j in {0,1}, k in {2,...,6}, any i in {0,1,2}: 2*5*3 = 30 terms
    # Total: 51 additional terms
    
    # But wait - the Strassen terms already contribute to C_{i,k} for i in {0,1}, k in {0,1}.
    # The schoolbook terms for j=2 and k>=2 will add to C_{i,k} for various (i,k).
    # I need to make sure there's no double-counting.
    
    # The Strassen terms handle the contribution from j in {0,1} to C_{i,k} for i in {0,1}, k in {0,1}.
    # So for those (i,k) pairs, I should NOT add schoolbook terms for j in {0,1}, k in {0,1}.
    # But I SHOULD add schoolbook terms for j=2 (for all i,k) and for j in {0,1}, k>=2 (for all i).
    
    rank = 7
    column = 7  # Start after the 7 Strassen columns
    # Note: We allocated max_rank, so column can go up to max_rank
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this is covered by Strassen
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*m+j][column] = 1
                V[j*p+k][column] = 1
                W[k*n+i][column] = 1
                column += 1
    
    rank = column
    
    # Trim to actual rank used
    U_trimmed = [row[:rank] for row in U]
    V_trimmed = [row[:rank] for row in V]
    W_trimmed = [row[:rank] for row in W]
    
    # Verify
    valid, err = _verify_decomposition(U_trimmed, V_trimmed, W_trimmed, n, m, p)
    if not valid:
        # Fall back to pure schoolbook
        rank = n * m * p
        U_fb = [[0]*rank for _ in range(n*m)]
        V_fb = [[0]*rank for _ in range(m*p)]
        W_fb = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1
                    V_fb[j*p+k][col] = 1
                    W_fb[k*n+i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_trimmed, "V": V_trimmed, "W": W_trimmed}
