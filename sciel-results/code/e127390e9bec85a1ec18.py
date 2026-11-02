"""Strassen-based rank-97 decomposition for 4x4 * 4x7 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _build_strassen_4x4x4():
    """
    Build the 49-term Strassen decomposition for 4x4 * 4x4 matrix multiplication.
    
    Uses nested Strassen: outer Strassen splits 4x4 into 2x2 blocks of 2x2 matrices,
    inner Strassen handles each 2x2 * 2x2 multiplication with rank 7.
    
    Returns list of 49 terms, each a dict with 'U', 'V', 'W' entries.
    U keys: (i, j) for A[i,j], i in 0..3, j in 0..3
    V keys: (j, k) for B[j,k], j in 0..3, k in 0..3
    W keys: (i, k) for C[i,k], i in 0..3, k in 0..3
    """
    terms = []
    
    # Outer Strassen: 7 combinations of 2x2 blocks
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Each block is 2x2.
    # A00[i,j] = A[2i, 2j], A01[i,j] = A[2i, 2j+1], A10[i,j] = A[2i+1, 2j], A11[i,j] = A[2i+1, 2j+1]
    # B00[i,j] = B[2i, 2j], B01[i,j] = B[2i, 2j+1], B10[i,j] = B[2i+1, 2j], B11[i,j] = B[2i+1, 2j+1]
    
    # Outer Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A10-A00)*(B00+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each outer M, we need to compute P*Q where P and Q are 2x2 matrices
    # (linear combinations of A blocks and B blocks respectively).
    # Then we use inner Strassen (rank 7) to decompose P*Q.
    
    # Inner Strassen for 2x2 * 2x2:
    # N1 = (P[0,0]+P[1,1])*(Q[0,0]+Q[1,1])
    # N2 = (P[1,0]+P[1,1])*Q[0,0]
    # N3 = P[0,0]*(Q[0,1]-Q[1,1])
    # N4 = P[1,1]*(Q[1,0]-Q[0,0])
    # N5 = (P[0,0]+P[1,0])*Q[1,1]
    # N6 = (P[0,1]-P[1,1])*(Q[1,0]+Q[1,1])
    # N7 = (P[1,0]-P[0,0])*(Q[0,0]+Q[0,1])
    
    # R[0,0] = N1 + N4 - N5 + N7
    # R[0,1] = N3 + N5
    # R[1,0] = N2 + N4
    # R[1,1] = N1 - N2 + N3 + N6
    
    # Each N is a product of two linear combinations of 2x2 entries.
    # For the tensor decomposition, each N contributes rank 1 terms.
    # Actually, each N = (sum of a's) * (sum of b's) where a's are P entries and b's are Q entries.
    # This expands to multiple rank-1 terms.
    
    # Wait, I need to think about this more carefully.
    # The tensor rank of 2x2x2 is 7, and Strassen achieves it with 7 terms.
    # Each Strassen term is of the form:
    #   (linear combo of A entries) * (linear combo of B entries) -> contributes to (linear combo of C entries)
    
    # For the inner Strassen, let P be the left 2x2 matrix and Q be the right 2x2 matrix.
    # P[a,b] and Q[b,c] are entries of these 2x2 matrices.
    # But P and Q themselves are linear combinations of the original A and B entries.
    
    # Let me define the outer Strassen combinations more explicitly.
    
    # For outer M1: P = A00 + A11, Q = B00 + B11
    # P[a,b] = A[2a, 2b] + A[2a+1, 2b+1]
    # Q[b,c] = B[2b, 2c] + B[2b+1, 2c+1]
    
    # For outer M2: P = A10 + A11, Q = B00
    # P[a,b] = A[2a+1, 2b] + A[2a+1, 2b+1]
    # Q[b,c] = B[2b, 2c]
    
    # For outer M3: P = A00, Q = B01 - B11
    # P[a,b] = A[2a, 2b]
    # Q[b,c] = B[2b, 2c+1] - B[2b+1, 2c+1]
    
    # For outer M4: P = A11, Q = B10 - B00
    # P[a,b] = A[2a+1, 2b+1]
    # Q[b,c] = B[2b+1, 2c] - B[2b, 2c]
    
    # For outer M5: P = A00 + A10, Q = B11
    # P[a,b] = A[2a, 2b] + A[2a+1, 2b]
    # Q[b,c] = B[2b+1, 2c+1]
    
    # For outer M6: P = A01 - A11, Q = B10 + B11
    # P[a,b] = A[2a, 2b+1] - A[2a+1, 2b+1]
    # Q[b,c] = B[2b+1, 2c] + B[2b+1, 2c+1]
    
    # For outer M7: P = A10 - A00, Q = B00 + B01
    # P[a,b] = A[2a+1, 2b] - A[2a, 2b]
    # Q[b,c] = B[2b, 2c] + B[2b, 2c+1]
    
    # Now, for each outer M, P[a,b] is a linear combo of A entries, and Q[b,c] is a linear combo of B entries.
    # The product M = P*Q gives a 2x2 matrix R where R[a,c] = sum_b P[a,b]*Q[b,c].
    # Each R[a,c] is a linear combination of products of A and B entries.
    
    # For the tensor decomposition, I need to express the full 4x4x4 tensor as a sum of rank-1 terms.
    # A rank-1 term is: U[i,j] * V[j,k] * W[i,k] where U, V, W are vectors.
    
    # Let me use the inner Strassen to decompose each outer M into 7 rank-1 terms.
    # For the inner Strassen, the "A" matrix is P (2x2) and the "B" matrix is Q (2x2).
    # But P and Q are not individual entries; they are linear combinations of the original A and B entries.
    
    # So for each inner Strassen term N, I need to expand it into rank-1 terms of the original A and B.
    
    # Let me define the 7 inner Strassen combinations for P and Q:
    # N1 = (P[0,0]+P[1,1])*(Q[0,0]+Q[1,1])
    # N2 = (P[1,0]+P[1,1])*Q[0,0]
    # N3 = P[0,0]*(Q[0,1]-Q[1,1])
    # N4 = P[1,1]*(Q[1,0]-Q[0,0])
    # N5 = (P[0,0]+P[1,0])*Q[1,1]
    # N6 = (P[0,1]-P[1,1])*(Q[1,0]+Q[1,1])
    # N7 = (P[1,0]-P[0,0])*(Q[0,0]+Q[0,1])
    
    # Each N is a product of two linear combinations.
    # For example, N1 = (P[0,0]+P[1,1])*(Q[0,0]+Q[1,1])
    # = P[0,0]*Q[0,0] + P[0,0]*Q[1,1] + P[1,1]*Q[0,0] + P[1,1]*Q[1,1]
    
    # But P[0,0] itself is a linear combo of A entries, and Q[0,0] is a linear combo of B entries.
    # So N1 expands to multiple products of A and B entries.
    
    # For the tensor decomposition, each product of a linear combo of A and a linear combo of B
    # that contributes to a specific C entry gives a rank-1 term.
    
    # Actually, I think the cleanest way is to just enumerate all the rank-1 terms directly.
    # For the 4x4x4 Strassen, there are 49 rank-1 terms total.
    
    # Let me define them explicitly.
    
    # For each outer M (0..6) and each inner Strassen term (0..6), I have one rank-1 term.
    # The U, V, W vectors for each term are determined by the outer and inner Strassen formulas.
    
    # Let me define the outer Strassen P and Q for each M:
    # M=0: P = A00+A11, Q = B00+B11
    # M=1: P = A10+A11, Q = B00
    # M=2: P = A00, Q = B01-B11
    # M=3: P = A11, Q = B10-B00
    # M=4: P = A00+A10, Q = B11
    # M=5: P = A01-A11, Q = B10+B11
    # M=6: P = A10-A00, Q = B00+B01
    
    # For each M, P is a 2x2 matrix where P[a,b] is a linear combo of A entries.
    # Q is a 2x2 matrix where Q[b,c] is a linear combo of B entries.
    
    # Let me define P and Q as dictionaries mapping (a,b) to a dict of A/B entries with coefficients.
    
    # For M=0:
    # P[0,0] = A[0,0] + A[1,1]
    # P[0,1] = A[0,1] + A[1,2]
    # P[1,0] = A[2,0] + A[3,1]
    # P[1,1] = A[2,1] + A[3,2]
    
    # Q[0,0] = B[0,0] + B[1,1]
    # Q[0,1] = B[0,1] + B[1,2]
    # Q[1,0] = B[2,0] + B[3,1]
    # Q[1,1] = B[2,1] + B[3,2]
    
    # Wait, I need to be more careful with the indexing.
    # A00[i,j] = A[2i, 2j], so A00[0,0] = A[0,0], A00[0,1] = A[0,2], A00[1,0] = A[2,0], A00[1,1] = A[2,2]
    # A11[i,j] = A[2i+1, 2j+1], so A11[0,0] = A[1,1], A11[0,1] = A[1,3], A11[1,0] = A[3,1], A11[1,1] = A[3,3]
    
    # So P = A00 + A11:
    # P[0,0] = A[0,0] + A[1,1]
    # P[0,1] = A[0,2] + A[1,3]
    # P[1,0] = A[2,0] + A[3,1]
    # P[1,1] = A[2,2] + A[3,3]
    
    # Q = B00 + B11:
    # Q[0,0] = B[0,0] + B[1,1]
    # Q[0,1] = B[0,2] + B[1,3]
    # Q[1,0] = B[2,0] + B[3,1]
    # Q[1,1] = B[2,2] + B[3,3]
    
    # OK, this is getting very detailed. Let me just code it up directly.
    
    # Define the 7 outer Strassen P and Q matrices.
    # Each P[a,b] is a dict: {(i,j): coeff} for A[i,j]
    # Each Q[b,c] is a dict: {(j,k): coeff} for B[j,k]
    
    outer_P = []
    outer_Q = []
    
    # M=0: P = A00+A11, Q = B00+B11
    P0 = {
        (0,0): {((0,0), 1), ((1,1), 1)},
        (0,1): {((0,2), 1), ((1,3), 1)},
        (1,0): {((2,0), 1), ((3,1), 1)},
        (1,1): {((2,2), 1), ((3,3), 1)},
    }
    Q0 = {
        (0,0): {((0,0), 1), ((1,1), 1)},
        (0,1): {((0,2), 1), ((1,3), 1)},
        (1,0): {((2,0), 1), ((3,1), 1)},
        (1,1): {((2,2), 1), ((3,3), 1)},
    }
    outer_P.append(P0)
    outer_Q.append(Q0)
    
    # M=1: P = A10+A11, Q = B00
    P1 = {
        (0,0): {((2,0), 1), ((3,1), 1)},  # A10[0,0]=A[2,0], A11[0,0]=A[1,1]... wait
    }
    
    # Let me redo this more carefully.
    # A10[i,j] = A[2i+1, 2j], A11[i,j] = A[2i+1, 2j+1]
    # A10[0,0] = A[1,0], A10[0,1] = A[1,2], A10[1,0] = A[3,0], A10[1,1] = A[3,2]
    # A11[0,0] = A[1,1], A11[0,1] = A[1,3], A11[1,0] = A[3,1], A11[1,1] = A[3,3]
    
    # P = A10+A11:
    # P[0,0] = A[1,0] + A[1,1]
    # P[0,1] = A[1,2] + A[1,3]
    # P[1,0] = A[3,0] + A[3,1]
    # P[1,1] = A[3,2] + A[3,3]
    
    P1 = {
        (0,0): {((1,0), 1), ((1,1), 1)},
        (0,1): {((1,2), 1), ((1,3), 1)},
        (1,0): {((3,0), 1), ((3,1), 1)},
        (1,1): {((3,2), 1), ((3,3), 1)},
    }
    Q1 = {
        (0,0): {((0,0), 1)},
        (0,1): {((0,2), 1)},
        (1,0): {((2,0), 1)},
        (1,1): {((2,2), 1)},
    }
    outer_P.append(P1)
    outer_Q.append(Q1)
    
    # M=2: P = A00, Q = B01-B11
    P2 = {
        (0,0): {((0,0), 1)},
        (0,1): {((0,2), 1)},
        (1,0): {((2,0), 1)},
        (1,1): {((2,2), 1)},
    }
    Q2 = {
        (0,0): {((0,1), 1), ((1,3), -1)},
        (0,1): {((0,3), 1), ((1,1), -1)},  # Wait, B01[i,j]=B[2i,2j+1], B11[i,j]=B[2i+1,2j+1]
        # B01[0,0]=B[0,1], B01[0,1]=B[0,3], B01[1,0]=B[2,1], B01[1,1]=B[2,3]
        # B11[0,0]=B[1,1], B11[0,1]=B[1,3], B11[1,0]=B[3,1], B11[1,1]=B[3,3]
        # Q = B01-B11:
        # Q[0,0] = B[0,1] - B[1,1]
        # Q[0,1] = B[0,3] - B[1,3]
        # Q[1,0] = B[2,1] - B[3,1]
        # Q[1,1] = B[2,3] - B[3,3]
    }
    Q2 = {
        (0,0): {((0,1), 1), ((1,1), -1)},
        (0,1): {((0,3), 1), ((1,3), -1)},
        (1,0): {((2,1), 1), ((3,1), -1)},
        (1,1): {((2,3), 1), ((3,3), -1)},
    }
    outer_P.append(P2)
    outer_Q.append(Q2)
    
    # M=3: P = A11, Q = B10-B00
    P3 = {
        (0,0): {((1,1), 1)},
        (0,1): {((1,3), 1)},
        (1,0): {((3,1), 1)},
        (1,1): {((3,3), 1)},
    }
    # B10[i,j]=B[2i+1,2j], B00[i,j]=B[2i,2j]
    # B10[0,0]=B[1,0], B10[0,1]=B[1,2], B10[1,0]=B[3,0], B10[1,1]=B[3,2]
    # B00[0,0]=B[0,0], B00[0,1]=B[0,2], B00[1,0]=B[2,0], B00[1,1]=B[2,2]
    # Q = B10-B00:
    # Q[0,0] = B[1,0] - B[0,0]
    # Q[0,1] = B[1,2] - B[0,2]
    # Q[1,0] = B[3,0] - B[2,0]
    # Q[1,1] = B[3,2] - B[2,2]
    Q3 = {
        (0,0): {((1,0), 1), ((0,0), -1)},
        (0,1): {((1,2), 1), ((0,2), -1)},
        (1,0): {((3,0), 1), ((2,0), -1)},
        (1,1): {((3,2), 1), ((2,2), -1)},
    }
    outer_P.append(P3)
    outer_Q.append(Q3)
    
    # M=4: P = A00+A10, Q = B11
    P4 = {
        (0,0): {((0,0), 1), ((1,0), 1)},
        (0,1): {((0,2), 1), ((1,2), 1)},
        (1,0): {((2,0), 1), ((3,0), 1)},
        (1,1): {((2,2), 1), ((3,2), 1)},
    }
    Q4 = {
        (0,0): {((1,1), 1)},
        (0,1): {((1,3), 1)},
        (1,0): {((3,1), 1)},
        (1,1): {((3,3), 1)},
    }
    outer_P.append(P4)
    outer_Q.append(Q4)
    
    # M=5: P = A01-A11, Q = B10+B11
    P5 = {
        (0,0): {((0,1), 1), ((1,1), -1)},
        (0,1): {((0,3), 1), ((1,3), -1)},
        (1,0): {((2,1), 1), ((3,1), -1)},
        (1,1): {((2,3), 1), ((3,3), -1)},
    }
    Q5 = {
        (0,0): {((1,0), 1), ((1,1), 1)},
        (0,1): {((1,2), 1), ((1,3), 1)},
        (1,0): {((3,0), 1), ((3,1), 1)},
        (1,1): {((3,2), 1), ((3,3), 1)},
    }
    outer_P.append(P5)
    outer_Q.append(Q5)
    
    # M=6: P = A10-A00, Q = B00+B01
    P6 = {
        (0,0): {((1,0), 1), ((0,0), -1)},
        (0,1): {((1,2), 1), ((0,2), -1)},
        (1,0): {((3,0), 1), ((2,0), -1)},
        (1,1): {((3,2), 1), ((2,2), -1)},
    }
    Q6 = {
        (0,0): {((0,0), 1), ((0,1), 1)},
        (0,1): {((0,2), 1), ((0,3), 1)},
        (1,0): {((2,0), 1), ((2,1), 1)},
        (1,1): {((2,2), 1), ((2,3), 1)},
    }
    outer_P.append(P6)
    outer_Q.append(Q6)
    
    # Now, for each outer M, apply inner Strassen.
    # Inner Strassen for P*Q:
    # N1 = (P[0,0]+P[1,1])*(Q[0,0]+Q[1,1])
    # N2 = (P[1,0]+P[1,1])*Q[0,0]
    # N3 = P[0,0]*(Q[0,1]-Q[1,1])
    # N4 = P[1,1]*(Q[1,0]-Q[0,0])
    # N5 = (P[0,0]+P[1,0])*Q[1,1]
    # N6 = (P[0,1]-P[1,1])*(Q[1,0]+Q[1,1])
    # N7 = (P[1,0]-P[0,0])*(Q[0,0]+Q[0,1])
    
    # R[0,0] = N1 + N4 - N5 + N7
    # R[0,1] = N3 + N5
    # R[1,0] = N2 + N4
    # R[1,1] = N1 - N2 + N3 + N6
    
    # Each N is a product of two linear combos.
    # For the tensor decomposition, each N contributes to specific C entries.
    # The C entries for the outer M are determined by the outer Strassen formula.
    
    # For outer M=0 (M1): contributes to C00 with +1 and C11 with +1
    # C00[a,c] = C[2a, 2c], C11[a,c] = C[2a+1, 2c+1]
    
    # For outer M=1 (M2): contributes to C10 with +1 and C11 with -1
    # C10[a,c] = C[2a+1, 2c], C11[a,c] = C[2a+1, 2c+1]
    
    # For outer M=2 (M3): contributes to C01 with +1 and C11 with +1
    # C01[a,c] = C[2a, 2c+1], C11[a,c] = C[2a+1, 2c+1]
    
    # For outer M=3 (M4): contributes to C00 with +1 and C10 with +1
    # C00[a,c] = C[2a, 2c], C10[a,c] = C[2a+1, 2c]
    
    # For outer M=4 (M5): contributes to C00 with -1 and C01 with +1
    # C00[a,c] = C[2a, 2c], C01[a,c] = C[2a, 2c+1]
    
    # For outer M=5 (M6): contributes to C11 with +1
    # C11[a,c] = C[2a+1, 2c+1]
    
    # For outer M=6 (M7): contributes to C00 with +1
    # C00[a,c] = C[2a, 2c]
    
    # Now, for each inner Strassen term N, it contributes to R[a,c] with a specific sign.
    # R[a,c] is the (a,c) entry of the 2x2 result matrix for the outer M.
    # The outer M's R[a,c] contributes to specific C entries as described above.
    
    # Let me define the inner Strassen terms and their contributions to R.
    
    # For each outer M (0..6) and inner term (0..6), I'll compute the U, V, W vectors.
    
    # The inner Strassen terms are:
    # N0: (P[0,0]+P[1,1])*(Q[0,0]+Q[1,1]) -> contributes to R[0,0] with +1, R[1,1] with +1
    # N1: (P[1,0]+P[1,1])*Q[0,0] -> contributes to R[1,0] with +1, R[1,1] with -1
    # N2: P[0,0]*(Q[0,1]-Q[1,1]) -> contributes to R[0,1] with +1, R[1,1] with +1
    # N3: P[1,1]*(Q[1,0]-Q[0,0]) -> contributes to R[0,0] with +1, R[1,0] with +1
    # N4: (P[0,0]+P[1,0])*Q[1,1] -> contributes to R[0,0] with -1, R[0,1] with +1
    # N5: (P[0,1]-P[1,1])*(Q[1,0]+Q[1,1]) -> contributes to R[1,1] with +1
    # N6: (P[1,0]-P[0,0])*(Q[0,0]+Q[0,1]) -> contributes to R[0,0] with +1
    
    # Wait, let me double-check the inner Strassen formulas.
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A10-A00)(B00+B01)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for the inner Strassen (with P as A and Q as B):
    # N0 = M1 = (P00+P11)(Q00+Q11) -> R00 += 1, R11 += 1
    # N1 = M2 = (P10+P11)Q00 -> R10 += 1, R11 -= 1
    # N2 = M3 = P00(Q01-Q11) -> R01 += 1, R11 += 1
    # N3 = M4 = P11(Q10-Q00) -> R00 += 1, R10 += 1
    # N4 = M5 = (P00+P10)Q11 -> R00 -= 1, R01 += 1
    # N5 = M6 = (P01-P11)(Q10+Q11) -> R11 += 1
    # N6 = M7 = (P10-P00)(Q00+Q01) -> R00 += 1
    
    # Now, for each outer M and inner N, the contribution to C is:
    # For outer M, R[a,c] contributes to specific C entries.
    # The inner N contributes to R[a,c] with a specific sign.
    # So the combined contribution to C[i,k] is the product of the outer and inner signs.
    
    # Let me define the outer M's R to C mapping:
    # M=0: R[0,0]->C[0,0] with +1, R[0,1]->C[0,2] with +1, R[1,0]->C[2,0] with +1, R[1,1]->C[2,2] with +1
    #       Also R[0,0]->C[1,1] with +1, R[0,1]->C[1,3] with +1, R[1,0]->C[3,1] with +1, R[1,1]->C[3,3] with +1
    #       Wait, that's not right. Let me reconsider.
    
    # For outer M=0 (M1): C00 = M1, C11 = M1
    # C00[a,c] = C[2a, 2c], so C00[0,0]=C[0,0], C00[0,1]=C[0,2], C00[1,0]=C[2,0], C00[1,1]=C[2,2]
    # C11[a,c] = C[2a+1, 2c+1], so C11[0,0]=C[1,1], C11[0,1]=C[1,3], C11[1,0]=C[3,1], C11[1,1]=C[3,3]
    # So M1 contributes to C[0,0], C[0,2], C[2,0], C[2,2] with +1, and C[1,1], C[1,3], C[3,1], C[3,3] with +1.
    
    # For outer M=1 (M2): C10 = M2, C11 = -M2
    # C10[a,c] = C[2a+1, 2c], so C10[0,0]=C[1,0], C10[0,1]=C[1,2], C10[1,0]=C[3,0], C10[1,1]=C[3,2]
    # C11[a,c] = C[2a+1, 2c+1], so C11[0,0]=C[1,1], C11[0,1]=C[1,3], C11[1,0]=C[3,1], C11[1,1]=C[3,3]
    # So M2 contributes to C[1,0], C[1,2], C[3,0], C[3,2] with +1, and C[1,1], C[1,3], C[3,1], C[3,3] with -1.
    
    # For outer M=2 (M3): C01 = M3, C11 = M3
    # C01[a,c] = C[2a, 2c+1], so C01[0,0]=C[0,1], C01[0,1]=C[0,3], C01[1,0]=C[2,1], C01[1,1]=C[2,3]
    # C11[a,c] = C[2a+1, 2c+1], so C11[0,0]=C[1,1], C11[0,1]=C[1,3], C11[1,0]=C[3,1], C11[1,1]=C[3,3]
    # So M3 contributes to C[0,1], C[0,3], C[2,1], C[2,3] with +1, and C[1,1], C[1,3], C[3,1], C[3,3] with +1.
    
    # For outer M=3 (M4): C00 = M4, C10 = M4
    # C00[a,c] = C[2a, 2c], so C00[0,0]=C[0,0], C00[0,1]=C[0,2], C00[1,0]=C[2,0], C00[1,1]=C[2,2]
    # C10[a,c] = C[2a+1, 2c], so C10[0,0]=C[1,0], C10[0,1]=C[1,2], C10[1,0]=C[3,0], C10[1,1]=C[3,2]
    # So M4 contributes to C[0,0], C[0,2], C[2,0], C[2,2] with +1, and C[1,0], C[1,2], C[3,0], C[3,2] with +1.
    
    # For outer M=4 (M5): C00 = -M5, C01 = M5
    # C00[a,c] = C[2a, 2c], so C00[0,0]=C[0,0], C00[0,1]=C[0,2], C00[1,0]=C[2,0], C00[1,1]=C[2,2]
    # C01[a,c] = C[2a, 2c+1], so C01[0,0]=C[0,1], C01[0,1]=C[0,3], C01[1,0]=C[2,1], C01[1,1]=C[2,3]
    # So M5 contributes to C[0,0], C[0,2], C[2,0], C[2,2] with -1, and C[0,1], C[0,3], C[2,1], C[2,3] with +1.
    
    # For outer M=5 (M6): C11 = M6
    # C11[a,c] = C[2a+1, 2c+1], so C11[0,0]=C[1,1], C11[0,1]=C[1,3], C11[1,0]=C[3,1], C11[1,1]=C[3,3]
    # So M6 contributes to C[1,1], C[1,3], C[3,1], C[3,3] with +1.
    
    # For outer M=6 (M7): C00 = M7
    # C00[a,c] = C[2a, 2c], so C00[0,0]=C[0,0], C00[0,1]=C[0,2], C00[1,0]=C[2,0], C00[1,1]=C[2,2]
    # So M7 contributes to C[0,0], C[0,2], C[2,0], C[2,2] with +1.
    
    # Now, for each outer M and inner N, the contribution to C[i,k] is:
    #   (outer sign for R[a,c] to C[i,k]) * (inner sign for N to R[a,c])
    
    # Let me define the inner N's contributions to R:
    # N0: R[0,0] += 1, R[1,1] += 1
    # N1: R[1,0] += 1, R[1,1] -= 1
    # N2: R[0,1] += 1, R[1,1] += 1
    # N3: R[0,0] += 1, R[1,0] += 1
    # N4: R[0,0] -= 1, R[0,1] += 1
    # N5: R[1,1] += 1
    # N6: R[0,0] += 1
    
    # Now I can compute the full U, V, W for each of the 49 terms.
    
    # For each outer M (0..6) and inner N (0..6):
    #   U: product of the P linear combos from the inner N
    #   V: product of the Q linear combos from the inner N
    #   W: sum of contributions to C entries
    
    # Let me define the inner N's P and Q linear combos:
    # N0: P_combo = P[0,0]+P[1,1], Q_combo = Q[0,0]+Q[1,1]
    # N1: P_combo = P[1,0]+P[1,1], Q_combo = Q[0,0]
    # N2: P_combo = P[0,0], Q_combo = Q[0,1]-Q[1,1]
    # N3: P_combo = P[1,1], Q_combo = Q[1,0]-Q[0,0]
    # N4: P_combo = P[0,0]+P[1,0], Q_combo = Q[1,1]
    # N5: P_combo = P[0,1]-P[1,1], Q_combo = Q[1,0]+Q[1,1]
    # N6: P_combo = P[1,0]-P[0,0], Q_combo = Q[0,0]+Q[0,1]
    
    # For each N, P_combo is a sum of P[a,b] entries, each of which is a linear combo of A entries.
    # Similarly for Q_combo.
    
    # The product P_combo * Q_combo is a sum of products of A and B entries.
    # Each such product contributes to specific C entries with a specific sign.
    
    # For the tensor decomposition, each (outer M, inner N) gives one rank-1 term.
    # The U vector is the P_combo expressed as coefficients of A entries.
    # The V vector is the Q_combo expressed as coefficients of B entries.
    # The W vector is the sum of contributions to C entries.
    
    # Let me code this up.
    
    # Define the inner N's P and Q index combinations:
    # N0: P_indices = [(0,0,1), (1,1,1)], Q_indices = [(0,0,1), (1,1,1)]
    # N1: P_indices = [(1,0,1), (1,1,1)], Q_indices = [(0,0,1)]
    # N2: P_indices = [(0,0,1)], Q_indices = [(0,1,1), (1,1,-1)]
    # N3: P_indices = [(1,1,1)], Q_indices = [(1,0,1), (0,0,-1)]
    # N4: P_indices = [(0,0,1), (1,0,1)], Q_indices = [(1,1,1)]
    # N5: P_indices = [(0,1,1), (1,1,-1)], Q_indices = [(1,0,1), (1,1,1)]
    # N6: P_indices = [(1,0,1), (0,0,-1)], Q_indices = [(0,0,1), (0,1,1)]
    
    inner_P_indices = [
        [(0,0,1), (1,1,1)],  # N0
        [(1,0,1), (1,1,1)],  # N1
        [(0,0,1)],            # N2
        [(1,1,1)],            # N3
        [(0,0,1), (1,0,1)],  # N4
        [(0,1,1), (1,1,-1)], # N5
        [(1,0,1), (0,0,-1)], # N6
    ]
    
    inner_Q_indices = [
        [(0,0,1), (1,1,1)],  # N0
        [(0,0,1)],            # N1
        [(0,1,1), (1,1,-1)], # N2
        [(1,0,1), (0,0,-1)], # N3
        [(1,1,1)],            # N4
        [(1,0,1), (1,1,1)],  # N5
        [(0,0,1), (0,1,1)],  # N6
    ]
    
    # Inner N's contributions to R[a,c]:
    inner_R_contrib = [
        {((0,0), 1), ((1,1), 1)},  # N0
        {((1,0), 1), ((1,1), -1)}, # N1
        {((0,1), 1), ((1,1), 1)},  # N2
        {((0,0), 1), ((1,0), 1)},  # N3
        {((0,0), -1), ((0,1), 1)}, # N4
        {((1,1), 1)},              # N5
        {((0,0), 1)},              # N6
    ]
    
    # Outer M's R to C mapping:
    # M=0: R[0,0]->C[0,0]+1, R[0,1]->C[0,2]+1, R[1,0]->C[2,0]+1, R[1,1]->C[2,2]+1
    #       R[0,0]->C[1,1]+1, R[0,1]->C[1,3]+1, R[1,0]->C[3,1]+1, R[1,1]->C[3,3]+1
    # M=1: R[0,0]->C[1,0]+1, R[0,1]->C[1,2]+1, R[1,0]->C[3,0]+1, R[1,1]->C[3,2]+1
    #       R[0,0]->C[1,1]-1, R[0,1]->C[1,3]-1, R[1,0]->C[3,1]-1, R[1,1]->C[3,3]-1
    # M=2: R[0,0]->C[0,1]+1, R[0,1]->C[0,3]+1, R[1,0]->C[2,1]+1, R[1,1]->C[2,3]+1
    #       R[0,0]->C[1,1]+1, R[0,1]->C[1,3]+1, R[1,0]->C[3,1]+1, R[1,1]->C[3,3]+1
    # M=3: R[0,0]->C[0,0]+1, R[0,1]->C[0,2]+1, R[1,0]->C[2,0]+1, R[1,1]->C[2,2]+1
    #       R[0,0]->C[1,0]+1, R[0,1]->C[1,2]+1, R[1,0]->C[3,0]+1, R[1,1]->C[3,2]+1
    # M=4: R[0,0]->C[0,0]-1, R[0,1]->C[0,2]-1, R[1,0]->C[2,0]-1, R[1,1]->C[2,2]-1
    #       R[0,0]->C[0,1]+1, R[0,1]->C[0,3]+1, R[1,0]->C[2,1]+1, R[1,1]->C[2,3]+1
    # M=5: R[0,0]->C[1,1]+1, R[0,1]->C[1,3]+1, R[1,0]->C[3,1]+1, R[1,1]->C[3,3]+1
    # M=6: R[0,0]->C[0,0]+1, R[0,1]->C[0,2]+1, R[1,0]->C[2,0]+1, R[1,1]->C[2,2]+1
    
    outer_R_to_C = [
        # M=0
        {((0,0), ((0,0), 1)), ((0,1), ((0,2), 1)), ((1,0), ((2,0), 1)), ((1,1), ((2,2), 1)),
         ((0,0), ((1,1), 1)), ((0,1), ((1,3), 1)), ((1,0), ((3,1), 1)), ((1,1), ((3,3), 1))},
        # M=1
        {((0,0), ((1,0), 1)), ((0,1), ((1,2), 1)), ((1,0), ((3,0), 1)), ((1,1), ((3,2), 1)),
         ((0,0), ((1,1), -1)), ((0,1), ((1,3), -1)), ((1,0), ((3,1), -1)), ((1,1), ((3,3), -1))},
        # M=2
        {((0,0), ((0,1), 1)), ((0,1), ((0,3), 1)), ((1,0), ((2,1), 1)), ((1,1), ((2,3), 1)),
         ((0,0), ((1,1), 1)), ((0,1), ((1,3), 1)), ((1,0), ((3,1), 1)), ((1,1), ((3,3), 1))},
        # M=3
        {((0,0), ((0,0), 1)), ((0,1), ((0,2), 1)), ((1,0), ((2,0), 1)), ((1,1), ((2,2), 1)),
         ((0,0), ((1,0), 1)), ((0,1), ((1,2), 1)), ((1,0), ((3,0), 1)), ((1,1), ((3,2), 1))},
        # M=4
        {((0,0), ((0,0), -1)), ((0,1), ((0,2), -1)), ((1,0), ((2,0), -1)), ((1,1), ((2,2), -1)),
         ((0,0), ((0,1), 1)), ((0,1), ((0,3), 1)), ((1,0), ((2,1), 1)), ((1,1), ((2,3), 1))},
        # M=5
        {((0,0), ((1,1), 1)), ((0,1), ((1,3), 1)), ((1,0), ((3,1), 1)), ((1,1), ((3,3), 1))},
        # M=6
        {((0,0), ((0,0), 1)), ((0,1), ((0,2), 1)), ((1,0), ((2,0), 1)), ((1,1), ((2,2), 1))},
    ]
    
    # Now, for each outer M and inner N, compute the U, V, W vectors.
    
    for m in range(7):
        P = outer_P[m]
        Q = outer_Q[m]
        R_to_C = outer_R_to_C[m]
        
        for n_idx in range(7):
            P_indices = inner_P_indices[n_idx]
            Q_indices = inner_Q_indices[n_idx]
            R_contrib = inner_R_contrib[n_idx]
            
            # Compute U vector: sum of P[a,b] for (a,b,coeff) in P_indices
            U_dict = {}
            for (a, b, coeff) in P_indices:
                for (ai, aj), p_coeff in P[(a, b)].items():
                    total_coeff = coeff * p_coeff
                    if (ai, aj) not in U_dict:
                        U_dict[(ai, aj)] = 0
                    U_dict[(ai, aj)] += total_coeff
            
            # Compute V vector: sum of Q[b,c] for (b,c,coeff) in Q_indices
            V_dict = {}
            for (b, c, coeff) in Q_indices:
                for (bj, bk), q_coeff in Q[(b, c)].items():
                    total_coeff = coeff * q_coeff
                    if (bj, bk) not in V_dict:
                        V_dict[(bj, bk)] = 0
                    V_dict[(bj, bk)] += total_coeff
            
            # Compute W vector: sum of contributions to C entries
            W_dict = {}
            for (ra, rc), r_coeff in R_contrib.items():
                if (ra, rc) in R_to_C:
                    for (ci, ck), c_coeff in R_to_C[(ra, rc)].items():
                        total_coeff = r_coeff * c_coeff
                        if (ci, ck) not in W_dict:
                            W_dict[(ci, ck)] = 0
                        W_dict[(ci, ck)] += total_coeff
            
            # Remove zero entries
            U_dict = {k: v for k, v in U_dict.items() if v != 0}
            V_dict = {k: v for k, v in V_dict.items() if v != 0}
            W_dict = {k: v for k, v in W_dict.items() if v != 0}
            
            terms.append({'U': U_dict, 'V': V_dict, 'W': W_dict})
    
    return terms


def construct():
    """
    Construct a rank-97 decomposition for 4x4 * 4x7 matrix multiplication.
    Uses Strassen (rank 49) for the 4x4*4x4 part and schoolbook (rank 48) for the 4x4*4x3 part.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Get Strassen terms for 4x4x4 (49 terms)
    strassen_terms = _build_strassen_4x4x4()
    
    # Build schoolbook terms for 4x4x3 (48 terms)
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(4, p):  # k = 4, 5, 6
                schoolbook_terms.append({
                    'U': {((i, j), 1)},
                    'V': {((j, k), 1)},
                    'W': {((i, k), 1)}
                })
    
    # Combine all terms
    all_terms = strassen_terms + schoolbook_terms
    rank = len(all_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, term in enumerate(all_terms):
        for (i, j), coeff in term['U'].items():
            U[i * m + j][r] = coeff
        for (j, k), coeff in term['V'].items():
            V[j * p + k][r] = coeff
        for (i, k), coeff in term['W'].items():
            W[k * n + i][r] = coeff
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
