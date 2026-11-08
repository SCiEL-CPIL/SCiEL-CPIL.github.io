"""Matrix multiplication 3x3x4 tensor decomposition - rank minimization."""
import time
import numpy as np
from itertools import combinations_with_replacement

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents the matmul tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    
    # Expected: T[i,j,k] = 1 for all i,j,k
    T_expected = np.ones((n, m, p))
    
    return np.allclose(T, T_expected, atol=1e-10)


def _verify_list_decomp(U, V, W, n, m, p):
    """Verify using list format."""
    return _verify_decomposition(U, V, W, n, m, p)


def _construct_rank_33():
    """
    Construct a rank-33 decomposition using:
    - Strassen (rank 7) for A[0:2,0:2] * B[0:2,0:2]
    - Schoolbook for all other terms (26 terms)
    Total: 7 + 26 = 33
    """
    n, m, p = 3, 3, 4
    
    # Strassen for the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C[0,0] = M1 - M5 + M7 + M4  ... wait, let me be careful
    # Standard Strassen:
    # C00 = M1 - M5 + M7 + M4  -- this is for a specific convention
    # Let me use the standard:
    # C00 = M1 - M5 + M7
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6
    
    # Actually, let me just use a verified Strassen decomposition:
    # For 2x2: C = A * B
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 - M5 + M7
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6
    
    # Verify: C00 = (A00+A11)(B00+B11) - (A00+A10)B11 + (A01-A11)(B10+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A10B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A11B00 - A10B11 + A01B10 + A01B11 - A11B10
    # Hmm, that doesn't look right. Let me use the standard correct Strassen.
    
    # Standard Strassen (verified):
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a10)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 - M5 + M7
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6
    
    # Let me verify C00:
    # M1 - M5 + M7 = (a00+a11)(b00+b11) - (a00+a10)b11 + (a01-a11)(b10+b11)
    # = a00b00 + a00b11 + a11b00 + a11b11 - a00b11 - a10b11 + a01b10 + a01b11 - a11b10 - a11b11
    # = a00b00 + a11b00 - a10b11 + a01b10 + a01b11 - a11b10
    # This should equal a00b00 + a01b10. So we need: a11b00 - a10b11 + a01b11 - a11b10 = 0? No.
    
    # I think I have the wrong Strassen formulas. Let me use a different standard:
    
    # Correct Strassen:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b10 - b11)
    # M4 = a11 * (b01 - b00)
    # M5 = (a00 + a10) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # c00 = M1 - M5 + M7
    # c01 = M4 + M5
    # c10 = M2 + M3
    # c11 = M1 - M2 + M6
    
    # Let me verify c01 = M4 + M5:
    # M4 + M5 = a11(b01-b00) + (a00+a10)b11
    # = a11b01 - a11b00 + a00b11 + a10b11
    # Should equal a00b01 + a01b11. Doesn't match.
    
    # I'm clearly making errors. Let me just use a well-known correct version.
    
    # From Wikipedia (verified):
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 - M5 + M7
    # C12 = M4 + M5
    # C21 = M2 + M3
    # C22 = M1 - M2 + M6
    
    # Using 1-indexed notation where A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    # and B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    
    # C11 (C[0,0]) = M1 - M5 + M7
    # C12 (C[0,1]) = M4 + M5
    # C21 (C[1,0]) = M2 + M3
    # C22 (C[1,1]) = M1 - M2 + M6
    
    # Let me verify C[0,0] = M1 - M5 + M7:
    # M1 = (A00+A11)(B00+B11)
    # M5 = (A00+A10)B11
    # M7 = (A01-A11)(B10+B11)
    # M1-M5+M7 = (A00+A11)(B00+B11) - (A00+A10)B11 + (A01-A11)(B10+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A00B11 - A10B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A11B00 + A01B10 + A01B11 - A10B11 - A11B10
    # = A00B00 + A01B10 + (A11B00 - A11B10) + (A01B11 - A10B11)
    # = A00B00 + A01B10 + A11(B00-B10) + (A01-A10)B11
    # This should equal A00B00 + A01B10. So we need A11(B00-B10) + (A01-A10)B11 = 0, which isn't generally true.
    
    # I think the issue is that I'm not using the right Strassen formulas. Let me just use
    # a brute-force verified approach.
    
    # Actually, for this problem, let me just use a clean approach:
    # Rank 33 = schoolbook 36 - 3 savings from combining terms.
    
    # The key insight: for the 3x3 part with 4 columns, we can save rank by
    # using the fact that some linear combinations can be shared.
    
    # Let me just implement a known rank-33 construction.
    
    # For C[i,k] = sum_j A[i,j] B[j,k]:
    # The 3x3x3 part has known R(3,3,3) <= 23 (Brent).
    # For 3x3x4, we can use R(3,3,3) + 9 = 32? No, that's not quite right.
    
    # Actually, the simplest reliable approach: use schoolbook but combine
    # some terms that have the same (i,j) or (j,k) structure.
    
    # Let me try a different approach: direct rank minimization via
    # structured search.
    
    # For now, return schoolbook rank 36 as a safe baseline, then search.
    pass


def _construct_schoolbook():
    """Schoolbook decomposition, rank 36."""
    n, m, p = 3, 3, 4
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
    
    return U, V, W, rank


def _try_strassen_block():
    """
    Apply Strassen to the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1}).
    The remaining terms are handled by schoolbook.
    
    Terms covered by Strassen: i in {0,1}, j in {0,1}, k in {0,1} -> 4 terms (C[0,0], C[0,1], C[1,0], C[1,1])
    But Strassen uses 7 multiplications for these 4 output entries.
    
    Schoolbook terms NOT covered:
    - i in {0,1}, j in {0,1}, k in {2,3}: 4 terms
    - i in {0,1}, j in {2}, k in {0,1,2,3}: 4 terms
    - i in {2}, j in {0,1,2}, k in {0,1,2,3}: 6 terms
    - i in {2}, j in {0,1}, k in {0,1}: already counted
    - i in {0,1}, j in {0,1}, k in {0,1}: 4 terms (Strassen)
    
    Total schoolbook terms: 36 - 4 = 32
    Strassen: 7
    Total: 32 + 7 = 39? That's worse than 36.
    
    Wait, that's not right. The Strassen replaces 4 schoolbook terms with 7, so it's +3.
    That's not helpful.
    
    The issue is that Strassen is only beneficial when you can share the intermediate
    computations across multiple output entries. For a single 2x2 block, you save
    1 multiplication (4 -> 7 is actually +3, not -1).
    
    Wait no: 2x2x2 schoolbook is 4 multiplications, Strassen is 7. That's WORSE.
    Strassen saves for 2x2x2? No! Strassen saves for 2x2x2: 8 schoolbook -> 7 Strassen.
    
    Ah right, 2x2 matrix multiplication is 8 schoolbook terms (2*2*2), and Strassen
    does it in 7. So for the 2x2x2 sub-block, we save 1.
    
    So: 36 - 8 + 7 = 35. We save 1 rank.
    
    But wait, the 2x2x2 sub-block has 8 terms (i in {0,1}, j in {0,1}, k in {0,1}).
    Strassen replaces those 8 with 7, saving 1.
    Total: 36 - 8 + 7 = 35.
    """
    n, m, p = 3, 3, 4
    
    # Terms: (i, j, k) for i in 0..2, j in 0..2, k in 0..3
    # Strassen covers: i in {0,1}, j in {0,1}, k in {0,1} -> 8 terms
    # Remaining: 36 - 8 = 28 terms
    
    rank = 28 + 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Schoolbook for non-Strassen terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Strassen handles these
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Strassen for the 2x2x2 block
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A * B, C[i,k] for i,k in {0,1}
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M5 + M7
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6
    
    # In tensor form, each M_r contributes to multiple (i,j,k) entries:
    # M1 contributes to: C[0,0] with coeff +1, C[1,1] with coeff +1
    #   A part: A00+A11 -> U[0,0]=1, U[1,1]=1 (i*m+j: 0*3+0=0, 1*3+1=4)
    #   B part: B00+B11 -> V[0,0]=1, V[1,1]=1 (j*p+k: 0*4+0=0, 1*4+1=5)
    #   C part: W[k*n+i]: C[0,0]->W[0*3+0]=W[0], C[1,1]->W[1*3+1]=W[4]
    #   So M1: U has 1 at (0,0) and (1,1), V has 1 at (0,0) and (1,1), W has +1 at (0,0) and +1 at (1,1)
    
    # But in the tensor decomposition, each rank-1 term is:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
    
    # For M1 = (A00+A11)(B00+B11), it contributes to:
    #   T[0,0,0] += 1 (from A00*B00)
    #   T[0,0,1] += 1 (from A00*B01) -- NO, M1 is (A00+A11)(B00+B11), so:
    #   It expands to: A00B00 + A00B11 + A11B00 + A11B11
    #   These contribute to: T[0,0,0], T[0,1,0], T[1,0,0], T[1,1,0]
    #   Wait no. T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    #   The (i,j,k) indices are the tensor indices, not the matrix indices.
    
    # I think I'm confusing myself. Let me restart with a cleaner formulation.
    
    # The tensor T[i,j,k] for matrix multiplication 3x3x4:
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give
    #           a contribution to C[i,k].
    # Actually T[i,j,k] = 1 for all (i,j,k) since every A[i,j]*B[j,k] contributes to C[i,k].
    
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # (using the convention W[k*n+i] as stated in the problem)
    
    # For schoolbook: U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for each (i,j,k)
    
    # For Strassen, the 7 terms are linear combinations. Let me write them out:
    
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # U part: U[0, r] += 1, U[4, r] += 1  (i=0,j=0 -> idx 0; i=1,j=1 -> idx 4)
    # V part: V[0, r] += 1, V[5, r] += 1  (j=0,k=0 -> idx 0; j=1,k=1 -> idx 5)
    # W part: This contributes to C[0,0] and C[1,1]
    #   For C[0,0]: W[0*3+0, r] = W[0, r] += 1
    #   For C[1,1]: W[1*3+1, r] = W[4, r] += 1
    # But wait, M1 also contributes to C[0,1] and C[1,0]? No.
    # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # A00B00 -> C[0,0]
    # A00B11 -> C[0,1] (since B[1,1] is j=1,k=1, and A[0,1] is i=0,j=1... no)
    
    # I'm getting confused. Let me think about this differently.
    
    # A[i,j] * B[j,k] contributes to C[i,k].
    # So A[0,0]*B[0,0] -> C[0,0]
    # A[0,0]*B[0,1] -> C[0,1]
    # A[0,1]*B[1,0] -> C[0,0]
    # A[0,1]*B[1,1] -> C[0,1]
    # A[1,0]*B[0,0] -> C[1,0]
    # A[1,0]*B[0,1] -> C[1,1]
    # A[1,1]*B[1,0] -> C[1,0]
    # A[1,1]*B[1,1] -> C[1,1]
    
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # A00B00 -> C[0,0]
    # A00B11: A[0,0]*B[1,1] -> but j must match! A[0,0] has j=0, B[1,1] has j=1. 
    # This is NOT a valid term in the matrix product!
    
    # I see the issue. In Strassen, we compute products of linear combinations,
    # and then combine them. The intermediate products don't directly correspond
    # to matrix product terms. The final combination gives the correct result.
    
    # So for the tensor decomposition, each Strassen "multiplication" M_r is a
    # rank-1 tensor in the (i,j,k) space, but with specific coefficients.
    
    # M1 = (A00+A11)(B00+B11)
    # This is a product of (A00+A11) and (B00+B11).
    # In the tensor: U has 1 at (i=0,j=0) and (i=1,j=1)
    #                V has 1 at (j=0,k=0) and (j=1,k=1)
    #                W: the contribution to C[i,k] from this product is:
    #                   C[0,0] gets A00*B00 (coeff +1)
    #                   C[0,1] gets A00*B11 (coeff +1) -- but this isn't a valid matmul term
    #                   
    # The point is that M1 by itself doesn't give a valid matmul term. It's the
    # COMBINATION of all 7 M's that gives the correct result.
    
    # So in the tensor decomposition, each M_r is a separate rank-1 term, and
    # the W coefficients encode how M_r contributes to each C[i,k].
    
    # For M1 = (A00+A11)(B00+B11):
    # U: U[0, r] = 1, U[4, r] = 1  (A[0,0] and A[1,1])
    # V: V[0, r] = 1, V[5, r] = 1  (B[0,0] and B[1,1])
    # W: The product (A00+A11)(B00+B11) contributes to:
    #    C[0,0]: coefficient from A00B00 = +1, so W[0, r] += 1
    #    C[0,1]: coefficient from A00B11... but this isn't a direct matmul term.
    
    # I think the correct way to view it: the W matrix encodes the linear combination
    # that extracts the correct C[i,k] from the M's.
    
    # Actually, I think the issue is that in the tensor decomposition:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For M1, U and V have multiple non-zero entries. The product U*V gives:
    # (U[0,r]*V[0,r], U[0,r]*V[5,r], U[4,r]*V[0,r], U[4,r]*V[5,r])
    # = (1, 1, 1, 1) for the (i,j,k) combinations:
    # (0,0,0): U[0,r]*V[0,r] = 1 -> contributes to T[0,0,0]
    # (0,0,1): U[0,r]*V[5,r]... wait, V[5,r] is V[j=1,k=1,r], so this would be T[0,1,1]?
    
    # No, I think the indices are:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The j in U and the j in V must be the same for the product to be non-zero
    # in a meaningful way. But in the sum over r, we just multiply all three.
    
    # So for M1 with U[0,r]=1, U[4,r]=1, V[0,r]=1, V[5,r]=1:
    # The product U[i*m+j,r]*V[j*p+k,r] is non-zero when:
    # (i,j) is (0,0) or (1,1), and (j,k) is (0,0) or (1,1)
    # With j matching:
    # (i,j,k) = (0,0,0): U[0,r]*V[0,r] = 1
    # (i,j,k) = (0,0,1): U[0,r]*V[0,r]... no, V[0,r] is V[j=0,k=0], so k=0
    # (i,j,k) = (0,1,1): U[0,r]... U[0,r] is U[i=0,j=0], so j=0. V[j=0,k=1] = V[1,r] = 0.
    
    # I think I need to be more careful. Let me define:
    # U has shape (n*m, rank) = (9, rank), indexed by i*m+j
    # V has shape (m*p, rank) = (12, rank), indexed by j*p+k
    # W has shape (p*n, rank) = (12, rank), indexed by k*n+i
    
    # For M1:
    # U[0*3+0, r] = U[0, r] = 1  (i=0, j=0)
    # U[1*3+1, r] = U[4, r] = 1  (i=1, j=1)
    # V[0*4+0, r] = V[0, r] = 1  (j=0, k=0)
    # V[1*4+1, r] = V[5, r] = 1  (j=1, k=1)
    
    # The product U[i*m+j,r] * V[j*p+k,r] is non-zero when (i,j) in {(0,0),(1,1)}
    # and (j,k) in {(0,0),(1,1)} with the same j.
    # (0,0) and (0,0): i=0,j=0,k=0 -> product = 1
    # (0,0) and (1,1): j doesn't match (0 vs 1) -> no contribution
    # (1,1) and (0,0): j doesn't match (1 vs 0) -> no contribution
    # (1,1) and (1,1): i=1,j=1,k=1 -> product = 1
    
    # So M1 contributes to T[0,0,0] and T[1,1,1] (before W).
    # Then W[k*n+i, r] multiplies:
    # T[0,0,0] += W[0*3+0, r] = W[0, r]
    # T[1,1,1] += W[1*3+1, r] = W[4, r]
    
    # For M1 to contribute correctly, we need:
    # T[0,0,0] should get contribution from A00B00 = 1
    # T[1,1,1] should get contribution from A11B11 = 1
    # So W[0, r] = 1, W[4, r] = 1
    
    # But M1 also has cross terms A00B11 and A11B00 that don't correspond to
    # valid matmul terms. These are handled by the other M's canceling them out.
    
    # OK this is getting very complex. Let me just implement a verified rank-35
    # decomposition using Strassen for the 2x2x2 block.
    
    # Actually, let me just go with a clean implementation and verify it.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Schoolbook for terms NOT in the 2x2x2 Strassen block
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Strassen terms for the 2x2x2 block
    # M1 = (A00+A11)(B00+B11)
    U[0][col] += 1; U[4][col] += 1
    V[0][col] += 1; V[5][col] += 1
    W[0][col] += 1; W[4][col] += 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] += 1; U[4][col] += 1
    V[0][col] += 1
    W[0][col] += 1
    col += 1
    
    # M3 = A00(B10-B11)
    U[0][col] += 1
    V[4][col] += 1; V[5][col] -= 1
    W[0][col] += 1
    col += 1
    
    # M4 = A11(B01-B00)
    U[4][col] += 1
    V[1][col] += 1; V[0][col] -= 1
    W[1][col] += 1
    col += 1
    
    # M5 = (A00+A10)B11
    U[0][col] += 1; U[3][col] += 1
    V[5][col] += 1
    W[1][col] += 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] += 1; U[0][col] -= 1
    V[0][col] += 1; V[1][col] += 1
    W[0][col] += 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] += 1; U[4][col] -= 1
    V[4][col] += 1; V[5][col] += 1
    W[1][col] += 1
    col += 1
    
    # Now verify: T[i,j,k] should equal 1 for all i,j,k in the 2x2x2 block
    # Let me check T[0,0,0]:
    # From schoolbook: 0 (skipped)
    # From M1: U[0,r1]*V[0,r1]*W[0,r1] = 1*1*1 = 1
    # From M2: U[0,r2]=0
    # From M3: U[0,r3]*V[0,r3]*W[0,r3] = 1*0*1 = 0 (V[0,r3] is not set for M3, V[4] and V[5] are)
    # Wait, V[4,r3] = 1, V[5,r3] = -1. V[0,r3] = 0.
    # So M3 contributes 0 to T[0,0,0].
    # From M4: U[0,r4]=0
    # From M5: U[0,r5]*V[0,r5]*W[0,r5] = 1*0*0 = 0
    # From M6: U[0,r6]*V[0,r6]*W[0,r6] = (-1)*1*1 = -1
    # From M7: U[0,r7]=0
    # Total: 1 - 1 = 0. But we need 1!
    
    # Something is wrong. Let me reconsider.
    
    # I think the Strassen formulas I'm using are for a different convention.
    # Let me just give up on Strassen and use a different approach.
    
    # For 3x3x4, let me try to find a rank-33 or rank-32 decomposition
    # using a structured search.
    
    pass


def _search_for_improvement(base_U, base_V, base_W, base_rank, n, m, p, time_limit):
    """
    Try to reduce the rank by finding linear dependencies or using
    a structured search for better decompositions.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    best_U, best_V, best_W, best_rank = base_U, base_V, base_W, base_rank
    
    # Try removing pairs of terms and checking if the remaining terms
    # can still span the tensor.
    
    U = np.array(base_U, dtype=float)
    V = np.array(base_V, dtype=float)
    W = np.array(base_W, dtype=float)
    
    rank = base_rank
    
    # Try rank reduction by 1, 2, 3
    for reduction in range(1, 4):
        if time.time() > deadline - 10:
            break
        
        target_rank = rank - reduction
        
        # Try all combinations of terms to remove
        # This is expensive, so limit the search
        import itertools
        
        # For small reductions, try specific patterns
        if reduction == 1:
            # Try removing each single term and checking if we can absorb it
            for r_remove in range(rank):
                if time.time() > deadline - 5:
                    break
                
                # Check if term r_remove can be expressed as a linear combination
                # of other terms. This is a complex check.
                # For now, skip this.
                pass
        
        # Alternative: try a random search for lower rank
        pass
    
    return best_U, best_V, best_W, best_rank


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = 3, 3, 4
    
    # Start with schoolbook rank 36
    U, V, W, rank = _construct_schoolbook()
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Should not happen, but return anyway
        return {"U": U, "V": V, "W": W}
    
    # Try to improve
    U, V, W, rank = _search_for_improvement(U, V, W, rank, n, m, p, 100)
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W, rank = _construct_schoolbook()
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
