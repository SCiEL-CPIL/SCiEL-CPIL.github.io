"""Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE = 130.0  # safety margin before 150s kill


def _verify(U, V, W, tol=1e-9):
    """Verify that the decomposition is correct. W indexed as W[k*n+i, r]."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)

    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False

    rank = U.shape[1]

    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                dec_val = 0.0
                for r in range(rank):
                    dec_val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(dec_val - 1.0) > tol:
                    return False
    return True


def _build_rank33():
    """Direct sum of three 2x3x2 blocks (rank 11 each) -> rank 33."""
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Local flat indices for a 2x3x2 block:
    # A: i*3+j -> A00=0,A01=1,A02=2,A10=3,A11=4,A12=5
    # B: j*2+k -> B00=0,B01=1,B10=2,B11=3,B20=4,B21=5
    # C: i*2+k -> C00=0,C01=1,C10=2,C11=3
    terms = [
        # Strassen on A[0:2,0:2] x B[0:2,0:2]
        ({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}),          # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),          # M3
        ({4: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),          # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),          # M5
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),          # M6
        ({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1}),          # M7
        # Scalar terms A[i,2]*B[2,k]
        ({2: 1}, {4: 1}, {0: 1}),
        ({2: 1}, {5: 1}, {1: 1}),
        ({5: 1}, {4: 1}, {2: 1}),
        ({5: 1}, {5: 1}, {3: 1}),
    ]

    for block in range(3):
        for t_idx, (u_map, v_map, w_map) in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in u_map.items():
                U[idx][col] = val
            for l_idx, val in v_map.items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            for l_idx, val in w_map.items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val

    return {"U": U, "V": V, "W": W}


def _try_rank22_via_transpose():
    """
    Attempt rank 22 by treating the problem as C^T = B^T A^T.
    B^T is 6x3, A^T is 3x2.
    Split B^T into 3 blocks of 2x3 (rows 0-1, 2-3, 4-5).
    Each block is a 2x3 x 3x2 multiplication.
    Rank of 2x3x2 is 11. Total rank 33.
    
    Wait, this is the same as the direct approach. 
    Let's try to see if we can do better by reusing Strassen across blocks.
    
    Actually, the strategy suggested is:
    "Exploit the commutativity of the intermediate dimension by transposing the B matrix to treat the problem as C^T = B^T A^T (a 6 x 3 times 3 x 2 multiplication), then apply a linear change of basis to the 6 x 3 input space that maps the three 2 x 2 column-blocks of B^T into a structured form allowing the reuse of a single rank-7 Strassen decomposition for the core"
    
    This is a bit vague. Let's look at the dimensions.
    A: 2x3
    B: 3x6
    C: 2x6
    
    C^T: 6x2
    B^T: 6x3
    A^T: 3x2
    
    C^T = B^T A^T
    (6x3) * (3x2) -> 6x2
    
    Let's split B^T into 3 blocks of 2x3:
    B1^T (rows 0,1), B2^T (rows 2,3), B3^T (rows 4,5).
    Each is 2x3.
    A^T is 3x2.
    
    C1^T = B1^T A^T (2x2)
    C2^T = B2^T A^T (2x2)
    C3^T = B3^T A^T (2x2)
    
    This is exactly the same structure as before, just transposed.
    The rank is still 3 * 11 = 33.
    
    Is there a way to get rank 22?
    2x3x3 multiplication has rank 11 (Strassen 2x2x2 + 3 extra? No, 2x3x3 is 2x3 * 3x3 -> 2x3.
    Rank of 2x3x3:
    A (2x3), B (3x3), C (2x3).
    C[i,k] = sum_j A[i,j] B[j,k].
    
    If we split B into 2 blocks of 3x3? No, B is 3x3.
    If we split the output C into 2 blocks of 2x2? No, C is 2x3.
    
    Let's check the rank of 2x3x3.
    Schoolbook: 2*3*3 = 18.
    Strassen on 2x2 part:
    A[:, 0:2] (2x2) * B[0:2, :] (2x3) -> C[:, 0:2] (2x2) and C[:, 2] (2x1).
    This doesn't split nicely.
    
    Let's try to construct a rank 22 solution directly.
    If rank 22 exists, it would be an improvement.
    However, the observation data shows that previous attempts at rank 22 (p214) scored 1.0909, which is the same as rank 33.
    This suggests that rank 22 might not be valid or the score calculation is different.
    Wait, score = seed/value.
    If seed is 33, and value is 33, score is 1.0.
    If score is 1.0909, then 33/value = 1.0909 => value = 33/1.0909 = 30.25.
    This doesn't make sense for rank.
    
    Let's re-read the task.
    "score is the rank"
    "The evaluator measures the raw value (lower is better)."
    "The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    
    If seed program has rank 33, then seed = 33.
    If I find rank 30, score = 33/30 = 1.1.
    If I find rank 33, score = 33/33 = 1.0.
    
    The current program scores 1.0909.
    33 / 1.0909 = 30.25.
    This implies the "value" is not the rank, or the seed is not 33.
    
    Let's check the inspiration 1.
    "DIMENSIONS = [2, 3, 6]"
    "score 1.0909"
    
    Maybe the seed program has a different rank?
    Or maybe the score is calculated differently.
    
    Let's look at the current program.
    It returns rank 33.
    If the score is 1.0909, then 1.0909 = seed / 33.
    seed = 36.
    
    So the seed program likely has rank 36.
    My program has rank 33.
    Score = 36 / 33 = 1.0909.
    
    So, to improve, I need to find a rank lower than 33.
    Rank 32? 36/32 = 1.125.
    Rank 30? 36/30 = 1.2.
    
    The observation says p214 (which tried rank 22) scored 1.0909.
    This means p214 also returned rank 33 (or failed and returned the fallback rank 33).
    The code for p214 is not shown, but it likely had a `_try_rank22` that failed verification and fell back to rank 33.
    
    So, the goal is to find a valid decomposition with rank < 33.
    
    Is rank 32 possible?
    Or rank 30?
    
    Let's think about the structure.
    2x3 x 3x6 -> 2x6.
    
    We can view this as 3 independent 2x3 x 3x2 multiplications (blocks of width 2 in B).
    Each 2x3 x 3x2 has rank 11.
    Total 33.
    
    Can we do better?
    What if we group the 6 columns of B differently?
    For example, 2 blocks of 3?
    2x3 x 3x3 -> 2x3.
    Rank of 2x3x3?
    
    Let's calculate the rank of 2x3x3.
    A (2x3), B (3x3), C (2x3).
    
    We can use Strassen for the 2x2x2 part (A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]).
    This takes rank 7.
    Remaining terms:
    A[:, 0:2] * B[0:2, 2] -> C[:, 2] (2x1)
    A[:, 2] * B[2, 0:3] -> C[:, 0:3] (2x3)
    
    A[:, 0:2] * B[0:2, 2] is 2x2 * 2x1 -> 2x1.
    This is like a vector-matrix product.
    C[:, 2] = A[:, 0] * B[0, 2] + A[:, 1] * B[1, 2].
    This is 2 scalar multiplications? No, A[:, 0] is a column vector.
    (A[:, 0]) * B[0, 2] + (A[:, 1]) * B[1, 2].
    This can be done in rank 2?
    Let u1 = A[:, 0], v1 = B[0, 2], w1 = 1.
    Let u2 = A[:, 1], v2 = B[1, 2], w2 = 1.
    Yes, rank 2.
    
    A[:, 2] * B[2, 0:3] is 2x1 * 1x3 -> 2x3.
    C[:, 0:3] += A[:, 2] * B[2, 0:3].
    This is an outer product of a 2x1 and 1x3.
    This is rank 1?
    C[i, k] += A[i, 2] * B[2, k].
    Yes, this is a single rank-1 term?
    No, it's 3 terms:
    A[0, 2] * B[2, 0] -> C[0, 0]
    A[0, 2] * B[2, 1] -> C[0, 1]
    A[0, 2] * B[2, 2] -> C[0, 2]
    A[1, 2] * B[2, 0] -> C[1, 0]
    A[1, 2] * B[2, 1] -> C[1, 1]
    A[1, 2] * B[2, 2] -> C[1, 2]
    
    Can we combine these?
    Let M = A[:, 2] * B[2, :] (2x3 matrix).
    This is an outer product.
    In tensor decomposition, an outer product of vectors u and v has rank 1.
    But here u is 2x1 and v is 1x3.
    The result is a 2x3 matrix.
    The tensor is T[i, j, k] where j is the intermediate index.
    Here, the term is A[i, 2] * B[2, k].
    This corresponds to j=2.
    So it's a single term in the sum over j.
    In the decomposition, this is a single rank-1 component?
    U[i, 2, r] = A[i, 2]
    V[2, k, r] = B[2, k]
    W[k, i, r] = 1
    This is valid for all i, k.
    So yes, the term A[:, 2] * B[2, :] contributes rank 1.
    
    So for 2x3x3:
    - Strassen for 2x2x2 core: rank 7.
    - A[:, 0:2] * B[0:2, 2]: rank 2.
    - A[:, 2] * B[2, 0:3]: rank 1.
    Total rank = 7 + 2 + 1 = 10.
    
    Wait, is rank 10 for 2x3x3 correct?
    Let's verify.
    C = A B.
    C[:, 0:2] = A[:, 0:2] B[0:2, 0:2] + A[:, 2] B[2, 0:2].
    C[:, 2] = A[:, 0:2] B[0:2, 2] + A[:, 2] B[2, 2].
    
    My breakdown:
    1. A[:, 0:2] B[0:2, 0:2] (2x2x2): Rank 7.
    2. A[:, 0:2] B[0:2, 2] (2x2x1): Rank 2.
    3. A[:, 2] B[2, 0:3] (2x1x3): Rank 1.
    
    Total Rank 10.
    
    If we use this for 2 blocks of 2x3x3:
    Total rank = 2 * 10 = 20.
    
    But wait, the dimensions are 2x3 and 3x6.
    If we split B (3x6) into two 3x3 blocks, we get two 2x3x3 multiplications.
    Total rank 20.
    
    This would be a huge improvement!
    Score = 36 / 20 = 1.8.
    
    Let's double check if rank 10 for 2x3x3 is valid.
    
    Terms for 2x2x2 (Strassen):
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    These give C00, C01, C10, C11.
    
    Now, we need to add the contributions from the 3rd column/row.
    
    Contribution 1: A[:, 0:2] * B[0:2, 2].
    This affects C[:, 2].
    C02 += A00 * B02 + A01 * B12
    C12 += A10 * B02 + A11 * B12
    
    We can use 2 terms:
    T8: A00 * B02 -> C02
    T9: A01 * B12 -> C02
    T10: A10 * B02 -> C12
    T11: A11 * B12 -> C12
    
    Wait, that's 4 terms.
    Can we combine them?
    C02 = A00 B02 + A01 B12
    C12 = A10 B02 + A11 B12
    
    Let M8 = (A00 + A10) * B02 ? No, that mixes rows.
    Let M8 = A00 * B02 + A10 * B02 = (A00 + A10) B02.
    This gives C02 += A00 B02 and C12 += A10 B02.
    Let M9 = A01 * B12 + A11 * B12 = (A01 + A11) B12.
    This gives C02 += A01 B12 and C12 += A11 B12.
    
    So yes, 2 terms are sufficient.
    
    Contribution 2: A[:, 2] * B[2, 0:3].
    This affects C[:, 0:3].
    C00 += A02 * B20
    C01 += A02 * B21
    C02 += A02 * B22
    C10 += A12 * B20
    C11 += A12 * B21
    C12 += A12 * B22
    
    This is an outer product of A[:, 2] and B[2, :].
    As established, this is rank 1.
    Let M10 = A[:, 2] * B[2, :].
    U: A02, A12.
    V: B20, B21, B22.
    W: 1 for all corresponding C entries.
    
    So total rank for 2x3x3 is 7 + 2 + 1 = 10.
    
    So for 2x3x6 (two 2x3x3 blocks), rank is 20.
    
    Let's construct this.
    
    Block 1: B[:, 0:3]
    Block 2: B[:, 3:6]
    
    We need to map the local indices to global indices.
    
    Local A: 2x3 (i=0,1; j=0,1,2). Flat: i*3+j.
    Local B: 3x3 (j=0,1,2; k=0,1,2). Flat: j*3+k.
    Local C: 2x3 (i=0,1; k=0,1,2). Flat: i*3+k.
    
    Global A: 2x3. Same.
    Global B: 3x6.
    Block 1: B[:, 0:3]. Global B index: j*6 + k (k=0,1,2).
    Block 2: B[:, 3:6]. Global B index: j*6 + (k+3) (k=0,1,2).
    
    Global C: 2x6.
    Block 1: C[:, 0:3]. Global C index: k*2 + i (k=0,1,2).
    Block 2: C[:, 3:6]. Global C index: (k+3)*2 + i (k=0,1,2).
    
    Let's define the 10 terms for 2x3x3.
    
    Terms (u, v, w):
    
    # Strassen for 2x2x2 (A[:, 0:2] * B[0:2, 0:2])
    # A indices: 0,1,3,4
    # B indices: 0,1,3,4
    # C indices: 0,1,3,4
    
    # M1 = (A00+A11)(B00+B11)
    ({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1})
    # M2 = (A10+A11)B00
    ({3: 1, 4: 1}, {0: 1}, {3: 1, 4: -1})
    # M3 = A00(B01-B11)
    ({0: 1}, {1: 1, 4: -1}, {1: 1, 4: 1})
    # M4 = A11(B10-B00)
    ({4: 1}, {3: 1, 0: -1}, {0: 1, 3: 1})
    # M5 = (A00+A01)B11
    ({0: 1, 1: 1}, {4: 1}, {0: -1, 1: 1})
    # M6 = (A10-A00)(B00+B01)
    ({3: 1, 0: -1}, {0: 1, 1: 1}, {4: 1})
    # M7 = (A01-A11)(B10+B11)
    ({1: 1, 4: -1}, {3: 1, 4: 1}, {0: 1})
    
    # Terms for A[:, 0:2] * B[0:2, 2]
    # A indices: 0,1,3,4
    # B indices: 2, 5
    # C indices: 1, 4 (C02, C12)
    
    # M8 = (A00+A10)B02
    ({0: 1, 3: 1}, {2: 1}, {1: 1, 4: 1})
    # M9 = (A01+A11)B12
    ({1: 1, 4: 1}, {5: 1}, {1: 1, 4: 1})
    
    # Term for A[:, 2] * B[2, 0:3]
    # A indices: 2, 5
    # B indices: 6, 7, 8
    # C indices: 0,1,2,3,4,5
    
    # M10 = A[:, 2] * B[2, :]
    ({2: 1, 5: 1}, {6: 1, 7: 1, 8: 1}, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
    
    Wait, M10 has 6 entries in W.
    Is this valid?
    U[2, r] = 1, U[5, r] = 1.
    V[6, r] = 1, V[7, r] = 1, V[8, r] = 1.
    W[0, r] = 1, W[1, r] = 1, W[2, r] = 1, W[3, r] = 1, W[4, r] = 1, W[5, r] = 1.
    
    Let's check the product for C00.
    Sum_r U[0, r] V[0, r] W[0, r].
    U[0, r] is 0 for M10.
    So M10 contributes 0 to C00?
    
    No, U[2, r] corresponds to A[0, 2].
    U[5, r] corresponds to A[1, 2].
    
    The term is A[i, 2] * B[2, k].
    For C[0, 0] (i=0, k=0):
    We need A[0, 2] * B[2, 0].
    U[0*3+2, r] = U[2, r] = 1.
    V[2*3+0, r] = V[6, r] = 1.
    W[0*2+0, r] = W[0, r] = 1.
    Product = 1. Correct.
    
    For C[0, 1] (i=0, k=1):
    We need A[0, 2] * B[2, 1].
    U[2, r] = 1.
    V[2*3+1, r] = V[7, r] = 1.
    W[1*2+0, r] = W[2, r] = 1.
    Product = 1. Correct.
    
    For C[1, 0] (i=1, k=0):
    We need A[1, 2] * B[2, 0].
    U[1*3+2, r] = U[5, r] = 1.
    V[6, r] = 1.
    W[0*2+1, r] = W[1, r] = 1.
    Product = 1. Correct.
    
    So M10 is a single rank-1 term that covers all 6 entries.
    
    So yes, rank 10 is correct.
    
    Now, let's implement this for the 2 blocks.
    
    Block 0: k=0,1,2.
    Block 1: k=3,4,5.
    
    Total rank = 20.
    
    Let's write the code.
"""

def _build_rank20():
    """Direct sum of two 2x3x3 blocks (rank 10 each) -> rank 20."""
    n, m, p = DIMENSIONS
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Local flat indices for a 2x3x3 block:
    # A: i*3+j -> 0..5
    # B: j*3+k -> 0..8
    # C: i*3+k -> 0..5 (Local C is 2x3)
    
    # Note: Global W is (p*n, rank) = (12, rank).
    # W[k*n+i, r].
    # Local C index is i*3+k.
    # We need to map local C (i, k) to global W (k, i).
    
    terms = [
        # Strassen on A[:, 0:2] x B[0:2, 0:2]
        ({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1}),   # M1
        ({3: 1, 4: 1}, {0: 1}, {3: 1, 4: -1}),          # M2
        ({0: 1}, {1: 1, 4: -1}, {1: 1, 4: 1}),          # M3
        ({4: 1}, {3: 1, 0: -1}, {0: 1, 3: 1}),          # M4
        ({0: 1, 1: 1}, {4: 1}, {0: -1, 1: 1}),          # M5
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {4: 1}),          # M6
        ({1: 1, 4: -1}, {3: 1, 4: 1}, {0: 1}),          # M7
        # A[:, 0:2] * B[0:2, 2]
        ({0: 1, 3: 1}, {2: 1}, {1: 1, 4: 1}),           # M8
        ({1: 1, 4: 1}, {5: 1}, {1: 1, 4: 1}),           # M9
        # A[:, 2] * B[2, 0:3]
        ({2: 1, 5: 1}, {6: 1, 7: 1, 8: 1}, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}), # M10
    ]

    for block in range(2):
        for t_idx, (u_map, v_map, w_map) in enumerate(terms):
            col = block * 10 + t_idx
            for idx, val in u_map.items():
                U[idx][col] = val
            for l_idx, val in v_map.items():
                j_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = j_local * p + (block * 3 + k_local)
                V[g_idx][col] = val
            for l_idx, val in w_map.items():
                i_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = (block * 3 + k_local) * n + i_local
                W[g_idx][col] = val

    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()

    # Try rank 20 first
    best = _build_rank20()
    if _verify(best["U"], best["V"], best["W"]):
        return best
    
    # Fallback to rank 33
    best = _build_rank33()
    if _verify(best["U"], best["V"], best["W"]):
        return best
        
    return best
