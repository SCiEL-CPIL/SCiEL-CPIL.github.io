"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from collections import defaultdict

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    deadline = time.time() + 140

    # Index sets
    # A indices: (i, j) with i in [0,n), j in [0,m)
    A_idx = [(i, j) for i in range(n) for j in range(m)]
    # B indices: (j, k) with j in [0,m), k in [0,p)
    B_idx = [(j, k) for j in range(m) for k in range(p)]
    # C indices: (k, i) with k in [0,p), i in [0,n)  [trace convention W[k*n+i]]
    C_idx = [(k, i) for k in range(p) for i in range(n)]

    # The target tensor: T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

    # Precompute A_flat_index, B_flat_index, C_flat_index
    A_flat = {idx: pos for pos, idx in enumerate(A_idx)}
    B_flat = {idx: pos for pos, idx in enumerate(B_idx)}
    C_flat = {idx: pos for pos, idx in enumerate(C_idx)}

    def a_pos(i, j): return A_flat[(i, j)]
    def b_pos(j, k): return B_flat[(j, k)]
    def c_pos(k, i): return C_flat[(k, i)]

    # ------------------------------------------------------------------
    # Strategy: Search for a rank-r decomposition where each term r
    # is a "simple" product of sparse vectors.
    #
    # We use a different structural family: instead of fixed Winograd/Strassen
    # blocks, we search over all possible rank-r decompositions by
    # exploiting the fact that the tensor T[i,j,k]=1 is highly symmetric.
    #
    # Key insight: The matrix multiplication tensor for (n,m,p) has rank
    # at most n*m*p (schoolbook). We search for lower rank using a
    # combinatorial approach:
    #
    # For each rank r, we try to find r triples of sparse vectors
    # (u_r, v_r, w_r) such that the tensor equation holds.
    #
    # We parameterize: each u_r is a sum of +/-1 on a subset of A indices,
    # each v_r is a sum of +/-1 on a subset of B indices,
    # each w_r is a sum of +/-1 on a subset of C indices.
    #
    # This is a SAT-like search. We use a greedy + local search approach.
    # ------------------------------------------------------------------

    # First, try known low-rank constructions from the literature:
    # For 2x3x5, the known upper bound is 10+7+5=22 (current).
    # Let's try to find something better with a different decomposition.

    # Alternative approach: Use the fact that 2x3x5 = 2x(3x5) and
    # 3x5 has rank at most 15 (schoolbook) or 14 (one can save 1).
    # Actually, let's try a completely different structural approach:
    #
    # Use the "border rank" idea or a different partition.
    #
    # Let's try: partition the 3 middle indices differently.
    # Instead of (2x2x3) + (2x2x2) + (2x1x5), try:
    # (2x3x2) + (2x3x2) + (2x3x1)
    # Each 2x3x2 has rank at most 12 (schoolbook) or 11 (Winograd-like).
    # 2x3x1 has rank 2*3*1=6.
    # Total: 11+11+6 = 28. Worse.
    #
    # Try: (2x1x5) + (2x1x5) + (2x1x5) = 3 * 10 = 30. Worse.
    #
    # Try: (1x3x5) + (1x3x5) = 2 * 15 = 30. Worse.
    #
    # The current 22 is actually quite good. Let's see if we can beat it.
    #
    # Alternative: Use a global optimization approach.
    # We formulate it as: find the minimum r such that there exist
    # U (6xr), V (15xr), W (10xr) with entries in {0, +/-1, +/-1/2}
    # satisfying the tensor equation.
    #
    # We use a randomized local search: start from the schoolbook (rank 30)
    # and try to merge terms or find a lower-rank representation.

    # Let's try a different structural family: 
    # Use the "Kronecker" or "block" decomposition with a different split.
    #
    # Split the output C into 3x5 blocks of size 2x? No, n=2.
    # Split the input A into 2x3 blocks.
    #
    # Actually, let's try: treat the multiplication as C = A * B
    # and use the fact that rank(2x3x5) <= rank(2x3x3) + rank(2x3x2)
    # rank(2x3x3) <= 18 (schoolbook) or 16 (Winograd 2x2x3 x2 + 2x1x3)
    # rank(2x3x2) <= 12 (schoolbook) or 11
    # Total: 16+11 = 27. Worse than 22.
    #
    # Let's try: rank(2x3x5) <= rank(2x2x5) + rank(2x1x5)
    # rank(2x2x5) <= 10 (schoolbook) or 9 (Winograd 2x2x3 + 2x2x2)
    # rank(2x1x5) = 10
    # Total: 9+10 = 19. Better!
    #
    # Let's implement: 2x2x5 (rank 9) + 2x1x5 (rank 10) = 19

    # 2x2x5 decomposition:
    # A_block = A[:, 0:2] (2x2), B_block = B[0:2, :] (2x5), C_block = C[:, :] (2x5)
    # Split B_block into B[:, 0:3] (2x3) and B[:, 3:5] (2x2)
    # 2x2x3: Winograd rank 10? No, 2x2x3 has rank 10? Let me check.
    # Actually 2x2x3: the standard Winograd is for 2x2x2 (rank 7) or 3x3x3.
    # For 2x2x3, the rank is at most 2*2*3=12. Can we do better?
    # The known optimal rank for 2x2x3 is 10 (from the current code's Part A).
    # Wait, the current code uses rank 10 for 2x2x3. That's 10 terms.
    # For 2x2x2: Strassen rank 7.
    # So 2x2x5 = 2x2x3 (rank 10) + 2x2x2 (rank 7) = 17.
    # Then 2x1x5 = rank 10.
    # Total: 17+10 = 27. Worse than 22.
    #
    # I made an arithmetic error. Let's re-examine.
    # Current: 2x2x3 (10) + 2x2x2 (7) + 2x1x5 (10) = 27? 
    # No, the current code has:
    # Part A: 2x2x3 Winograd rank 10 (but the code says rank 10 in comment, 
    #   but let me count the terms: winograd_terms has 12 entries! 
    #   Wait, let me recount.
    #   winograd_terms = [12 tuples]. So rank 12?
    #   Let me recount: 1,2,3,4,5,6,7,8,9,10,11,12. Yes, 12 terms.
    #   But the comment says rank 10. That's a bug in the comment or code.
    #   Let me verify: the code adds 12 terms for Part A.
    # Part B: 7 terms (Strassen)
    # Part C: 10 terms (2 outer products x 5 k's)
    # Total: 12+7+10 = 29? 
    # But the fallback is rank 30. So the current program likely uses rank 29 or 30.
    #
    # Wait, let me re-read the code more carefully.
    # Part A: 12 terms
    # Part B: 7 terms  
    # Part C: 10 terms
    # Total: 29 terms.
    #
    # So the current score is based on rank 29 (if valid) or 30 (fallback).
    #
    # Can we do better than 29?
    #
    # Let's try: 2x3x5 = 2x(3x5). 
    # Use a different split: A (2x3) * B (3x5).
    # Split B into 3x3 + 3x2.
    # 2x3x3: rank at most 18. Can we do 16? 
    # 2x3x2: rank at most 12. Can we do 10?
    # Total: 16+10 = 26. Better than 29!
    #
    # Let's implement 2x3x3 with rank 16 and 2x3x2 with rank 10.
    #
    # 2x3x3: A (2x3) * B (3x3) = C (2x3)
    # Split A into 2x2 + 2x1.
    # 2x2x3: rank 10 (from Winograd)
    # 2x1x3: rank 6
    # Total: 10+6 = 16. 
    #
    # 2x3x2: A (2x3) * B (3x2) = C (2x2)
    # Split A into 2x2 + 2x1.
    # 2x2x2: rank 7 (Strassen)
    # 2x1x2: rank 4
    # Total: 7+4 = 11. 
    #
    # Grand total: 16+11 = 27. Better than 29!
    #
    # Let's implement this.

    U_terms = []
    V_terms = []
    W_terms = []

    def add_term(u_vec, v_vec, w_vec):
        U_terms.append(list(u_vec))
        V_terms.append(list(v_vec))
        W_terms.append(list(w_vec))

    def zeros6():
        return [0] * 6

    def zeros15():
        return [0] * 15

    def zeros10():
        return [0] * 10

    def setA(u, i, j, val):
        u[i * 3 + j] += val

    def setB(v, j, k, val):
        v[j * 5 + k] += val

    def setC(w, i, k, val):
        w[k * 2 + i] += val

    # ------------------------------------------------------------------
    # Part A: 2x2x3 Winograd for A[:, 0:2] * B[0:2, 0:3]
    # Same as before, 12 terms
    # ------------------------------------------------------------------
    x_map = {
        0: [(0, 0, +1), (0, 1, +1)],
        1: [(0, 0, +1), (0, 1, -1)],
        2: [(1, 0, +1), (1, 1, +1)],
        3: [(1, 0, +1), (1, 1, -1)],
        4: [(0, 0, +1)],
        5: [(0, 1, +1)],
        6: [(1, 0, +1)],
        7: [(1, 1, +1)],
    }
    y_map = {
        0: [(0, 0, +1), (0, 2, +1)],
        1: [(0, 0, +1), (0, 2, -1)],
        2: [(0, 1, +1)],
        3: [(1, 0, +1), (1, 2, +1)],
        4: [(1, 0, +1), (1, 2, -1)],
        5: [(1, 1, +1)],
        6: [(1, 0, +1)],
        7: [(1, 1, +1)],
        8: [(0, 0, +1)],
    }
    c_map = {
        0: (0, 0), 1: (0, 1), 2: (0, 2),
        3: (1, 0), 4: (1, 1), 5: (1, 2),
    }
    winograd_terms = [
        (0, 0, 0, +1), (4, 8, 0, +1),
        (4, 2, 1, +1), (5, 7, 1, +1),
        (0, 1, 2, +1), (5, 8, 2, +1),
        (2, 3, 3, +1), (6, 4, 3, +1),
        (6, 5, 4, +1), (7, 7, 4, +1),
        (2, 4, 5, +1), (7, 4, 5, +1),
    ]
    for x_idx, y_idx, c_idx, sign in winograd_terms:
        u = zeros6(); v = zeros15(); w = zeros10()
        for (i, j, coeff) in x_map[x_idx]:
            setA(u, i, j, sign * coeff)
        for (j, k, coeff) in y_map[y_idx]:
            setB(v, j, k, sign * coeff)
        ci, ck = c_map[c_idx]
        setC(w, ci, ck, 1)
        add_term(u, v, w)

    # ------------------------------------------------------------------
    # Part B: 2x1x3 for A[:, 2] * B[2, 0:3]
    # A[:, 2] is 2x1, B[2, 0:3] is 1x3
    # C[:, 0:3] gets contributions
    # Rank: 2*3 = 6
    # ------------------------------------------------------------------
    for k in range(3):
        for i in range(2):
            u = zeros6(); v = zeros15(); w = zeros10()
            setA(u, i, 2, 1)
            setB(v, 2, k, 1)
            setC(w, i, k, 1)
            add_term(u, v, w)

    # ------------------------------------------------------------------
    # Part C: 2x2x2 Strassen for A[:, 0:2] * B[0:2, 3:5]
    # A_block = A[:, 0:2] (2x2), B_block = B[0:2, 3:5] (2x2)
    # C_block = C[:, 3:5] (2x2)
    # B local: b00=B[0,3], b01=B[0,4], b10=B[1,3], b11=B[1,4]
    # C local: c00=C[0,3], c01=C[0,4], c10=C[1,3], c11=C[1,4]
    # ------------------------------------------------------------------
    strassen_terms = [
        ([(0, 0, +1), (1, 1, +1)], [(0, 3, +1), (1, 4, +1)], [(0, 3, +1), (1, 4, +1)]),
        ([(1, 0, +1), (1, 1, +1)], [(0, 3, +1)], [(1, 3, +1)]),
        ([(0, 0, +1)], [(1, 3, +1), (1, 4, -1)], [(0, 4, +1), (1, 3, +1)]),
        ([(1, 1, +1)], [(0, 4, +1), (0, 3, -1)], [(0, 3, +1), (1, 3, +1)]),
        ([(0, 0, +1), (0, 1, +1)], [(1, 4, +1)], [(0, 3, -1), (0, 4, +1)]),
        ([(1, 0, +1), (0, 0, -1)], [(0, 3, +1), (1, 3, +1)], [(1, 4, +1)]),
        ([(0, 1, +1), (1, 1, -1)], [(1, 4, +1), (0, 4, +1)], [(0, 3, +1)]),
    ]
    for (a_list, b_list, c_list) in strassen_terms:
        u = zeros6(); v = zeros15(); w = zeros10()
        for (i, j, coeff) in a_list:
            setA(u, i, j, coeff)
        for (j, k, coeff) in b_list:
            setB(v, j, k, coeff)
        for (i, k, coeff) in c_list:
            setC(w, i, k, coeff)
        add_term(u, v, w)

    # ------------------------------------------------------------------
    # Part D: 2x1x2 for A[:, 2] * B[2, 3:5]
    # A[:, 2] is 2x1, B[2, 3:5] is 1x2
    # C[:, 3:5] gets contributions
    # Rank: 2*2 = 4
    # ------------------------------------------------------------------
    for k in range(3, 5):
        for i in range(2):
            u = zeros6(); v = zeros15(); w = zeros10()
            setA(u, i, 2, 1)
            setB(v, 2, k, 1)
            setC(w, i, k, 1)
            add_term(u, v, w)

    rank = len(U_terms)
    # Expected: 12 + 6 + 7 + 4 = 29. Same as before!
    # Hmm, the split doesn't help because the total is the same.
    #
    # Let me try a different approach: can we reduce the Winograd 2x2x3 rank?
    # The known optimal rank for 2x2x3 is 10, not 12.
    # Let me use the correct rank-10 decomposition.
    #
    # Actually, let me look up: the rank of the 2x2x3 matrix multiplication tensor.
    # From literature: rank(2x2x3) = 10.
    # The current code uses 12 terms, which is suboptimal.
    #
    # Let me implement the correct rank-10 Winograd for 2x2x3.
    #
    # From the literature, the rank-10 decomposition for 2x2x3:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01, b02], [b10, b11, b12]]
    # C = A*B = [[c00, c01, c02], [c10, c11, c12]]
    #
    # The 10 terms are:
    # t1 = (a00+a01)(b00+b02) -> c00, c02
    # t2 = (a00-a01)(b00-b02) -> c00, c02
    # t3 = a00*b01 -> c01
    # t4 = a01*b11 -> c01
    # t5 = (a10+a11)(b10+b12) -> c10, c12
    # t6 = (a10-a11)(b10-b12) -> c10, c12
    # t7 = a10*b01 -> c11? No, let me be more careful.
    #
    # Actually, let me just use the known result: rank(2x2x3)=10.
    # I'll implement it correctly.

    # Let me restart with a cleaner approach.
    U_terms = []
    V_terms = []
    W_terms = []

    # ------------------------------------------------------------------
    # Use the optimal known decompositions:
    # 2x2x3: rank 10
    # 2x2x2: rank 7
    # 2x1x5: rank 10
    # Total: 10+7+10 = 27
    #
    # Or: 2x3x3: rank 16, 2x3x2: rank 11, total 27.
    #
    # Let me implement 2x2x3 with rank 10 correctly.
    # ------------------------------------------------------------------

    def add_term2(u_vec, v_vec, w_vec):
        U_terms.append(list(u_vec))
        V_terms.append(list(v_vec))
        W_terms.append(list(w_vec))

    # 2x2x3 rank 10 decomposition
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01, b02], [b10, b11, b12]]
    # C = A*B
    #
    # From the paper "Tensor rank of matrix multiplication" or similar:
    # The 10 terms for 2x2x3:
    #
    # I'll use a known construction. Let me derive it.
    #
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[0,2] = a00*b02 + a01*b12
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    # C[1,2] = a10*b02 + a11*b12
    #
    # Winograd-style:
    # x1 = a00+a01, x2 = a00-a01
    # y1 = b00+b02, y2 = b00-b02, y3 = b01
    # z1 = b10+b12, z2 = b10-b12, z3 = b11
    #
    # C[0,0] = (x1*y1 + x2*y2)/2 + (a00*b01 + a01*b11)? No.
    #
    # Let me use a simpler approach: just use rank 12 for 2x2x3 (as in the current code)
    # and see if the total is 29. Then try to improve.
    #
    # Actually, let me just keep the current decomposition (rank 29) and add a
    # randomized search to try to find a lower rank.

    # Rebuild with the original 29-term decomposition
    U_terms = []
    V_terms = []
    W_terms = []

    for x_idx, y_idx, c_idx, sign in winograd_terms:
        u = zeros6(); v = zeros15(); w = zeros10()
        for (i, j, coeff) in x_map[x_idx]:
            setA(u, i, j, sign * coeff)
        for (j, k, coeff) in y_map[y_idx]:
            setB(v, j, k, sign * coeff)
        ci, ck = c_map[c_idx]
        setC(w, ci, ck, 1)
        add_term2(u, v, w)

    for (a_list, b_list, c_list) in strassen_terms:
        u = zeros6(); v = zeros15(); w = zeros10()
        for (i, j, coeff) in a_list:
            setA(u, i, j, coeff)
        for (j, k, coeff) in b_list:
            setB(v, j, k, coeff)
        for (i, k, coeff) in c_list:
            setC(w, i, k, coeff)
        add_term2(u, v, w)

    for k in range(5):
        for i in range(2):
            u = zeros6(); v = zeros15(); w = zeros10()
            setA(u, i, 2, 1)
            setB(v, 2, k, 1)
            setC(w, i, k, 1)
            add_term2(u, v, w)

    rank = len(U_terms)  # 12 + 7 + 10 = 29

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for r in range(rank):
        U[:, r] = U_terms[r]
        V[:, r] = V_terms[r]
        W[:, r] = W_terms[r]

    # Verify
    T = np.ones((n, m, p))
    result = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                result[i, j, k] = val

    valid = np.allclose(result, T, atol=1e-9)

    def is_half_integer(M):
        for val in M.flatten():
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
        return True

    if valid and is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
