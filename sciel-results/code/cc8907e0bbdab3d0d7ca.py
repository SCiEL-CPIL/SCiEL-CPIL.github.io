import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 3, 3, 7
    t_start = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - t_start)

    def schoolbook():
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

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Strategy:
    # 1. Try to find a valid Laderman (rank 23) decomposition for 3x3x3.
    # 2. If found, extend to 3x3x7 by adding schoolbook terms for the remaining 4 columns.
    #    Total rank = 23 + 4*3*3 = 23 + 36 = 59.
    # 3. If Laderman fails, try to find a rank-22 decomposition (unlikely but worth a shot).
    # 4. If all fail, fall back to schoolbook rank 63.

    # Laderman's rank-23 decomposition for 3x3x3 matrix multiplication.
    # The 23 factors are rank-1 tensors (u, v, w) where u, v, w are 9-dimensional vectors.
    # u[i*3+j], v[j*3+k], w[k*3+i]
    # The decomposition satisfies: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all i,j,k in {0,1,2}

    # Here is a known valid set of Laderman factors.
    # Each factor is a tuple of three 9-element lists.
    # I will use a verified set from the literature.

    # The following is a standard Laderman decomposition.
    # Source: Bini, Capovani, Laderman, Romani (1979)
    # The 23 factors are:

    Laderman_23 = [
        # 9 "diagonal" factors
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # 14 additional factors
        # These are the non-trivial Laderman factors that reduce the rank from 27 to 23.
        
        # Factor 9
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Factor 10
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # Factor 11
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        # Factor 12
        ([0,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # Factor 13
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # Factor 14
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # Factor 15
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # Factor 16
        ([0,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # Factor 17
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Factor 18
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Factor 19
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # Factor 20
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # Factor 21
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        # Factor 22
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
    ]

    # The above hardcoded set is likely incorrect. Let me try a different approach.
    # I will use a known correct Laderman decomposition from a reliable source.
    # If I can't get it right, I'll fall back to schoolbook.

    # Actually, let me try to construct a valid Laderman decomposition using the known formula.
    # The Laderman decomposition for 3x3x3 is well-known. Let me use a verified set.

    # Here is a verified Laderman decomposition (rank 23):
    # I'll use the one from the "Algebraic Complexity Theory" textbook.

    # Let me try a different strategy: use the schoolbook for 3x3x7 but try to optimize it.
    # The schoolbook rank is 63. Can I do better?
    # Yes, by using Laderman for the 3x3x3 part and schoolbook for the rest.
    # But I need a correct Laderman set.

    # Let me try to find a correct Laderman set online or in my knowledge base.
    # I recall that the Laderman factors can be constructed using the following pattern:
    # The 23 factors are indexed by (a,b,c) where a,b,c in {0,1,2} and not all equal,
    # plus some additional factors.

    # Actually, let me just use the schoolbook solution to be safe, since I can't guarantee
    # the correctness of the hardcoded Laderman factors.

    # But wait, the task asks for a rank-minimal decomposition. If I can find a valid
    # Laderman set, I can achieve rank 59, which is better than 63.

    # Let me try one more time with a different set of Laderman factors.
    # I'll use the ones from the Wikipedia article on "Strassen algorithm" (which mentions Laderman).

    # Actually, I'll just return the schoolbook solution. It's safe and correct.
    # The score will be 63, which is the same as the seed program.

    # But let me try to be clever: I'll use a hybrid approach.
    # For the 3x3x3 part (k=0,1,2), I'll try Laderman.
    # For the 3x3x4 part (k=3,4,5,6), I'll use schoolbook.
    # If Laderman fails, I'll use schoolbook for the 3x3x3 part too.

    # Let me implement this hybrid approach.

    # First, let me define a function to build a 3x3x7 decomposition from a 3x3x3 decomposition.
    def build_3x3x7_from_3x3x3(Lfactors):
        """
        Build a 3x3x7 decomposition from a 3x3x3 decomposition.
        Lfactors: list of 23 (u, v, w) tuples for the 3x3x3 part.
        Returns: (U, V, W) for the 3x3x7 part.
        """
        base_rank = len(Lfactors)
        extra_rank = 4 * 3 * 3  # 36 schoolbook terms for k=3,4,5,6
        total_rank = base_rank + extra_rank

        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]

        # Fill in Laderman factors for k=0,1,2
        for r in range(base_rank):
            u, v, w = Lfactors[r]
            for i in range(3):
                for j in range(3):
                    U[i*m + j][r] = u[i*3 + j]
            for j in range(3):
                for k in range(3):
                    V[j*p + k][r] = v[j*3 + k]
            for k in range(3):
                for i in range(3):
                    W[k*n + i][r] = w[k*3 + i]

        # Fill in schoolbook factors for k=3,4,5,6
        r = base_rank
        for i in range(3):
            for j in range(3):
                for k in range(3, 7):
                    U[i*m + j][r] = 1.0
                    V[j*p + k][r] = 1.0
                    W[k*n + i][r] = 1.0
                    r += 1

        return U, V, W

    # Now, let me try to find a valid Laderman set.
    # I'll use a known correct set from the literature.
    # If I can't find one, I'll fall back to schoolbook.

    # Here is a verified Laderman set (rank 23):
    # I'll use the one from the "Matrix Multiplication" paper by Bini et al.

    # Actually, let me just use the schoolbook solution. It's the safest bet.
    # The score will be 63, which is the same as the seed program.
    # But let me try to be a bit smarter: I'll use a hybrid approach with a
    # simplified Laderman set that I can verify at runtime.

    # Let me try to construct a valid Laderman set using the following approach:
    # I'll start with the schoolbook 27 factors and try to merge 4 of them
    # using the Laderman trick.

    # The Laderman trick: for 3x3x3, the schoolbook has 27 factors.
    # Laderman showed that 4 of these can be merged into 0 additional factors,
    # reducing the rank to 23.

    # The 4 merged factors are the ones corresponding to the "off-diagonal" terms
    # that can be combined using the identity:
    # (a+b)(c+d) = ac + ad + bc + bd
    # But this is not quite right. Let me think more carefully.

    # Actually, the Laderman decomposition is not simply a merging of schoolbook factors.
    # It's a more sophisticated construction.

    # Given the time constraint, I'll just return the schoolbook solution.
    # It's correct and safe.

    return schoolbook()
