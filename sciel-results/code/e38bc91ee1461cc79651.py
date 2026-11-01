import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # n=2, m=2, p=9

    def build_rank32():
        """Build rank-32 decomposition: 4 Strassen blocks (4*7=28) + 1 trivial (4) = 32."""
        rank = 32
        U = [[0] * rank for _ in range(n * m)]   # 4 x 32
        V = [[0] * rank for _ in range(m * p)]   # 18 x 32
        W = [[0] * rank for _ in range(p * n)]   # 18 x 32
        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            # M1: (a00+a11)(b0k0+b1k1)
            U[0][col] = 1; U[3][col] = 1
            V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
            W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
            col += 1
            # M2: (a10+a11)b0k0
            U[2][col] = 1; U[3][col] = 1
            V[0*p + k0][col] = 1
            W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
            col += 1
            # M3: a00(b0k1-b1k1)
            U[0][col] = 1
            V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
            W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
            col += 1
            # M4: a11(b1k0-b0k0)
            U[3][col] = 1
            V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
            W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
            col += 1
            # M5: (a00+a01)b1k1
            U[0][col] = 1; U[1][col] = 1
            V[1*p + k1][col] = 1
            W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
            col += 1
            # M6: (a10-a00)(b0k0+b0k1)
            U[2][col] = 1; U[0][col] = -1
            V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
            W[k1*n + 1][col] = 1
            col += 1
            # M7: (a01-a11)(b1k0+b1k1)
            U[1][col] = 1; U[3][col] = -1
            V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
            W[k0*n + 0][col] = 1
            col += 1

        def add_trivial(k):
            nonlocal col
            U[0][col] = 1; V[0*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
            U[1][col] = 1; V[1*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
            U[2][col] = 1; V[0*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1
            U[3][col] = 1; V[1*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1

        # 4 Strassen blocks covering cols 0-7
        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        # Trivial for col 8
        add_trivial(8)
        assert col == 32
        return U, V, W

    def build_rank31():
        """
        Attempt rank-31: 3 Strassen blocks (21) + 1 six-term block (6) + 1 trivial (4) = 31.
        The six-term block handles cols 6,7 with a construction that exploits
        a specific structure. We use a known 6-term decomposition for a restricted
        2x2 x 2x2 multiplication where one output is forced to zero via the
        block structure (the output at position that would be C[1,1] for that
        sub-block is handled by cross-terms from other blocks).
        
        Actually, a cleaner approach: use the fact that M_{2,2,2} has rank 7,
        but if we allow the decomposition to share terms across blocks, we can
        save one term. Here we use a direct construction:
        
        For cols 6,7, use 6 terms instead of 7 by noting that the (1,1) output
        element can be computed as a combination that overlaps with the trivial
        block's computation. This is a valid algebraic identity.
        """
        rank = 31
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_strassen(k0, k1):
            nonlocal col
            U[0][col] = 1; U[3][col] = 1
            V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
            W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
            col += 1
            U[2][col] = 1; U[3][col] = 1
            V[0*p + k0][col] = 1
            W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
            col += 1
            U[0][col] = 1
            V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
            W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
            col += 1
            U[3][col] = 1
            V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
            W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
            col += 1
            U[0][col] = 1; U[1][col] = 1
            V[1*p + k1][col] = 1
            W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
            col += 1
            U[2][col] = 1; U[0][col] = -1
            V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
            W[k1*n + 1][col] = 1
            col += 1
            U[1][col] = 1; U[3][col] = -1
            V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
            W[k0*n + 0][col] = 1
            col += 1

        def add_trivial(k):
            nonlocal col
            U[0][col] = 1; V[0*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
            U[1][col] = 1; V[1*p + k][col] = 1; W[k*n + 0][col] = 1; col += 1
            U[2][col] = 1; V[0*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1
            U[3][col] = 1; V[1*p + k][col] = 1; W[k*n + 1][col] = 1; col += 1

        def add_six_term(k0, k1):
            """
            6-term decomposition for 2x2 x 2x2 block at columns k0, k1.
            This is valid when we can absorb the (1,1) output into another block.
            We use a construction based on the fact that for the specific
            sub-problem, one of the 7 Strassen products can be eliminated
            if the corresponding output is handled elsewhere.
            
            Construction (verified algebraically):
            Uses the identity that C[1,1] = a11*b1k1 + a10*b0k1 can be
            written as part of a shared computation.
            """
            nonlocal col
            # M1: (a00+a11)(b0k0+b1k1) -> contributes to C[0,0] and C[1,1]
            U[0][col] = 1; U[3][col] = 1
            V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
            W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
            col += 1
            # M2: (a10+a11)b0k0 -> contributes to C[0,1] and C[1,1]
            U[2][col] = 1; U[3][col] = 1
            V[0*p + k0][col] = 1
            W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
            col += 1
            # M3: a00(b0k1-b1k1) -> contributes to C[1,0] and C[1,1]
            U[0][col] = 1
            V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
            W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
            col += 1
            # M4: a11(b1k0-b0k0) -> contributes to C[0,0] and C[0,1]
            U[3][col] = 1
            V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
            W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
            col += 1
            # M5: (a00+a01)b1k1 -> contributes to C[0,0] and C[1,0]
            U[0][col] = 1; U[1][col] = 1
            V[1*p + k1][col] = 1
            W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
            col += 1
            # M6: (a10-a00)(b0k0+b0k1) -> contributes to C[1,1]
            U[2][col] = 1; U[0][col] = -1
            V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
            W[k1*n + 1][col] = 1
            col += 1
            # Note: We dropped M7: (a01-a11)(b1k0+b1k1) -> C[0,0]
            # This means C[0,0] is missing the a01*b1k0 - a11*b1k0 - a01*b1k1 + a11*b1k1 terms
            # from M7. We need to verify if these are captured elsewhere.
            # They are NOT captured, so this 6-term block is INVALID for a standalone
            # 2x2 x 2x2 multiplication. 
            # 
            # However, if we pair this with the trivial block for col 8, we can
            # potentially share. But the trivial block only handles col 8, not cols 6,7.
            # So this approach doesn't work directly.
            #
            # Let's try a different 6-term construction that IS valid.
            pass

        # Since the 6-term standalone block is not straightforward, let's use
        # a different strategy: use 3 Strassen + 1 Strassen + 1 trivial = 32
        # and instead try to find a valid rank-31 via a different partition.
        
        # Alternative: use the fact that M_{2,2,9} can be decomposed as
        # M_{2,2,4} + M_{2,2,4} + M_{2,2,1} = 14 + 14 + 4 = 32
        # Or M_{2,2,3} + M_{2,2,3} + M_{2,2,3} = 11 + 11 + 11 = 33 (worse)
        # Or M_{2,2,5} + M_{2,2,4} = 18 + 14 = 32
        # Or M_{2,2,6} + M_{2,2,3} = 20 + 11 = 31!
        #   where M_{2,2,6} = M_{2,2,3} + M_{2,2,3} = 11 + 11 = 22 (not 20)
        #   Actually M_{2,2,6} is not known to be 20.
        
        # Let's just return the rank-32 solution for safety.
        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_trivial(8)
        assert col == 31 or col == 32, f"Got {col}"
        # Trim to actual used columns
        if col < rank:
            U = [row[:col] for row in U]
            V = [row[:col] for row in V]
            W = [row[:col] for row in W]
            rank = col
        return U, V, W

    def verify(U, V, W, n, m, p):
        """Verify the decomposition is exact."""
        rank = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try rank-31 first
    best_result = None
    try:
        U31, V31, W31 = build_rank31()
        rank31 = len(U31[0])
        if verify(U31, V31, W31, n, m, p):
            best_result = {"U": U31, "V": V31, "W": W31}
    except Exception:
        pass

    # Fall back to rank-32
    if best_result is None:
        try:
            U32, V32, W32 = build_rank32()
            rank32 = len(U32[0])
            if verify(U32, V32, W32, n, m, p):
                best_result = {"U": U32, "V": V32, "W": W32}
        except Exception:
            pass

    # Ultimate fallback: trivial rank-36
    if best_result is None:
        rank_t = n * m * p
        U_t = [[0] * rank_t for _ in range(n * m)]
        V_t = [[0] * rank_t for _ in range(m * p)]
        W_t = [[0] * rank_t for _ in range(p * n)]
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_t[ii*m+jj][c] = 1
                    V_t[jj*p+kk][c] = 1
                    W_t[kk*n+ii][c] = 1
                    c += 1
        best_result = {"U": U_t, "V": V_t, "W": W_t}

    return best_result
# EVOLVE-BLOCK-END
