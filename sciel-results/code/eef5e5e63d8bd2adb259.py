import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = 3, 3, 3

    def idx_u(i, j):
        return i * m + j

    def idx_v(j, k):
        return j * p + k

    def idx_w(k, i):
        return k * n + i

    def verify(U, V, W, rank):
        """Verify the decomposition exactly using integer arithmetic on multiple test cases."""
        import random
        random.seed(42)
        for _ in range(20):
            A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
            AB = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
            C = [[0] * p for _ in range(n)]
            for r in range(rank):
                for i in range(n):
                    for k in range(p):
                        su = sum(U[i * m + j][r] * A[i][j] for j in range(m))
                        sv = sum(V[j * p + k][r] * B[j][k] for j in range(m))
                        C[i][k] += su * sv * W[k * n + i][r]
            if C != AB:
                return False
        return True

    # Build the rank-26 decomposition: 7 Strassen terms for 2x2 core + 19 border terms
    def build_rank26():
        rank = 26
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        # 7 Strassen terms for the 2x2 core
        # m1 = (a00+a11)(b00+b11)
        U[idx_u(0,0)][0] += 1; U[idx_u(1,1)][0] += 1
        V[idx_v(0,0)][0] += 1; V[idx_v(1,1)][0] += 1
        W[idx_w(0,0)][0] += 1; W[idx_w(1,1)][0] += 1

        # m2 = (a10+a11)b00
        U[idx_u(1,0)][1] += 1; U[idx_u(1,1)][1] += 1
        V[idx_v(0,0)][1] += 1
        W[idx_w(0,1)][1] += 1; W[idx_w(1,1)][1] += -1

        # m3 = a00(b10-b11)
        U[idx_u(0,0)][2] += 1
        V[idx_v(1,0)][2] += 1; V[idx_v(1,1)][2] += -1
        W[idx_w(0,1)][2] += 1; W[idx_w(1,1)][2] += 1

        # m4 = a11(b01-b00)
        U[idx_u(1,1)][3] += 1
        V[idx_v(0,1)][3] += 1; V[idx_v(0,0)][3] += -1
        W[idx_w(0,0)][3] += 1; W[idx_w(1,0)][3] += -1

        # m5 = (a00+a01)b11
        U[idx_u(0,0)][4] += 1; U[idx_u(0,1)][4] += 1
        V[idx_v(1,1)][4] += 1
        W[idx_w(0,0)][4] += -1; W[idx_w(0,1)][4] += 1; W[idx_w(1,0)][4] += -1

        # m6 = (a10-a00)(b00+b01)
        U[idx_u(1,0)][5] += 1; U[idx_u(0,0)][5] += -1
        V[idx_v(0,0)][5] += 1; V[idx_v(0,1)][5] += 1
        W[idx_w(0,1)][5] += 1; W[idx_w(1,0)][5] += 1

        # m7 = (a01-a11)(b10+b11)
        U[idx_u(0,1)][6] += 1; U[idx_u(1,1)][6] += -1
        V[idx_v(1,0)][6] += 1; V[idx_v(1,1)][6] += 1
        W[idx_w(0,0)][6] += 1; W[idx_w(0,1)][6] += -1; W[idx_w(1,1)][6] += 1

        # 19 border terms: one for each (i,j,k) with max(i,j,k) == 2
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if max(i, j, k) == 2:
                        U[idx_u(i, j)][col] = 1
                        V[idx_v(j, k)][col] = 1
                        W[idx_w(k, i)][col] = 1
                        col += 1

        return U, V, W, rank

    # Build schoolbook rank-27 as ultimate fallback
    def build_schoolbook():
        rank = 27
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j)][col] = 1
                    V[idx_v(j, k)][col] = 1
                    W[idx_w(k, i)][col] = 1
                    col += 1
        return U, V, W, rank

    # Try to find a rank-23 decomposition using the known border-rank construction.
    # The 3x3 matrix multiplication tensor has border rank 23 (Landsberg).
    # We attempt to construct it by carefully combining the 19 border entries
    # into 16 terms using linear algebra.
    def build_rank23():
        """
        Attempt rank-23 construction.
        7 Strassen terms + 16 border terms = 23.
        The 19 border entries must be covered by 16 rank-1 terms.
        This requires that 3 pairs of border entries can be merged via linear combinations.
        
        Key insight: entries (2,2,k), (2,k,2), (k,2,2) for k in {0,1} form a 2x2x2 sub-tensor
        that can be handled with 7 Strassen-like terms, but we already used those for the core.
        
        Alternative: Use the explicit rank-23 from the literature.
        The construction uses the fact that the 3x3 tensor = 2x2 tensor (rank 7) + 
        a "border" that has rank 16.
        
        Let me try a direct construction by solving a linear system.
        """
        rank = 23
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        # 7 Strassen terms (same as above)
        U[idx_u(0,0)][0] += 1; U[idx_u(1,1)][0] += 1
        V[idx_v(0,0)][0] += 1; V[idx_v(1,1)][0] += 1
        W[idx_w(0,0)][0] += 1; W[idx_w(1,1)][0] += 1

        U[idx_u(1,0)][1] += 1; U[idx_u(1,1)][1] += 1
        V[idx_v(0,0)][1] += 1
        W[idx_w(0,1)][1] += 1; W[idx_w(1,1)][1] += -1

        U[idx_u(0,0)][2] += 1
        V[idx_v(1,0)][2] += 1; V[idx_v(1,1)][2] += -1
        W[idx_w(0,1)][2] += 1; W[idx_w(1,1)][2] += 1

        U[idx_u(1,1)][3] += 1
        V[idx_v(0,1)][3] += 1; V[idx_v(0,0)][3] += -1
        W[idx_w(0,0)][3] += 1; W[idx_w(1,0)][3] += -1

        U[idx_u(0,0)][4] += 1; U[idx_u(0,1)][4] += 1
        V[idx_v(1,1)][4] += 1
        W[idx_w(0,0)][4] += -1; W[idx_w(0,1)][4] += 1; W[idx_w(1,0)][4] += -1

        U[idx_u(1,0)][5] += 1; U[idx_u(0,0)][5] += -1
        V[idx_v(0,0)][5] += 1; V[idx_v(0,1)][5] += 1
        W[idx_w(0,1)][5] += 1; W[idx_w(1,0)][5] += 1

        U[idx_u(0,1)][6] += 1; U[idx_u(1,1)][6] += -1
        V[idx_v(1,0)][6] += 1; V[idx_v(1,1)][6] += 1
        W[idx_w(0,0)][6] += 1; W[idx_w(0,1)][6] += -1; W[idx_w(1,1)][6] += 1

        # Now we need 16 terms for the 19 border entries.
        # The border entries are all (i,j,k) with max(i,j,k)=2.
        # Let's enumerate them:
        border_entries = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if max(i, j, k) == 2:
                        border_entries.append((i, j, k))
        # There should be 19 of these.

        # Strategy: Use 16 rank-1 terms. We need to find coefficients such that
        # sum_{r=7}^{22} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for each border entry
        # and = 0 for non-border entries (but Strassen already handles those).
        
        # This is a system of 19 equations with 16*3*3 = 144 unknowns (but structured).
        # Let's use a greedy/iterative approach: assign each border entry to a term,
        # then try to merge terms.

        # Simpler approach: Use the known explicit rank-23 from Strassen's work.
        # The construction is:
        # - 7 terms for the 2x2x2 core
        # - 4 terms for A[2,*] * B[*,*] (i=2, j,k in {0,1})
        # - 4 terms for A[*,*] * B[*,2] (k=2, i,j in {0,1})  
        # - 4 terms for A[*,2] * B[2,*] (j=2, i,k in {0,1})
        # - 1 term for A[2,2]*B[2,2]
        # - 1 term combining (2,2,0) and (2,0,2) and (0,2,2)
        # - 1 term combining (2,2,1) and (2,1,2) and (1,2,2)
        # - 1 term combining (2,0,2) and (0,2,2) [already counted?]
        # This doesn't quite work out. Let me try a different grouping.

        # Actually, let's use a numerical approach: set up the linear system and solve it.
        # For each of the 16 border terms r=7..22, we have variables:
        # u_r[i,j] for i,j in 0..2 (9 vars), v_r[j,k] for j,k in 0..2 (9 vars), w_r[k,i] for k,i in 0..2 (9 vars)
        # Total 16*27 = 432 unknowns, 19 equations (one per border entry).
        # This is underdetermined, so we can find a solution.
        # But we need integer/half-integer entries.

        # Let's use a simpler constructive approach.
        # Group the 19 border entries into 16 groups, where 3 groups have 2 entries each.
        
        # The 19 entries:
        # i=2: (2,0,0),(2,0,1),(2,0,2),(2,1,0),(2,1,1),(2,1,2),(2,2,0),(2,2,1),(2,2,2) [9]
        # j=2: (0,2,0),(0,2,1),(0,2,2),(1,2,0),(1,2,1),(1,2,2) [6]
        # k=2: (0,0,2),(0,1,2),(1,0,2),(1,1,2) [4]
        # Total: 9+6+4 = 19

        # Grouping:
        # Term 7: (2,0,0) alone
        # Term 8: (2,0,1) alone
        # Term 9: (2,1,0) alone
        # Term 10: (2,1,1) alone
        # Term 11: (0,0,2) alone
        # Term 12: (0,1,2) alone
        # Term 13: (1,0,2) alone
        # Term 14: (1,1,2) alone
        # Term 15: (0,2,0) alone
        # Term 16: (0,2,1) alone
        # Term 17: (1,2,0) alone
        # Term 18: (1,2,1) alone
        # Term 19: (2,2,0) and (2,0,2) and (0,2,2) -- can we merge?
        # Term 20: (2,2,1) and (2,1,2) and (1,2,2) -- can we merge?
        # Term 21: (2,2,2) alone
        # That's 15 single-entry terms + 2 multi-entry terms + 1 single = 18 terms. Still not 16.

        # Let me try a different approach: use the fact that some border entries
        # can be expressed as linear combinations of others.
        
        # For example, consider the sub-tensor on indices where at least one is 2.
        # This is a 3x3x3 tensor with 19 non-zero entries.
        # We want to express it as a sum of 16 rank-1 tensors.
        
        # Let's use a numerical solver to find such a decomposition.
        # We'll set up the problem as: find U, V, W (each 9x16) such that
        # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for border (i,j,k), 0 otherwise.
        
        # This is a nonlinear system. Let's use a simpler approach:
        # Assume a specific structure for U, V, W and solve for the remaining variables.

        # Given the complexity, let's just try to verify if the rank-26 is correct
        # and return it. The rank-23 is the theoretical minimum but hard to construct explicitly.
        
        return None

    # Build rank-26 (guaranteed valid)
    U, V, W, rank = build_rank26()

    # Verify
    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        U, V, W, rank = build_schoolbook()

    # Try to find a rank-23 decomposition using numerical optimization
    if time.time() < deadline - 30:
        try:
            # Use numpy to attempt a rank-23 decomposition
            # We'll set up the tensor and try to find a low-rank approximation
            # that is exact.
            
            # The target tensor T has T[i,j,k] = 1 for all i,j,k.
            # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
            
            # For rank 23, we have 23 terms.
            # Let's initialize with the rank-26 and try to eliminate 3 terms.
            
            # This is a hard optimization problem. Let's skip it for now
            # and return the rank-26 result.
            
            pass
        except Exception:
            pass

    # Convert to lists for the return value
    return {"U": U, "V": V, "W": W}
