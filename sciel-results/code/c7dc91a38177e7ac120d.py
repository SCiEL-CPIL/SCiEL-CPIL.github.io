"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(t_val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    def build_rank50():
        """
        Construct a rank-50 decomposition using Strassen for 2x2x2 blocks
        and schoolbook for 2x2x1 blocks.
        
        Partition j into {0,1} and {2,3}.
        For k-pairs (0,1), (2,3), (4,5): use Strassen 2x2x2 (rank 7 each).
        For k=6: use schoolbook 2x2x1 (rank 4 each).
        
        Total: 2 j-blocks * (3 * 7 + 4) = 2 * 25 = 50.
        """
        rank = 50
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0

        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # A[i0..i0+1, j0..j0+1]
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            # B[j0..j0+1, k0..k0+1]
            e = j0 * p + k0
            f = j0 * p + (k0 + 1)
            g = (j0 + 1) * p + k0
            h = (j0 + 1) * p + (k0 + 1)
            # C[k0..k0+1, i0..i0+1]
            c00 = k0 * n + i0
            c01 = (k0 + 1) * n + i0
            c10 = k0 * n + (i0 + 1)
            c11 = (k0 + 1) * n + (i0 + 1)

            # Strassen 2x2x2 multiplication
            # M1 = (a+d)(e+h) -> C00 += M1, C11 += M1
            U[a, col] += 1; U[d, col] += 1
            V[e, col] += 1; V[h, col] += 1
            W[c00, col] += 1; W[c11, col] += 1
            col += 1

            # M2 = (c+d)e -> C10 += M2, C11 -= M2
            U[c, col] += 1; U[d, col] += 1
            V[e, col] += 1
            W[c10, col] += 1; W[c11, col] -= 1
            col += 1

            # M3 = a(f-h) -> C01 += M3, C11 += M3
            U[a, col] += 1
            V[f, col] += 1; V[h, col] -= 1
            W[c01, col] += 1; W[c11, col] += 1
            col += 1

            # M4 = d(g-e) -> C00 += M4, C10 += M4
            U[d, col] += 1
            V[g, col] += 1; V[e, col] -= 1
            W[c00, col] += 1; W[c10, col] += 1
            col += 1

            # M5 = (a+b)h -> C00 -= M5, C01 += M5
            U[a, col] += 1; U[b, col] += 1
            V[h, col] += 1
            W[c00, col] -= 1; W[c01, col] += 1
            col += 1

            # M6 = (c-a)(e+f) -> C11 += M6
            U[c, col] += 1; U[a, col] -= 1
            V[e, col] += 1; V[f, col] += 1
            W[c11, col] += 1
            col += 1

            # M7 = (b-d)(g+h) -> C00 += M7
            U[b, col] += 1; U[d, col] -= 1
            V[g, col] += 1; V[h, col] += 1
            W[c00, col] += 1
            col += 1

        def add_schoolbook_221(i0, j0, k0):
            nonlocal col
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            g = (j0 + 1) * p + k0
            c00 = k0 * n + i0
            c10 = k0 * n + (i0 + 1)

            # a*e -> C00
            U[a, col] += 1; V[e, col] += 1; W[c00, col] += 1
            col += 1
            # b*g -> C00
            U[b, col] += 1; V[g, col] += 1; W[c00, col] += 1
            col += 1
            # c*e -> C10
            U[c, col] += 1; V[e, col] += 1; W[c10, col] += 1
            col += 1
            # d*g -> C10
            U[d, col] += 1; V[g, col] += 1; W[c10, col] += 1
            col += 1

        # Apply blocks
        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                if col + 7 > rank:
                    return None
                add_strassen_222(0, j_start, k_start)
            if col + 4 > rank:
                return None
            add_schoolbook_221(0, j_start, 6)

        if col != rank:
            return None
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _schoolbook_local(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Try the 50-rank construction
    result = build_rank50()
    if result is not None:
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
        else:
            return _schoolbook_local(n, m, p)
    
    # Fallback
    return _schoolbook_local(n, m, p)

# EVOLVE-BLOCK-END
