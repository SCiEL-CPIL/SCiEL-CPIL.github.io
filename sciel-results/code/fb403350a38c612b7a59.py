import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def get_T_u_v_w(u, v, w):
        i = u // m
        j = u % m
        j2 = v // p
        k = v % p
        k2 = w // n
        i2 = w % n
        if i == i2 and j == j2 and k == k2:
            return Fraction(1)
        return Fraction(0)

    def check_solution(U, V, W):
        max_err = Fraction(0)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    val = Fraction(0)
                    for r in range(U.shape[1]):
                        val += U[u, r] * V[v, r] * W[w, r]
                    err = abs(val - Fraction(1))
                    if err > max_err:
                        max_err = err
        # Check random zero entries
        import random
        random.seed(42)
        for _ in range(50):
            u = random.randint(0, n*m - 1)
            v = random.randint(0, m*p - 1)
            w = random.randint(0, p*n - 1)
            if get_T_u_v_w(u, v, w) == 0:
                val = Fraction(0)
                for r in range(U.shape[1]):
                    val += U[u, r] * V[v, r] * W[w, r]
                err = abs(val)
                if err > max_err:
                    max_err = err
        return max_err

    def als_step(U, V, W, rank):
        # Update U
        for i in range(n):
            for j in range(m):
                u = i * m + j
                B = [[Fraction(0) for _ in range(rank)] for _ in range(p)]
                b = [Fraction(1) for _ in range(p)]
                for k in range(p):
                    v = j * p + k
                    w = k * n + i
                    for r in range(rank):
                        B[k][r] = V[v, r] * W[w, r]
                # Solve B @ U[u, :] = b using Gaussian elimination or similar
                # Since rank is small, we can use a simple solver
                # We'll use a simplified approach: just update if we can, else skip
                # For now, we'll just do a few iterations of gradient descent or similar
                # Actually, implementing a full Fraction-based linear solver is complex.
                # Let's stick to the float-based search but verify with Fractions?
                # No, the search needs to be exact to converge to integers.
                # I will use a simple integer/half-integer snapping approach after float ALS.
                pass

    # Since exact Fraction-based ALS is complex and slow, we use float ALS but with stricter verification.
    # And we limit the search to ranks 35 and 34.
    
    def float_als(U, V, W, rank, max_iters=20):
        for _ in range(max_iters):
            # Update U
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    B = np.zeros((p, rank))
                    b = np.ones(p)
                    for k in range(p):
                        v = j * p + k
                        w = k * n + i
                        for r in range(rank):
                            B[k, r] = V[v, r] * W[w, r]
                    U[u, :], _, _, _ = np.linalg.lstsq(B, b, rcond=None)
            # Update V
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    B = np.zeros((n, rank))
                    b = np.ones(n)
                    for i in range(n):
                        u = i * m + j
                        w = k * n + i
                        for r in range(rank):
                            B[i, r] = U[u, r] * W[w, r]
                    V[v, :], _, _, _ = np.linalg.lstsq(B, b, rcond=None)
            # Update W
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    B = np.zeros((m, rank))
                    b = np.ones(m)
                    for j in range(m):
                        u = i * m + j
                        v = j * p + k
                        for r in range(rank):
                            B[j, r] = U[u, r] * V[v, r]
                    W[w, :], _, _, _ = np.linalg.lstsq(B, b, rcond=None)

    def check_float_solution(U, V, W):
        # Convert to Fractions for exact check
        U_f = [[Fraction(x).limit_denominator(1000) for x in row] for row in U]
        V_f = [[Fraction(x).limit_denominator(1000) for x in row] for row in V]
        W_f = [[Fraction(x).limit_denominator(1000) for x in row] for row in W]
        return check_solution(np.array(U_f), np.array(V_f), np.array(W_f))

    # Try ranks 35 and 34
    for rank in [35, 34]:
        if time.time() > deadline:
            break
        for restart in range(10):
            if time.time() > deadline:
                break
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            float_als(U, V, W, rank)
            # Snap to integers/half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            err = check_float_solution(U_snap, V_snap, W_snap)
            if err == 0:
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}

    # Fallback to schoolbook rank 36
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
