"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p):
    """Fast verification using integer arithmetic.
    U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    Check: for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def _verify_via_matrices(U, V, W, n, m, p, trials=5):
    """Verify by actually multiplying random integer matrices."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    for _ in range(trials):
        A = np.random.randint(-2, 3, (n, m)).astype(np.int64)
        B = np.random.randint(-2, 3, (m, p)).astype(np.int64)
        C_expected = A @ B
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # = sum_c (sum_j A[i,j] * U[i*m+j,c]) * (sum_j B[j,k] * V[j*p+k,c]) * W[k*n+i,c]
        # Let u_c = U^T a, v_c = V^T b, then C[i,k] = sum_c u_c * v_c * W[k*n+i,c]
        a_flat = A.flatten()
        b_flat = B.flatten()
        u = Ua.T @ a_flat  # rank
        v = Va.T @ b_flat  # rank
        # C[i,k] = sum_c u[c]*v[c]*W[k*n+i,c]
        C_calc = np.zeros((n, p), dtype=np.int64)
        for i in range(n):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += u[c] * v[c] * Wa[k*n+i, c]
                C_calc[i, k] = s
        if not np.array_equal(C_calc, C_expected):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety

    schoolbook_rank = n * m * p  # 60
    # Build schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    best_rank = schoolbook_rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]

    # ---- Strategy: random low-rank integer search ----
    # For each candidate rank r, try to find integer U,V,W (entries in {-1,0,1} mostly)
    # such that the tensor identity holds. Use randomized search with local repair.
    #
    # The tensor has n*m*p = 60 nonzero entries each equal to 1.
    # We want sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all (i,j,k).

    def _random_search(rank, budget):
        """Randomized search for integer decomposition of given rank."""
        nonlocal best_rank, best_U, best_V, best_W
        # Represent each factor as a 2D array of small integers
        # Use numpy for speed
        Ua = np.zeros((n * m, rank), dtype=np.int64)
        Va = np.zeros((m * p, rank), dtype=np.int64)
        Wa = np.zeros((p * n, rank), dtype=np.int64)
        # Precompute index mapping for the 60 tensor positions
        positions = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    positions.append((i * m + j, j * p + k, k * n + i))
        npos = len(positions)

        def _residual(Ua, Va, Wa):
            """Compute max absolute residual over all positions."""
            # For each position (a,b,c) we need sum_k Ua[a,k]*Va[b,k]*Wa[c,k] == 1
            maxr = 0
            for (a, b, c) in positions:
                s = 0
                for k in range(rank):
                    s += Ua[a, k] * Va[b, k] * Wa[c, k]
                r = abs(s - 1)
                if r > maxr:
                    maxr = r
            return maxr

        # Random init with small integers
        for attempt in range(40):
            if time.time() > deadline - 20:
                return
            # Initialize: assign each of the 60 positions a random column
            # and set the three entries to 1, then add random noise
            Ua.fill(0)
            Va.fill(0)
            Wa.fill(0)
            for (a, b, c) in positions:
                col = random.randint(0, rank - 1)
                Ua[a, col] = 1
                Va[b, col] = 1
                Wa[c, col] = 1
            # Add random small perturbations
            for _ in range(rank // 2):
                a = random.randint(0, n * m - 1)
                b = random.randint(0, m * p - 1)
                c = random.randint(0, p * n - 1)
                ca = random.randint(0, rank - 1)
                cb = random.randint(0, rank - 1)
                cc = random.randint(0, rank - 1)
                Ua[a, ca] += random.choice([-1, 1])
                Va[b, cb] += random.choice([-1, 1])
                Wa[c, cc] += random.choice([-1, 1])

            # Local repair: iteratively fix positions with wrong residual
            improved = True
            stall = 0
            while improved and stall < 30 and time.time() < deadline - 20:
                improved = False
                for (a, b, c) in positions:
                    if time.time() > deadline - 20:
                        break
                    # compute residual
                    s = 0
                    for k in range(rank):
                        s += Ua[a, k] * Va[b, k] * Wa[c, k]
                    res = s - 1
                    if res == 0:
                        continue
                    # Try to fix by adjusting one entry
                    # Choose a random column and one of the three factors
                    col = random.randint(0, rank - 1)
                    cur_u = Ua[a, col]
                    cur_v = Va[b, col]
                    cur_w = Wa[c, col]
                    # We want to change res by -res.
                    # Changing Ua[a,col] by du changes s by du*cur_v*cur_w
                    # Try small du, dv, dw
                    found = False
                    for du in (-2, -1, 1, 2):
                        for dv in (-2, -1, 1, 2):
                            for dw in (-2, -1, 1, 2):
                                if du == 0 and dv == 0 and dw == 0:
                                    continue
                                new_u = cur_u + du
                                new_v = cur_v + dv
                                new_w = cur_w + dw
                                if abs(new_u) > 3 or abs(new_v) > 3 or abs(new_w) > 3:
                                    continue
                                # delta in s from this column
                                old_contrib = cur_u * cur_v * cur_w
                                new_contrib = new_u * new_v * new_w
                                delta = new_contrib - old_contrib
                                if res + delta == 0:
                                    Ua[a, col] = new_u
                                    Va[b, col] = new_v
                                    Wa[c, col] = new_w
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if not found:
                        stall += 1
                        if stall > 20:
                            break

            # Check full residual
            maxr = _residual(Ua, Va, Wa)
            if maxr == 0:
                # Success! Convert to list of lists
                rU = Ua.tolist()
                rV = Va.tolist()
                rW = Wa.tolist()
                if rank < best_rank:
                    best_rank = rank
                    best_U = rU
                    best_V = rV
                    best_W = rW
                return

    # Try decreasing ranks from schoolbook-1 down to some floor
    for target_rank in range(schoolbook_rank - 1, 20, -1):
        if time.time() > deadline - 30:
            break
        _random_search(target_rank, 2)

    # Final verification of best
    if _verify_fast(best_U, best_V, best_W, n, m, p):
        return {"U": best_U, "V": best_V, "W": best_W}

    # Fallback to schoolbook
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
