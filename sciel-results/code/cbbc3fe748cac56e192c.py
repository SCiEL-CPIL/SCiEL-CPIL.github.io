"""Matrix multiplication 2x3x3 rank-minimal decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 3]
DEADLINE = time.time() + 140  # Safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = [[0]*rank_school for _ in range(n*m)]
    V_s = [[0]*rank_school for _ in range(m*p)]
    W_s = [[0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    best = {"U": U_s, "V": V_s, "W": W_s}
    best_rank = rank_school

    # The constraint: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # Strategy: Fix U and V randomly (or structurally), then solve for W linearly.
    # For fixed U, V: each constraint gives a linear equation in W entries.
    # Constraint (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is linear in W[k*n+i, :] for each (i,j,k).
    # For fixed i,k: we have m equations (one per j) that involve W[k*n+i, :]
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for j=0..m-1
    # This is a system of m equations in p*n unknowns (W has p*n rows, each of length rank)
    # Wait, W[k*n+i, c] is one entry. For fixed (i,k), W[k*n+i, :] is a vector of length rank.
    # The m equations are: for j=0..m-1: dot(A_j, W[k*n+i,:]) = 1 where A_j[c] = U[i*m+j,c]*V[j*p+k,c]
    # So for each (i,k), we have m linear equations in rank unknowns (the row W[k*n+i,:])
    # If m < rank, we can have solutions.

    # Let's try: pick U, V with random half-integer values, then solve for W.
    # Then check if W entries are close to half-integers.

    def try_decomposition(rank, U, V):
        """Given U (n*m x rank) and V (m*p x rank), solve for W (p*n x rank)."""
        W = np.zeros((p*n, rank))
        for i in range(n):
            for k in range(p):
                # For j=0..m-1: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # Build system: A @ w = 1 where A[j,c] = U[i*m+j,c]*V[j*p+k,c]
                A = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # Solve A @ w = ones(m) for w (length rank)
                # Underdetermined if rank > m. Use least squares or pick a particular solution.
                try:
                    w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = w
                except:
                    return None
        return W

    def is_half_integer(arr, tol=1e-6):
        """Check if all entries are within tol of half-integers."""
        doubled = 2 * arr
        rounded = np.round(doubled)
        return np.all(np.abs(doubled - rounded) < tol)

    def snap_to_half_integer(arr):
        return np.round(2 * arr) / 2.0

    # Search for rank from 10 upward
    for rank in range(10, best_rank):
        if time.time() > DEADLINE:
            break
        found = False
        num_attempts = 500
        for attempt in range(num_attempts):
            if time.time() > DEADLINE:
                break
            # Random U and V with half-integer entries
            vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
            U = np.random.choice(vals, size=(n*m, rank))
            V = np.random.choice(vals, size=(m*p, rank))
            # Make sure no row is all zeros (to avoid degenerate cases)
            for i in range(n*m):
                if np.all(U[i] == 0):
                    U[i, 0] = 1
            for i in range(m*p):
                if np.all(V[i] == 0):
                    V[i, 0] = 1

            W = try_decomposition(rank, U, V)
            if W is None:
                continue

            # Check if W is close to half-integer
            if is_half_integer(W):
                W_snapped = snap_to_half_integer(W)
                # Verify the decomposition exactly
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W_snapped[k*n+i, c]
                            if abs(s - 1.0) > 1e-9:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
                    V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
                    W_list = [[float(W_snapped[i, c]) for c in range(rank)] for i in range(p*n)]
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    found = True
                    break
        if found:
            continue  # Try lower rank... wait, we're going upward. If found at rank, try rank-1?
            # Actually we're iterating rank from 10 upward. If we find at rank=10, great.
            # If not found at 10, try 11, etc.

    return best
