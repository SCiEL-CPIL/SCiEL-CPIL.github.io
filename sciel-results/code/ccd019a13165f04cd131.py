"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = t0 + 130.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - t0

    def schoolbook():
        """Return schoolbook decomposition of rank n*m*p = 48."""
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
        return U, V, W

    def verify_decomp(U, V, W, rank):
        """Exact verification using Fraction arithmetic."""
        Uf = [[Fraction(x) for x in row] for row in U]
        Vf = [[Fraction(x) for x in row] for row in V]
        Wf = [[Fraction(x) for x in row] for row in W]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def round_to_half(vals):
        """Round list of floats to nearest half-integer, return as int or float."""
        out = []
        for x in vals:
            r = round(x * 2) / 2
            if r == int(r):
                out.append(int(r))
            else:
                out.append(r)
        return out

    def round_mat(M):
        """Round a 2D list of floats to half-integers."""
        return [round_to_half(row) for row in M]

    def try_optimize(rank, budget_sec=10.0, seed=0):
        """Try to find a rank-`rank` decomposition using alternating least squares.
           Returns (U, V, W) as lists of lists, or None."""
        if elapsed() > DEADLINE - budget_sec:
            return None

        np.random.seed(seed)

        # Initialize: take first `rank` columns of schoolbook + noise
        U0, V0, W0 = schoolbook()
        U = np.array(U0, dtype=np.float64)[:, :rank]
        V = np.array(V0, dtype=np.float64)[:, :rank]
        W = np.array(W0, dtype=np.float64)[:, :rank]

        # Add small noise to avoid exact degenerate start
        U += np.random.randn(*U.shape) * 0.05
        V += np.random.randn(*V.shape) * 0.05
        W += np.random.randn(*W.shape) * 0.05

        # We'll do alternating least squares.
        # For fixed V, W: solve for U row by row (each row is a rank-rank vector).
        # Equation: for each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # For fixed (i,j): we have p equations in rank unknowns U[i*m+j,0..rank-1].
        # Let a_c = sum over... actually:
        # For fixed i,j: the equation over k is:
        #   sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
        # So define for each k: b_c(k) = V[j*p+k,c] * W[k*n+i,c]
        # Then: sum_c U[i*m+j,c] * b_c(k) = 1 for k=0..p-1
        # This is a linear system: B * u = ones(p), where B[k,c] = b_c(k), u = U[i*m+j,:]
        # B is p x rank, u is rank x 1. If rank >= p, we can find a solution.

        for it in range(300):
            if elapsed() > DEADLINE - 5:
                return None

            # Update U: for each (i,j), solve linear system
            for i in range(n):
                for j in range(m):
                    # Build p x rank matrix B
                    B = np.zeros((p, rank), dtype=np.float64)
                    for k in range(p):
                        for c in range(rank):
                            B[k, c] = V[j * p + k, c] * W[k * n + i, c]
                    # Solve B * u = ones(p) using lstsq
                    try:
                        u, *_ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                    except:
                        pass

            # Update V: for each (j,k), solve linear system
            for j in range(m):
                for k in range(p):
                    # Build m x rank matrix B
                    B = np.zeros((n * m, rank), dtype=np.float64)  # Actually need to think about this
                    # For fixed (j,k): equation is sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for each i
                    # So: B[i,c] = U[i*m+j,c] * W[k*n+i,c], unknown is v = V[j*p+k,:]
                    # n equations in rank unknowns
                    B = np.zeros((n, rank), dtype=np.float64)
                    for i in range(n):
                        for c in range(rank):
                            B[i, c] = U[i * m + j, c] * W[k * n + i, c]
                    try:
                        v, *_ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                    except:
                        pass

            # Update W: for each (i,k), solve linear system
            for i in range(n):
                for k in range(p):
                    # For fixed (i,k): equation is sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for each j
                    # So: B[j,c] = U[i*m+j,c] * V[j*p+k,c], unknown is w = W[k*n+i,:]
                    # m equations in rank unknowns
                    B = np.zeros((m, rank), dtype=np.float64)
                    for j in range(m):
                        for c in range(rank):
                            B[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    try:
                        w, *_ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                    except:
                        pass

            # Check convergence: compute max residual
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        res = abs(1.0 - s)
                        if res > max_res:
                            max_res = res

            if max_res < 1e-9:
                # Round to half-integers and verify
                U_r = round_mat(U.tolist())
                V_r = round_mat(V.tolist())
                W_r = round_mat(W.tolist())
                if verify_decomp(U_r, V_r, W_r, rank):
                    return U_r, V_r, W_r

            # Also periodically check rounding without convergence
            if it % 50 == 49:
                U_r = round_mat(U.tolist())
                V_r = round_mat(V.tolist())
                W_r = round_mat(W.tolist())
                if verify_decomp(U_r, V_r, W_r, rank):
                    return U_r, V_r, W_r

        return None

    # First: get schoolbook as fallback
    U_sch, V_sch, W_sch = schoolbook()

    # Try to find lower rank: 47, 46, ...
    for target_rank in range(47, max(n*m, m*p, p*n) - 1, -1):
        if elapsed() > DEADLINE - 15:
            break
        found = try_optimize(target_rank, budget_sec=8.0, seed=42 + target_rank)
        if found is not None:
            U, V, W = found
            if verify_decomp(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}

    # Return schoolbook
    return {"U": U_sch, "V": V_sch, "W": W_sch}
# EVOLVE-BLOCK-END
