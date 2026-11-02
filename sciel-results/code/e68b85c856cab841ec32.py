"""Matrix multiplication 2x3x4 rank-minimal exact decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s

    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24

    # Build schoolbook solution (rank 24)
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank

    # Try to find rank 23 or 22 using a structured algebraic search
    for target_rank in [23, 22]:
        if time.time() > deadline:
            break
        result = _search_rank(n, m, p, target_rank, deadline)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break

    return best_result


def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction to avoid floating point issues."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(len(U[0])):
                    s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if s != Fraction(1):
                    return False
    return True


def _verify_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float verification."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _snap_to_rational(U, V, W, n, m, p, max_denom=8):
    """Snap to rationals with small denominators and verify exactly."""
    for denom in [1, 2, 4, 8, 16]:
        U_snap = np.round(np.array(U) * denom) / denom
        V_snap = np.round(np.array(V) * denom) / denom
        W_snap = np.round(np.array(W) * denom) / denom
        # Check if actually rational with this denom
        if np.allclose(U_snap * denom, np.round(U_snap * denom), atol=1e-6) and \
           np.allclose(V_snap * denom, np.round(V_snap * denom), atol=1e-6) and \
           np.allclose(W_snap * denom, np.round(W_snap * denom), atol=1e-6):
            U_int = np.round(U_snap * denom).astype(int)
            V_int = np.round(V_snap * denom).astype(int)
            W_int = np.round(W_snap * denom).astype(int)
            # Verify exactly
            U_frac = [[Fraction(a, denom) for a in row] for row in U_int]
            V_frac = [[Fraction(a, denom) for a in row] for row in V_int]
            W_frac = [[Fraction(a, denom) for a in row] for row in W_int]
            if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                return U_frac, V_frac, W_frac
    return None


def _search_rank(n, m, p, rank, deadline):
    """
    Structured search: start from schoolbook rank-24 and attempt to merge
    two rank-1 terms into one via a 2x2 sub-decomposition, or use a
    randomized algebraic perturbation with exact rational verification.
    """
    um, vm, wm = n*m, m*p, p*n

    # Strategy: Use the fact that for 2x3x4, we can try to find a rank-23
    # decomposition by perturbing the schoolbook. We use a local algebraic
    # approach: pick two schoolbook terms and try to replace them with one
    # term plus a correction, using the freedom in the remaining rank-22 terms.

    # More practically: use a randomized search over integer/half-integer
    # matrices with a smart initialization based on schoolbook.

    np.random.seed(12345)

    # Initialize with schoolbook truncated to `rank` columns, then add noise
    for attempt in range(200):
        if time.time() > deadline:
            break

        # Start from schoolbook, drop the last (24-rank) columns, add small perturbation
        U0 = np.zeros((um, rank))
        V0 = np.zeros((vm, rank))
        W0 = np.zeros((wm, rank))

        # Copy first `rank` schoolbook columns
        for r in range(rank):
            # Schoolbook column r corresponds to (i,j,k) = (r//12, (r//4)%3, r%4)
            # Actually let's just index properly
            i = r // (m*p)
            j = (r // p) % m
            k = r % p
            U0[i*m+j, r] = 1
            V0[j*p+k, r] = 1
            W0[k*n+i, r] = 1

        # Add random perturbation to the last few columns to allow flexibility
        # The dropped terms need to be compensated
        noise_scale = 0.5
        for r in range(max(0, rank-4), rank):
            U0[:, r] += np.random.choice([-1, -0.5, 0, 0.5, 1], size=um) * noise_scale
            V0[:, r] += np.random.choice([-1, -0.5, 0, 0.5, 1], size=vm) * noise_scale
            W0[:, r] += np.random.choice([-1, -0.5, 0, 0.5, 1], size=wm) * noise_scale

        # Quick float check
        if _verify_float(U0, V0, W0, n, m, p, tol=1e-4):
            snapped = _snap_to_rational(U0, V0, W0, n, m, p)
            if snapped is not None:
                Uf, Vf, Wf = snapped
                return {"U": Uf, "V": Vf, "W": Wf}

        # Try a local gradient step to reduce residual
        # Compute residual
        residuals = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    residuals[idx] = np.dot(U0[i*m+j], V0[j*p+k] * W0[k*n+i]) - 1.0
                    idx += 1

        if np.max(np.abs(residuals)) < 1e-6:
            snapped = _snap_to_rational(U0, V0, W0, n, m, p)
            if snapped is not None:
                Uf, Vf, Wf = snapped
                return {"U": Uf, "V": Vf, "W": Wf}

        # Simple coordinate descent: adjust one entry at a time to reduce max residual
        for step in range(50):
            if time.time() > deadline:
                break
            # Find worst (i,j,k)
            worst_idx = np.argmax(np.abs(residuals))
            wi = worst_idx // (m*p)
            wj = (worst_idx // p) % m
            wk = worst_idx % p

            # Try adjusting U[wi*m+wj, r] for each r
            improved = False
            for r in range(rank):
                for delta in [-1.0, -0.5, 0.5, 1.0]:
                    old = U0[wi*m+wj, r]
                    U0[wi*m+wj, r] = old + delta
                    # Recompute this residual
                    new_val = np.dot(U0[wi*m+wj], V0[wj*p+wk] * W0[wk*n+wi])
                    if abs(new_val - 1.0) < abs(residuals[worst_idx]):
                        residuals[worst_idx] = new_val - 1.0
                        improved = True
                        break
                    else:
                        U0[wi*m+wj, r] = old
                if improved:
                    break

            if not improved:
                # Try V
                for r in range(rank):
                    for delta in [-1.0, -0.5, 0.5, 1.0]:
                        old = V0[wj*p+wk, r]
                        V0[wj*p+wk, r] = old + delta
                        new_val = np.dot(U0[wi*m+wj], V0[wj*p+wk] * W0[wk*n+wi])
                        if abs(new_val - 1.0) < abs(residuals[worst_idx]):
                            residuals[worst_idx] = new_val - 1.0
                            improved = True
                            break
                        else:
                            V0[wj*p+wk, r] = old
                    if improved:
                        break

            if not improved:
                # Try W
                for r in range(rank):
                    for delta in [-1.0, -0.5, 0.5, 1.0]:
                        old = W0[wk*n+wi, r]
                        W0[wk*n+wi, r] = old + delta
                        new_val = np.dot(U0[wi*m+wj], V0[wj*p+wk] * W0[wk*n+wi])
                        if abs(new_val - 1.0) < abs(residuals[worst_idx]):
                            residuals[worst_idx] = new_val - 1.0
                            improved = True
                            break
                        else:
                            W0[wk*n+wi, r] = old
                    if improved:
                        break

            if np.max(np.abs(residuals)) < 1e-6:
                snapped = _snap_to_rational(U0, V0, W0, n, m, p)
                if snapped is not None:
                    Uf, Vf, Wf = snapped
                    return {"U": Uf, "V": Vf, "W": Wf}
                break

            if not improved:
                break

    return None

# EVOLVE-BLOCK-END
