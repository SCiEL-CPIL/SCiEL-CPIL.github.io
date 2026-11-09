"""Matrix multiplication 4x4x6 rank-minimal exact decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _build_strassen():
    """Build the explicit block-Strassen rank-84 decomposition (guaranteed valid)."""
    n, m, p = DIMENSIONS

    rank = 84
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    A_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
    }
    B_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (0, 2): (0, 1, 4, 5),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
        (1, 2): (2, 3, 4, 5),
    }
    C_blocks = {
        (0, 0): (0, 1, 0, 1),
        (0, 1): (0, 1, 2, 3),
        (0, 2): (0, 1, 4, 5),
        (1, 0): (2, 3, 0, 1),
        (1, 1): (2, 3, 2, 3),
        (1, 2): (2, 3, 4, 5),
    }

    pairs = [
        ((0, 0), (0, 0), (0, 0)),
        ((0, 1), (1, 0), (0, 0)),
        ((0, 0), (0, 1), (0, 1)),
        ((0, 1), (1, 1), (0, 1)),
        ((0, 0), (0, 2), (0, 2)),
        ((0, 1), (1, 2), (0, 2)),
        ((1, 0), (0, 0), (1, 0)),
        ((1, 1), (1, 0), (1, 0)),
        ((1, 0), (0, 1), (1, 1)),
        ((1, 1), (1, 1), (1, 1)),
        ((1, 0), (0, 2), (1, 2)),
        ((1, 1), (1, 2), (1, 2)),
    ]

    # Strassen 2x2x2 terms: local index 0=(0,0),1=(0,1),2=(1,0),3=(1,1)
    strassen_terms = [
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]},
    ]

    col = 0
    for (a_idx, b_idx, c_idx) in pairs:
        a_r1, a_r2, a_c1, a_c2 = A_blocks[a_idx]
        b_r1, b_r2, b_c1, b_c2 = B_blocks[b_idx]
        c_r1, c_r2, c_c1, c_c2 = C_blocks[c_idx]

        for term in strassen_terms:
            for local_a, coeff in term['u']:
                lr, lc = divmod(local_a, 2)
                U[(a_r1 + lr) * m + (a_c1 + lc), col] += coeff
            for local_b, coeff in term['v']:
                lr, lc = divmod(local_b, 2)
                V[(b_r1 + lr) * p + (b_c1 + lc), col] += coeff
            for local_c, coeff in term['w']:
                lr, lc = divmod(local_c, 2)
                # W[k*n + i] where k=col of C, i=row of C
                W[(c_c1 + lc) * n + (c_r1 + lr), col] += coeff
            col += 1

    return U, V, W


def _verify_full(U, V, W, n, m, p):
    """Check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                if abs(val - 1.0) > 1e-6:
                    return False
    return True


def _verify_random(U, V, W, n, m, p, trials=5):
    """Verify by actually multiplying random integer matrices."""
    rank = U.shape[1]
    for _ in range(trials):
        A = np.random.randint(0, 5, (n, m), dtype=float)
        B = np.random.randint(0, 5, (m, p), dtype=float)
        C = A @ B
        # C[i,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]*A[i,j]*B[j,k]
        C2 = np.zeros((n, p), dtype=float)
        for r in range(rank):
            Au = A * U[:, r].reshape(n, m)
            Bv = B * V[:, r].reshape(m, p)
            C2 += W[:, r].reshape(p, n).T * (Au @ Bv)
        if not np.allclose(C2, C, atol=1e-5):
            return False
    return True


def _snap_to_halfint(U, V, W):
    """Snap near-integer/half-integer values exactly; return None if any value is far off."""
    for M in (U, V, W):
        for r in range(M.shape[1]):
            for row in range(M.shape[0]):
                v = M[row, r]
                if v == 0:
                    continue
                # nearest half-integer
                h = np.round(2 * v) / 2
                if abs(v - h) > 1e-3:
                    return None
                M[row, r] = h
    return (U, V, W)


def _try_optimize(U0, V0, W0, target_rank, n, m, p, deadline, seed):
    """Attempt to find a rank-target decomposition via iterative refinement.
    Returns (U,V,W) or None."""
    if target_rank >= U0.shape[1]:
        return None
    rng = np.random.RandomState(seed)
    rank = target_rank

    # Initialize from a random subset of the valid rank-84 columns
    idx = rng.choice(U0.shape[1], size=rank, replace=False)
    U = U0[:, idx].copy()
    V = V0[:, idx].copy()
    W = W0[:, idx].copy()

    # Add a small random perturbation to break symmetry
    U += rng.randn(n * m, rank) * 1e-3
    V += rng.randn(m * p, rank) * 1e-3
    W += rng.randn(p * n, rank) * 1e-3

    # Precompute index lists
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    def compute_errors():
        """Compute error for all triples vectorized using einsum."""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return T.flatten() - 1.0

    # Optimization loop
    lr = 0.05
    for it in range(10000):
        if time.time() > deadline:
            break
        e = compute_errors()
        max_err_idx = int(np.argmax(np.abs(e)))
        max_err = e[max_err_idx]
        
        if abs(max_err) < 1e-9:
            break
        
        i, j, k = triples[max_err_idx]
        
        # Gradients
        dU = V[j * p + k, :] * W[k * n + i, :]
        dV = U[i * m + j, :] * W[k * n + i, :]
        dW = U[i * m + j, :] * V[j * p + k, :]
        
        # Update
        U[i * m + j, :] -= lr * max_err * dU
        V[j * p + k, :] -= lr * max_err * dV
        W[k * n + i, :] -= lr * max_err * dW

    # Final polish
    lr = 0.01
    for it in range(5000):
        if time.time() > deadline:
            break
        e = compute_errors()
        max_err_idx = int(np.argmax(np.abs(e)))
        max_err = e[max_err_idx]
        if abs(max_err) < 1e-11:
            break
        i, j, k = triples[max_err_idx]
        dU = V[j * p + k, :] * W[k * n + i, :]
        dV = U[i * m + j, :] * W[k * n + i, :]
        dW = U[i * m + j, :] * V[j * p + k, :]
        U[i * m + j, :] -= lr * max_err * dU
        V[j * p + k, :] -= lr * max_err * dV
        W[k * n + i, :] -= lr * max_err * dW

    if not _verify_full(U, V, W, n, m, p):
        return None
    snapped = _snap_to_halfint(U.copy(), V.copy(), W.copy())
    if snapped is None:
        return None
    U, V, W = snapped
    if not _verify_full(U, V, W, n, m, p):
        return None
    if not _verify_random(U, V, W, n, m, p):
        return None
    return (U, V, W)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    n, m, p = DIMENSIONS

    # Guaranteed baseline: block-Strassen rank 84
    U, V, W = _build_strassen()
    best = (U.copy(), V.copy(), W.copy())
    best_rank = 84

    # Try to find lower ranks via optimization
    # We try a range of ranks. Since optimization is now faster, we can try more.
    # We'll try descending ranks.
    for target_rank in range(70, 84):
        if time.time() > deadline:
            break
        # Try multiple seeds for each rank to increase chance of success
        for seed in range(target_rank, target_rank + 5):
             if time.time() > deadline:
                 break
             res = _try_optimize(best[0], best[1], best[2], target_rank, n, m, p, deadline, seed)
             if res is not None:
                 best = res
                 best_rank = target_rank
                 break # Move to next lower rank
        
    U, V, W = best

    # Convert to list-of-lists as required by interface
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
