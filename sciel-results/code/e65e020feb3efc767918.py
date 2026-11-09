"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _to_halfint_list(mat):
    """Convert a float matrix to a list-of-lists of integers/half-integers if possible, else round."""
    # Try to snap to nearest 0.5
    doubled = np.round(mat * 2) / 2.0
    out = []
    for row in doubled:
        r = []
        for val in row:
            if abs(val - round(val)) < 1e-9:
                r.append(int(round(val)))
            else:
                r.append(val)
        out.append(r)
    return out


def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact (up to numerical tolerance)."""
    rank = U.shape[1]
    # Tensor entry T[i*m+j, j*p+k, k*n+i] should be 1, else 0.
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # We can check via: for each r, the outer product U[:,r] * V[:,r] * W[:,r]
    # More efficiently: build the full tensor? That's n*m * m*p * p*n which for 2,3,10 is 6*30*20=3600, fine.
    # But let's just check a random sample or all non-zero positions.
    # Actually the tensor is sparse. Non-zero entries are at (i*m+j, j*p+k, k*n+i) for i,j,k.
    # Let's verify all such positions sum to 1, and a few zero positions sum to 0.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    # Check some zero positions: pick random triples that are NOT of the form above
    rng = np.random.default_rng(42)
    for _ in range(50):
        a = rng.integers(0, n * m)
        b = rng.integers(0, m * p)
        c = rng.integers(0, p * n)
        # Check if this is a "valid" triple
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if not (j == j2 and k == k2 and i == i2):
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > 1e-6:
                return False
    return True


def _try_als(n, m, p, target_rank, max_iter=200, tol=1e-10, n_restarts=20, deadline=None):
    """
    Try to find a rank-target_rank decomposition using ALS with random restarts.
    Returns (U, V, W) numpy arrays if found, else None.
    """
    rng = np.random.default_rng()
    best = None
    best_res = np.inf
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        U = rng.standard_normal((n * m, target_rank))
        V = rng.standard_normal((m * p, target_rank))
        W = rng.standard_normal((p * n, target_rank))
        # Normalize columns to avoid scale issues
        for r in range(target_rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 0: U[:, r] /= norm_u
            if norm_v > 0: V[:, r] /= norm_v
            if norm_w > 0: W[:, r] /= norm_w
        
        prev_res = np.inf
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            # Solve for U: fix V, W
            # For each column a of U: sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c]
            # This is a linear system. Build A where A[ab, r] = V[b,r]*W[c,r] for the fixed a? 
            # Actually, T is sparse. Let's build the full linear system for U.
            # Number of equations = n*m * m*p * p*n but sparse.
            # Better: iterate over all (a,b,c) with T[a,b,c]=1.
            # For each such triple, equation: sum_r U[a,r] * V[b,r] * W[c,r] = 1
            # Group by a: for fixed a, we have equations over (b,c) where T[a,b,c]=1.
            # But T[a,b,c]=1 only when a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
            # For fixed a = i*m+j, the valid (b,c) are: for each k, b=j*p+k, c=k*n+i.
            # So there are p equations for each a.
            # Let's build the matrix for each a.
            for a in range(n * m):
                i = a // m
                j = a % m
                # Equations: for k in 0..p-1: b = j*p+k, c = k*n+i
                # sum_r U[a,r] * V[b,r] * W[c,r] = 1
                # Build p x target_rank matrix M where M[k,r] = V[j*p+k, r] * W[k*n+i, r]
                M = np.zeros((p, target_rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                # Solve M @ U[a,:] = ones(p)
                # Use least squares
                U[a, :], res, rank_a, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
            
            # Solve for V: fix U, W
            for b in range(m * p):
                j = b // p
                k = b % p
                # Equations: for i in 0..n-1: a=i*m+j, c=k*n+i
                # sum_r U[a,r] * V[b,r] * W[c,r] = 1
                M = np.zeros((n, target_rank))
                for i in range(n):
                    a = i * m + j
                    c = k * n + i
                    M[i, :] = U[a, :] * W[c, :]
                V[b, :], res, rank_v, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
            
            # Solve for W: fix U, V
            for c in range(p * n):
                k = c // n
                i = c % n
                # Equations: for j in 0..m-1: a=i*m+j, b=j*p+k
                # sum_r U[a,r] * V[b,r] * W[c,r] = 1
                M = np.zeros((m, target_rank))
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    M[j, :] = U[a, :] * V[b, :]
                W[c, :], res, rank_w, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
            
            # Check residual
            # Compute max error over non-zero positions
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            # Also check zero positions (a few)
            for _ in range(10):
                a = rng.integers(0, n * m)
                b = rng.integers(0, m * p)
                c = rng.integers(0, p * n)
                i = a // m; j = a % m
                j2 = b // p; k = b % p
                k2 = c // n; i2 = c % n
                if not (j == j2 and k == k2 and i == i2):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > max_err:
                        max_err = abs(val)
            
            if max_err < tol:
                # Found a solution!
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2.0
                V_snap = np.round(V * 2) / 2.0
                W_snap = np.round(W * 2) / 2.0
                if _validate(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
                else:
                    # Return the float solution (the problem allows integer/half-integer, but let's check if raw floats are acceptable? No, contract says integer/half-integer)
                    # For now, return None and keep searching
                    pass
            
            if max_err < best_res:
                best_res = max_err
                best = (U.copy(), V.copy(), W.copy())
            
            if max_err < tol * 100:
                break
        
        if deadline is not None and time.time() > deadline:
            break
    
    return best


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Try to find a low-rank decomposition
    # Known lower bound for 2x3xp is 2p when p is a power of 2? Actually for 2x3x5, rank is 10.
    # For 2x3x10, rank is likely 20.
    for target_rank in [20, 21, 22, 24, 30, 40]:
        if time.time() > deadline - 10:
            break
        result = _try_als(n, m, p, target_rank, max_iter=100, tol=1e-10, n_restarts=10, deadline=deadline - 10)
        if result is not None:
            U, V, W = result
            return {
                "U": _to_halfint_list(U),
                "V": _to_halfint_list(V),
                "W": _to_halfint_list(W)
            }
    
    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {
        "U": _to_halfint_list(U),
        "V": _to_halfint_list(V),
        "W": _to_halfint_list(W)
    }

# EVOLVE-BLOCK-END
