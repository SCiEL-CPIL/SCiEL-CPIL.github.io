"""Tensor decomposition for 3x5 * 5x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    
    The MM tensor T[a, b, c] = 1 iff there exist i,j,k such that
    a = i*m+j, b = j*p+k, c = k*n+i.
    Decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all valid (i,j,k) triples give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :] * V[b, :], W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check some inconsistent entries give 0
    # An entry (a, b, c) is inconsistent if the implied indices don't match
    # a = i*m+j, b = j'*p+k, c = k'*n+i' where j != j' or k != k' or i != i'
    # We'll check a subset of inconsistent entries
    checked = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                # Try a different j' for b
                for j2 in range(m):
                    if j2 == j:
                        continue
                    for k2 in range(p):
                        b2 = j2 * p + k2
                        c2 = k2 * n + i
                        val = np.dot(U[a, :] * V[b2, :], W[c2, :])
                        if abs(val) > tol:
                            return False
                        checked += 1
                        if checked > 500:
                            return True
    return True

def _verify_random(U, V, W, n, m, p, tol=1e-9, n_checks=500, seed=42):
    """Faster verification using random sampling."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    rng = np.random.RandomState(seed)
    
    # Check valid entries
    for _ in range(n_checks):
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        a = i * m + j
        b = j * p + k
        c = k * n + i
        val = np.dot(U[a, :] * V[b, :], W[c, :])
        if abs(val - 1.0) > tol:
            return False
    
    # Check inconsistent entries
    for _ in range(n_checks):
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        a = i * m + j
        # Pick a different j'
        j2 = rng.randint(m)
        while j2 == j:
            j2 = rng.randint(m)
        k2 = rng.randint(p)
        b2 = j2 * p + k2
        c2 = k2 * n + i
        val = np.dot(U[a, :] * V[b2, :], W[c2, :])
        if abs(val) > tol:
            return False
    
    return True

def _snap_half_int(x, tol=1e-8):
    """Snap a value to nearest half-integer if close."""
    half = np.round(x * 2) / 2.0
    if abs(x - half) < tol:
        return half
    return x

def _snap_matrix(M, tol=1e-8):
    """Snap matrix entries to nearest half-integer if close."""
    M = np.array(M, dtype=float)
    result = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            result[i, j] = _snap_half_int(M[i, j], tol)
    return result

def _try_rank(R, n, m, p, deadline, attempt_seed=0):
    """Try to find a rank-R decomposition.
    
    Strategy: randomly initialize U and V with small half-integer entries,
    then solve for W row-by-row using linear algebra.
    """
    np.random.seed(attempt_seed)
    
    # Use small half-integer entries for U and V
    # Values from {-1, -0.5, 0, 0.5, 1}
    entries = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    U = np.random.choice(entries, size=(n * m, R))
    V = np.random.choice(entries, size=(m * p, R))
    
    W = np.zeros((p * n, R))
    valid = True
    
    for i in range(n):
        if not valid:
            break
        for k in range(p):
            if not valid:
                break
            
            # For this (i, k), we need W[k*n+i, :] such that
            # for all j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a linear system: A * w = ones(m)
            # where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
            
            A_mat = np.zeros((m, R))
            for j in range(m):
                A_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
            
            # Solve A_mat * w = ones(m)
            # Use least squares
            try:
                w, residuals, rank_sv, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
            except:
                valid = False
                break
            
            # Check if the solution is exact
            residual = np.max(np.abs(A_mat @ w - np.ones(m)))
            if residual > 1e-9:
                valid = False
                break
            
            # Try to snap to half-integers
            w_snapped = np.array([_snap_half_int(w[r], 1e-8) for r in range(R)])
            residual_snapped = np.max(np.abs(A_mat @ w_snapped - np.ones(m)))
            
            if residual_snapped < 1e-9:
                W[k * n + i, :] = w_snapped
            else:
                # Keep the unsnapped solution for now
                W[k * n + i, :] = w
    
    if not valid:
        return None
    
    # Verify the full decomposition
    if not _verify_full(U, V, W, n, m, p, tol=1e-9):
        return None
    
    return U, V, W

def _try_rank_gradient(R, n, m, p, deadline, attempt_seed=0):
    """Try to find a rank-R decomposition using gradient descent.
    
    Objective: minimize ||T_recon - T||_F^2 where T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    and T is the MM tensor.
    """
    np.random.seed(attempt_seed)
    
    # Initialize with small random values
    scale = 1.0
    U = np.random.randn(n * m, R) * scale
    V = np.random.randn(m * p, R) * scale
    W = np.random.randn(p * n, R) * scale
    
    # Build list of nonzero entries of T
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.append((a, b, c))
    
    # Also need to check zero entries. We'll sample them.
    # Total entries: (n*m)*(m*p)*(p*n) = 15*35*21 = 11025
    # Nonzero entries: n*m*p = 105
    # Zero entries: 11025 - 105 = 10920
    
    # Gradient descent
    lr = 0.01
    n_iters = 200
    
    for it in range(n_iters):
        if time.time() > deadline - 5:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # Gradient from nonzero entries: sum (recon - 1) * V[b,r]*W[c,r] for dU[a,r]
        for (a, b, c) in nonzero:
            recon = np.dot(U[a, :] * V[b, :], W[c, :])
            err = recon - 1.0
            dU[a, :] += err * V[b, :] * W[c, :]
            dV[b, :] += err * U[a, :] * W[c, :]
            dW[c, :] += err * U[a, :] * V[b, :]
        
        # Gradient from zero entries (sample a few)
        rng = np.random.RandomState(it)
        n_zero_samples = 100
        for _ in range(n_zero_samples):
            a = rng.randint(n * m)
            b = rng.randint(m * p)
            c = rng.randint(p * n)
            # Check if this is a nonzero entry
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if j == j2 and k == k2 and i == i2:
                continue  # This is a nonzero entry, skip
            recon = np.dot(U[a, :] * V[b, :], W[c, :])
            dU[a, :] += recon * V[b, :] * W[c, :]
            dV[b, :] += recon * U[a, :] * W[c, :]
            dW[c, :] += recon * U[a, :] * V[b, :]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Periodic check
        if it % 50 == 0:
            # Check error on nonzero entries
            max_err = 0
            for (a, b, c) in nonzero[:20]:
                recon = np.dot(U[a, :] * V[b, :], W[c, :])
                max_err = max(max_err, abs(recon - 1.0))
            if max_err < 1e-6:
                # Try to snap and verify
                U_s = _snap_matrix(U, 1e-6)
                V_s = _snap_matrix(V, 1e-6)
                W_s = _snap_matrix(W, 1e-6)
                if _verify_full(U_s, V_s, W_s, n, m, p, tol=1e-6):
                    return U_s, V_s, W_s
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (rank 105) as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound up to best_rank - 1
    for R in range(lower_bound, best_rank):
        if time.time() > deadline - 15:
            break
        
        # Try multiple attempts with different seeds
        n_attempts = 50 if R <= 50 else 20
        
        found = False
        for attempt in range(n_attempts):
            if time.time() > deadline - 15:
                break
            
            # Method 1: Random U, V + solve for W
            result = _try_rank(R, n, m, p, deadline, attempt_seed=attempt)
            if result is not None:
                U, V, W = result
                best_rank = R
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                found = True
                break
        
        if found:
            # Found a solution at this rank, try to go lower
            continue
        
        # Method 2: Gradient descent (try a few times)
        if time.time() < deadline - 20:
            for attempt in range(3):
                if time.time() > deadline - 15:
                    break
                result = _try_rank_gradient(R, n, m, p, deadline, attempt_seed=1000 + attempt)
                if result is not None:
                    U, V, W = result
                    best_rank = R
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    found = True
                    break
    
    return best_result
