"""Bootstrap SEARCH for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_rank(U, V, W, n=2, m=5, p=6):
    """Check if given factor matrices produce correct matmul tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # On-diagonal: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                if abs(val - 1.0) > 1e-8:
                    return False
    
    # Off-diagonal: for (i,j,k) where the indices don't form a valid matmul entry
    # T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i' should be 0 unless j==j' and k==k' and i==i'
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # Already checked on-diagonal above
                # Check off-diagonal: vary one index at a time
                # Vary i: i2 != i, same j, same k
                for i2 in range(n):
                    if i2 == i:
                        continue
                    a2 = i2 * m + j
                    c2 = k * n + i2
                    val = np.dot(U[a2] * V[b], W[c2])
                    if abs(val) > 1e-8:
                        return False
                # Vary k: k2 != k, same i, same j
                for k2 in range(p):
                    if k2 == k:
                        continue
                    b2 = j * p + k2
                    c2 = k2 * n + i
                    val = np.dot(U[a] * V[b2], W[c2])
                    if abs(val) > 1e-8:
                        return False
                # Vary j: j2 != j, need to check combinations
                for j2 in range(m):
                    if j2 == j:
                        continue
                    # When j changes, the coupling breaks
                    # T[i,j2,k] with a=i*m+j2, b=j*p+k (j!=j2), c=k*n+i
                    # This is off-diagonal in the j-index
                    a2 = i * m + j2
                    val = np.dot(U[a2] * V[b], W[c])
                    if abs(val) > 1e-8:
                        return False
                    # T[i,j,k2] with a=i*m+j, b=j2*p+k2, c=k2*n+i
                    for k2 in range(p):
                        b2 = j2 * p + k2
                        c2 = k2 * n + i
                        val = np.dot(U[a] * V[b2], W[c2])
                        if abs(val) > 1e-8:
                            return False
    return True

def _schoolbook_rank():
    """Return the standard schoolbook decomposition."""
    n, m, p = DIMENSIONS
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

def _build_kr_matrices(n, m, p):
    """Precompute Kronecker-product design matrices for ALS."""
    nm, mp, pn = n * m, m * p, p * n
    
    # For solving W: rows indexed by (a,b) = (i*m+j, j*p+k), only valid when j matches
    # KR_W[a*mp+b, c] = U[a,c] * V[b,c] -> this is outer product structure
    # We need: for each c in pn, target has 1 at (a,b) where a=i*m+j, b=j*p+k, c=k*n+i
    # KR_W is (nm*mp) x rank, entry (a*mp+b, c) = U[a,c]*V[b,c]
    # This equals kron(V, U) reshaped: actually it's (U ⊗ V) in a specific ordering
    # KR_W[a*mp+b, :] = U[a,:] * V[b,:] (element-wise)
    # This is the Khatri-Rao product structure
    
    # Precompute index mappings
    # For W-solve: for each c=k*n+i, the valid (a,b) pairs are a=i*m+j, b=j*p+k for j in range(m)
    W_targets = {}
    for i in range(n):
        for k in range(p):
            c = k * n + i
            indices = []
            for j in range(m):
                a = i * m + j
                b = j * p + k
                indices.append(a * mp + b)
            W_targets[c] = indices
    
    # For V-solve: for each b=j*p+k, valid (c,a) pairs: c=k*n+i, a=i*m+j
    V_targets = {}
    for j in range(m):
        for k in range(p):
            b = j * p + k
            indices = []
            for i in range(n):
                c = k * n + i
                a = i * m + j
                indices.append(c * nm + a)
            V_targets[b] = indices
    
    # For U-solve: for each a=i*m+j, valid (b,c) pairs: b=j*p+k, c=k*n+i
    U_targets = {}
    for i in range(n):
        for j in range(m):
            a = i * m + j
            indices = []
            for k in range(p):
                b = j * p + k
                c = k * n + i
                indices.append(b * pn + c)
            U_targets[a] = indices
    
    return W_targets, V_targets, U_targets


def _als_solve(U, V, W, n=2, m=5, p=6, max_iter=80, tol=1e-9, deadline=None):
    """ALS for tensor decomposition. Returns (U, V, W, success)."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    U, V, W = U.astype(np.float64).copy(), V.astype(np.float64).copy(), W.astype(np.float64).copy()
    
    W_targets, V_targets, U_targets = _build_kr_matrices(n, m, p)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            return U, V, W, False
        
        # Solve for W: KR_W[a*mp+b, :] = U[a,:] * V[b,:]
        # Vectorized: KR_W = (U[:,None,:] * V[None,:,:]).reshape(nm*mp, rank)
        KR_W = (U[:, None, :] * V[None, :, :]).reshape(nm * mp, rank)
        KR_W_T = KR_W.T  # (rank, nm*mp)
        
        W_new = np.zeros((pn, rank))
        for c in range(pn):
            tgt = np.zeros(nm * mp)
            for idx in W_targets[c]:
                tgt[idx] = 1.0
            try:
                W_new[c] = np.linalg.lstsq(KR_W_T, tgt, rcond=None)[0]
            except:
                return U, V, W, False
        W = W_new
        
        # Solve for V: KR_V[c*nm+a, :] = W[c,:] * U[a,:]
        KR_V = (W[:, None, :] * U[None, :, :]).reshape(pn * nm, rank)
        KR_V_T = KR_V.T
        
        V_new = np.zeros((mp, rank))
        for b in range(mp):
            tgt = np.zeros(pn * nm)
            for idx in V_targets[b]:
                tgt[idx] = 1.0
            try:
                V_new[b] = np.linalg.lstsq(KR_V_T, tgt, rcond=None)[0]
            except:
                return U, V, W, False
        V = V_new
        
        # Solve for U: KR_U[b*pn+c, :] = V[b,:] * W[c,:]
        KR_U = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, rank)
        KR_U_T = KR_U.T
        
        U_new = np.zeros((nm, rank))
        for a in range(nm):
            tgt = np.zeros(mp * pn)
            for idx in U_targets[a]:
                tgt[idx] = 1.0
            try:
                U_new[a] = np.linalg.lstsq(KR_U_T, tgt, rcond=None)[0]
            except:
                return U, V, W, False
        U = U_new
        
        # Check convergence (vectorized)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i * m + j, j * p + k, k * n + i
                    err = max(err, abs(np.dot(U[a] * V[b], W[c]) - 1.0))
        if err < tol:
            # Quick off-diagonal check
            off = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a, b, c = i * m + j, j * p + k, k * n + i
                        for i2 in range(n):
                            if i2 == i: continue
                            val = np.dot(U[i2*m+j] * V[b], W[k*n+i2])
                            off = max(off, abs(val))
                        for k2 in range(p):
                            if k2 == k: continue
                            val = np.dot(U[a] * V[j*p+k2], W[k2*n+i])
                            off = max(off, abs(val))
            if off < tol:
                return U, V, W, True
    return U, V, W, False

def _polish(U, V, W, n=2, m=5, p=6, max_iter=50, tol=1e-12, deadline=None):
    """High-precision polish pass with tighter tolerance."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    U, V, W = U.astype(np.float64).copy(), V.astype(np.float64).copy(), W.astype(np.float64).copy()
    
    W_targets, V_targets, U_targets = _build_kr_matrices(n, m, p)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            return U, V, W, False
        
        KR_W = (U[:, None, :] * V[None, :, :]).reshape(nm * mp, rank)
        KR_W_T = KR_W.T
        W_new = np.zeros((pn, rank))
        for c in range(pn):
            tgt = np.zeros(nm * mp)
            for idx in W_targets[c]:
                tgt[idx] = 1.0
            try:
                W_new[c] = np.linalg.lstsq(KR_W_T, tgt, rcond=None)[0]
            except:
                return U, V, W, False
        W = W_new
        
        KR_V = (W[:, None, :] * U[None, :, :]).reshape(pn * nm, rank)
        KR_V_T = KR_V.T
        V_new = np.zeros((mp, rank))
        for b in range(mp):
            tgt = np.zeros(pn * nm)
            for idx in V_targets[b]:
                tgt[idx] = 1.0
            try:
                V_new[b] = np.linalg.lstsq(KR_V_T, tgt, rcond=None)[0]
            except:
                return U, V, W, False
        V = V_new
        
        KR_U = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, rank)
        KR_U_T = KR_U.T
        U_new = np.zeros((nm, rank))
        for a in range(nm):
            tgt = np.zeros(mp * pn)
            for idx in U_targets[a]:
                tgt[idx] = 1.0
            try:
                U_new[a] = np.linalg.lstsq(KR_U_T, tgt, rcond=None)[0]
            except:
                return U, V, W, False
        U = U_new
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i * m + j, j * p + k, k * n + i
                    err = max(err, abs(np.dot(U[a] * V[b], W[c]) - 1.0))
        if err < tol:
            return U, V, W, True
    return U, V, W, False


def _try_rank(r, n=2, m=5, p=6, n_trials=3, deadline=None):
    """Try rank-r decomposition via ALS with polish."""
    nm, mp, pn = n * m, m * p, p * n
    rng = np.random.default_rng(42)
    for trial in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        U = rng.standard_normal((nm, r)) * 0.1
        V = rng.standard_normal((mp, r)) * 0.1
        W = rng.standard_normal((pn, r)) * 0.1
        U, V, W, ok = _als_solve(U, V, W, n, m, p, 80, 1e-7, deadline)
        if ok:
            # Polish with tighter tolerance
            U, V, W, ok2 = _polish(U, V, W, n, m, p, 40, 1e-11, deadline)
            if ok2:
                res = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                if _verify_rank(res["U"], res["V"], res["W"], n, m, p):
                    return res
            else:
                # Even without polish convergence, try to verify
                res = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                if _verify_rank(res["U"], res["V"], res["W"], n, m, p):
                    return res
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 125  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    fb = _schoolbook_rank()
    for r in [59, 58, 57]:
        if time.time() > deadline:
            break
        res = _try_rank(r, n, m, p, 3, deadline)
        if res is not None:
            return res
    return fb

# EVOLVE-BLOCK-END
