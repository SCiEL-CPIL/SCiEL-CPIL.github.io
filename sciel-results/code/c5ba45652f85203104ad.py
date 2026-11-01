import time
import numpy as np
from scipy.optimize import least_squares

DEADLINE = 120.0
START = time.time()

def _time_left():
    return DEADLINE - (time.time() - START)

def _build_schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                r += 1
    return U, V, W

def _build_strassen_222():
    """Rank-7 Strassen for 2x2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    # P1 = (A11+A22)(B11+B22)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # P2 = (A21+A22)A11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # P3 = A11(B12-B22)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # P4 = A22(B21-B11)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # P5 = (A11+A12)B22
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[3,4]=1
    # P6 = (A21-A11)(B11+B12)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # P7 = (A12-A22)(B21+B22)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[2,6]=-1
    return U, V, W

def _embed(t_U, t_V, t_W, n_src, m_src, p_src, n_tgt, m_tgt, p_tgt, n_map, m_map, p_map):
    """Embed a small tensor into a larger space."""
    rank = t_U.shape[1]
    U = np.zeros((n_tgt * m_tgt, rank))
    V = np.zeros((m_tgt * p_tgt, rank))
    W = np.zeros((p_tgt * n_tgt, rank))
    for i in range(n_src):
        for j in range(m_src):
            U[n_map[i]*m_tgt + m_map[j], :] = t_U[i*m_src+j, :]
    for j in range(m_src):
        for k in range(p_src):
            V[m_map[j]*p_tgt + p_map[k], :] = t_V[j*p_src+k, :]
    for k in range(p_src):
        for i in range(n_src):
            W[p_map[k]*n_tgt + n_map[i], :] = t_W[k*n_src+i, :]
    return U, V, W

def _tensor_sum(U_list, V_list, W_list, n, m, p):
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    return U, V, W

def _build_50_rank():
    """Build the rank-50 decomposition (Strassen-based)."""
    S_U, S_V, S_W = _build_strassen_222()
    
    # Build 2x4x4: four 2x2x2 Strassen blocks
    U_parts, V_parts, W_parts = [], [], []
    for m_off in [0, 2]:
        for p_off in [0, 2]:
            u, v, w = _embed(S_U, S_V, S_W, 2,2,2, 2,4,4,
                             [0,1], [m_off, m_off+1], [p_off, p_off+1])
            U_parts.append(u); V_parts.append(v); W_parts.append(w)
    U244, V244, W244 = _tensor_sum(U_parts, V_parts, W_parts, 2,4,4)
    
    # Build 2x4x3: two 2x2x3 blocks, each = Strassen(2x2x2) + schoolbook(2x2x1)
    sch_U, sch_V, sch_W = _build_schoolbook(2, 2, 1)
    # 2x2x3 = embed Strassen into p=[0,1] + embed schoolbook into p=[2]
    u_s, v_s, w_s = _embed(S_U, S_V, S_W, 2,2,2, 2,2,3, [0,1],[0,1],[0,1])
    u_sc, v_sc, w_sc = _embed(sch_U, sch_V, sch_W, 2,2,1, 2,2,3, [0,1],[0,1],[2])
    U223 = np.hstack([u_s, u_sc])
    V223 = np.hstack([v_s, v_sc])
    W223 = np.hstack([w_s, w_sc])
    
    U_parts, V_parts, W_parts = [], [], []
    for m_off in [0, 2]:
        u, v, w = _embed(U223, V223, W223, 2,2,3, 2,4,3,
                         [0,1], [m_off, m_off+1], [0,1,2])
        U_parts.append(u); V_parts.append(v); W_parts.append(w)
    U243, V243, W243 = _tensor_sum(U_parts, V_parts, W_parts, 2,4,3)
    
    # 2x4x7 = 2x4x4 (p=[0..3]) + 2x4x3 (p=[4..6])
    u1, v1, w1 = _embed(U244, V244, W244, 2,4,4, 2,4,7,
                        [0,1],[0,1,2,3],[0,1,2,3])
    u2, v2, w2 = _embed(U243, V243, W243, 2,4,3, 2,4,7,
                        [0,1],[0,1,2,3],[4,5,6])
    U = np.hstack([u1, u2])
    V = np.hstack([v1, v2])
    W = np.hstack([w1, w2])
    return U, V, W

def _verify(U, V, W, n=2, m=4, p=7):
    """Check decomposition correctness on random inputs."""
    np.random.seed(12345)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    C_calc = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_calc[i, k] = s
    return np.allclose(C_calc, C_true, atol=1e-8)

def _try_rank_r(r, n=2, m=4, p=7, max_iters=300, tol=1e-6, seed=0):
    """
    Try to find a rank-r decomposition using iterative Gauss-Seidel.
    The constraint is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    (unit tensor: each output entry gets exactly 1 from one rank-1 term, 0 from others).
    Actually the unit tensor has T[i,j,k] = 1 iff this is a valid "atom" position.
    For matrix mult, T[i,j,k] = 1 for all (i,j,k) — no wait.
    
    The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k.
    We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for ALL (i,j,k).
    
    This is a system of n*m*p = 56 quadratic equations in 3*r variables (well, 3*(n*m + m*p + p*n) entries, but only r columns each).
    
    Gauss-Seidel: fix all but one variable, solve for it.
    If we fix U and V, then for each (i,j,k): W[k*n+i, r] = 1 / (U[i*m+j,r] * V[j*p+k,r]) — but this must be consistent across all (j) for a given (i,k,r).
    
    Better: treat it as: for each rank slice r, we have vectors u_r (n*m), v_r (m*p), w_r (p*n).
    Constraint: for all (i,j,k): sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1.
    
    Iterative approach:
    1. Initialize with random values.
    2. For each (i,j,k), compute residual = 1 - sum_r u_r[i*m+j]*v_r[j*p+k]*w_r[k*n+i].
    3. Distribute correction to one of the three factors (cycle through i,j,k or r).
    
    Actually, a cleaner approach: use the fact that for a given r, if we fix all other ranks, the contribution of rank r to entry (i,j,k) is u_r[i*m+j]*v_r[j*p+k]*w_r[k*n+i]. We want the sum over r to equal 1.
    
    Let's use a different strategy: start from the rank-50 solution and try to merge two rank-1 terms into one, reducing rank by 1.
    """
    # Start from rank-50 solution and try to reduce
    if r >= 50:
        U0, V0, W0 = _build_50_rank()
        if r == 50:
            return U0, V0, W0
        # Truncate to rank r (just drop last columns) — won't be exact but gives starting point
        U0 = U0[:, :r]
        V0 = V0[:, :r]
        W0 = W0[:, :r]
    else:
        rng = np.random.RandomState(seed)
        U0 = rng.randn(n*m, r) * 0.1
        V0 = rng.randn(m*p, r) * 0.1
        W0 = rng.randn(p*n, r) * 0.1
    
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()
    
    # Precompute index triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Gauss-Seidel iterations
    for it in range(max_iters):
        if _time_left() < 20:
            break
        # Compute residuals
        residuals = np.zeros((n, m, p))
        for idx, (i, j, k) in enumerate(triples):
            s = 0.0
            for r_idx in range(r):
                s += U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx]
            residuals[i, j, k] = 1.0 - s
        
        if np.max(np.abs(residuals)) < tol:
            break
        
        # Update: for each triple, adjust one variable
        # Strategy: cycle through which variable to update
        phase = it % 3
        for idx, (i, j, k) in enumerate(triples):
            res = residuals[i, j, k]
            if abs(res) < tol:
                continue
            if phase == 0:
                # Update U[i*m+j, :]
                # res = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # We want to reduce |res|. Adjust U[i*m+j, r] by -res * V[j*p+k,r]*W[k*n+i,r] / (sum_r V[j*p+k,r]^2*W[k*n+i,r]^2)
                denom = 0.0
                for r_idx in range(r):
                    denom += V[j*p+k, r_idx]**2 * W[k*n+i, r_idx]**2
                if denom < 1e-12:
                    continue
                correction = res / denom
                for r_idx in range(r):
                    U[i*m+j, r_idx] -= correction * V[j*p+k, r_idx] * W[k*n+i, r_idx]
            elif phase == 1:
                # Update V[j*p+k, :]
                denom = 0.0
                for r_idx in range(r):
                    denom += U[i*m+j, r_idx]**2 * W[k*n+i, r_idx]**2
                if denom < 1e-12:
                    continue
                correction = res / denom
                for r_idx in range(r):
                    V[j*p+k, r_idx] -= correction * U[i*m+j, r_idx] * W[k*n+i, r_idx]
            else:
                # Update W[k*n+i, :]
                denom = 0.0
                for r_idx in range(r):
                    denom += U[i*m+j, r_idx]**2 * V[j*p+k, r_idx]**2
                if denom < 1e-12:
                    continue
                correction = res / denom
                for r_idx in range(r):
                    W[k*n+i, r_idx] -= correction * U[i*m+j, r_idx] * V[j*p+k, r_idx]
    
    return U, V, W

def construct():
    t0 = time.time()
    
    # First, get the known rank-50 solution
    U50, V50, W50 = _build_50_rank()
    if not _verify(U50, V50, W50):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook(2, 4, 7)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U50, V50, W50
    best_rank = 50
    
    # Try to find rank 49
    if _time_left() > 30:
        for attempt in range(5):
            if _time_left() < 25:
                break
            seed = 42 + attempt
            U49, V49, W49 = _try_rank_r(49, seed=seed)
            if _verify(U49, V49, W49):
                best_U, best_V, best_W = U49, V49, W49
                best_rank = 49
                break
    
    # Try rank 48 if time permits
    if best_rank == 49 and _time_left() > 30:
        for attempt in range(3):
            if _time_left() < 25:
                break
            seed = 100 + attempt
            U48, V48, W48 = _try_rank_r(48, seed=seed)
            if _verify(U48, V48, W48):
                best_U, best_V, best_W = U48, V48, W48
                best_rank = 48
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
