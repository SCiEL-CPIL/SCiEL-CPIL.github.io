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

def _flatten_params(U, V, W):
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _unflatten_params(params, n, m, p, rank):
    size_u = n * m * rank
    size_v = m * p * rank
    U = params[:size_u].reshape(n * m, rank)
    V = params[size_u:size_u + size_v].reshape(m * p, rank)
    W = params[size_u + size_v:].reshape(p * n, rank)
    return U, V, W

def _residual_func(params, n, m, p, rank):
    U, V, W = _unflatten_params(params, n, m, p, rank)
    # Compute tensor T_calc
    # T_calc[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T_calc to equal the unit tensor T_true where T_true[i,j,k]=1 for all i,j,k?
    # No, for matrix multiplication, the tensor T is defined such that
    # C_ik = sum_j A_ij B_jk.
    # The decomposition is C_ik = sum_r (sum_j U_ij^r A_ij) (sum_k V_jk^r B_jk) W_ki^r ?
    # Wait, the standard definition is:
    # C_ik = sum_j sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} A_{ij} B_{jk}
    # So the tensor T has entries T_{ijk} = sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}
    # And we require T_{ijk} = 1 for all i, j, k?
    # Let's check: C_ik = sum_j A_ij B_jk.
    # If T_{ijk} = 1, then sum_j sum_r ... = sum_j 1 * A_ij B_jk = sum_j A_ij B_jk.
    # Yes, the target tensor is the all-ones tensor of shape (n, m, p).
    
    T_calc = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over r
                T_calc[i, j, k] = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
    
    # Target is 1.0 everywhere
    residual = T_calc - 1.0
    return residual.flatten()

def _try_rank_direct(r, n=2, m=4, p=7, max_iter=200, seed=0):
    """
    Try to find a rank-r decomposition using scipy.optimize.least_squares.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize with random sparse vectors
    # Scale: since we need sum of products to be 1, and we have r terms,
    # each term contributes ~1/r. So entries should be ~ sqrt(1/r) or similar.
    # Let's try normal distribution with small std.
    scale = 1.0 / np.sqrt(r)
    
    U0 = rng.randn(n * m, r) * scale
    V0 = rng.randn(m * p, r) * scale
    W0 = rng.randn(p * n, r) * scale
    
    params0 = _flatten_params(U0, V0, W0)
    
    def func(params):
        if _time_left() < 10:
            # Return large residual to stop optimization
            return np.ones(n*m*p) * 1e6
        return _residual_func(params, n, m, p, r)
    
    try:
        res = least_squares(func, params0, method='trf', max_nfev=10000, xtol=1e-8, ftol=1e-8, gtol=1e-8)
    except Exception:
        return None
    
    if res.cost < 1e-10:
        U, V, W = _unflatten_params(res.x, n, m, p, r)
        # Verify
        if _verify(U, V, W):
            return U, V, W
    return None

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
    
    # Try to find lower ranks using direct optimization
    # We'll try ranks 49 down to 45
    # Limit time per rank
    time_per_rank = 10.0
    
    for r in range(49, 44, -1):
        if _time_left() < time_per_rank + 5:
            break
            
        found = False
        for attempt in range(5):
            if _time_left() < time_per_rank / 2 + 5:
                break
            seed = r * 100 + attempt
            result = _try_rank_direct(r, seed=seed)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = r
                found = True
                break
        
        if found:
            # If we found a lower rank, we can stop or try even lower
            # But let's try to go lower if time permits
            continue
        else:
            # If we failed to find rank r, it's likely that r is too low
            # or we just didn't find it. Given the difficulty, we might stop.
            # However, maybe r=49 is possible but hard.
            # Let's assume if 49 fails, lower ones are less likely to succeed quickly.
            # But we can try a few more seeds for 49.
            pass
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
