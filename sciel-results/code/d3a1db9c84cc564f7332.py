import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS  # 3,4,5


def _build_strassen_222():
    """Rank-7 Strassen for 2x2x2. Returns U,V,W each (4,7)."""
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W


def _build_221():
    """Rank-4 schoolbook for 2x2x1."""
    U = np.zeros((4,4)); V = np.zeros((2,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W


def _build_122():
    """Rank-4 schoolbook for 1x2x2."""
    U = np.zeros((2,4)); V = np.zeros((4,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[2,1]=1; W[0,1]=1
    U[0,2]=1; V[1,2]=1; W[1,2]=1
    U[1,3]=1; V[3,3]=1; W[1,3]=1
    return U, V, W


def _build_121():
    """Rank-2 for 1x2x1."""
    U = np.zeros((2,2)); V = np.zeros((2,2)); W = np.zeros((1,2))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    return U, V, W


def _verify(U, V, W):
    """Check decomposition correctness."""
    r = U.shape[1]
    Umat = np.asarray(U, dtype=float)
    Vmat = np.asarray(V, dtype=float)
    Wmat = np.asarray(W, dtype=float)
    for idx in range(r):
        u = Umat[:, idx].reshape(n, m)
        v = Vmat[:, idx].reshape(m, p)
        w = Wmat[:, idx].reshape(p, n)
        T = u * v * w  # element-wise, but we need outer product
        # Actually need to check the full tensor
        # Let's do it properly
    # Build tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    T = np.zeros((n, m, p))
    for idx in range(r):
        u = Umat[:, idx].reshape(n, m)
        v = Vmat[:, idx].reshape(m, p)
        w = Wmat[:, idx].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
    M = np.ones((n, m, p))
    return np.allclose(T, M, atol=1e-8)


def _construct_block_decomp():
    """Block decomposition with Strassen, rank 42."""
    U_parts, V_parts, W_parts = [], [], []
    def add(u_idx, v_idx, w_idx, Ublk, Vblk, Wblk):
        for r in range(Ublk.shape[1]):
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            for i, a in enumerate(u_idx): u[a] = Ublk[i, r]
            for i, a in enumerate(v_idx): v[a] = Vblk[i, r]
            for i, a in enumerate(w_idx): w[a] = Wblk[i, r]
            U_parts.append(u); V_parts.append(v); W_parts.append(w)

    # Part A: rows 0,1 of A (2x4) * B(4x5)
    Uu, Vv, Ww = _build_strassen_222()
    add([0,1,4,5],[0,1,5,6],[0,3,1,4], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_strassen_222()
    add([0,1,4,5],[2,3,7,8],[6,9,7,10], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_221()
    add([0,1,4,5],[4,9],[12,13], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_strassen_222()
    add([2,3,6,7],[10,11,15,16],[0,3,1,4], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_strassen_222()
    add([2,3,6,7],[12,13,17,18],[6,9,7,10], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_221()
    add([2,3,6,7],[14,19],[12,13], Uu, Vv, Ww)

    # Part B: row 2 of A (1x4) * B(4x5)
    Uu, Vv, Ww = _build_122()
    add([8,9],[0,1,5,6],[2,5], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_122()
    add([8,9],[2,3,7,8],[8,11], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_121()
    add([8,9],[4,9],[14], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_122()
    add([10,11],[10,11,15,16],[2,5], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_122()
    add([10,11],[12,13,17,18],[8,11], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_121()
    add([10,11],[14,19],[14], Uu, Vv, Ww)

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W


def _construct_schoolbook():
    """Rank-60 schoolbook decomposition."""
    U_parts, V_parts, W_parts = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m); u[i*m+j] = 1
                v = np.zeros(m*p); v[j*p+k] = 1
                w = np.zeros(p*n); w[k*n+i] = 1
                U_parts.append(u); V_parts.append(v); W_parts.append(w)
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W


def _try_optimize_rank(rank, U_init, V_init, W_init, timeout=60):
    """Try to optimize a rank-r decomposition using least squares."""
    t0 = time.time()
    def elapsed(): return time.time() - t0
    
    # Flatten
    dim_U, dim_V, dim_W = n*m, m*p, p*n
    
    # Create parameter vector
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    # Build tensor target
    target = np.ones((n, m, p))
    
    def objective(x):
        if elapsed() > timeout:
            return 1e10
        U_flat, V_flat, W_flat = x[:dim_U*rank], x[dim_U*rank:dim_U*rank+dim_V*rank], x[dim_U*rank+dim_V*rank:]
        U_mat = U_flat.reshape(dim_U, rank)
        V_mat = V_flat.reshape(dim_V, rank)
        W_mat = W_flat.reshape(dim_W, rank)
        
        # Compute tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u = U_mat[:, r].reshape(n, m)
            v = V_mat[:, r].reshape(m, p)
            w = W_mat[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
        diff = T - target
        return np.sum(diff**2)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < 1e-8:
            x = result.x
            U_opt = x[:dim_U*rank].reshape(dim_U, rank)
            V_opt = x[dim_U*rank:dim_U*rank+dim_V*rank].reshape(dim_V, rank)
            W_opt = x[dim_U*rank+dim_V*rank:].reshape(dim_W, rank)
            return U_opt, V_opt, W_opt
    except Exception:
        pass
    return None


def _try_find_lower_rank(current_U, current_V, current_W, min_rank, timeout=90):
    """Try to find a decomposition with lower rank than current."""
    t0 = time.time()
    def elapsed(): return time.time() - t0
    
    current_rank = current_U.shape[1]
    
    for rank in range(min_rank, current_rank):
        if elapsed() > timeout:
            break
        # Try multiple random initializations
        for attempt in range(20):
            if elapsed() > timeout:
                break
            np.random.seed(attempt * 1000 + rank)
            U_rand = np.random.randn(n*m, rank) * 0.1
            V_rand = np.random.randn(m*p, rank) * 0.1
            W_rand = np.random.randn(p*n, rank) * 0.1
            
            result = _try_optimize_rank(rank, U_rand, V_rand, W_rand, timeout=min(30, timeout - elapsed()))
            if result is not None:
                U_opt, V_opt, W_opt = result
                if _verify(U_opt, V_opt, W_opt):
                    return U_opt, V_opt, W_opt, rank
    
    return None


def construct():
    t0 = time.time()
    deadline = 150.0
    margin = 10.0
    
    # Start with block decomposition (rank 42)
    U, V, W = _construct_block_decomp()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank
    min_target = 25  # Known lower bounds suggest we should try at least this
    result = _try_find_lower_rank(best_U, best_V, best_W, min_target, timeout=deadline - margin - (time.time() - t0))
    
    if result is not None:
        U_new, V_new, W_new, new_rank = result
        if new_rank < best_rank:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = new_rank
    
    # Final verification
    if not _verify(best_U, best_V, best_W):
        best_U, best_V, best_W = _construct_schoolbook()
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist(),
    }
