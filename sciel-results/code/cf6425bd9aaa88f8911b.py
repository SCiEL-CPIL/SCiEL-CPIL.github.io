import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_np(U_np, V_np, W_np, n, m, p, tol=1e-6):
    """Vectorized verification using numpy."""
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    """Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere."""
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def _build_block_config(n, m, p, i0, j0, k_offsets):
    """Build a specific block configuration."""
    blocks = []
    for ko in k_offsets:
        if ko >= 0 and ko + 2 <= p and i0 + 2 <= n and j0 + 2 <= m:
            blocks.append((i0, j0, ko))
    return blocks

def _try_rank_reduction(U_init, V_init, W_init, n, m, p, target_rank, time_limit=20.0):
    """
    Attempt to find a decomposition of rank target_rank using optimization.
    Uses vectorized residual computation for speed.
    """
    start_time = time.time()
    
    U_np = np.array(U_init, dtype=float)
    V_np = np.array(V_init, dtype=float)
    W_np = np.array(W_init, dtype=float)
    
    nm = n * m
    mp = m * p
    pn = p * n
    total_params = nm * target_rank + mp * target_rank + pn * target_rank
    
    # Precompute index mappings for fast residual
    # For each (i,j,k), we need U[i*m+j], V[j*p+k], W[k*n+i]
    idx_U = np.array([[i*m + j for j in range(m)] for i in range(n)])  # (n,m)
    idx_V = np.array([[j*p + k for k in range(p)] for j in range(m)])  # (m,p)
    idx_W = np.array([[k*n + i for i in range(n)] for k in range(p)])  # (p,n)
    
    def residuals(params):
        U_flat = params[:nm*target_rank]
        V_flat = params[nm*target_rank : nm*target_rank + mp*target_rank]
        W_flat = params[nm*target_rank + mp*target_rank : ]
        
        U_opt = U_flat.reshape(nm, target_rank)
        V_opt = V_flat.reshape(mp, target_rank)
        W_opt = W_flat.reshape(pn, target_rank)
        
        # Vectorized: for each (i,j,k), compute dot(U[i*m+j], V[j*p+k]*W[k*n+i])
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                u_row = U_opt[i*m+j]  # (target_rank,)
                for k in range(p):
                    vw = V_opt[j*p+k] * W_opt[k*n+i]  # (target_rank,)
                    res[i,j,k] = np.dot(u_row, vw) - 1.0
        return res.flatten()
    
    # Generate multiple initializations
    initializations = []
    
    # 1. Direct truncation
    if U_np.shape[1] >= target_rank:
        initializations.append(np.concatenate([
            U_np[:, :target_rank].flatten(),
            V_np[:, :target_rank].flatten(),
            W_np[:, :target_rank].flatten()
        ]))
    
    # 2. Random perturbation of truncation
    if U_np.shape[1] >= target_rank:
        base = np.concatenate([
            U_np[:, :target_rank].flatten(),
            V_np[:, :target_rank].flatten(),
            W_np[:, :target_rank].flatten()
        ])
        for scale in [0.05, 0.1, 0.2]:
            initializations.append(base + np.random.normal(0, scale, total_params))
    
    # 3. Drop different random columns
    orig_rank = U_np.shape[1]
    for _ in range(10):
        if orig_rank > target_rank:
            drop_idx = np.random.choice(orig_rank, size=orig_rank - target_rank, replace=False)
            keep_idx = [i for i in range(orig_rank) if i not in set(drop_idx.tolist())]
            initializations.append(np.concatenate([
                U_np[:, keep_idx].flatten(),
                V_np[:, keep_idx].flatten(),
                W_np[:, keep_idx].flatten()
            ]))
    
    # 4. Random initialization
    for _ in range(5):
        U_rand = np.random.normal(0, 1, (nm, target_rank))
        V_rand = np.random.normal(0, 1, (mp, target_rank))
        W_rand = np.random.normal(0, 1, (pn, target_rank))
        initializations.append(np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()]))
    
    # 5. Scale-adjusted: try to make the product work by scaling
    if U_np.shape[1] >= target_rank:
        base_U = U_np[:, :target_rank].copy()
        base_V = V_np[:, :target_rank].copy()
        base_W = W_np[:, :target_rank].copy()
        # Try different scalings
        for s in [0.5, 1.0, 2.0]:
            initializations.append(np.concatenate([
                (base_U * s).flatten(),
                (base_V * s).flatten(),
                (base_W * s).flatten()
            ]))
    
    best_result = None
    best_cost = float('inf')
    
    for x_init in initializations:
        if time.time() - start_time > time_limit:
            break
        if len(x_init) != total_params:
            continue
            
        try:
            result = least_squares(
                residuals, x_init, method='lm',
                max_nfev=800, xtol=1e-8, ftol=1e-8, gtol=1e-8
            )
            if result.cost < best_cost:
                best_cost = result.cost
                if result.cost < 1e-10:
                    U_opt = result.x[:nm*target_rank].reshape(nm, target_rank)
                    V_opt = result.x[nm*target_rank : nm*target_rank + mp*target_rank].reshape(mp, target_rank)
                    W_opt = result.x[nm*target_rank + mp*target_rank : ].reshape(pn, target_rank)
                    
                    if _verify_np(U_opt, V_opt, W_opt, n, m, p, tol=1e-5):
                        return U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
            elif result.cost < 1e-8:
                # Close but not exact - try a polish
                U_opt = result.x[:nm*target_rank].reshape(nm, target_rank)
                V_opt = result.x[nm*target_rank : nm*target_rank + mp*target_rank].reshape(mp, target_rank)
                W_opt = result.x[nm*target_rank + mp*target_rank : ].reshape(pn, target_rank)
                
                # Polish with tighter tolerance
                try:
                    result2 = least_squares(
                        residuals, result.x, method='lm',
                        max_nfev=1000, xtol=1e-10, ftol=1e-10, gtol=1e-10
                    )
                    if result2.cost < 1e-12:
                        U_opt = result2.x[:nm*target_rank].reshape(nm, target_rank)
                        V_opt = result2.x[nm*target_rank : nm*target_rank + mp*target_rank].reshape(mp, target_rank)
                        W_opt = result2.x[nm*target_rank + mp*target_rank : ].reshape(pn, target_rank)
                        if _verify_np(U_opt, V_opt, W_opt, n, m, p, tol=1e-5):
                            return U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
                except Exception:
                    pass
        except Exception:
            pass
    
    return None

def _try_all_block_configs(n, m, p):
    """Try various block configurations and return the best one."""
    best_rank = n * m * p
    best_result = None
    
    # All possible 2x2x2 block placements
    # i0 in {0,1}, j0 in {0,1}, k0 in {0,1,...,6}
    # Must be non-overlapping
    
    # Try all combinations of up to 4 non-overlapping blocks
    all_k_positions = list(range(0, p - 1))  # 0..6
    all_i_positions = list(range(0, n - 1))  # 0,1
    all_j_positions = list(range(0, m - 1))  # 0,1
    
    # Greedy: try different sets
    configs_to_try = []
    
    # Standard: 4 blocks at k=0,2,4,6 with i0=0,j0=0
    configs_to_try.append([(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)])
    
    # Try i0=1
    configs_to_try.append([(1, 0, 0), (1, 0, 2), (1, 0, 4), (1, 0, 6)])
    
    # Try j0=1
    configs_to_try.append([(0, 1, 0), (0, 1, 2), (0, 1, 4), (0, 1, 6)])
    
    # Mixed i
    configs_to_try.append([(0, 0, 0), (1, 0, 2), (0, 0, 4), (1, 0, 6)])
    configs_to_try.append([(0, 0, 0), (0, 0, 2), (1, 0, 4), (1, 0, 6)])
    
    # Mixed j
    configs_to_try.append([(0, 0, 0), (0, 1, 2), (0, 0, 4), (0, 1, 6)])
    configs_to_try.append([(0, 0, 0), (0, 0, 2), (0, 1, 4), (0, 1, 6)])
    
    # 3 blocks (might allow better optimization)
    configs_to_try.append([(0, 0, 0), (0, 0, 2), (0, 0, 4)])
    configs_to_try.append([(0, 0, 0), (0, 0, 2), (1, 0, 4)])
    configs_to_try.append([(0, 0, 0), (0, 1, 2), (0, 0, 4)])
    configs_to_try.append([(0, 0, 0), (1, 0, 2), (0, 0, 4)])
    
    # 2 blocks
    configs_to_try.append([(0, 0, 0), (0, 0, 2)])
    configs_to_try.append([(0, 0, 0), (1, 0, 2)])
    
    for blocks in configs_to_try:
        if time.time() - _START > DEADLINE - 30:
            break
        try:
            U, V, W = _build_strassen_embedding(n, m, p, blocks)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    return best_result, best_rank

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    # Try various block configurations
    config_result, config_rank = _try_all_block_configs(n, m, p)
    if config_result:
        best_result = config_result
        best_rank = config_rank
    
    # Now try to reduce rank by 1 using optimization
    if best_rank > 1 and _time_left() > 40:
        U_cur = best_result["U"]
        V_cur = best_result["V"]
        W_cur = best_result["W"]
        
        target = best_rank - 1
        res = _try_rank_reduction(U_cur, V_cur, W_cur, n, m, p, target, 
                                  time_limit=min(35, _time_left() - 10))
        if res:
            U_new, V_new, W_new = res
            best_rank = target
            best_result = {"U": U_new, "V": V_new, "W": W_new}
    
    # Try to reduce by another 1
    if best_rank > 2 and _time_left() > 25:
        U_cur = best_result["U"]
        V_cur = best_result["V"]
        W_cur = best_result["W"]
        
        target = best_rank - 1
        res = _try_rank_reduction(U_cur, V_cur, W_cur, n, m, p, target,
                                  time_limit=min(20, _time_left() - 5))
        if res:
            U_new, V_new, W_new = res
            best_rank = target
            best_result = {"U": U_new, "V": V_new, "W": W_new}
    
    # Final verification
    if not _verify(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        best_result = {"U": U_school, "V": V_school, "W": W_school}
        best_rank = n * m * p
    
    return best_result
