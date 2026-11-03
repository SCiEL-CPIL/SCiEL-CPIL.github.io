"""Matrix multiplication 3,3,5 tensor decomposition search."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np
from fractions import Fraction
from itertools import product

# EVOLVE-BLOCK-START

def construct():
    """Construct a rank-minimal tensor decomposition for (3,3,5) matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank) = (9, rank)
    - V has shape (m*p, rank) = (15, rank)  
    - W has shape (p*n, rank) = (15, rank)
    
    The decomposition satisfies: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    where T[i,j,k] = 1 iff i = a*m+b, j = b*p+c, k = c*n+a for some a,b,c.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    def build_tensor():
        """Build the (n,m,p) matrix multiplication tensor."""
        T = np.zeros((n*m, m*p, p*n))
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i = a * m + b
                    j = b * p + c
                    k = c * n + a
                    T[i, j, k] = 1
        return T
    
    def verify(U, V, W):
        """Verify that U, V, W form a valid decomposition."""
        T = build_tensor()
        rank = U.shape[1] if hasattr(U, 'shape') else len(U[0])
        # Compute sum of outer products
        recon = np.zeros((n*m, m*p, p*n))
        for c in range(rank):
            if hasattr(U, 'shape'):
                uc = U[:, c]
                vc = V[:, c]
                wc = W[:, c]
            else:
                uc = np.array([U[i][c] for i in range(n*m)])
                vc = np.array([V[i][c] for i in range(m*p)])
                wc = np.array([W[i][c] for i in range(p*n)])
            recon += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.allclose(recon, T, atol=1e-10)
    
    def schoolbook():
        """Schoolbook decomposition (rank = n*m*p = 45)."""
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
        return {"U": U, "V": V, "W": W}
    
    def snap_to_half_integers(M, tol=1e-8):
        """Try to snap matrix entries to half-integers. Returns None if not close enough."""
        if not hasattr(M, 'shape'):
            M = np.array(M)
        rounded = np.round(2 * M) / 2.0
        if np.max(np.abs(M - rounded)) < tol:
            return rounded
        return None
    
    def verify_rational(U, V, W):
        """Verify with exact rational arithmetic."""
        T = build_tensor()
        rank = U.shape[1]
        recon = np.zeros((n*m, m*p, p*n))
        for c in range(rank):
            recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(recon, T, atol=1e-10)
    
    def als_decompose(T, r, max_iter=300, n_restarts=10, seed=42):
        """Alternating least squares for tensor rank-r decomposition."""
        I, J, K = T.shape
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
                
            rng = np.random.RandomState(seed + restart)
            U = rng.randn(I, r)
            V = rng.randn(J, r)
            W = rng.randn(K, r)
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(I):
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for j in range(J):
                        for k in range(K):
                            if T[i, j, k] != 0:
                                num += T[i, j, k] * V[j, :] * W[k, :]
                                den += V[j, :]**2 * W[k, :]**2
                    mask = den > 1e-12
                    U[i, mask] = num[mask] / den[mask]
                    U[i, ~mask] = 0
                
                # Update V
                for j in range(J):
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for i in range(I):
                        for k in range(K):
                            if T[i, j, k] != 0:
                                num += T[i, j, k] * U[i, :] * W[k, :]
                                den += U[i, :]**2 * W[k, :]**2
                    mask = den > 1e-12
                    V[j, mask] = num[mask] / den[mask]
                    V[j, ~mask] = 0
                
                # Update W
                for k in range(K):
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for i in range(I):
                        for j in range(J):
                            if T[i, j, k] != 0:
                                num += T[i, j, k] * U[i, :] * V[j, :]
                                den += U[i, :]**2 * V[j, :]**2
                    mask = den > 1e-12
                    W[k, mask] = num[mask] / den[mask]
                    W[k, ~mask] = 0
                
                # Check convergence
                recon = np.zeros((I, J, K))
                for c in range(r):
                    recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                err = np.linalg.norm(recon - T) / max(1, np.linalg.norm(T))
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    return U, V, W
        
        return best_U, best_V, best_W
    
    def try_rank_als(r, n_restarts=15, seed_base=0):
        """Try to find rank-r decomposition using ALS with multiple initializations."""
        T = build_tensor()
        best_result = None
        best_err = float('inf')
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Different initialization strategies
            rng = np.random.RandomState(seed_base + restart)
            
            if restart % 3 == 0:
                # Random dense
                U = rng.randn(9, r)
                V = rng.randn(15, r)
                W = rng.randn(15, r)
            elif restart % 3 == 1:
                # Sparse random
                U = np.zeros((9, r))
                V = np.zeros((15, r))
                W = np.zeros((15, r))
                for c in range(r):
                    for _ in range(2 + rng.randint(2)):
                        U[rng.randint(9), c] = rng.choice([-1, 1])
                    for _ in range(2 + rng.randint(3)):
                        V[rng.randint(15), c] = rng.choice([-1, 1])
                    for _ in range(2 + rng.randint(3)):
                        W[rng.randint(15), c] = rng.choice([-1, 1])
            else:
                # Structured based on index patterns
                U = np.zeros((9, r))
                V = np.zeros((15, r))
                W = np.zeros((15, r))
                for c in range(r):
                    a = c % 3
                    b = (c // 3) % 3
                    c_idx = (c // 9) % 5
                    U[a*3 + b, c] = 1.0
                    V[b*5 + c_idx, c] = 1.0
                    W[c_idx*3 + a, c] = 1.0
                    # Add some noise
                    if c < r - 3:
                        U[a*3 + b, c] += 0.5 * rng.randn()
                        V[b*5 + c_idx, c] += 0.5 * rng.randn()
                        W[c_idx*3 + a, c] += 0.5 * rng.randn()
            
            # Run ALS from this initialization
            U_cur, V_cur, W_cur = U, V, W
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(9):
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for j in range(15):
                        for k in range(15):
                            if T[i, j, k] != 0:
                                num += T[i, j, k] * V_cur[j, :] * W_cur[k, :]
                                den += V_cur[j, :]**2 * W_cur[k, :]**2
                    mask = den > 1e-12
                    U_cur[i, mask] = num[mask] / den[mask]
                    U_cur[i, ~mask] = 0
                
                # Update V
                for j in range(15):
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for i in range(9):
                        for k in range(15):
                            if T[i, j, k] != 0:
                                num += T[i, j, k] * U_cur[i, :] * W_cur[k, :]
                                den += U_cur[i, :]**2 * W_cur[k, :]**2
                    mask = den > 1e-12
                    V_cur[j, mask] = num[mask] / den[mask]
                    V_cur[j, ~mask] = 0
                
                # Update W
                for k in range(15):
                    num = np.zeros(r)
                    den = np.zeros(r)
                    for i in range(9):
                        for j in range(15):
                            if T[i, j, k] != 0:
                                num += T[i, j, k] * U_cur[i, :] * V_cur[j, :]
                                den += U_cur[i, :]**2 * V_cur[j, :]**2
                    mask = den > 1e-12
                    W_cur[k, mask] = num[mask] / den[mask]
                    W_cur[k, ~mask] = 0
                
                # Check convergence
                recon = np.zeros((9, 15, 15))
                for c in range(r):
                    recon += np.einsum('i,j,k->ijk', U_cur[:, c], V_cur[:, c], W_cur[:, c])
                err = np.linalg.norm(recon - T) / max(1, np.linalg.norm(T))
                
                if err < best_err:
                    best_err = err
                    best_result = (U_cur.copy(), V_cur.copy(), W_cur.copy())
                
                if err < 1e-10:
                    return U_cur, V_cur, W_cur
        
        if best_result is not None and best_err < 1e-8:
            return best_result
        return None
    
    def try_gauge_optimization(r, n_restarts=20, seed_base=1000):
        """Try gauge-fixed optimization using least squares approach."""
        from scipy.optimize import least_squares
        
        dim_u, dim_v, dim_w = 9, 15, 15
        
        # Precompute equations
        eqs = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    eqs.append((u_idx, v_idx, w_idx))
        
        def make_objective(gauge):
            def objective(params):
                if gauge == 0:
                    U_rest = params[:r*(dim_u-1)].reshape(r, dim_u-1)
                    V = params[r*(dim_u-1):r*(dim_u-1+dim_v)].reshape(r, dim_v)
                    W = params[r*(dim_u-1+dim_v):].reshape(r, dim_w)
                    U = np.zeros((r, dim_u))
                    U[:, 0] = 1.0
                    U[:, 1:] = U_rest
                elif gauge == 1:
                    U = params[:r*dim_u].reshape(r, dim_u)
                    V_rest = params[r*dim_u:r*dim_u+r*(dim_v-1)].reshape(r, dim_v-1)
                    W = params[r*dim_u+r*(dim_v-1):].reshape(r, dim_w)
                    V = np.zeros((r, dim_v))
                    V[:, 0] = 1.0
                    V[:, 1:] = V_rest
                else:
                    U = params[:r*dim_u].reshape(r, dim_u)
                    V = params[r*dim_u:r*dim_u+r*dim_v].reshape(r, dim_v)
                    W_rest = params[r*dim_u+r*dim_v:r*dim_u+r*dim_v+r*(dim_w-1)].reshape(r, dim_w-1)
                    W = np.zeros((r, dim_w))
                    W[:, 0] = 1.0
                    W[:, 1:] = W_rest
                
                residuals = np.zeros(len(eqs))
                for idx, (u_idx, v_idx, w_idx) in enumerate(eqs):
                    val = np.sum(U[:, u_idx] * V[:, v_idx] * W[:, w_idx])
                    residuals[idx] = val - 1.0
                return residuals
            return objective
        
        best_result = None
        best_cost = float('inf')
        
        for gauge in range(3):
            if time.time() > deadline - 10:
                break
            
            objective = make_objective(gauge)
            
            if gauge == 0:
                num_vars = r * (dim_u - 1 + dim_v + dim_w)
            elif gauge == 1:
                num_vars = r * (dim_u + dim_v - 1 + dim_w)
            else:
                num_vars = r * (dim_u + dim_v + dim_w - 1)
            
            for attempt in range(n_restarts):
                if time.time() > deadline - 5:
                    break
                
                rng = np.random.RandomState(seed_base + gauge * 100 + attempt)
                x0 = rng.uniform(-1, 1, num_vars)
                
                try:
                    result = least_squares(objective, x0, method='lm', max_nfev=3000)
                    
                    if result.cost < best_cost:
                        best_cost = result.cost
                        
                        # Reconstruct matrices
                        if gauge == 0:
                            U_rest = result.x[:r*(dim_u-1)].reshape(r, dim_u-1)
                            V = result.x[r*(dim_u-1):r*(dim_u-1+dim_v)].reshape(r, dim_v)
                            W = result.x[r*(dim_u-1+dim_v):].reshape(r, dim_w)
                            U = np.zeros((r, dim_u))
                            U[:, 0] = 1.0
                            U[:, 1:] = U_rest
                        elif gauge == 1:
                            U = result.x[:r*dim_u].reshape(r, dim_u)
                            V_rest = result.x[r*dim_u:r*dim_u+r*(dim_v-1)].reshape(r, dim_v-1)
                            W = result.x[r*dim_u+r*(dim_v-1):].reshape(r, dim_w)
                            V = np.zeros((r, dim_v))
                            V[:, 0] = 1.0
                            V[:, 1:] = V_rest
                        else:
                            U = result.x[:r*dim_u].reshape(r, dim_u)
                            V = result.x[r*dim_u:r*dim_u+r*dim_v].reshape(r, dim_v)
                            W_rest = result.x[r*dim_u+r*dim_v:r*dim_u+r*dim_v+r*(dim_w-1)].reshape(r, dim_w-1)
                            W = np.zeros((r, dim_w))
                            W[:, 0] = 1.0
                            W[:, 1:] = W_rest
                        
                        if result.cost < 1e-10:
                            # Try to snap to half-integers
                            U_snap = snap_to_half_integers(U)
                            V_snap = snap_to_half_integers(V)
                            W_snap = snap_to_half_integers(W)
                            
                            if U_snap is not None and V_snap is not None and W_snap is not None:
                                if verify_rational(U_snap, V_snap, W_snap):
                                    return U_snap, V_snap, W_snap
                            
                            # Check if close to half-integers with tighter tolerance
                            U_snap2 = snap_to_half_integers(U, tol=1e-6)
                            V_snap2 = snap_to_half_integers(V, tol=1e-6)
                            W_snap2 = snap_to_half_integers(W, tol=1e-6)
                            
                            if U_snap2 is not None and V_snap2 is not None and W_snap2 is not None:
                                if verify_rational(U_snap2, V_snap2, W_snap2):
                                    return U_snap2, V_snap2, W_snap2
                        
                        best_result = (U, V, W)
                        
                except Exception:
                    continue
        
        if best_result is not None and best_cost < 1e-8:
            return best_result
        return None
    
    # Main search logic
    # Try ranks from low to high, with time budget allocation
    # Allocate more time to lower ranks since they're more valuable
    
    T = build_tensor()
    
    # Try rank 44 first (most likely improvement over schoolbook 45)
    if time.time() < deadline - 60:
        result = try_rank_als(44, n_restarts=20, seed_base=0)
        if result is not None:
            U_res, V_res, W_res = result
            U_snap = snap_to_half_integers(U_res)
            V_snap = snap_to_half_integers(V_res)
            W_snap = snap_to_half_integers(W_res)
            if U_snap is not None and V_snap is not None and W_snap is not None:
                if verify_rational(U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        result = try_gauge_optimization(44, n_restarts=15, seed_base=1000)
        if result is not None:
            U_res, V_res, W_res = result
            U_snap = snap_to_half_integers(U_res)
            V_snap = snap_to_half_integers(V_res)
            W_snap = snap_to_half_integers(W_res)
            if U_snap is not None and V_snap is not None and W_snap is not None:
                if verify_rational(U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Try rank 43
    if time.time() < deadline - 40:
        result = try_rank_als(43, n_restarts=15, seed_base=100)
        if result is not None:
            U_res, V_res, W_res = result
            U_snap = snap_to_half_integers(U_res)
            V_snap = snap_to_half_integers(V_res)
            W_snap = snap_to_half_integers(W_res)
            if U_snap is not None and V_snap is not None and W_snap is not None:
                if verify_rational(U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        result = try_gauge_optimization(43, n_restarts=10, seed_base=2000)
        if result is not None:
            U_res, V_res, W_res = result
            U_snap = snap_to_half_integers(U_res)
            V_snap = snap_to_half_integers(V_res)
            W_snap = snap_to_half_integers(W_res)
            if U_snap is not None and V_snap is not None and W_snap is not None:
                if verify_rational(U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Try rank 42
    if time.time() < deadline - 25:
        result = try_rank_als(42, n_restarts=10, seed_base=200)
        if result is not None:
            U_res, V_res, W_res = result
            U_snap = snap_to_half_integers(U_res)
            V_snap = snap_to_half_integers(V_res)
            W_snap = snap_to_half_integers(W_res)
            if U_snap is not None and V_snap is not None and W_snap is not None:
                if verify_rational(U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Fallback: schoolbook rank 45
    return schoolbook()

# EVOLVE-BLOCK-END
