import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    # Precompute the target tensor in flattened form
    # T[i, j, k] = 1 if (i,j) is valid U index, (j,k) valid V index, (k,i) valid W index
    # U index: i * m + j where i in [0,n), j in [0,m)
    # V index: j * p + k where j in [0,m), k in [0,p)
    # W index: k * n + i where k in [0,p), i in [0,n)
    
    # We need to figure out which (u_idx, v_idx, w_idx) triples have T=1
    # u_idx = i*m + j, v_idx = j*p + k, w_idx = k*n + i
    # So for each (i,j,k), we have a triple (i*m+j, j*p+k, k*n+i)
    
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                support.append((u_idx, v_idx, w_idx))
    
    support = np.array(support)
    n_support = len(support)
    
    # Target tensor is sparse - only these positions are 1, rest are 0
    # Total entries = nm * mp * pn = 12 * 24 * 18 = 5184
    total_entries = nm * mp * pn
    
    def build_target_vec():
        """Build the target tensor as a flat vector of length total_entries"""
        target = np.zeros(total_entries)
        for (u_idx, v_idx, w_idx) in support:
            # Flat index in C-order for 3D array of shape (nm, mp, pn)
            flat_idx = u_idx * (mp * pn) + v_idx * pn + w_idx
            target[flat_idx] = 1.0
        return target
    
    target_vec = build_target_vec()
    
    def compute_residual(params, r):
        """Compute residual for given parameters and rank r"""
        # params is flat array of size r*(nm + mp + pn)
        u = params[:nm*r].reshape(nm, r)
        v = params[nm*r:nm*r+mp*r].reshape(mp, r)
        w = params[nm*r+mp*r:].reshape(pn, r)
        
        # Compute T_recon[i, j, k] = sum_c u[i,c] * v[j,c] * w[k,c]
        # Using einsum: 'ic,jc,kc->ijk'
        T_recon = np.einsum('ic,jc,kc->ijk', u, v, w, optimize=True)
        T_flat = T_recon.flatten()
        
        return T_flat - target_vec
    
    def compute_residual_sparse(params, r):
        """Compute residual only on support entries (faster)"""
        u = params[:nm*r].reshape(nm, r)
        v = params[nm*r:nm*r+mp*r].reshape(mp, r)
        w = params[nm*r+mp*r:].reshape(pn, r)
        
        # For each support entry (u_idx, v_idx, w_idx), compute sum_c u[u_idx,c]*v[v_idx,c]*w[w_idx,c]
        # This can be vectorized
        u_vals = u[support[:, 0], :]  # shape (n_support, r)
        v_vals = v[support[:, 1], :]  # shape (n_support, r)
        w_vals = w[support[:, 2], :]  # shape (n_support, r)
        
        # T_recon at support = sum_c u_vals[:,c] * v_vals[:,c] * w_vals[:,c]
        T_support = np.sum(u_vals * v_vals * w_vals, axis=1)
        
        # Residual at support should be 1
        # But we also need to check that non-support entries are 0
        # For now, just check support
        return T_support - 1.0
    
    def verify_solution(u, v, w):
        """Verify that the solution exactly decomposes the tensor"""
        T_recon = np.einsum('ic,jc,kc->ijk', u, v, w, optimize=True)
        T_flat = T_recon.flatten()
        
        # Check support entries are 1
        for (u_idx, v_idx, w_idx) in support:
            flat_idx = u_idx * (mp * pn) + v_idx * pn + w_idx
            if abs(T_flat[flat_idx] - 1.0) > 1e-9:
                return False
        
        # Check non-support entries are 0
        # Create mask of support entries
        support_set = set((u, v, w) for (u, v, w) in support)
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    if (i, j, k) not in support_set:
                        flat_idx = i * (mp * pn) + j * pn + k
                        if abs(T_flat[flat_idx]) > 1e-9:
                            return False
        return True
    
    def snap_to_half_integer(x):
        """Snap values to nearest half-integer"""
        return np.round(x * 2) / 2.0
    
    # Lower bound for rank: max(nm, mp, pn) = max(12, 24, 18) = 24
    min_rank = max(nm, mp, pn)
    
    # Known upper bound: schoolbook rank = n*m*p = 72
    # But we can do better. For 3x4x6, known results suggest rank around 24-36
    
    best_result = None
    best_rank = None
    
    # Try ranks from lower bound upward
    for r in range(min_rank, min_rank + 30):  # Try up to rank 54
        if time.time() > deadline:
            break
        
        # Multiple random restarts for each rank
        n_restarts = 20 if r <= 30 else 10
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + r * 1000 + restart)
            
            # Initialize with small random values
            scale = 0.5
            u_init = np.random.randn(nm, r) * scale
            v_init = np.random.randn(mp, r) * scale
            w_init = np.random.randn(pn, r) * scale
            
            params_init = np.concatenate([u_init.flatten(), v_init.flatten(), w_init.flatten()])
            
            # Use least_squares with the full residual
            def res_func(x):
                return compute_residual(x, r)
            
            try:
                result = least_squares(
                    res_func, 
                    params_init, 
                    method='trf', 
                    max_nfev=2000,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12
                )
                
                if result.success or result.cost < 1e-8:
                    # Extract solution
                    u_sol = result.x[:nm*r].reshape(nm, r)
                    v_sol = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
                    w_sol = result.x[nm*r+mp*r:].reshape(pn, r)
                    
                    # Try snapping to half-integers
                    u_snap = snap_to_half_integer(u_sol)
                    v_snap = snap_to_half_integer(v_sol)
                    w_snap = snap_to_half_integer(w_sol)
                    
                    # Verify snapped solution
                    if verify_solution(u_snap, v_snap, w_snap):
                        return {
                            "U": u_snap.tolist(),
                            "V": v_snap.tolist(),
                            "W": w_snap.tolist()
                        }
                    
                    # Also verify original (unsnapped) solution for debugging
                    if verify_solution(u_sol, v_sol, w_sol):
                        # Found a valid solution but not half-integer
                        # Keep it but continue searching for half-integer
                        if best_result is None:
                            best_result = {
                                "U": u_sol.tolist(),
                                "V": v_sol.tolist(),
                                "W": w_sol.tolist()
                            }
                            best_rank = r
                            
            except Exception:
                pass
    
    # If no half-integer solution found, return best floating-point solution if valid
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook decomposition (rank = n*m*p = 72)
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
