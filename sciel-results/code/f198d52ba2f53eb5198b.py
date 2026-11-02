import time
import sys
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 130  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    alpha = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    beta = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    w_coeff = [
        {},
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for s in range(1, 8):
        for t in range(4):
            c = 4 * (s - 1) + t
            k_a = 0 if t in [0, 2] else 1
            r_b = 1 if t in [1, 3] else 0
            c_b = 1 if t in [2, 3] else 0
            k_c = 0 if t in [0, 1] else 1
            
            for (i, j), val in alpha[s].items():
                if val != 0:
                    u_idx = i * m + 2 * j + k_a
                    U[u_idx][c] += val
            
            for (i, j), val in beta[s].items():
                if val != 0:
                    v_row = 2 * i + r_b
                    v_col = 2 * j + c_b
                    v_idx = v_row * p + v_col
                    V[v_idx][c] += val
            
            for (i, j), val in w_coeff[s].items():
                if val != 0:
                    w_col = 2 * j + k_c
                    w_row = i
                    w_idx = w_col * n + w_row
                    W[w_idx][c] += val
    
    return U, V, W

def residuals(x, n=2, m=4, p=4, rank=24):
    """Calculate residuals for the optimization problem using vectorized operations."""
    U = x[:n*m*rank].reshape(n, m, rank)
    V = x[n*m*rank : n*m*rank + m*p*rank].reshape(m, p, rank)
    W = x[n*m*rank + m*p*rank : ].reshape(p, n, rank)
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    return T_recon.flatten() - 1.0

def try_rank_reduction_from_strassen(target_rank, n=2, m=4, p=4, max_restarts=20, prev_solution=None):
    """Try to reduce rank from a previous solution to target_rank."""
    if time_up():
        return None
    
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    best_result = None
    best_error = float('inf')
    
    # Start with previous solution or Strassen as base
    if prev_solution is not None:
        U_s = np.array(prev_solution['U'], dtype=float)
        V_s = np.array(prev_solution['V'], dtype=float)
        W_s = np.array(prev_solution['W'], dtype=float)
    else:
        U_strassen, V_strassen, W_strassen = strassen_decomposition()
        U_s = np.array(U_strassen, dtype=float)
        V_s = np.array(V_strassen, dtype=float)
        W_s = np.array(W_strassen, dtype=float)
    
    src_rank = U_s.shape[1]
    
    for restart in range(max_restarts):
        if time_up():
            break
        
        # Initialize from previous solution with perturbation
        if restart == 0:
            # Truncate or pad the previous solution
            if target_rank <= src_rank:
                U_init = U_s[:, :target_rank].copy()
                V_init = V_s[:, :target_rank].copy()
                W_init = W_s[:, :target_rank].copy()
            else:
                U_init = np.hstack([U_s, np.random.randn(n*m, target_rank - src_rank) * 0.1])
                V_init = np.hstack([V_s, np.random.randn(m*p, target_rank - src_rank) * 0.1])
                W_init = np.hstack([W_s, np.random.randn(p*n, target_rank - src_rank) * 0.1])
        else:
            # Add random perturbations
            scale = 0.03 + 0.01 * restart
            keep = min(target_rank, src_rank)
            U_init = U_s[:, :keep].copy() + np.random.randn(n*m, keep) * scale
            V_init = V_s[:, :keep].copy() + np.random.randn(m*p, keep) * scale
            W_init = W_s[:, :keep].copy() + np.random.randn(p*n, keep) * scale
            if target_rank > keep:
                U_init = np.hstack([U_init, np.random.randn(n*m, target_rank - keep) * 0.1])
                V_init = np.hstack([V_init, np.random.randn(m*p, target_rank - keep) * 0.1])
                W_init = np.hstack([W_init, np.random.randn(p*n, target_rank - keep) * 0.1])
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), method='lm', max_nfev=5000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x.copy()
                
            if err < 1e-9:
                U = best_result[:size_U].reshape(n*m, target_rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception as e:
            pass
            
    # If we got close but not exact, try a polish pass
    if best_result is not None and best_error < 0.1 and not time_up():
        try:
            result2 = opt.least_squares(residuals, best_result, args=(n, m, p, target_rank),
                                         method='trf', max_nfev=20000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            err2 = np.linalg.norm(result2.fun)
            if err2 < 1e-9:
                U = result2.x[:size_U].reshape(n*m, target_rank)
                V = result2.x[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = result2.x[size_U + size_V : ].reshape(p*n, target_rank)
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
        except Exception:
            pass
            
    return None

def find_low_rank_decomposition(target_rank, n=2, m=4, p=4, max_restarts=30):
    """Use scipy.optimize.least_squares to find a decomposition of target_rank."""
    n, m, p = 2, 4, 4
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    best_result = None
    best_error = float('inf')
    
    for restart in range(max_restarts):
        if time_up():
            break
            
        # Use structured random initialization based on tensor structure
        if restart == 0:
            # Initialize with sparse structure
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            for r in range(target_rank):
                # Randomly assign non-zero entries
                u_rows = np.random.choice(n*m, size=max(1, n*m//4), replace=False)
                v_rows = np.random.choice(m*p, size=max(1, m*p//4), replace=False)
                w_rows = np.random.choice(p*n, size=max(1, p*n//4), replace=False)
                U[u_rows, r] = np.random.choice([-1, 1], size=len(u_rows))
                V[v_rows, r] = np.random.choice([-1, 1], size=len(v_rows))
                W[w_rows, r] = np.random.choice([-1, 1], size=len(w_rows))
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        else:
            # Random initialization with varying scales
            scale = 0.1 + 0.05 * (restart % 5)
            x0 = np.random.randn(total_size) * scale
        
        try:
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), method='lm', max_nfev=8000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x.copy()
                
            if err < 1e-9:
                # Found a solution
                U = best_result[:size_U].reshape(n*m, target_rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                # Verify
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception as e:
            pass
    
    # Polish pass if we got close
    if best_result is not None and best_error < 0.1 and not time_up():
        try:
            result2 = opt.least_squares(residuals, best_result, args=(n, m, p, target_rank),
                                         method='trf', max_nfev=20000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            err2 = np.linalg.norm(result2.fun)
            if err2 < 1e-9:
                U = result2.x[:size_U].reshape(n*m, target_rank)
                V = result2.x[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = result2.x[size_U + size_V : ].reshape(p*n, target_rank)
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = 2, 4, 4
    best_result = None
    best_rank = float('inf')
    
    # Start with Strassen
    U, V, W = strassen_decomposition()
    valid, err = verify_decomposition(U, V, W, n, m, p)
    if valid:
        best_rank = 28
        best_result = {"U": U, "V": V, "W": W}
    
    # Greedy sequential rank reduction: 28 -> 27 -> 26 -> 25 -> 24
    # Each step uses the previous solution as warm start
    prev_sol = best_result
    for target_rank in [27, 26, 25, 24]:
        if time_up():
            break
        if prev_sol is None:
            break
        
        res = try_rank_reduction_from_strassen(target_rank, n, m, p, prev_solution=prev_sol)
        if res is not None:
            U2, V2, W2 = res
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = target_rank
                best_result = {"U": U2, "V": V2, "W": W2}
                prev_sol = best_result
            else:
                # Keep trying with the same prev_sol
                pass
        else:
            # Try random search as fallback for this rank
            if not time_up():
                res = find_low_rank_decomposition(target_rank, n, m, p)
                if res is not None:
                    U2, V2, W2 = res
                    valid, err = verify_decomposition(U2, V2, W2, n, m, p)
                    if valid:
                        best_rank = target_rank
                        best_result = {"U": U2, "V": V2, "W": W2}
                        prev_sol = best_result
    
    if best_result is None:
        # Fallback: naive decomposition
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
