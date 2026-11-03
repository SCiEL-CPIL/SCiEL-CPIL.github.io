import numpy as np
import time
from itertools import product
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _build_strassen56():
    n, m, p = DIMENSIONS
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for s in range(2):
        for j in range(4):
            a_idx = [0 * m + (2*s + 0), 0 * m + (2*s + 1), 1 * m + (2*s + 0), 1 * m + (2*s + 1)]
            b_idx = [(2*s + 0) * p + (2*j + 0), (2*s + 0) * p + (2*j + 1), (2*s + 1) * p + (2*j + 0), (2*s + 1) * p + (2*j + 1)]
            c_idx = [(2*j + 0) * n + 0, (2*j + 1) * n + 0, (2*j + 0) * n + 1, (2*j + 1) * n + 1]
            
            U[a_idx[0]][col] += 1
            U[a_idx[3]][col] += 1
            U[a_idx[2]][col+1] += 1
            U[a_idx[3]][col+1] += 1
            U[a_idx[0]][col+2] += 1
            U[a_idx[3]][col+3] += 1
            U[a_idx[0]][col+4] += 1
            U[a_idx[1]][col+4] += 1
            U[a_idx[2]][col+5] += 1
            U[a_idx[0]][col+5] += -1
            U[a_idx[1]][col+6] += 1
            U[a_idx[3]][col+6] += -1
            
            V[b_idx[0]][col] += 1
            V[b_idx[3]][col] += 1
            V[b_idx[0]][col+1] += 1
            V[b_idx[1]][col+2] += 1
            V[b_idx[3]][col+2] += -1
            V[b_idx[2]][col+3] += 1
            V[b_idx[0]][col+3] += -1
            V[b_idx[3]][col+4] += 1
            V[b_idx[0]][col+5] += 1
            V[b_idx[1]][col+5] += 1
            V[b_idx[2]][col+6] += 1
            V[b_idx[3]][col+6] += 1
            
            W[c_idx[0]][col] += 1
            W[c_idx[0]][col+3] += 1
            W[c_idx[0]][col+4] += -1
            W[c_idx[0]][col+6] += 1
            W[c_idx[1]][col+2] += 1
            W[c_idx[1]][col+4] += 1
            W[c_idx[2]][col+1] += 1
            W[c_idx[2]][col+3] += 1
            W[c_idx[3]][col] += 1
            W[c_idx[3]][col+1] += -1
            W[c_idx[3]][col+2] += 1
            W[c_idx[3]][col+5] += 1
            
            col += 7
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p):
    rank = len(U[1]) if U.shape[1] > 0 else 0
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    return max_err < 1e-9

def _try_reduce_rank(U, V, W, n, m, p, target_rank):
    if _time_remaining() < 10.0:
        return None
    
    # Flatten U, V, W
    U_flat = U.flatten()
    V_flat = V.flatten()
    W_flat = W.flatten()
    
    # We remove (56 - target_rank) columns.
    # Let's remove the last columns.
    remove_count = 56 - target_rank
    U_new = U[:, :target_rank]
    V_new = V[:, :target_rank]
    W_new = W[:, :target_rank]
    
    # Define the residual function
    def residuals(params):
        U_flat = params[:U_new.size]
        V_flat = params[U_new.size:U_new.size + V_new.size]
        W_flat = params[U_new.size + V_new.size:]
        
        U_r = U_flat.reshape(U_new.shape)
        V_r = V_flat.reshape(V_new.shape)
        W_r = W_flat.reshape(W_new.shape)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(target_rank):
                        s += U_r[i*m + j, r] * V_r[j*p + k, r] * W_r[k*n + i, r]
                    res.append(s - 1.0)
        return np.array(res)
    
    initial_params = np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()])
    
    try:
        result = least_squares(residuals, initial_params, method='lm', max_nfev=1000)
        if result.success and np.max(np.abs(result.fun)) < 1e-8:
            U_opt = result.x[:U_new.size].reshape(U_new.shape)
            V_opt = result.x[U_new.size:U_new.size + V_new.size].reshape(V_new.shape)
            W_opt = result.x[U_new.size + V_new.size:].reshape(W_new.shape)
            return {"U": U_opt, "V": V_opt, "W": W_opt}
    except Exception:
        pass
    return None

def construct():
    n, m, p = DIMENSIONS
    best_result = _build_strassen56()
    
    if not _validate(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        return best_result
    
    # Try to reduce rank
    for target_rank in [50, 48, 44, 40]:
        if _time_remaining() < 5.0:
            break
        res = _try_reduce_rank(best_result["U"], best_result["V"], best_result["W"], n, m, p, target_rank)
        if res is not None:
            # Check if it's valid with integers/half-integers?
            # The problem requires integer/half-integer entries.
            # The LS solution is float. We need to round.
            U_rounded = np.round(res["U"] * 2) / 2
            V_rounded = np.round(res["V"] * 2) / 2
            W_rounded = np.round(res["W"] * 2) / 2
            
            if _validate(U_rounded, V_rounded, W_rounded, n, m, p):
                best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                break
    
    return best_result
