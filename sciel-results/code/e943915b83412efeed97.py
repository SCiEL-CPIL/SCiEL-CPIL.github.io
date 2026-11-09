# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N)
_YS = np.repeat(_VALS, _N)    # y for each cell
_XS = np.tile(_VALS, _N)      # x for each cell

# Precompute projection indices
_IDX_D = (_XS - _YS) + 9
_N_D = 19
_IDX_H0 = _XS
_N_H0 = 10
_IDX_H1 = _XS + _YS
_N_H1 = 19
_IDX_H2 = _XS + 2 * _YS
_N_H2 = 28

def _entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count vector (summing to 1)."""
    p = counts
    s = p.sum()
    if s <= 1e-30:
        return 0.0
    q = p / s
    q = q[q > 1e-15]
    if len(q) == 0:
        return 0.0
    return float(-(q * np.log2(q)).sum())

def _compute_value(P):
    """Compute D/M ratio for a given distribution P (10x10)."""
    p = P.ravel()
    
    d_counts = np.zeros(_N_D)
    np.add.at(d_counts, _IDX_D, p)
    H_D = _entropy_from_counts(d_counts)
    
    h0_counts = np.zeros(_N_H0)
    np.add.at(h0_counts, _IDX_H0, p)
    H_0 = _entropy_from_counts(h0_counts)
    
    h1_counts = np.zeros(_N_H1)
    np.add.at(h1_counts, _IDX_H1, p)
    H_1 = _entropy_from_counts(h1_counts)
    
    h2_counts = np.zeros(_N_H2)
    np.add.at(h2_counts, _IDX_H2, p)
    H_2 = _entropy_from_counts(h2_counts)
    
    M = max(H_0, H_1, H_2)
    if M <= 1e-9:
        return 0.0
        
    return H_D / M

def _log_softmax(theta):
    """Convert log-weights to probabilities."""
    # Subtract max for numerical stability
    theta_max = np.max(theta)
    exp_theta = np.exp(theta - theta_max)
    sum_exp = np.sum(exp_theta)
    if sum_exp == 0:
        return np.ones(len(theta)) / len(theta)
    return exp_theta / sum_exp

def _neg_value_from_log(theta):
    """Objective function for optimizer: negative value from log-weights."""
    P = _log_softmax(theta).reshape(_N, _N)
    return -_compute_value(P)

def _generate_initializations():
    """Generate a list of initial log-weight vectors."""
    inits = []
    
    # 1. Uniform
    inits.append(np.zeros(_N * _N))
    
    # 2. Center Cross (Row 4,5 and Col 4,5)
    P_cross = np.zeros((_N, _N))
    P_cross[4, :] = 1
    P_cross[5, :] = 1
    P_cross[:, 4] = 1
    P_cross[:, 5] = 1
    P_cross /= P_cross.sum()
    inits.append(np.log(P_cross.ravel() + 1e-15))
    
    # 3. Frame (Border)
    P_frame = np.zeros((_N, _N))
    P_frame[0, :] = 1
    P_frame[-1, :] = 1
    P_frame[:, 0] = 1
    P_frame[:, -1] = 1
    P_frame /= P_frame.sum()
    inits.append(np.log(P_frame.ravel() + 1e-15))
    
    # 4. Diagonals
    for c in range(-4, 5):
        P_diag = np.zeros((_N, _N))
        for y in range(_N):
            x = y + c
            if 0 <= x < _N:
                P_diag[y, x] = 1
        if P_diag.sum() > 0:
            P_diag /= P_diag.sum()
            inits.append(np.log(P_diag.ravel() + 1e-15))
    
    # 5. Random Sparse Sets
    np.random.seed(42)
    for _ in range(10):
        P_sparse = np.zeros((_N, _N))
        num_cells = np.random.randint(5, 15)
        indices = np.random.choice(_N * _N, num_cells, replace=False)
        P_sparse.ravel()[indices] = 1
        P_sparse /= P_sparse.sum()
        inits.append(np.log(P_sparse.ravel() + 1e-15))
        
    return inits

def _optimize_from_init(init_theta, time_budget):
    """Run Nelder-Mead optimization from a single initialization."""
    t_start = time.time()
    
    # Initial evaluation
    best_val = -_neg_value_from_log(init_theta)
    best_theta = init_theta.copy()
    
    # Use Nelder-Mead
    # Initial simplex size: small perturbation
    # To speed up, we limit maxiter
    max_iter = 2000
    
    try:
        res = minimize(_neg_value_from_log, init_theta, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True})
        
        if res.fun < -best_val:
            best_val = -res.fun
            best_theta = res.x
    except Exception:
        pass
        
    # Check time
    elapsed = time.time() - t_start
    if elapsed > time_budget:
        return best_theta, best_val
        
    # If time remains, do a second pass with tighter tolerance
    remaining = time_budget - elapsed
    if remaining > 5:
        try:
            res2 = minimize(_neg_value_from_log, best_theta, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-8, 'adaptive': True})
            if res2.fun < -best_val:
                best_val = -res2.fun
                best_theta = res2.x
        except Exception:
            pass
            
    return best_theta, best_val

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    t_start = time.time()
    time_limit = 620.0  # Safety margin
    
    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_val = _compute_value(best_P)
    
    inits = _generate_initializations()
    
    # Allocate time budget per initialization
    # We have ~15 inits. 620s / 15 = ~40s each.
    # But Nelder-Mead might be fast. Let's give 20s each.
    time_per_init = 20.0
    
    for init_theta in inits:
        if time.time() - t_start > time_limit - 10:
            break
            
        try:
            theta_opt, val_opt = _optimize_from_init(init_theta, time_per_init)
            if val_opt > best_val:
                best_val = val_opt
                P_opt = _log_softmax(theta_opt).reshape(n, n)
                best_P = P_opt
        except Exception:
            pass
            
    # Final Polish: Use remaining time to polish the best result
    if time.time() - t_start < time_limit - 5:
        remaining = time_limit - (time.time() - t_start) - 2
        if remaining > 10:
            # Convert best_P to log-weights
            log_weights = np.log(best_P.ravel() + 1e-15)
            try:
                theta_final, val_final = _optimize_from_init(log_weights, remaining)
                if val_final > best_val:
                    best_val = val_final
                    best_P = _log_softmax(theta_final).reshape(n, n)
            except Exception:
                pass

    # Final normalization and check
    P = np.clip(best_P, 0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        P /= s
        
    return P

# EVOLVE-BLOCK-END
