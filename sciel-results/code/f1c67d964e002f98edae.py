# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _precompute_pooling_matrices(n=10):
    """Precompute pooling matrices for slopes -1, 0, 1, 2.
    Each matrix A_r has shape (k_r, n*n) where k_r is number of unique values.
    A_r[v, idx] = 1 if cell idx maps to value v under slope r.
    """
    y_idx, x_idx = np.indices((n, n))
    flat_idx = np.arange(n * n)
    
    matrices = {}
    for r in [-1, 0, 1, 2]:
        vals = (x_idx + r * y_idx).flatten()
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        k = len(unique_vals)
        A = np.zeros((k, n * n), dtype=np.float64)
        A[inverse, flat_idx] = 1.0
        matrices[r] = (A, unique_vals, inverse)
    
    return matrices


def _compute_all_entropies(P_flat, matrices):
    """Compute H_{-1}, H_0, H_1, H_2 from flattened P."""
    entropies = {}
    for r in [-1, 0, 1, 2]:
        A, _, _ = matrices[r]
        probs = A @ P_flat
        probs = np.maximum(probs, 0)
        mask = probs > 1e-15
        if mask.sum() == 0:
            entropies[r] = 0.0
        else:
            p = probs[mask]
            entropies[r] = -np.sum(p * np.log2(p))
    return entropies


def _compute_ratio(P_flat, matrices):
    """Compute D/M ratio."""
    ent = _compute_all_entropies(P_flat, matrices)
    D = ent[-1]
    M = max(ent[0], ent[1], ent[2])
    if M < 1e-9:
        return 0.0, D, M
    return D / M, D, M


def _gradient_log_ratio(P_flat, matrices):
    """Compute gradient of log(D) - log(M) w.r.t. P_flat.
    This is the gradient of the log-ratio objective.
    """
    ent = _compute_all_entropies(P_flat, matrices)
    D = ent[-1]
    M = max(ent[0], ent[1], ent[2])
    
    if D < 1e-15 or M < 1e-15:
        return np.zeros_like(P_flat)
    
    # d(log D)/dP = (1/D) * dD/dP
    # dD/dP[y,x] = -(log2(q_v) + 1/ln(2)) where q_v is the pooled prob for cell (y,x)
    
    # For numerator: slope -1
    A_num, _, inv_num = matrices[-1]
    q_num = A_num @ P_flat
    q_num = np.maximum(q_num, 1e-15)
    # dD/dP_flat[i] = -(log2(q_num[inv_num[i]]) + 1/ln(2))
    log_q_num = np.log2(q_num)
    dD_dP = -(log_q_num[inv_num] + 1.0 / np.log(2))
    dlogD_dP = dD_dP / D
    
    # For denominator: max of H_0, H_1, H_2
    # We use the maximum one (or average if tied)
    max_r = None
    max_val = -1
    for r in [0, 1, 2]:
        if ent[r] > max_val:
            max_val = ent[r]
            max_r = r
    
    A_den, _, inv_den = matrices[max_r]
    q_den = A_den @ P_flat
    q_den = np.maximum(q_den, 1e-15)
    log_q_den = np.log2(q_den)
    dH_den_dP = -(log_q_den[inv_den] + 1.0 / np.log(2))
    dlogM_dP = dH_den_dP / M
    
    return dlogD_dP - dlogM_dP


def _project_simplex(P, n=10):
    """Project P onto simplex: non-negative, sum to 1."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s > 1e-15:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
    return P


def _gradient_ascent(P, matrices, n, max_iters=2000, time_limit=300):
    """Run gradient ascent on log-ratio objective with adaptive step size."""
    start_time = time.time()
    P_flat = P.flatten().copy()
    
    best_P_flat = P_flat.copy()
    best_ratio, _, _ = _compute_ratio(P_flat, matrices)
    
    step_size = 0.02
    patience = 0
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradient
        grad = _gradient_log_ratio(P_flat, matrices)
        
        # Gradient step
        P_new_flat = P_flat + step_size * grad
        P_new = _project_simplex(P_new_flat.reshape(n, n), n)
        P_new_flat = P_new.flatten()
        
        # Compute new ratio
        new_ratio, _, _ = _compute_ratio(P_new_flat, matrices)
        
        if new_ratio > best_ratio + 1e-10:
            best_ratio = new_ratio
            best_P_flat = P_new_flat.copy()
            patience = 0
            no_improve_count = 0
        else:
            patience += 1
            no_improve_count += 1
        
        # Adaptive step size
        if patience > 50:
            step_size *= 0.85
            patience = 0
        
        if no_improve_count > 500:
            break
        
        P_flat = P_new_flat
    
    return best_P_flat.reshape(n, n), best_ratio


def _random_init(n, rng):
    """Generate a random initial distribution on the simplex."""
    P = rng.dirichlet(np.ones(n * n))
    return P.reshape(n, n)


def _structured_init(n, rng):
    """Generate structured initial distributions that may have good ratio properties."""
    inits = []
    
    # Uniform
    inits.append(np.ones((n, n)) / (n * n))
    
    # Diagonal emphasis (x - y = 0 has high weight)
    for diag_weight in [0.1, 0.3, 0.5]:
        P = np.ones((n, n)) * 0.01
        for i in range(n):
            P[i, i] = diag_weight
        P = P / P.sum()
        inits.append(P)
    
    # Anti-diagonal emphasis
    for anti_weight in [0.1, 0.3, 0.5]:
        P = np.ones((n, n)) * 0.01
        for i in range(n):
            P[i, n - 1 - i] = anti_weight
        P = P / P.sum()
        inits.append(P)
    
    # Corner emphasis
    for corner_weight in [0.2, 0.4]:
        P = np.ones((n, n)) * 0.01
        P[0, 0] = corner_weight
        P[n-1, n-1] = corner_weight
        P[0, n-1] = corner_weight
        P[n-1, 0] = corner_weight
        P = P / P.sum()
        inits.append(P)
    
    # Random sparse
    for _ in range(5):
        P = np.zeros((n, n))
        num_nonzero = rng.integers(10, 50)
        idx = rng.choice(n * n, size=num_nonzero, replace=False)
        P.flat[idx] = rng.dirichlet(np.ones(num_nonzero))
        P = P / P.sum()
        inits.append(P)
    
    return inits


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimize P to maximize H_{-1} / max(H_0, H_1, H_2) using precomputed
    pooling matrices and gradient ascent on the log-ratio objective.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 620  # safety margin
    
    # Precompute pooling matrices
    matrices = _precompute_pooling_matrices(n)
    
    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = _structured_init(n, rng)
    
    # Add some random inits
    for _ in range(10):
        candidates.append(_random_init(n, rng))
    
    best_P = None
    best_ratio = -np.inf
    
    # Evaluate all initial candidates
    for P_init in candidates:
        if time.time() > deadline - 100:
            break
        P_flat = P_init.flatten()
        ratio, _, _ = _compute_ratio(P_flat, matrices)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P_init.copy()
    
    # Run gradient ascent from each promising initial point
    # Sort candidates by initial ratio and try top ones
    candidate_ratios = []
    for P_init in candidates:
        P_flat = P_init.flatten()
        ratio, _, _ = _compute_ratio(P_flat, matrices)
        candidate_ratios.append((ratio, P_init))
    
    candidate_ratios.sort(key=lambda x: x[0], reverse=True)
    
    # Try gradient ascent from top 10 initial points
    for ratio, P_init in candidate_ratios[:10]:
        if time.time() > deadline - 50:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 30:
            break
        
        # Allocate time based on how many attempts remain
        attempt_time = min(remaining_time / 2, 120)
        
        P_opt, ratio_opt = _gradient_ascent(
            P_init, matrices, n,
            max_iters=5000,
            time_limit=attempt_time
        )
        
        if ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_P = P_opt.copy()
    
    # Final polish: run more iterations from the best solution
    if best_P is not None and time.time() < deadline - 30:
        remaining_time = deadline - time.time() - 10
        if remaining_time > 20:
            P_final, ratio_final = _gradient_ascent(
                best_P, matrices, n,
                max_iters=10000,
                time_limit=remaining_time
            )
            if ratio_final > best_ratio:
                best_ratio = ratio_final
                best_P = P_final.copy()
    
    # Ensure validity
    if best_P is None:
        best_P = np.ones((n, n)) / (n * n)
    
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 1e-15:
        best_P = best_P / s
    else:
        best_P = np.ones((n, n)) / (n * n)
    
    return best_P


# EVOLVE-BLOCK-END
