# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_entropies_from_marginals(marginals):
    """Compute Shannon entropy in bits for a list of marginal distributions."""
    entropies = []
    for m in marginals:
        m = m[m > 1e-15]
        if len(m) == 0:
            entropies.append(0.0)
        else:
            h = -np.sum(m * np.log2(m))
            entropies.append(h)
    return entropies


def _precompute_indices():
    """Precompute flattened indices for the 4 relevant linear projections."""
    n = 10
    # P[y, x] corresponds to flat index y * n + x
    # We need indices for:
    # D: x - y (range -9 to 9, shift by 9 to 0..18)
    # M0: x (range 0 to 9)
    # M1: x + y (range 0 to 18)
    # M2: x + 2y (range 0 to 27)
    
    idx = np.arange(100)
    y = idx // n
    x = idx % n
    
    idx_d = (x - y + 9).astype(int)
    idx_m0 = x.astype(int)
    idx_m1 = (x + y).astype(int)
    idx_m2 = (x + 2 * y).astype(int)
    
    return idx_d, idx_m0, idx_m1, idx_m2


# Global precomputed indices
_IDX_D, _IDX_M0, _IDX_M1, _IDX_M2 = _precompute_indices()


def _evaluate(P_flat):
    """
    Evaluate the ratio D/M for a flattened P vector.
    Returns ratio, D, H0, H1, H2, and the index of the max H in M.
    """
    P = P_flat.reshape(10, 10)
    P = np.clip(P, 0.0, None)
    total = P.sum()
    if total <= 1e-9:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0
    P = P / total
    P_flat_norm = P.flatten()
    
    # Compute marginals
    m_d = np.bincount(_IDX_D, weights=P_flat_norm, minlength=19)
    m_0 = np.bincount(_IDX_M0, weights=P_flat_norm, minlength=10)
    m_1 = np.bincount(_IDX_M1, weights=P_flat_norm, minlength=19)
    m_2 = np.bincount(_IDX_M2, weights=P_flat_norm, minlength=28)
    
    # Compute entropies
    D = 0.0
    m_d_pos = m_d[m_d > 1e-15]
    if len(m_d_pos) > 0:
        D = -np.sum(m_d_pos * np.log2(m_d_pos))
        
    H_0 = 0.0
    m_0_pos = m_0[m_0 > 1e-15]
    if len(m_0_pos) > 0:
        H_0 = -np.sum(m_0_pos * np.log2(m_0_pos))
        
    H_1 = 0.0
    m_1_pos = m_1[m_1 > 1e-15]
    if len(m_1_pos) > 0:
        H_1 = -np.sum(m_1_pos * np.log2(m_1_pos))
        
    H_2 = 0.0
    m_2_pos = m_2[m_2 > 1e-15]
    if len(m_2_pos) > 0:
        H_2 = -np.sum(m_2_pos * np.log2(m_2_pos))
        
    # Determine max H and its index
    H_vals = np.array([H_0, H_1, H_2])
    max_idx = np.argmax(H_vals)
    M = H_vals[max_idx]
    
    if M <= 1e-9:
        return 0.0, D, H_0, H_1, H_2, max_idx
        
    ratio = D / M
    return ratio, D, H_0, H_1, H_2, max_idx


def _objective(P_flat):
    """Objective function for minimization (negative ratio)."""
    ratio, _, _, _, _, _ = _evaluate(P_flat)
    return -ratio


def _gradient(P_flat):
    """
    Compute gradient of -ratio with respect to P_flat.
    ratio = D / M, where M = max(H_0, H_1, H_2).
    grad ratio = (grad D * M - D * grad M) / M^2
    grad M = grad H_active
    """
    ratio, D, H_0, H_1, H_2, max_idx = _evaluate(P_flat)
    
    if ratio == 0.0:
        return np.zeros(100)
        
    M = [H_0, H_1, H_2][max_idx]
    
    # We need gradients of D and the active H with respect to P_flat.
    # H(P) = - sum p_i log2(p_i) = - (1/ln 2) sum p_i ln(p_i)
    # dH/dp_j = - (1/ln 2) (ln(p_j) + 1)
    # Note: p_i here are the marginal probabilities, not the cell probabilities.
    # Let q_k be the marginal probability for value k.
    # q_k = sum_{i: proj(i)=k} P_i
    # d q_k / d P_j = 1 if proj(j)=k else 0.
    # dH/dP_j = sum_k (dH/dq_k) * (dq_k/dP_j) = (dH/dq_{proj(j)}) * 1
    # So dH/dP_j = -(1/ln 2) * (ln(q_{proj(j)}) + 1)
    
    # Precompute ln(q) for each projection
    def calc_grad_H(idx_proj, q):
        # q is the marginal vector
        # We need ln(q) for each cell j based on its projection idx_proj[j]
        # Handle q=0: if q=0, the term is 0 in entropy sum, but gradient is undefined?
        # Actually if q=0, p_j must be 0 for all j in that bin.
        # If P_j > 0, then q > 0.
        # We only compute gradient for P_j > 0 effectively, or handle 0s.
        # If P_j = 0, we can set gradient to 0 or large negative to push it up?
        # Standard practice: if P_j=0, set gradient to -1/ln(2) * 1 = -1/ln(2) (approx) 
        # because ln(0) is -inf, but the limit of p ln p is 0.
        # Let's be safe: replace 0s in q with small epsilon for log calculation, 
        # but only for cells that have mass?
        # Actually, if P_j > 0, q_{proj(j)} > 0.
        # If P_j == 0, the gradient contribution is 0 in the sum? No, dH/dP_j is defined.
        # If P_j=0, and we increase it, H changes.
        # Let's just compute for all j.
        
        q_safe = np.where(q > 1e-15, q, 1.0) # Avoid log(0)
        # If q is 0, ln(q) is -inf. But if P_j is 0, it doesn't contribute to H.
        # However, the gradient tells us how to change P.
        # If q=0, any increase in P_j (which contributes to this q) will increase H significantly.
        # So gradient should be large negative (to increase P).
        # Let's use a small floor for q in log.
        q_log = np.log2(q_safe)
        # For cells where q was 0, we want a strong negative gradient.
        # log2(1e-15) is about -50. So -(-50+1) = 49.
        
        # Map back to cell indices
        grad_cells = q_log[idx_proj] + 1.0
        # dH/dP_j = -grad_cells / ln(2)
        return -grad_cells / np.log(2)
    
    # Compute marginals again (or reuse from _evaluate, but _evaluate doesn't return them)
    P = np.clip(P_flat, 0.0, None)
    total = P.sum()
    if total <= 1e-9:
        return np.zeros(100)
    P_norm = P / total
    P_flat_norm = P_norm.flatten()
    
    m_d = np.bincount(_IDX_D, weights=P_flat_norm, minlength=19)
    m_0 = np.bincount(_IDX_M0, weights=P_flat_norm, minlength=10)
    m_1 = np.bincount(_IDX_M1, weights=P_flat_norm, minlength=19)
    m_2 = np.bincount(_IDX_M2, weights=P_flat_norm, minlength=28)
    
    grad_D = calc_grad_H(_IDX_D, m_d)
    
    if max_idx == 0:
        grad_M = calc_grad_H(_IDX_M0, m_0)
    elif max_idx == 1:
        grad_M = calc_grad_H(_IDX_M1, m_1)
    else:
        grad_M = calc_grad_H(_IDX_M2, m_2)
        
    # ratio = D / M
    # d(ratio)/dP = (M * dD - D * dM) / M^2
    # d(-ratio)/dP = (D * dM - M * dD) / M^2
    
    grad_ratio = (D * grad_M - M * grad_D) / (M * M)
    
    return -grad_ratio


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Uses full 100-parameter optimization with L-BFGS-B and multiple restarts.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    start_time = time.time()
    deadline = 600.0
    
    best_P = np.ones(100) / 100.0
    best_ratio, _, _, _, _, _ = _evaluate(best_P)
    
    # Random restarts
    np.random.seed(42)
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 10:
            break
            
        # Random initialization
        # Use Dirichlet for diverse starts
        alpha = np.random.uniform(0.5, 2.0, 100)
        P_init = np.random.dirichlet(alpha)
        
        # Try to optimize
        try:
            # Bounds: P >= 0
            bounds = [(0.0, 1.0) for _ in range(100)]
            
            result = minimize(
                _objective,
                P_init,
                method='L-BFGS-B',
                jac=_gradient,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            P_opt = result.x
            P_opt = np.clip(P_opt, 0.0, None)
            P_opt_sum = P_opt.sum()
            if P_opt_sum > 1e-9:
                P_opt = P_opt / P_opt_sum
            else:
                continue
                
            ratio_opt, _, _, _, _, _ = _evaluate(P_opt)
            
            if ratio_opt > best_ratio:
                best_ratio = ratio_opt
                best_P = P_opt.copy()
                
        except Exception:
            continue
            
    # Final validation and normalization
    best_P = best_P.reshape(10, 10)
    best_P = np.clip(best_P, 0.0, None)
    total = best_P.sum()
    if total <= 1e-9:
        best_P = np.ones((10, 10)) / 100.0
    else:
        best_P = best_P / total
        
    return best_P


# EVOLVE-BLOCK-END
