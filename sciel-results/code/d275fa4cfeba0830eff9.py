# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N)
_YS = np.repeat(_VALS, _N)
_XS = np.tile(_VALS, _N)
_XY = _XS + _YS
_X2Y = _XS + 2 * _YS
_XM1Y = _XS - _YS  # range -9..9

# Pre-compute bin indices for efficiency
_IDX_X = _XS
_IDX_XY = _XY
_IDX_X2Y = _X2Y
_IDX_XM1Y = _XM1Y + 9  # Shift to 0..18

def _entropy(p):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    s = p.sum()
    if s <= 1e-30:
        return 0.0
    q = p / s
    q = q[q > 0.0]
    if len(q) == 0:
        return 0.0
    return float(-(q * np.log2(q)).sum())

def _compute_value(P):
    """Compute the ratio D/M for a given distribution P using hard max."""
    p = P.ravel()
    total = p.sum()
    if total <= 1e-30:
        return 0.0

    # D: distribution of x - y (slope -1), 19 bins
    d_bins = np.zeros(19)
    np.add.at(d_bins, _IDX_XM1Y, p)
    hd = _entropy(d_bins)

    # H_0: distribution of x, 10 bins
    h0_bins = np.zeros(10)
    np.add.at(h0_bins, _IDX_X, p)
    h0 = _entropy(h0_bins)

    # H_1: distribution of x + y, 19 bins
    h1_bins = np.zeros(19)
    np.add.at(h1_bins, _IDX_XY, p)
    h1 = _entropy(h1_bins)

    # H_2: distribution of x + 2y, 28 bins
    h2_bins = np.zeros(28)
    np.add.at(h2_bins, _IDX_X2Y, p)
    h2 = _entropy(h2_bins)

    m = max(h0, h1, h2)
    if m <= 1e-9:
        return 0.0
    return hd / m

def _build_from_theta(theta):
    """Convert log-probability parameters to a probability distribution."""
    theta_shifted = theta - np.max(theta)
    p = np.exp(theta_shifted)
    s = p.sum()
    if s <= 1e-30:
        return np.ones(_N * _N) / (_N * _N)
    p = p / s
    return p

def _soft_max(vals, t=10.0):
    """
    Approximate max(vals) using log-sum-exp.
    Larger t makes the approximation sharper (closer to max).
    """
    # Shift for numerical stability
    v = np.array(vals)
    v_max = np.max(v)
    # If all are very small/negative, handle carefully
    # log-sum-exp: v_max + log(sum(exp(v - v_max)))
    # We want max, so we use a high temperature parameter in the exponent?
    # Actually, max(x) = lim_{t->inf} (1/t) log(sum(exp(t*x)))
    # So we compute (1/t) * log(sum(exp(t*x)))
    # To avoid overflow, we use: (1/t) * (t*v_max + log(sum(exp(t*(v - v_max)))))
    # = v_max + (1/t) * log(sum(exp(t*(v - v_max))))
    
    # If t is large, exp(t*(v-v_max)) will be 1 for max and 0 for others.
    # log(sum) will be log(1) = 0. So result is v_max.
    # For t=10, it's a smooth approximation.
    
    # Let's use a simpler robust soft-max:
    # soft_max = v_max + (1/t) * log(sum(exp(t * (v - v_max))))
    
    # Handle case where v_max is -inf or similar
    if not np.isfinite(v_max):
        return 0.0
        
    # Scale
    scaled = t * (v - v_max)
    # Clip to avoid underflow/overflow issues in exp, though v-v_max <= 0 so exp <= 1
    # exp(scaled) is safe as long as scaled is not too negative (underflow to 0 is fine)
    exp_sum = np.sum(np.exp(scaled))
    if exp_sum <= 0:
        return v_max
    return v_max + (1.0/t) * np.log(exp_sum)

def _objective_soft(theta, t=10.0):
    """
    Negative of the soft-max value we want to maximize.
    Uses soft-max for M to ensure differentiability.
    """
    p = _build_from_theta(theta)
    
    # Compute entropies
    # D
    d_bins = np.zeros(19)
    np.add.at(d_bins, _IDX_XM1Y, p)
    hd = _entropy(d_bins)
    
    # H0
    h0_bins = np.zeros(10)
    np.add.at(h0_bins, _IDX_X, p)
    h0 = _entropy(h0_bins)
    
    # H1
    h1_bins = np.zeros(19)
    np.add.at(h1_bins, _IDX_XY, p)
    h1 = _entropy(h1_bins)
    
    # H2
    h2_bins = np.zeros(28)
    np.add.at(h2_bins, _IDX_X2Y, p)
    h2 = _entropy(h2_bins)
    
    m_soft = _soft_max([h0, h1, h2], t)
    if m_soft <= 1e-9:
        return 0.0
    return -hd / m_soft

def _greedy_refine(P, n_iter=200, deadline=None):
    """
    Greedy coordinate descent refinement: try small perturbations to each cell
    and keep if it improves the hard-max objective.
    """
    best_P = P.copy()
    best_val = _compute_value(best_P)
    n_cells = _N * _N
    p_flat = best_P.ravel().copy()
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n_cells):
            old_val = p_flat[i]
            if old_val < 1e-15:
                continue
            # Try increasing and decreasing
            for factor in [0.5, 0.7, 1.3, 2.0]:
                new_val = old_val * factor
                if new_val < 0:
                    continue
                p_flat[i] = new_val
                total = p_flat.sum()
                if total <= 1e-30:
                    p_flat[i] = old_val
                    continue
                P_test = (p_flat / total).reshape(_N, _N)
                val_test = _compute_value(P_test)
                if val_test > best_val + 1e-12:
                    best_val = val_test
                    best_P = P_test.copy()
                    p_flat = best_P.ravel().copy()
                    improved = True
                    break
                else:
                    p_flat[i] = old_val
        if not improved:
            break
    
    return best_P, best_val

def _simulated_annealing(P, deadline=None, n_steps=20000):
    """
    Simulated annealing that swaps probability mass between pairs of cells.
    This treats the problem as a discrete combinatorial optimization over support
    structure, using the entropy ratio as a step function to escape local minima.
    """
    rng = np.random.RandomState(12345)
    n_cells = _N * _N
    p_flat = P.ravel().copy()
    best_P = P.copy()
    best_val = _compute_value(best_P)
    
    T_start = 0.1
    T_end = 1e-6
    n_active = np.sum(p_flat > 1e-8)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Cooling schedule
        progress = step / max(n_steps, 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Select two cells to swap mass
        # Bias toward active cells
        active_mask = p_flat > 1e-10
        if np.sum(active_mask) < 2:
            active_mask = np.ones(n_cells, dtype=bool)
        
        # With high probability, pick from active; occasionally explore
        if rng.random() < 0.85:
            active_indices = np.where(active_mask)[0]
            i = active_indices[rng.randint(len(active_indices))]
            j = active_indices[rng.randint(len(active_indices))]
        else:
            i = rng.randint(n_cells)
            j = rng.randint(n_cells)
        
        if i == j:
            continue
        
        # Determine swap amount: fraction of smaller cell's mass
        pi = p_flat[i]
        pj = p_flat[j]
        if pi < 1e-12 or pj < 1e-12:
            continue
        
        # Try multiple swap fractions
        best_local_val = best_val
        best_local_p = None
        
        for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            swap_amt = min(pi, pj) * frac
            if swap_amt < 1e-15:
                continue
            
            new_pi = pi - swap_amt
            new_pj = pj + swap_amt
            
            if new_pi < 0 or new_pj < 0:
                continue
            
            p_flat[i] = new_pi
            p_flat[j] = new_pj
            
            total = p_flat.sum()
            if total <= 1e-30:
                p_flat[i] = pi
                p_flat[j] = pj
                continue
            
            P_test = (p_flat / total).reshape(_N, _N)
            val_test = _compute_value(P_test)
            
            if val_test > best_local_val:
                best_local_val = val_test
                best_local_p = p_flat.copy()
            else:
                p_flat[i] = pi
                p_flat[j] = pj
        
        # Accept the best local move if it improves
        if best_local_p is not None:
            p_flat = best_local_p
            if best_local_val > best_val:
                best_val = best_local_val
                best_P = (p_flat / p_flat.sum()).reshape(_N, _N).copy()
        
        # With some probability, accept a worse move to escape local minima
        elif T > T_end * 10:
            # Pick a random perturbation
            i = rng.randint(n_cells)
            j = rng.randint(n_cells)
            if i != j and p_flat[i] > 1e-10 and p_flat[j] > 1e-10:
                frac = rng.uniform(0.1, 0.5)
                swap_amt = min(p_flat[i], p_flat[j]) * frac
                if swap_amt > 1e-15:
                    old_i = p_flat[i]
                    old_j = p_flat[j]
                    p_flat[i] -= swap_amt
                    p_flat[j] += swap_amt
                    
                    total = p_flat.sum()
                    if total > 1e-30:
                        P_test = (p_flat / total).reshape(_N, _N)
                        val_test = _compute_value(P_test)
                        delta_val = val_test - best_val
                        
                        # Metropolis criterion
                        if delta_val > 0 or rng.random() < np.exp(delta_val / T):
                            if val_test > best_val:
                                best_val = val_test
                                best_P = P_test.copy()
                        else:
                            p_flat[i] = old_i
                            p_flat[j] = old_j
    
    return best_P, best_val

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    Uses multiple random restarts with L-BFGS-B in log-probability space
    to maximize D/M where D = H_{x-y} and M = max(H_x, H_{x+y}, H_{x+2y}).
    Uses a soft-max surrogate for M during optimization to handle non-smoothness.
    
    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1.
    """
    deadline = time.time() + 550  # Safety margin before 600s limit
    n_cells = _N * _N
    
    best_P = np.ones((_N, _N), dtype=np.float64) / n_cells
    best_val = _compute_value(best_P)
    
    rng = np.random.RandomState(42)
    
    # Strategy 1: Start from uniform and optimize with soft-max objective (higher t for sharper max)
    theta0 = np.zeros(n_cells)
    result = minimize(
        lambda th: _objective_soft(th, t=30.0), theta0, method='L-BFGS-B',
        options={'maxiter': 800, 'ftol': 1e-10, 'gtol': 1e-9}
    )
    if time.time() > deadline:
        return best_P
    
    p_opt = _build_from_theta(result.x)
    P_opt = p_opt.reshape(_N, _N)
    val_opt = _compute_value(P_opt)
    if val_opt > best_val:
        best_val = val_opt
        best_P = P_opt.copy()
    
    # Strategy 2: Multiple random restarts
    n_restarts = 50 # Increased restarts as soft-max may converge faster/more reliably
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random initialization in log-space
        if restart % 3 == 0:
            # Random sparse initialization
            theta = rng.randn(n_cells) * 2.0
        elif restart % 3 == 1:
            # Concentrate on anti-diagonal
            theta = np.full(n_cells, -10.0)
            for y in range(_N):
                for x in range(_N):
                    if x + y == 9:
                        theta[y * _N + x] = 0.0
                    elif abs(x - y) >= 5:
                        theta[y * _N + x] = -2.0
        else:
            # Concentrate on corners
            theta = np.full(n_cells, -10.0)
            corners = [(0, 0), (0, 9), (9, 0), (9, 9)]
            for (cy, cx) in corners:
                theta[cy * _N + cx] = 0.0
            for y in range(_N):
                for x in range(_N):
                    if abs(x - y) >= 4:
                        theta[y * _N + x] = -3.0
        
        result = minimize(
            _objective_soft, theta, method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        p_opt = _build_from_theta(result.x)
        P_opt = p_opt.reshape(_N, _N)
        val_opt = _compute_value(P_opt)
        if val_opt > best_val:
            best_val = val_opt
            best_P = P_opt.copy()
    
    # Strategy 3: Refine the best solution with higher precision
    if best_val > 0 and time.time() < deadline - 90:
        p_flat = best_P.ravel()
        theta_best = np.log(p_flat + 1e-15)
        
        # Use a much higher t for a sharper approximation of max
        result = minimize(
            lambda th: _objective_soft(th, t=100.0), theta_best, method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        p_opt = _build_from_theta(result.x)
        P_opt = p_opt.reshape(_N, _N)
        val_opt = _compute_value(P_opt)
        if val_opt > best_val:
            best_val = val_opt
            best_P = P_opt.copy()
        
        # Second polish with even higher t
        if time.time() < deadline - 60:
            p_flat = best_P.ravel()
            theta_best2 = np.log(p_flat + 1e-15)
            result2 = minimize(
                lambda th: _objective_soft(th, t=200.0), theta_best2, method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            p_opt2 = _build_from_theta(result2.x)
            P_opt2 = p_opt2.reshape(_N, _N)
            val_opt2 = _compute_value(P_opt2)
            if val_opt2 > best_val:
                best_val = val_opt2
                best_P = P_opt2.copy()
    
    # Strategy 4: Greedy coordinate descent refinement
    if time.time() < deadline - 100:
        best_P, best_val = _greedy_refine(best_P, n_iter=300, deadline=deadline - 80)
    
    # Strategy 5: Simulated annealing to escape local minima
    if time.time() < deadline - 60:
        sa_deadline = deadline - 40
        best_P, best_val = _simulated_annealing(best_P, deadline=sa_deadline, n_steps=30000)
    
    # Strategy 6: Final high-precision polish after SA
    if best_val > 0 and time.time() < deadline - 20:
        p_flat = best_P.ravel()
        theta_best3 = np.log(p_flat + 1e-15)
        result3 = minimize(
            lambda th: _objective_soft(th, t=500.0), theta_best3, method='L-BFGS-B',
            options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13}
        )
        p_opt3 = _build_from_theta(result3.x)
        P_opt3 = p_opt3.reshape(_N, _N)
        val_opt3 = _compute_value(P_opt3)
        if val_opt3 > best_val:
            best_val = val_opt3
            best_P = P_opt3.copy()
    
    # Final cleanup
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s <= 1e-9:
        best_P = np.ones((_N, _N), dtype=np.float64) / n_cells
    else:
        best_P /= s
    
    return best_P


# EVOLVE-BLOCK-END
