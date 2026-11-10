import numpy as np
import scipy.optimize as optimize
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 590.0

    def calc_prob(support, weights):
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Pairwise sums and weights for Z = X1 + X2
        s_sums = support[:, None] + support[None, :]
        p_prods = p[:, None] * p[None, :]
        
        s_sums_flat = s_sums.flatten()
        p_prods_flat = p_prods.flatten()
        
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        p_sorted = p_prods_flat[sort_idx]
        p_prefix = np.cumsum(p_sorted)
        
        # Thresholds for X3, X4
        # Condition: X1 + X2 < 2*X4 - X3
        # Let s_k be value of X3, s_l be value of X4.
        # Threshold is 2*s_l - s_k.
        # We sum p_k * p_l * P(Z < 2*s_l - s_k)
        
        s_k_grid = support[:, None]
        s_l_grid = support[None, :]
        thresholds = 2.0 * s_l_grid - s_k_grid
        
        # Searchsorted on sorted s_sums
        # Number of elements < threshold
        idxs = np.searchsorted(s_sorted, thresholds, side='left')
        
        prob_z_less = np.zeros_like(idxs, dtype=np.float64)
        mask = idxs > 0
        if np.any(mask):
            prob_z_less[mask] = p_prefix[np.maximum(idxs[mask] - 1, 0)]
            
        p_k_grid = p[:, None]
        p_l_grid = p[None, :]
        
        total_prob = np.sum(p_k_grid * p_l_grid * prob_z_less)
        
        return float(total_prob)

    def objective_n2(params):
        # Support fixed at [0, 1]
        # Params: [p_0]
        # p_1 = 1 - p_0
        p0 = params[0]
        if p0 < 1e-9 or p0 > 1 - 1e-9:
            return -0.0
        support = np.array([0.0, 1.0])
        weights = np.array([p0, 1.0 - p0])
        return -calc_prob(support, weights)

    def objective_n3(params):
        # Support [0, a, 1]
        # Params: [p_0, p_a, a]
        # p_1 = 1 - p_0 - p_a
        p0, pa, a = params
        if p0 < 1e-9 or pa < 1e-9 or p0 + pa > 1 - 1e-9 or a < 1e-9 or a > 1 - 1e-9:
            return -0.0
        support = np.array([0.0, a, 1.0])
        weights = np.array([p0, pa, 1.0 - p0 - pa])
        return -calc_prob(support, weights)

    def objective_n4(params):
        # Support [0, a1, a2, 1]
        # Params: [p_0, p_a1, p_a2, a1, a2]
        # p_1 = 1 - p_0 - p_a1 - p_a2
        p0, p1, p2, a1, a2 = params
        if (p0 < 1e-9 or p1 < 1e-9 or p2 < 1e-9 or 
            p0 + p1 + p2 > 1 - 1e-9 or 
            a1 < 1e-9 or a2 < 1e-9 or a1 > 1 - 1e-9 or a2 > 1 - 1e-9):
            return -0.0
        support = np.array([0.0, a1, a2, 1.0])
        weights = np.array([p0, p1, p2, 1.0 - p0 - p1 - p2])
        return -calc_prob(support, weights)

    def objective_n5(params):
        # Support [0, a1, a2, a3, 1]
        # Params: [p_0, p_a1, p_a2, p_a3, a1, a2, a3]
        # p_1 = 1 - p_0 - p_a1 - p_a2 - p_a3
        p0, p1, p2, p3, a1, a2, a3 = params
        if (p0 < 1e-9 or p1 < 1e-9 or p2 < 1e-9 or p3 < 1e-9 or 
            p0 + p1 + p2 + p3 > 1 - 1e-9 or 
            a1 < 1e-9 or a2 < 1e-9 or a3 < 1e-9 or 
            a1 > 1 - 1e-9 or a2 > 1 - 1e-9 or a3 > 1 - 1e-9):
            return -0.0
        support = np.array([0.0, a1, a2, a3, 1.0])
        weights = np.array([p0, p1, p2, p3, 1.0 - p0 - p1 - p2 - p3])
        return -calc_prob(support, weights)

    best_score = -1.0
    best_support = None
    best_weights = None

    # Optimize n=2
    if time.time() < deadline:
        res2 = optimize.differential_evolution(objective_n2, bounds=[(1e-6, 1-1e-6)], seed=42, maxiter=1000, tol=1e-8, polish=True)
        score2 = -res2.fun
        support2 = np.array([0.0, 1.0])
        weights2 = np.array([res2.x[0], 1.0 - res2.x[0]])
        if score2 > best_score:
            best_score = score2
            best_support = support2.copy()
            best_weights = weights2.copy()

    # Optimize n=3
    if time.time() < deadline:
        bounds3 = [(1e-6, 0.9), (1e-6, 0.9), (1e-6, 0.9)]
        res3 = optimize.differential_evolution(objective_n3, bounds=bounds3, seed=42, maxiter=1000, tol=1e-8, polish=True)
        score3 = -res3.fun
        p0, pa, a = res3.x
        if p0 > 1e-9 and pa > 1e-9 and p0+pa < 1-1e-9 and a > 1e-9 and a < 1-1e-9:
            support3 = np.array([0.0, a, 1.0])
            weights3 = np.array([p0, pa, 1.0 - p0 - pa])
            s = calc_prob(support3, weights3)
            if s > best_score:
                best_score = s
                best_support = support3.copy()
                best_weights = weights3.copy()

    # Optimize n=4
    if time.time() < deadline:
        bounds4 = [(1e-6, 0.5), (1e-6, 0.5), (1e-6, 0.5), (1e-6, 0.9), (0.1, 0.9)]
        res4 = optimize.differential_evolution(objective_n4, bounds=bounds4, seed=42, maxiter=1000, tol=1e-8, polish=True)
        score4 = -res4.fun
        p0, p1, p2, a1, a2 = res4.x
        if (p0 > 1e-9 and p1 > 1e-9 and p2 > 1e-9 and 
            p0+p1+p2 < 1-1e-9 and 
            a1 > 1e-9 and a2 > 1e-9 and a1 < 1-1e-9 and a2 < 1-1e-9):
            support4 = np.array([0.0, a1, a2, 1.0])
            weights4 = np.array([p0, p1, p2, 1.0 - p0 - p1 - p2])
            s = calc_prob(support4, weights4)
            if s > best_score:
                best_score = s
                best_support = support4.copy()
                best_weights = weights4.copy()

    # Optimize n=5
    if time.time() < deadline:
        bounds5 = [(1e-6, 0.4), (1e-6, 0.4), (1e-6, 0.4), (1e-6, 0.4), 
                   (0.01, 0.4), (0.1, 0.6), (0.3, 0.9)]
        res5 = optimize.differential_evolution(objective_n5, bounds=bounds5, seed=42, maxiter=800, tol=1e-8, polish=True)
        score5 = -res5.fun
        p0, p1, p2, p3, a1, a2, a3 = res5.x
        if (p0 > 1e-9 and p1 > 1e-9 and p2 > 1e-9 and p3 > 1e-9 and 
            p0+p1+p2+p3 < 1-1e-9 and 
            a1 > 1e-9 and a2 > 1e-9 and a3 > 1e-9 and 
            a1 < 1-1e-9 and a2 < 1-1e-9 and a3 < 1-1e-9):
            support5 = np.array([0.0, a1, a2, a3, 1.0])
            weights5 = np.array([p0, p1, p2, p3, 1.0 - p0 - p1 - p2 - p3])
            s = calc_prob(support5, weights5)
            if s > best_score:
                best_score = s
                best_support = support5.copy()
                best_weights = weights5.copy()

    # If no valid result found (should not happen), fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    # Ensure non-negative and normalized
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    sum_w = np.sum(best_weights)
    if sum_w > 1e-9:
        best_weights = best_weights / sum_w
    else:
        best_weights = np.ones(len(best_support)) / len(best_support)

    return best_support, best_weights
