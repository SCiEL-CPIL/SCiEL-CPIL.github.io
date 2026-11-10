import numpy as np
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
    deadline = start_time + 620.0  # Safety margin: 630s limit
    
    def compute_P(support, weights):
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Pairwise sums and weights for Z = X1 + X2
        s_sums = support[:, None] + support[None, :]
        p_prods = p[:, None] * p[None, :]
        
        # Flatten and sort
        s_sums_flat = s_sums.flatten()
        p_prods_flat = p_prods.flatten()
        
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        p_sorted = p_prods_flat[sort_idx]
        
        # Prefix sums of probabilities for Z = X1 + X2
        p_prefix = np.cumsum(p_sorted)
        
        # Vectorized calculation for X3 and X4
        T = 2.0 * support[None, :] - support[:, None]
        idx = np.searchsorted(s_sorted, T, side='left')
        
        p_prefix_pad = np.zeros(n * n + 1)
        p_prefix_pad[1:] = p_prefix
        prob_z_less = p_prefix_pad[idx]
        
        total_prob = np.sum(p[:, None] * p[None, :] * prob_z_less)
        return total_prob

    def optimize_weights_fixed_support(support, time_limit, rng):
        """Optimize weights on a fixed support grid using projected gradient ascent."""
        n = len(support)
        p = np.ones(n) / n  # Uniform initialization
        
        best_p = p.copy()
        best_score = compute_P(support, p)
        
        # Precompute structures for fast score evaluation
        s_sums = support[:, None] + support[None, :]
        s_sums_flat = s_sums.flatten()
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        
        T = 2.0 * support[None, :] - support[:, None]
        idx = np.searchsorted(s_sorted, T, side='left')
        
        # Precompute the index matrix
        idx_flat = idx.flatten()
        
        # Precompute p_prefix for current p
        def score_and_grad(p_curr):
            p_curr = np.maximum(p_curr, 1e-12)
            p_curr = p_curr / np.sum(p_curr)
            
            # Compute score
            p_prods = p_curr[:, None] * p_curr[None, :]
            p_prods_flat = p_prods.flatten()
            p_sorted = p_prods_flat[sort_idx]
            p_prefix = np.cumsum(p_sorted)
            p_prefix_pad = np.zeros(n * n + 1)
            p_prefix_pad[1:] = p_prefix
            prob_z_less = p_prefix_pad[idx]
            score = np.sum(p_curr[:, None] * p_curr[None, :] * prob_z_less)
            
            # Compute gradient via finite differences
            grad = np.zeros(n)
            eps = 1e-4
            for i in range(n):
                p_plus = p_curr.copy()
                p_plus[i] += eps
                p_plus = p_plus / np.sum(p_plus)
                
                p_prods_plus = p_plus[:, None] * p_plus[None, :]
                p_prods_plus_flat = p_prods_plus.flatten()
                p_sorted_plus = p_prods_plus_flat[sort_idx]
                p_prefix_plus = np.cumsum(p_sorted_plus)
                p_prefix_pad_plus = np.zeros(n * n + 1)
                p_prefix_pad_plus[1:] = p_prefix_plus
                prob_z_less_plus = p_prefix_pad_plus[idx]
                score_plus = np.sum(p_plus[:, None] * p_plus[None, :] * prob_z_less_plus)
                
                p_minus = p_curr.copy()
                p_minus[i] = max(0, p_minus[i] - eps)
                p_minus_sum = np.sum(p_minus)
                if p_minus_sum < 1e-12:
                    grad[i] = 0
                else:
                    p_minus = p_minus / p_minus_sum
                    p_prods_minus = p_minus[:, None] * p_minus[None, :]
                    p_prods_minus_flat = p_prods_minus.flatten()
                    p_sorted_minus = p_prods_minus_flat[sort_idx]
                    p_prefix_minus = np.cumsum(p_sorted_minus)
                    p_prefix_pad_minus = np.zeros(n * n + 1)
                    p_prefix_pad_minus[1:] = p_prefix_minus
                    prob_z_less_minus = p_prefix_pad_minus[idx]
                    score_minus = np.sum(p_minus[:, None] * p_minus[None, :] * prob_z_less_minus)
                
                grad[i] = (score_plus - score_minus) / (2 * eps)
            
            return score, grad
        
        # Projected gradient ascent with adaptive step size
        lr = 0.1
        for iteration in range(2000):
            if time.time() > time_limit:
                break
            
            score, grad = score_and_grad(p)
            
            if score > best_score:
                best_score = score
                best_p = p.copy()
            
            # Gradient step
            p_new = p + lr * grad
            p_new = np.maximum(p_new, 0)
            p_sum = np.sum(p_new)
            if p_sum < 1e-12:
                p_new = np.ones(n) / n
            else:
                p_new = p_new / p_sum
            
            score_new, _ = score_and_grad(p_new)
            
            if score_new > score:
                p = p_new
            else:
                lr *= 0.9
                if lr < 1e-6:
                    # Try to escape with perturbation
                    p = p + rng.normal(0, 0.05, n)
                    p = np.maximum(p, 0)
                    p = p / np.sum(p)
                    lr = 0.05
        
        return best_p, best_score

    def refine_support(support, weights, rng, time_limit):
        """Refine both support and weights using alternating optimization."""
        n = len(support)
        current_support = support.copy()
        current_weights = weights.copy()
        current_score = compute_P(current_support, current_weights)
        
        step_s = 0.05
        step_w = 0.1
        stagnation = 0
        last_improve = time.time()
        
        while time.time() < time_limit:
            # Alternate between support and weight updates
            if rng.random() < 0.4:
                # Update support
                new_support = current_support.copy()
                idx = rng.integers(n)
                new_support[idx] += rng.normal(0, step_s)
                new_support = np.maximum(new_support, 0)
                new_support = np.sort(new_support)
                
                new_score = compute_P(new_support, current_weights)
                
                if new_score > current_score:
                    current_support = new_support
                    current_score = new_score
                    stagnation = 0
                    last_improve = time.time()
                else:
                    stagnation += 1
                    if stagnation > 20:
                        step_s *= 0.9
                        stagnation = 0
                        if step_s < 1e-5:
                            break
            else:
                # Update weights
                new_weights = current_weights.copy()
                new_weights += rng.normal(0, step_w, n)
                new_weights = np.maximum(new_weights, 0)
                w_sum = np.sum(new_weights)
                if w_sum < 1e-9:
                    continue
                new_weights = new_weights / w_sum
                
                new_score = compute_P(current_support, new_weights)
                
                if new_score > current_score:
                    current_weights = new_weights
                    current_score = new_score
                    stagnation = 0
                    last_improve = time.time()
                else:
                    stagnation += 1
                    if stagnation > 20:
                        step_w *= 0.9
                        stagnation = 0
                        if step_w < 1e-5:
                            break
            
            if time.time() - last_improve > 60:
                break
        
        return current_support, current_weights, current_score

    rng = np.random.default_rng(123)
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    # Phase 1: Try different support configurations
    support_configs = [
        np.linspace(0, 1, 8),
        np.linspace(0, 2, 10),
        np.geomspace(0.01, 1, 12),
        np.sort(rng.uniform(0, 1.5, 10)),
        np.sort(rng.uniform(0, 1, 12)),
        np.linspace(0, 0.5, 8),
    ]
    
    for support in support_configs:
        if time.time() > deadline - 60:
            break
        
        n = len(support)
        
        # Initialize weights
        weights_init = np.ones(n) / n
        
        # Optimize weights on this fixed support
        time_per_opt = min(30, (deadline - 60 - time.time()) / 2)
        p_opt, score_opt = optimize_weights_fixed_support(support, time.time() + time_per_opt, rng)
        
        if score_opt > best_score:
            best_score = score_opt
            best_support = support.copy()
            best_weights = p_opt.copy()
        
        # Refine support and weights jointly
        time_per_refine = min(20, (deadline - 30 - time.time()) / 3)
        if time.time() < deadline - 30:
            s_ref, w_ref, score_ref = refine_support(support, p_opt, rng, time.time() + time_per_refine)
            
            if score_ref > best_score:
                best_score = score_ref
                best_support = s_ref.copy()
                best_weights = w_ref.copy()
    
    # Phase 2: Final polish with small steps
    if time.time() < deadline - 5 and best_support is not None:
        n = len(best_support)
        current_support = best_support.copy()
        current_weights = best_weights.copy()
        current_score = best_score
        
        step_s = 1e-3
        step_w = 1e-3
        polish_deadline = deadline - 5
        
        while time.time() < polish_deadline:
            # Small support perturbation
            if rng.random() < 0.5:
                new_support = current_support.copy()
                idx = rng.integers(n)
                new_support[idx] += rng.normal(0, step_s)
                new_support = np.maximum(new_support, 0)
                new_score = compute_P(new_support, current_weights)
                
                if new_score > current_score:
                    current_support = new_support
                    current_score = new_score
                else:
                    step_s *= 0.95
                    if step_s < 1e-6:
                        step_s = 1e-6
            else:
                # Small weight perturbation
                new_weights = current_weights.copy()
                new_weights += rng.normal(0, step_w, n)
                new_weights = np.maximum(new_weights, 0)
                w_sum = np.sum(new_weights)
                if w_sum < 1e-9:
                    continue
                new_weights = new_weights / w_sum
                
                new_score = compute_P(current_support, new_weights)
                
                if new_score > current_score:
                    current_weights = new_weights
                    current_score = new_score
                else:
                    step_w *= 0.95
                    if step_w < 1e-6:
                        step_w = 1e-6
        
        if current_score > best_score:
            best_score = current_score
            best_support = current_support
            best_weights = current_weights
    
    # Final cleanup
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    
    # Ensure valid return
    best_support = best_support.flatten()
    best_weights = best_weights.flatten()
    
    # Verify validity
    if len(best_support) != len(best_weights) or len(best_support) == 0:
        best_support = np.arange(8, dtype=np.float64)
        best_weights = np.ones(8, dtype=np.float64) / 8
    
    if np.any(np.isnan(best_support)) or np.any(np.isinf(best_support)):
        best_support = np.arange(8, dtype=np.float64)
        best_weights = np.ones(8, dtype=np.float64) / 8
    
    if np.any(np.isnan(best_weights)) or np.any(np.isinf(best_weights)):
        best_weights = np.ones(len(best_support), dtype=np.float64) / len(best_support)
    
    if np.sum(best_weights) <= 1e-9:
        best_weights = np.ones(len(best_support), dtype=np.float64) / len(best_support)
    
    return best_support, best_weights
