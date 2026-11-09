import numpy as np
import time
from scipy.optimize import linprog

def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shape: expect (5, 100)
    if prizes.ndim == 1:
        P = prizes
        w = weights.reshape(5, -1) if weights.ndim == 1 else weights
    else:
        P = prizes.sum(axis=0)
        w = weights
    
    n = P.shape[0]
    C = w.shape[0] if w.ndim == 2 else 5
    cap = 25.0
    
    # Time budget
    t0 = time.time()
    deadline = t0 + 310.0  # Safety margin before 330s kill
    
    eps = 1e-9
    
    def sanitize(h):
        h = np.asarray(h, dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        return np.clip(h, 1e-9, 1e6)
    
    # Precompute weight statistics
    wmax = w.max(axis=0)
    wsum = w.sum(axis=0)
    wmean = w.mean(axis=0)
    
    # --- 1. LP Relaxation for Dual Prices ---
    # Maximize P^T x subject to w x <= cap, 0 <= x <= 1
    # linprog minimizes, so minimize -P^T x
    dual_prices = np.ones(C)  # fallback: uniform prices
    try:
        c_obj = -P
        A_ub = w  # shape (C, n)
        b_ub = np.full(C, cap)
        bounds = [(0, 1)] * n
        # Use highs method for speed and robustness
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            # Extract dual variables (shadow prices)
            # In HiGHS, for inequality constraints Ax <= b, the marginals are the duals
            if hasattr(res, 'ineqlin') and res.ineqlin is not None:
                if hasattr(res.ineqlin, 'marginals'):
                    dual_prices = np.abs(res.ineqlin.marginals)
                    # Ensure non-negative and finite
                    dual_prices = np.where(np.isfinite(dual_prices) & (dual_prices > 0), dual_prices, 1e-9)
    except Exception:
        pass
    
    # Reduced cost: P[i] - sum_c dual[c] * w[c,i]
    # High reduced cost means the item is "cheap" relative to its value given the constraint prices
    reduced_costs = P - w.T @ dual_prices
    
    # --- 2. Greedy Frequency Estimation ---
    # Build a few greedy solutions to identify "commonly chosen" items
    def get_greedy_freq(P, w, cap, C, n, iterations=20):
        item_freq = np.zeros(n)
        rng = np.random.RandomState(42)
        wmax_local = w.max(axis=0)
        for _ in range(iterations):
            remaining = np.full(C, cap)
            used = np.zeros(n, dtype=bool)
            while True:
                fits_mask = np.all(w <= remaining[:, None], axis=0) & ~used
                if not fits_mask.any():
                    break
                fit_indices = np.where(fits_mask)[0]
                # Use value density for greedy choice
                scores = P[fit_indices] / (wmax_local[fit_indices] + eps)
                # Add slight randomness
                scores += 0.05 * rng.rand(len(fit_indices))
                best_local = np.argmax(scores)
                idx = fit_indices[best_local]
                used[idx] = True
                item_freq[idx] += 1
                remaining -= w[:, idx]
        return item_freq
    
    greedy_freq = get_greedy_freq(P, w, cap, C, n)
    greedy_norm = greedy_freq / (greedy_freq.max() + eps)
    
    # --- 3. Pareto Dominance Filter ---
    # Items that are not dominated by others get a boost
    dominance_count = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if P[j] >= P[i] and np.all(w[:, j] <= w[:, i]):
                if P[j] > P[i] or np.any(w[:, j] < w[:, i]):
                    dominance_count[i] += 1
    tau = 2.0
    pareto_weight = np.exp(-dominance_count / tau)
    
    # --- 4. Candidate Generation ---
    # Base components
    ratio_max = P / (wmax + eps)
    ratio_sum = P / (wsum + eps)
    rc_pos = np.maximum(reduced_costs, eps)
    
    candidates = [
        # Pure Ratio
        sanitize(ratio_max),
        sanitize(ratio_max ** 2),
        sanitize(ratio_sum),
        # Pure LP Reduced Cost
        sanitize(rc_pos),
        sanitize(rc_pos ** 2),
        # Blends of Ratio and LP
        sanitize(0.5 * ratio_max + 0.5 * rc_pos),
        sanitize(0.7 * ratio_max + 0.3 * rc_pos),
        sanitize(0.3 * ratio_max + 0.7 * rc_pos),
        sanitize(ratio_max * rc_pos),
        # Greedy Boosted
        sanitize(ratio_max * (1.0 + greedy_norm)),
        sanitize(rc_pos * (1.0 + greedy_norm)),
        sanitize(0.5 * ratio_max * (1.0 + greedy_norm) + 0.5 * rc_pos * (1.0 + greedy_norm)),
        # Pareto Weighted
        sanitize(ratio_max * pareto_weight),
        sanitize(rc_pos * pareto_weight),
        sanitize(ratio_max ** 2 * pareto_weight),
        sanitize(rc_pos ** 2 * pareto_weight),
        # Complex Blend: Ratio * LP * Pareto
        sanitize(ratio_max * rc_pos * pareto_weight),
        sanitize(ratio_max * (1.0 + greedy_norm) * pareto_weight),
    ]
    
    # --- 5. Colony Simulation ---
    def fast_colony_sim(h, seed, n_iter=30, n_ants=20):
        rng = np.random.RandomState(seed)
        pheromone = np.ones(n)
        best_total = 0.0
        
        for it in range(n_iter):
            ant_totals = np.zeros(n_ants)
            
            for a in range(n_ants):
                remaining = np.full(C, cap)
                used = np.zeros(n, dtype=bool)
                total = 0.0
                
                while True:
                    # Vectorized fit check
                    fits_mask = np.all(w <= remaining[:, None], axis=0) & ~used
                    if not fits_mask.any():
                        break
                    
                    fit_indices = np.where(fits_mask)[0]
                    probs = pheromone[fit_indices] * h[fit_indices]
                    probs = probs / probs.sum()
                    
                    idx = rng.choice(fit_indices, p=probs)
                    used[idx] = True
                    remaining -= w[:, idx]
                    total += P[idx]
                
                ant_totals[a] = total
            
            iter_best = ant_totals.max()
            best_total = max(best_total, iter_best)
            pheromone = pheromone * 0.9 + iter_best / 10.0
        
        return best_total
    
    # --- 6. Evaluation and Search ---
    best_h = candidates[0]
    best_score = -1.0
    
    # Phase 1: Evaluate initial candidates
    for h in candidates:
        if time.time() > deadline - 60.0:
            break
        score = fast_colony_sim(h, 42)
        if score > best_score:
            best_score = score
            best_h = h
            
    # Phase 2: Refinement around best candidate
    if time.time() < deadline - 30.0:
        # Identify which base feature was dominant in best_h? 
        # Simpler: Just try variations of the top components
        # 1. Power adjustments on the best candidate
        for alpha in [0.5, 0.8, 1.2, 1.5, 2.0]:
            if time.time() > deadline - 30.0:
                break
            h_test = sanitize(best_h ** alpha)
            score = fast_colony_sim(h_test, 42)
            if score > best_score:
                best_score = score
                best_h = h_test
        
        # 2. Blend best_h with pure LP and pure Ratio
        # We don't know which one best_h is derived from, so we blend best_h with the other strong components
        for beta in [0.2, 0.5, 0.8]:
            if time.time() > deadline - 30.0:
                break
            # Blend with LP
            h_lp_blend = sanitize(beta * best_h + (1 - beta) * rc_pos)
            score = fast_colony_sim(h_lp_blend, 42)
            if score > best_score:
                best_score = score
                best_h = h_lp_blend
            
            # Blend with Ratio
            h_ratio_blend = sanitize(beta * best_h + (1 - beta) * ratio_max)
            score = fast_colony_sim(h_ratio_blend, 42)
            if score > best_score:
                best_score = score
                best_h = h_ratio_blend
                
        # 3. Try adding greedy frequency to the current best
        for gamma in [0.5, 1.0, 2.0]:
            if time.time() > deadline - 30.0:
                break
            h_greedy_blend = sanitize(best_h * (1.0 + gamma * greedy_norm))
            score = fast_colony_sim(h_greedy_blend, 42)
            if score > best_score:
                best_score = score
                best_h = h_greedy_blend
                
        # 4. Try Pareto weighting on the current best
        if time.time() < deadline - 30.0:
            h_pareto_blend = sanitize(best_h * pareto_weight)
            score = fast_colony_sim(h_pareto_blend, 42)
            if score > best_score:
                best_score = score
                best_h = h_pareto_blend

    # Phase 3: Final Polish / Perturbation
    if time.time() < deadline - 10.0:
        # Small random perturbations to escape local optima
        rng = np.random.RandomState(123)
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            noise = 1.0 + 0.05 * rng.randn(n)
            h_perturbed = sanitize(best_h * noise)
            score = fast_colony_sim(h_perturbed, 42)
            if score > best_score:
                best_score = score
                best_h = h_perturbed

    return best_h
