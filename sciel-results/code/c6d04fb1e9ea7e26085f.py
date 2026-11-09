"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

CAPACITY = 25.0
N_ITEMS = 100


def _solve_lp(prizes, weights, capacity, n_cons, n_items):
    """
    Solve the LP relaxation of the multiple-constraint knapsack.
    Returns (objective_value, dual_prices) where dual_prices has shape (n_cons,).
    If LP fails, returns (0, None).
    """
    try:
        # Objective: maximize sum of total_prize * x_i
        total_prize = prizes.sum(axis=0)
        c = -total_prize  # linprog minimizes
        
        # Constraints: for each constraint c, sum_j weights[c,j] * x_j <= capacity
        A_ub = weights  # shape (n_cons, n_items)
        b_ub = np.full(n_cons, capacity)
        
        # Bounds: 0 <= x_i <= 1
        bounds = [(0, 1)] * n_items
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            # Dual prices are in res.ineqlin.marginals (negative of shadow prices for <= constraints)
            # For maximization with <= constraints, shadow prices = -marginals
            dual = -res.ineqlin.marginals
            # Ensure non-negative (shadow prices should be >= 0 for <= constraints in max)
            dual = np.maximum(dual, 0)
            return res.fun * -1, dual
        else:
            return 0.0, None
    except Exception:
        return 0.0, None


def _reduced_cost_heuristic(prizes, weights, dual_prices, n_cons, n_items):
    """
    Compute reduced cost heuristic based on LP dual prices.
    reduced_cost_i = total_prize_i - sum_c dual_c * weight_c,i
    Items with positive reduced cost are "profitable" relative to the LP optimum.
    We use max(reduced_cost, small_positive) as the heuristic.
    """
    total_prize = prizes.sum(axis=0)
    # reduced_cost_i = total_prize_i - dot(dual, weights[:, i])
    # weights is (n_cons, n_items), dual is (n_cons,)
    # dot(dual, weights[:, i]) for all i = dual @ weights
    dual_weight_cost = dual_prices @ weights  # shape (n_items,)
    reduced_costs = total_prize - dual_weight_cost
    
    # The heuristic should be higher for more attractive items.
    # In ACO, we want to select items that are "efficient".
    # If reduced_cost is high, it means the item is "undervalued" by the LP duals, 
    # or simply has high net value.
    # However, standard reduced cost for max problem: if RC > 0, entering variable improves objective.
    # So higher RC is better.
    
    # But we need to be careful: if all RCs are negative, we still need a ranking.
    # Let's shift them so the minimum is 1e-9.
    min_rc = np.min(reduced_costs)
    if min_rc < 0:
        reduced_costs = reduced_costs - min_rc + 1e-9
    
    return reduced_costs


def _simulate_ant(heuristic, prizes, weights, capacity, n_items, n_cons):
    """
    Simulate a single ant using greedy selection based on heuristic.
    Returns total prize.
    """
    remaining = np.full(n_cons, capacity)
    selected = np.zeros(n_items, dtype=bool)
    total_prize = 0.0
    
    # Precompute total prize per item for quick lookup
    item_prizes = prizes.sum(axis=0)
    
    while True:
        # Find feasible items: not selected and weights fit in remaining capacity
        feasible_mask = ~selected
        for c in range(n_cons):
            feasible_mask &= (weights[c] <= remaining[c])
        
        if not np.any(feasible_mask):
            break
        
        # Among feasible items, pick the one with highest heuristic
        # Use a masked argmax
        masked_heuristic = np.where(feasible_mask, heuristic, -np.inf)
        best_idx = int(np.argmax(masked_heuristic))
        
        if masked_heuristic[best_idx] == -np.inf:
            break
            
        # Select this item
        selected[best_idx] = True
        total_prize += item_prizes[best_idx]
        remaining -= weights[:, best_idx]
        
    return total_prize


def _evaluate_heuristic(heuristic, prizes, weights, capacity, n_items, n_cons):
    """
    Evaluate a heuristic by simulating a few ants.
    """
    s = 0.0
    n_ants = 5  # Reduced for speed
    for _ in range(n_ants):
        s += _simulate_ant(heuristic, prizes, weights, capacity, n_items, n_cons)
    return s / n_ants


def _compute_heuristics(prizes, weights, deadline):
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n_cons, n_items = prizes.shape
    eps = 1e-9
    
    # 1. Compute LP dual prices
    lp_obj, dual_prices = _solve_lp(prizes, weights, CAPACITY, n_cons, n_items)
    
    # 2. Generate candidate heuristics
    total_prize = prizes.sum(axis=0)
    sum_w = weights.sum(axis=0)
    max_w = weights.max(axis=0)
    
    # Candidate 1: Reduced Cost based on LP duals
    if dual_prices is not None:
        rc_heur = _reduced_cost_heuristic(prizes, weights, dual_prices, n_cons, n_items)
    else:
        rc_heur = total_prize / (max_w + eps)
    
    # Candidate 2: Standard prize/weight ratio (max weight)
    ratio_max = total_prize / (max_w + eps)
    
    # Candidate 3: Prize/weight ratio (sum weight)
    ratio_sum = total_prize / (sum_w + eps)
    
    # Candidate 4: Geometric mean of ratios
    geo_ratio = np.sqrt(ratio_max * ratio_sum)
    
    # Candidate 5: Mixed LP and ratio
    # Normalize rc_heur and ratio_max to [0,1] for mixing
    rc_norm = (rc_heur - rc_heur.min()) / (rc_heur.max() - rc_heur.min() + eps)
    ratio_norm = (ratio_max - ratio_max.min()) / (ratio_max.max() - ratio_max.min() + eps)
    
    mixed_heur = 0.5 * rc_norm + 0.5 * ratio_norm
    
    pool = [
        rc_heur,
        ratio_max,
        ratio_sum,
        geo_ratio,
        mixed_heur,
        rc_heur * ratio_norm,  # Product: high RC and high ratio
    ]
    
    # 3. Evaluate candidates and pick the best
    best_h = pool[0].copy()
    best_score = -np.inf
    
    for h in pool:
        if time.time() > deadline:
            break
        hh = np.asarray(h, dtype=np.float64)
        hh = np.where(np.isfinite(hh), hh, 1e-9)
        hh = np.maximum(hh, 1e-9)
        sc = _evaluate_heuristic(hh, prizes, weights, CAPACITY, n_items, n_cons)
        if sc > best_score:
            best_score = sc
            best_h = hh.copy()
    
    # 4. Local refinement: scaling
    for mult in (0.5, 1.5, 2.0):
        if time.time() > deadline:
            break
        cand = best_h * mult
        cand = np.maximum(cand, 1e-9)
        sc = _evaluate_heuristic(cand, prizes, weights, CAPACITY, n_items, n_cons)
        if sc > best_score:
            best_score = sc
            best_h = cand
    
    best_h = np.where(np.isfinite(best_h), best_h, 1e-9)
    best_h = np.maximum(best_h, 1e-9)
    return best_h


def heuristics(prizes, weights):
    start_time = time.time()
    try:
        p = np.asarray(prizes, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if p.ndim != 2 or w.ndim != 2:
            raise ValueError("bad shape")
        if p.shape != w.shape:
            raise ValueError("shape mismatch")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(w)):
            raise ValueError("non-finite")
        n_cons, n_items = p.shape

        deadline = start_time + 290.0  # safety margin before the 330s kill
        result = _compute_heuristics(p, w, deadline)
        result = np.asarray(result, dtype=np.float64)
        if result.ndim != 1 or result.shape[0] != n_items:
            raise ValueError("bad result shape")
        if not np.all(np.isfinite(result)):
            raise ValueError("non-finite result")
        return result
    except Exception:
        try:
            p = np.asarray(prizes, dtype=np.float64)
            n_items = p.shape[1] if p.ndim == 2 else len(p)
        except Exception:
            n_items = N_ITEMS
        return np.ones(n_items, dtype=np.float64)
# EVOLVE-BLOCK-END
