"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores per item using LP dual prices (shadow prices).
    
    prizes: shape (5, 100) - prize values for 5 constraints, 100 items
    weights: shape (5, 100) - weight values for 5 constraints, 100 items
    
    Returns: array of shape (100,) with one score per item.
    """
    # Deadline guard: track elapsed time
    start_time = time.time()
    DEADLINE = 300.0  # seconds
    SAFETY_MARGIN = 10.0  # return at least 10s before deadline
    
    n_constraints = prizes.shape[0]  # 5
    n_items = prizes.shape[1]  # 100
    
    # Total prize per item (sum across constraints)
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Weights matrix: shape (5, 100)
    # Capacity per constraint: 25
    capacity = np.full(n_constraints, 25.0)
    
    # Solve LP relaxation:
    # Maximize sum_j c_j * x_j
    # Subject to: sum_j w_ij * x_j <= b_i for each constraint i
    #             0 <= x_j <= 1 for each item j
    #
    # In standard form for linprog (minimize):
    # Minimize -sum_j c_j * x_j
    # Subject to: sum_j w_ij * x_j <= b_i
    #             -1 <= x_j <= 0  (we'll handle bounds separately)
    
    c = -total_prize  # objective coefficients for minimization
    
    # A_ub: shape (n_constraints, n_items) = weights
    A_ub = weights.copy()
    b_ub = capacity
    
    # Bounds: 0 <= x_j <= 1
    bounds = [(0, 1)] * n_items
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.status == 0:  # Success
            # Dual prices (shadow prices) for the <= constraints
            # These are in result.ineqlin.marginals (for HiGHS)
            # or we can extract from result
            # In scipy linprog with method='highs', duals are in result.ineqlin.marginals
            if hasattr(result, 'ineqlin') and hasattr(result.ineqlin, 'marginals'):
                dual_prices = np.abs(result.ineqlin.marginals)
            else:
                # Fallback: try to get from other attributes
                # For older scipy or different method
                if hasattr(result, 'x') and hasattr(result, 'slacks'):
                    # Use a simple heuristic if duals not available
                    dual_prices = np.ones(n_constraints)
                else:
                    dual_prices = np.ones(n_constraints)
        else:
            # LP failed, use uniform dual prices
            dual_prices = np.ones(n_constraints)
    except Exception:
        # If LP fails for any reason, use uniform dual prices
        dual_prices = np.ones(n_constraints)
    
    # Check deadline
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        # Return a simple fallback
        return np.maximum(total_prize / (weights.sum(axis=0) + 1e-9), 1e-9)
    
    # Compute reduced costs:
    # reduced_cost_j = prize_j - sum_i dual_i * weight_ij
    # This represents the net benefit of item j after accounting for the
    # "cost" of consuming constraint resources at their shadow prices.
    
    # dual_prices: shape (5,)
    # weights: shape (5, 100)
    # weighted_weight_cost: shape (100,) = sum_i dual_i * weight_ij
    weighted_weight_cost = dual_prices @ weights  # shape (100,)
    
    # Reduced cost
    reduced_cost = total_prize - weighted_weight_cost
    
    # The heuristic should be positive and larger is more attractive.
    # Items with high positive reduced cost are most "efficient" relative to the LP optimum.
    # Items with negative reduced cost are less efficient.
    
    # We want scores to be positive. Use a transformation that:
    # 1. Maps reduced costs to positive values
    # 2. Preserves ordering (higher reduced cost -> higher score)
    # 3. Gives more weight to items with higher reduced costs
    
    # Option 1: Exponential transformation to ensure positivity and amplify differences
    # score_j = exp(alpha * reduced_cost_j) for some alpha
    
    # But we need to be careful about numerical stability.
    # Let's normalize reduced costs first.
    
    rc_min = reduced_cost.min()
    rc_max = reduced_cost.max()
    rc_range = rc_max - rc_min
    
    if rc_range > 1e-12:
        # Normalize to [0, 1]
        rc_norm = (reduced_cost - rc_min) / rc_range
    else:
        # All reduced costs are approximately equal
        rc_norm = np.full(n_items, 0.5)
    
    # Use an exponential transformation to amplify differences
    # This makes items with higher reduced costs much more attractive
    alpha = 2.0  # Tuning parameter
    scores = np.exp(alpha * rc_norm)
    
    # Ensure scores are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    # Check deadline again
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
