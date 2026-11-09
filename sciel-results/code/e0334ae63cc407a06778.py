# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1..X4 are i.i.d. with law mu.

    Approach (New family: Numerical optimization of a 3-point measure with variable support)
    -----------------------------------------------------------------
    We restrict to a 3-point measure on {0, 1, a} with probabilities p0, p1, p2.
    We treat the third support point 'a' as a free optimization variable (a > 0).
    We run the optimization from two initial guesses (a > 1 and a < 1) to cover
    both orderings of the support points and select the best result.
    """
    
    def calc_prob(p1, p2, a):
        p0 = 1.0 - p1 - p2
        if p0 < -1e-12 or p1 < -1e-12 or p2 < -1e-12:
            return -1e10
            
        # Support
        s = np.array([0.0, 1.0, a], dtype=np.float64)
        # Weights
        w = np.array([p0, p1, p2], dtype=np.float64)
        
        # Calculate P[X1+X2+X3 < 2*X4]
        # P = sum_l w_l * P(Sum3 < 2*s_l)
        total_prob = 0.0
        
        for l_idx in range(3):
            target = 2.0 * s[l_idx]
            # Sum over all combinations of i, j, m
            # n=3, so 3^3=27 combinations. Very fast.
            sum_prob = 0.0
            for i in range(3):
                for j in range(3):
                    for m in range(3):
                        s_sum = s[i] + s[j] + s[m]
                        if s_sum < target:
                            sum_prob += w[i] * w[j] * w[m]
            total_prob += w[l_idx] * sum_prob
            
        return total_prob

    def objective(params):
        p1, p2, a = params
        if a < 1e-6:
            return 1e10 # Penalty for a ~ 0 (degenerate)
        p = calc_prob(p1, p2, a)
        return -p

    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0]},       # p1 >= 0
        {'type': 'ineq', 'fun': lambda x: x[1]},       # p2 >= 0
        {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]} # p0 >= 0
    ]
    
    # Bounds
    # p1, p2 in [0, 1], a in [1e-6, 10]
    bounds = [(0, 1), (0, 1), (1e-6, 10)]
    
    best_res = None
    best_score = -np.inf
    
    # Run 1: a > 1
    x0_1 = np.array([0.2, 0.3, 2.0])
    res1 = minimize(objective, x0_1, method='SLSQP', bounds=bounds, constraints=constraints)
    if res1.success and res1.fun < best_score:
        best_score = res1.fun
        best_res = res1
        
    # Run 2: a < 1
    x0_2 = np.array([0.2, 0.3, 0.5])
    res2 = minimize(objective, x0_2, method='SLSQP', bounds=bounds, constraints=constraints)
    if res2.success and res2.fun < best_score:
        best_score = res2.fun
        best_res = res2
        
    if best_res is None:
        # Fallback to 2-point optimal
        p_star = (1 + np.sqrt(33)) / 16
        support = np.array([0.0, 1.0], dtype=np.float64)
        weights = np.array([1-p_star, p_star], dtype=np.float64)
        return support, weights

    p1 = best_res.x[0]
    p2 = best_res.x[1]
    a = best_res.x[2]
    p0 = 1.0 - p1 - p2
    
    # Clip and normalize
    p0 = max(0.0, p0)
    p1 = max(0.0, p1)
    p2 = max(0.0, p2)
    
    total = p0 + p1 + p2
    p0 /= total
    p1 /= total
    p2 /= total
    
    support = np.array([0.0, 1.0, a], dtype=np.float64)
    weights = np.array([p0, p1, p2], dtype=np.float64)
    
    return support, weights


# EVOLVE-BLOCK-END
