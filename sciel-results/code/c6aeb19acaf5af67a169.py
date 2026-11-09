# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1..X4 are i.i.d. with law mu.

    Approach (new family: numerical optimization of a 3-point measure)
    -----------------------------------------------------------------
    We restrict to a 3-point measure on {0, 1, 2} with probabilities p0, p1, p2.
    We numerically optimize the probability P[X1+X2+X3 < 2*X4] over the simplex.
    This allows the distribution to "spread" mass to 2, which can increase the
    probability of satisfying the inequality when X4 is large, while maintaining
    sufficient mass at 0 and 1 to satisfy the inequality when X4 is small.
    """
    support = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    
    # Define the objective function to minimize (negative probability)
    def objective(params):
        p1, p2 = params
        p0 = 1.0 - p1 - p2
        if p0 < -1e-12 or p1 < -1e-12 or p2 < -1e-12:
            return 1e10  # Penalty for invalid probabilities
        
        # Calculate P[X1+X2+X3 < 2*X4]
        # Terms for X4=0: 0
        # Terms for X4=1: p1 * (P[Sum=0] + P[Sum=1])
        #   P[Sum=0] = p0^3
        #   P[Sum=1] = 3 * p1 * p0^2
        # Terms for X4=2: p2 * (P[Sum=0] + P[Sum=1] + P[Sum=2] + P[Sum=3])
        #   P[Sum=0] = p0^3
        #   P[Sum=1] = 3 * p1 * p0^2
        #   P[Sum=2] = 3 * p1^2 * p0 + 3 * p2 * p0^2
        #   P[Sum=3] = p1^3 + 6 * p1 * p2 * p0
        
        sum_0 = p0**3
        sum_1 = 3 * p1 * p0**2
        sum_2 = 3 * p1**2 * p0 + 3 * p2 * p0**2
        sum_3 = p1**3 + 6 * p1 * p2 * p0
        
        prob = p1 * (sum_0 + sum_1) + p2 * (sum_0 + sum_1 + sum_2 + sum_3)
        return -prob

    # Constraints: p1 >= 0, p2 >= 0, p1 + p2 <= 1
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0]},       # p1 >= 0
        {'type': 'ineq', 'fun': lambda x: x[1]},       # p2 >= 0
        {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]} # p0 >= 0
    ]
    
    # Bounds for p1, p2
    bounds = [(0, 1), (0, 1)]
    
    # Initial guess: uniform
    x0 = np.array([1/3, 1/3])
    
    # Optimize
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    p1 = res.x[0]
    p2 = res.x[1]
    p0 = 1.0 - p1 - p2
    
    # Clip small negatives due to numerical precision
    p0 = max(0.0, p0)
    p1 = max(0.0, p1)
    p2 = max(0.0, p2)
    
    # Normalize just in case
    total = p0 + p1 + p2
    p0 /= total
    p1 /= total
    p2 /= total
    
    weights = np.array([p0, p1, p2], dtype=np.float64)
    
    return support, weights


# EVOLVE-BLOCK-END
