# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _parse_distribution(distribution_spec, support, constraints):
    """
    Parse the distribution specification into a probability vector over the 16 types.
    Types are (v1, v2) with v1, v2 in {0,1,2,3}.
    """
    n_types = 16
    # Try to extract probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        # Could be {'type': 'uniform'} or {'type': 'product', 'marginals': [...]}
        dist_type = distribution_spec.get('type', 'uniform')
        if dist_type == 'uniform':
            return np.ones(n_types) / n_types
        elif dist_type == 'product':
            # Product of marginals
            marginals = distribution_spec.get('marginals', [0.25, 0.25, 0.25, 0.25])
            if len(marginals) == 2:
                p1 = np.array(marginals[0], dtype=float)
                p2 = np.array(marginals[1], dtype=float)
                # Ensure they sum to 1
                p1 = p1 / p1.sum()
                p2 = p2 / p2.sum()
                joint = np.outer(p1, p2)
                return joint.flatten()
            else:
                return np.ones(n_types) / n_types
        elif dist_type == 'explicit':
            probs = distribution_spec.get('probs', None)
            if probs is not None:
                probs = np.array(probs, dtype=float)
                probs = probs / probs.sum()
                return probs
        # Fallback to uniform
        return np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if probs.shape == (16,):
            probs = probs / probs.sum()
            return probs
        elif probs.shape == (4, 4):
            probs = probs / probs.sum()
            return probs.flatten()
    # Default to uniform
    return np.ones(n_types) / n_types


def _type_index(v1, v2):
    """Map (v1, v2) to index 0..15 where v1, v2 in {0,1,2,3}."""
    return v1 * 4 + v2


def _construct_baseline(p):
    """
    Construct a baseline mechanism: allocate good 1 to highest value for good 1,
    allocate good 2 to highest value for good 2.
    This is a simple heuristic that may not be optimal but is feasible.
    """
    n_types = 16
    # allocations[i, j] = probability of allocating good j to type i
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    
    # For each good, allocate to the type with the highest value
    for good in range(2):
        values = np.array([t[good] for t in product(range(4), repeat=2)])
        max_val = values.max()
        # Allocate with probability 1 to types with max value for this good
        for i, (v1, v2) in enumerate(product(range(4), repeat=2)):
            if v1 == max_val and good == 0:
                allocations[i, good] = 1.0
            elif v2 == max_val and good == 1:
                allocations[i, good] = 1.0
    
    # Compute payments using the single-bidder incentive compatibility condition
    # For a single bidder, the payment for type i is:
    # p_i = v_i * x_i - sum_{j < i} (v_j - v_{j-1}) * x_{j-1} in the single-good case
    # For multi-good, we use the standard formula:
    # p_i = sum_g v_{i,g} * x_{i,g} - sum_{j} [something]
    # Actually, for a single additive bidder with multiple goods, the IC condition is:
    # u_i = v_i . x_i - p_i
    # u_i >= u_j for all i, j (truthfulness)
    # p_i = v_i . x_i - u_i
    # where u_i is determined by the IC constraints.
    
    # Let's compute utilities using the standard approach:
    # Sort types by value (but for multi-dimensional, this is complex)
    # Instead, use the fact that for a single bidder, the payment can be computed as:
    # p_i = sum_g v_{i,g} * x_{i,g} - sum_{j != i} (something)
    
    # Simpler approach: use the LP to find optimal payments given allocations
    # But for baseline, let's use a simple heuristic:
    # p_i = sum_g v_{i,g} * x_{i,g} (pay the value, which gives zero utility)
    # This is not necessarily IC, but let's check
    
    for i in range(n_types):
        v1, v2 = i // 4, i % 4
        payments[i] = v1 * allocations[i, 0] + v2 * allocations[i, 1]
    
    return allocations, payments


def _verify_mechanism(allocations, payments, p, support, constraints, epsilon=1e-8):
    """
    Verify that the mechanism is:
    1. Feasible: allocations in [0,1], sum of allocations <= 1 (or whatever constraint)
    2. Nonnegative payments
    3. Individual rationality: u_i >= 0
    4. Incentive compatible: u_i >= u_j for all i, j (truthfulness)
    """
    n_types = 16
    types = list(product(range(4), repeat=2))
    
    # Check feasibility
    if np.any(allocations < -epsilon) or np.any(allocations > 1 + epsilon):
        return False, "Infeasible allocations"
    
    # Check nonnegative payments
    if np.any(payments < -epsilon):
        return False, "Negative payments"
    
    # Compute utilities
    utilities = np.zeros(n_types)
    for i, (v1, v2) in enumerate(types):
        utilities[i] = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
    
    # Check individual rationality
    if np.any(utilities < -epsilon):
        return False, "IR violated"
    
    # Check incentive compatibility
    # For truthfulness, reporting truthfully must be optimal
    # u_i >= u_j for all i, j where i is the true type and j is the reported type
    # Actually, IC means: for all i, j: u_i(i) >= u_i(j)
    # where u_i(j) is the utility of type i when reporting j
    # u_i(j) = v_i . x_j - p_j
    
    for i, (v1, v2) in enumerate(types):
        for j, (w1, w2) in enumerate(types):
            u_i_j = v1 * allocations[j, 0] + v2 * allocations[j, 1] - payments[j]
            if u_i_j > utilities[i] + epsilon:
                return False, f"IC violated: type {i} prefers reporting {j}"
    
    return True, "Valid"


def _compute_revenue(allocations, payments, p):
    """Compute expected revenue."""
    n_types = 16
    types = list(product(range(4), repeat=2))
    revenue = 0.0
    for i, (v1, v2) in enumerate(types):
        revenue += p[i] * payments[i]
    return revenue


def _solve_lp_for_optimal_revenue(p, support, constraints):
    """
    Solve the LP to find the optimal revenue mechanism.
    This is the benchmark for normalization.
    """
    from scipy.optimize import linprog
    
    n_types = 16
    types = list(product(range(4), repeat=2))
    
    # Variables: x_{i,g} for i in 0..15, g in 0,1 (32 variables)
    # and p_i for i in 0..15 (16 variables)
    # Total: 48 variables
    
    # Objective: maximize sum_i p_i * (v_{i,1} * x_{i,1} + v_{i,2} * x_{i,2} - p_i)
    # = maximize sum_i p_i * v_{i,1} * x_{i,1} + sum_i p_i * v_{i,2} * x_{i,2} - sum_i p_i^2
    # Wait, revenue is sum_i p_i * payment_i
    # payment_i = v_i . x_i - u_i
    # We want to maximize sum_i p_i * (v_i . x_i - u_i)
    # Subject to IC and IR constraints
    
    # This is a complex LP. Let's use a simpler approach:
    # The optimal revenue for a single bidder with multi-dimensional types
    # can be found by solving an LP with variables x_{i,g} and u_i
    
    # Variables: x_{i,0}, x_{i,1} for each i, and u_i for each i
    # Total: 16*2 + 16 = 48 variables
    
    # c = -[p_i * v_{i,0} for i] - [p_i * v_{i,1} for i] + [p_i for i]
    # because we maximize sum_i p_i * (v_i . x_i) - sum_i p_i * u_i
    
    c = np.zeros(48)
    for i in range(n_types):
        v1, v2 = types[i]
        c[i] = -p[i] * v1      # coefficient for x_{i,0}
        c[n_types + i] = -p[i] * v2  # coefficient for x_{i,1}
        c[2 * n_types + i] = -p[i]   # coefficient for u_i
    
    # Constraints:
    # 1. 0 <= x_{i,g} <= 1
    # 2. u_i >= 0 (IR)
    # 3. IC: u_i >= v_i . x_j - v_j . x_j + u_j for all i, j
    #    i.e., u_i - u_j >= v_i . x_j - v_j . x_j
    
    A_ub = []
    b_ub = []
    
    # IC constraints: for all i, j: u_i - u_j >= v_i . x_j - v_j . x_j
    # Rearranged: -u_i + u_j + v_i . x_j - v_j . x_j <= 0
    # Wait, let's be careful.
    # IC: u_i >= v_i . x_j - p_j = v_i . x_j - (v_j . x_j - u_j)
    # u_i >= v_i . x_j - v_j . x_j + u_j
    # u_i - u_j >= v_i . x_j - v_j . x_j
    # -u_i + u_j + v_i . x_j - v_j . x_j <= 0
    
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            v1_i, v2_i = types[i]
            v1_j, v2_j = types[j]
            row = np.zeros(48)
            row[2 * n_types + i] = -1  # -u_i
            row[2 * n_types + j] = 1   # +u_j
            row[j] = v1_i - v1_j       # (v_i - v_j) . x_j for good 0
            row[n_types + j] = v2_i - v2_j  # for good 1
            A_ub.append(row)
            b_ub.append(0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: 0 <= x_{i,g} <= 1, u_i >= 0
    bounds = []
    for i in range(n_types):
        bounds.append((0, 1))  # x_{i,0}
        bounds.append((0, 1))  # x_{i,1}
    for i in range(n_types):
        bounds.append((0, None))  # u_i >= 0
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.status != 0:
        # Fallback: use baseline revenue
        allocations, payments = _construct_baseline(p)
        return _compute_revenue(allocations, payments, p)
    
    # Extract optimal revenue
    x_opt = result.x[:2 * n_types].reshape(n_types, 2)
    u_opt = result.x[2 * n_types:]
    
    # Compute payments
    payments = np.zeros(n_types)
    for i in range(n_types):
        v1, v2 = types[i]
        payments[i] = v1 * x_opt[i, 0] + v2 * x_opt[i, 1] - u_opt[i]
    
    revenue = _compute_revenue(x_opt, payments, p)
    return revenue


def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 75.0  # Safety margin
    
    # Parse distribution
    p = _parse_distribution(distribution_spec, support, constraints)
    
    # Try to solve the LP for optimal mechanism
    try:
        optimal_revenue = _solve_lp_for_optimal_revenue(p, support, constraints)
    except Exception:
        optimal_revenue = None
    
    # Construct baseline mechanism
    allocations, payments = _construct_baseline(p)
    
    # Verify and adjust if needed
    valid, msg = _verify_mechanism(allocations, payments, p, support, constraints)
    if not valid:
        # Fallback: zero allocation, zero payment
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
    
    # Check time
    if time.time() > deadline:
        return {'allocations': allocations, 'payments': payments}
    
    return {'allocations': allocations, 'payments': payments}

# EVOLVE-BLOCK-END
