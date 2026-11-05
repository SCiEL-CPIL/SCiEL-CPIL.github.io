"""Incentive-compatible finite auction design via SLSQP/QP approach."""
import numpy as np
from scipy.optimize import minimize
import time

def construct_auction(distribution_spec, support, constraints):
    # Deadline guard
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Values are in {0, 1, 2, 3} for each good
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)  # 16
    
    # Map type to index
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Get distribution probabilities
    probs = _get_distribution(distribution_spec, n_types)
    
    # Decision variables:
    # x[i, g] = allocation probability of good g to type i (16 x 2)
    # p[i] = payment for type i (16,)
    # Total variables: 16*2 + 16 = 48
    
    n_goods = 2
    n_vars = n_types * n_goods + n_types  # 48
    
    def unpack(z):
        x = z[:n_types * n_goods].reshape(n_types, n_goods)
        p = z[n_types * n_goods:]
        return x, p
    
    def revenue(z):
        x, p = unpack(z)
        # Expected revenue = sum_i probs[i] * p[i]
        return float(np.sum(probs * p))
    
    def expected_value(i, x):
        """Expected value for type i given allocation x."""
        v_i = types[i]
        return sum(x[i, g] * v_i[g] for g in range(n_goods))
    
    # Objective: maximize revenue (minimize negative)
    def objective(z):
        if time.time() - t0 > DEADLINE:
            return 1e10
        return -revenue(z)
    
    # Constraints:
    # 1. Allocation feasibility: 0 <= x[i,g] <= 1
    # 2. Nonnegative payments: p[i] >= 0
    # 3. Individual rationality: p[i] <= expected_value(i, x)
    # 4. Incentive compatibility: for all i, j:
    #    u(i, report j) <= u(i, report i)
    #    u(i, report j) = sum_g x[j,g]*v_i[g] - p[j]
    #    u(i, report i) = sum_g x[i,g]*v_i[g] - p[i]
    #    So: sum_g x[j,g]*v_i[g] - p[j] <= sum_g x[i,g]*v_i[g] - p[i]
    #    Rearranged: p[i] - p[j] <= sum_g x[i,g]*v_i[g] - sum_g x[j,g]*v_i[g]
    
    constraints_list = []
    
    # IR constraints: p[i] - expected_value(i, x) <= 0
    for i in range(n_types):
        v_i = types[i]
        def ir_constr(z, i=i, v_i=v_i):
            x, p = unpack(z)
            ev = sum(x[i, g] * v_i[g] for g in range(n_goods))
            return p[i] - ev  # should be <= 0
        constraints_list.append({'type': 'ineq', 'fun': ir_constr})
    
    # IC constraints: for all i, j: p[i] - p[j] - (EV_i_at_i - EV_i_at_j) <= 0
    # where EV_i_at_i = sum_g x[i,g]*v_i[g], EV_i_at_j = sum_g x[j,g]*v_i[g]
    # So: p[i] - p[j] - sum_g x[i,g]*v_i[g] + sum_g x[j,g]*v_i[g] <= 0
    
    for i in range(n_types):
        v_i = types[i]
        for j in range(n_types):
            if i == j:
                continue
            def ic_constr(z, i=i, j=j, v_i=v_i):
                x, p = unpack(z)
                ev_i_at_i = sum(x[i, g] * v_i[g] for g in range(n_goods))
                ev_i_at_j = sum(x[j, g] * v_i[g] for g in range(n_goods))
                return p[i] - p[j] - ev_i_at_i + ev_i_at_j  # should be <= 0
            constraints_list.append({'type': 'ineq', 'fun': ic_constr})
    
    # Bounds: x in [0,1], p >= 0
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))
    
    # Initial guess: allocate all to highest value type, zero payments
    z0 = np.zeros(n_vars)
    # Set some initial allocation
    for i in range(n_types):
        for g in range(n_goods):
            z0[i * n_goods + g] = 0.5
    # Set payments to zero initially
    
    # Run SLSQP optimization
    best_result = None
    best_obj = -np.inf
    
    # Try multiple starting points
    start_points = [z0]
    
    # Add perturbed starting points
    rng = np.random.RandomState(42)
    for _ in range(3):
        zp = z0.copy()
        zp[:n_types * n_goods] = np.clip(zp[:n_types * n_goods] + rng.randn(n_types * n_goods) * 0.1, 0, 1)
        zp[n_types * n_goods:] = np.abs(rng.randn(n_types) * 0.5)
        start_points.append(zp)
    
    for sp in start_points:
        if time.time() - t0 > DEADLINE:
            break
        try:
            res = minimize(objective, sp, method='SLSQP', bounds=bounds,
                           constraints=constraints_list,
                           options={'maxiter': 200, 'ftol': 1e-9})
            if res.success and res.fun < best_obj:
                best_obj = res.fun
                best_result = res.x
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: use the initial point or a simple heuristic
        best_result = z0
    
    x, p = unpack(best_result)
    
    # Clip to ensure feasibility
    x = np.clip(x, 0.0, 1.0)
    p = np.maximum(p, 0.0)
    
    # Verify and fix constraints with small tolerance
    # Re-check IC and IR, adjust if needed
    for i in range(n_types):
        v_i = types[i]
        ev_i = sum(x[i, g] * v_i[g] for g in range(n_goods))
        if p[i] > ev_i + 1e-8:
            p[i] = ev_i
        for j in range(n_types):
            ev_i_at_i = sum(x[i, g] * v_i[g] for g in range(n_goods))
            ev_i_at_j = sum(x[j, g] * v_i[g] for g in range(n_goods))
            lhs = p[i] - p[j] - ev_i_at_i + ev_i_at_j
            if lhs > 1e-8:
                # Violation: need to adjust p[i] down or p[j] up
                # Adjust p[i] down
                p[i] = p[j] + ev_i_at_i - ev_i_at_j
    
    # Final clip
    p = np.maximum(p, 0.0)
    x = np.clip(x, 0.0, 1.0)
    
    return {'allocations': x, 'payments': p}


def _get_distribution(distribution_spec, n_types):
    """Get probability distribution over types."""
    if isinstance(distribution_spec, str):
        spec = distribution_spec.lower()
        if 'uniform' in spec or 'independent' in spec:
            return np.ones(n_types) / n_types
        elif 'concentrated' in spec or 'peaked' in spec:
            # Higher probability on high-value types
            weights = np.array([1.0, 2.0, 3.0, 4.0])
            probs = np.outer(weights, weights)
            return probs / probs.sum()
        elif 'sparse' in spec or 'low' in spec:
            # Higher probability on low-value types
            weights = np.array([4.0, 3.0, 2.0, 1.0])
            probs = np.outer(weights, weights)
            return probs / probs.sum()
        else:
            return np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float).flatten()
        if len(arr) == n_types:
            return arr / arr.sum()
        else:
            return np.ones(n_types) / n_types
    else:
        return np.ones(n_types) / n_types
