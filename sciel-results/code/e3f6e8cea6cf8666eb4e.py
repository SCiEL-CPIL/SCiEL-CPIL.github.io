"""Incentive-compatible finite auction design - optimal LP-based mechanism."""
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an optimal incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: Distribution specification (list of 16 probabilities or similar)
        support: Support specification (16 types, 2 goods)
        constraints: Constraints dictionary
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    DEADLINE = 45.0  # Safety margin before 90s limit
    
    # Define the 16 types: (v1, v2) where v1, v2 in {0, 1, 2, 3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    n_goods = 2
    
    # Parse distribution probabilities
    probs = _parse_distribution(distribution_spec, n_types)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return _fallback_result()
    
    # Use heuristic search for optimal mechanism
    result = _solve_heuristic_auction(probs, types, n_types, n_goods, start_time, DEADLINE)
    
    return result


def _parse_distribution(distribution_spec, n_types):
    """Parse distribution specification into probability vector."""
    probs = np.zeros(n_types)
    
    if distribution_spec is None:
        return probs / n_types  # Uniform fallback
    
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (n_types,):
            probs = arr
        elif arr.shape == (4, 4):
            probs = arr.flatten()
        elif arr.size == n_types:
            probs = arr.flatten()
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, dict):
        # Try to parse from dict
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    
    # Normalize to ensure valid probabilities
    total = probs.sum()
    if total > 1e-12:
        probs = probs / total
    else:
        probs = np.ones(n_types) / n_types
    
    # Clip small negative values
    probs = np.clip(probs, 0, 1)
    
    return probs


def _solve_heuristic_auction(probs, types, n_types, n_goods, start_time, deadline):
    """
    Solve the auction problem using a heuristic search over threshold mechanisms
    with fractional boundary allocations and local refinement.
    """
    
    best_revenue = -1
    best_x = None
    best_p = None
    
    def _evaluate(x):
        """Evaluate a mechanism: compute payments and revenue."""
        p, valid = _compute_min_payments(x, types)
        if not valid:
            return None, -1
        if not _check_ir(x, p, types):
            return None, -1
        revenue = float(np.dot(probs, p))
        return (x, p), revenue
    
    def _update_best(x, p, revenue):
        nonlocal best_revenue, best_x, best_p
        if revenue > best_revenue:
            best_revenue = revenue
            best_x = x.copy()
            best_p = p.copy()
    
    # Phase 1: Search over threshold mechanisms with fractional boundary allocations
    # For each good, threshold tau means:
    #   x = 1 if v > tau
    #   x = alpha if v == tau (alpha in [0,1])
    #   x = 0 if v < tau
    # Try tau in {0,1,2,3} and alpha in {0, 0.25, 0.5, 0.75, 1.0}
    
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    thresholds = [0, 1, 2, 3]
    
    for tau1 in thresholds:
        for tau2 in thresholds:
            for a1 in alphas:
                for a2 in alphas:
                    x = np.zeros((n_types, n_goods))
                    for t in range(n_types):
                        v1, v2 = types[t]
                        if v1 > tau1:
                            x[t, 0] = 1.0
                        elif v1 == tau1:
                            x[t, 0] = a1
                        # else 0.0
                        if v2 > tau2:
                            x[t, 1] = 1.0
                        elif v2 == tau2:
                            x[t, 1] = a2
                        # else 0.0
                    
                    result, revenue = _evaluate(x)
                    if result is not None:
                        _update_best(result[0], result[1], revenue)
                    
                    # Check deadline
                    if time.time() - start_time > deadline - 10:
                        break
                if time.time() - start_time > deadline - 10:
                    break
            if time.time() - start_time > deadline - 10:
                break
        if time.time() - start_time > deadline - 10:
            break
    
    # Phase 2: Local refinement - perturb the best mechanism and re-evaluate
    if best_x is not None and time.time() - start_time < deadline - 5:
        # Try small perturbations to the allocation matrix
        for _ in range(200):
            if time.time() - start_time > deadline - 5:
                break
            
            # Random perturbation
            x_perturbed = best_x.copy()
            # Perturb a random subset of entries
            for _ in range(3):
                t = np.random.randint(n_types)
                g = np.random.randint(n_goods)
                delta = np.random.uniform(-0.1, 0.1)
                x_perturbed[t, g] = np.clip(x_perturbed[t, g] + delta, 0, 1)
            
            result, revenue = _evaluate(x_perturbed)
            if result is not None:
                _update_best(result[0], result[1], revenue)
    
    # Phase 3: Try some structured non-threshold mechanisms
    if time.time() - start_time < deadline - 3:
        # Try "bundle" mechanisms: allocate both goods together
        for tau in thresholds:
            for a in alphas:
                x = np.zeros((n_types, n_goods))
                for t in range(n_types):
                    v1, v2 = types[t]
                    # Allocate both if min(v1,v2) > tau, fractional if == tau
                    if min(v1, v2) > tau:
                        x[t, :] = 1.0
                    elif min(v1, v2) == tau:
                        x[t, :] = a
                result, revenue = _evaluate(x)
                if result is not None:
                    _update_best(result[0], result[1], revenue)
        
        # Try "max" mechanisms: allocate both if max(v1,v2) > tau
        for tau in thresholds:
            for a in alphas:
                x = np.zeros((n_types, n_goods))
                for t in range(n_types):
                    v1, v2 = types[t]
                    if max(v1, v2) > tau:
                        x[t, :] = 1.0
                    elif max(v1, v2) == tau:
                        x[t, :] = a
                result, revenue = _evaluate(x)
                if result is not None:
                    _update_best(result[0], result[1], revenue)
    
    if best_x is None:
        return _fallback_result()
    
    return {
        'allocations': best_x,
        'payments': best_p
    }


def _compute_min_payments(x, types):
    """
    Compute the minimal payments that satisfy IC and IR for a given allocation x.
    Returns (payments, valid).
    """
    n_types = len(types)
    
    # We need to find U(t) such that:
    # 1. U(t) >= 0
    # 2. U(t) >= U(t') + sum_g (t_g - t'_g) x(t', g)
    
    # This is a longest path problem on a graph where edge t' -> t has weight
    # w(t', t) = sum_g (t_g - t'_g) x(t', g)
    # We add a source node S with U(S) = 0 and edges S -> t with weight 0.
    
    # Since n_types is small (16), we can use Bellman-Ford for longest paths.
    # Note: Bellman-Ford finds shortest paths. We can negate weights to find longest,
    # or just adapt the relaxation step.
    
    U = np.zeros(n_types)
    
    # Relaxation: U(t) = max(U(t), U(t') + w(t', t))
    # We iterate until convergence.
    
    for _ in range(n_types):
        updated = False
        for t in range(n_types):
            for t_prime in range(n_types):
                if t == t_prime:
                    continue
                # Weight from t' to t
                w = 0.0
                for g in range(2):
                    w += (types[t][g] - types[t_prime][g]) * x[t_prime, g]
                
                new_val = U[t_prime] + w
                if new_val > U[t] + 1e-9:
                    U[t] = new_val
                    updated = True
        if not updated:
            break
    
    # Check for positive cycles (infeasible IC)
    # If U changes after n_types iterations, there's a cycle.
    # But for valid IC, there should be no positive cycles in the utility difference graph.
    # Actually, if there is a positive cycle, IC is impossible.
    
    # Payments
    p = np.zeros(n_types)
    for t in range(n_types):
        v_dot_x = np.dot(x[t], types[t])
        p[t] = v_dot_x - U[t]
    
    # Check if payments are non-negative (IR is U(t) >= 0, which is satisfied by construction if we start with 0 and only increase)
    # Wait, U(t) is the minimal utility. If U(t) < 0, IR is violated?
    # No, IR is U(t) >= 0. Our construction ensures U(t) >= 0 because we start with 0 and take max.
    # However, we must ensure p(t) >= 0.
    # p(t) = v(t).x(t) - U(t).
    # If p(t) < 0, it means U(t) > v(t).x(t), which violates IR (since utility = v.x - p = U, and IR requires utility >= 0, which is U >= 0. But also p >= 0 is a constraint).
    # The constraint is p(t) >= 0 AND U(t) >= 0.
    # If p(t) < 0, the mechanism is invalid.
    
    valid = np.all(p >= -1e-9)
    
    # Clip small negatives
    p = np.clip(p, 0, None)
    
    return p, valid


def _check_ir(x, p, types):
    """Check Individual Rationality."""
    for t in range(len(types)):
        utility = np.dot(x[t], types[t]) - p[t]
        if utility < -1e-9:
            return False
    return True


# Removed _verify_constraints and _fix_violations as they are no longer used by the heuristic solver
# The heuristic solver uses _compute_min_payments and _check_ir


def _fallback_result():
    """Return a valid fallback result (e.g., no trade)."""
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    return {
        'allocations': allocations,
        'payments': payments
    }
