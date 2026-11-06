"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    deadline = t0 + 82.0  # Safety margin before 90s limit

    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16

    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'weights' in distribution_spec:
            probs = np.array(distribution_spec['weights'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            for key in distribution_spec:
                if isinstance(distribution_spec[key], (list, tuple, np.ndarray)):
                    probs = np.array(distribution_spec[key], dtype=float)
                    break
            else:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types

    if probs.shape[0] != n_types:
        if probs.size == n_types:
            probs = probs.reshape(n_types)
        else:
            probs = np.ones(n_types) / n_types

    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types

    # Value matrix: V[t, g] = value of good g for type t
    V = np.array([[t[0], t[1]] for t in types], dtype=float)  # (16, 2)

    # Variable indexing:
    # x[g, t_rep] -> index = g * n_types + t_rep
    # p[t_rep]    -> index = 2 * n_types + t_rep

    def x_idx(g, t_rep):
        return g * n_types + t_rep

    def p_idx(t_rep):
        return 2 * n_types + t_rep

    n_vars = 3 * n_types  # 48

    # Objective: maximize sum_{t} probs[t] * p[t]
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[p_idx(t)] = -probs[t]

    A_ub_rows = []
    b_ub_vals = []

    # 1. IC Constraints:
    # u(t, t) >= u(t, t_rep) for all t, t_rep
    # sum_g V[t,g]*x[g,t] - p(t) >= sum_g V[t,g]*x[g,t_rep] - p(t_rep)
    # -sum_g V[t,g]*x[g,t] + sum_g V[t,g]*x[g,t_rep] + p(t) - p(t_rep) <= 0
    for t in range(n_types):
        for t_rep in range(n_types):
            if t == t_rep:
                continue
            row = np.zeros(n_vars)
            for g in range(2):
                row[x_idx(g, t)] -= V[t, g]
                row[x_idx(g, t_rep)] += V[t, g]
            row[p_idx(t)] += 1.0
            row[p_idx(t_rep)] -= 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)

    # 2. IR Constraints:
    # u(t, t) >= 0
    # -sum_g V[t,g]*x[g,t] + p(t) <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        for g in range(2):
            row[x_idx(g, t)] -= V[t, g]
        row[p_idx(t)] += 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)

    # Bounds: 0 <= x <= 1, p >= 0
    bounds = []
    for g in range(2):
        for t_rep in range(n_types):
            bounds.append((0.0, 1.0))
    for t_rep in range(n_types):
        bounds.append((0.0, None))

    # Check deadline
    if time.time() > deadline - 10.0:
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

    # Solve LP with tighter tolerance for better precision
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True, 'time_limit': deadline - 15.0,
                                  'dual_feasibility_tolerance': 1e-12,
                                  'primal_feasibility_tolerance': 1e-12})

        if result.success:
            x_opt = result.x
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)

            for t in range(n_types):
                for g in range(2):
                    allocations[t, g] = x_opt[x_idx(g, t)]
                payments[t] = x_opt[p_idx(t)]

            # Clip small numerical errors
            allocations = np.clip(allocations, 0.0, 1.0)
            payments = np.maximum(payments, 0.0)

            # Robust post-processing: verify and repair constraints
            eps = 1e-8
            feasible = True

            # Check IR constraints
            for t in range(n_types):
                u_tt = sum(V[t, g] * allocations[t, g] for g in range(2)) - payments[t]
                if u_tt < -eps:
                    feasible = False
                    break

            # Check IC constraints
            if feasible:
                for t in range(n_types):
                    u_tt = sum(V[t, g] * allocations[t, g] for g in range(2)) - payments[t]
                    for t_rep in range(n_types):
                        if t == t_rep:
                            continue
                        u_t_trep = sum(V[t, g] * allocations[t_rep, g] for g in range(2)) - payments[t_rep]
                        if u_t_trep - u_tt > eps:
                            feasible = False
                            break
                    if not feasible:
                        break

            if not feasible:
                # Try to repair with a secondary LP: minimize payment changes subject to constraints
                if time.time() < deadline - 5.0:
                    repaired = _repair_mechanism(allocations, payments, V, n_types, eps)
                    if repaired is not None:
                        allocations, payments = repaired
                        # Final verification
                        feasible = _verify_mechanism(allocations, payments, V, n_types, eps)

            if not feasible:
                # Fallback: conservative mechanism (allocate nothing, charge nothing)
                allocations = np.zeros((n_types, 2))
                payments = np.zeros(n_types)

            return {'allocations': allocations, 'payments': payments}
        else:
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}

    except Exception:
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}


def _verify_mechanism(allocations, payments, V, n_types, eps=1e-8):
    """Verify that the mechanism satisfies IC and IR constraints."""
    for t in range(n_types):
        # IR: u(t,t) >= 0
        u_tt = sum(V[t, g] * allocations[t, g] for g in range(2)) - payments[t]
        if u_tt < -eps:
            return False

    for t in range(n_types):
        u_tt = sum(V[t, g] * allocations[t, g] for g in range(2)) - payments[t]
        for t_rep in range(n_types):
            if t == t_rep:
                continue
            u_t_trep = sum(V[t, g] * allocations[t_rep, g] for g in range(2)) - payments[t_rep]
            if u_t_trep - u_tt > eps:
                return False
    return True


def _repair_mechanism(allocations, payments, V, n_types, eps=1e-8):
    """
    Try to repair a nearly-feasible mechanism by finding minimal payment adjustments.
    We keep allocations fixed and only adjust payments to satisfy IC and IR.
    """
    # Variables: delta_p[t] for each type t
    # We want to minimize sum |delta_p[t]| but that's not linear.
    # Instead, minimize sum delta_p[t] (since we want to keep payments close)
    # with constraints that ensure feasibility.
    #
    # Actually, let's use a simpler approach: solve for payments directly
    # given fixed allocations, minimizing revenue loss.
    #
    # For fixed allocations, the IC and IR constraints become:
    # IR: p(t) <= sum_g V[t,g]*x[g,t]
    # IC: p(t) - p(t_rep) <= sum_g V[t,g]*(x[g,t] - x[g,t_rep])
    #
    # We want to maximize sum probs[t]*p[t] subject to these.
    # But we don't have probs here. Let's just find any feasible payments.
    #
    # Simpler: set p(t) = max over t_rep of [sum_g V[t,g]*x[g,t_rep] - sum_g V[t_rep,g]*x[g,t_rep] + p(t_rep)]
    # This is getting complex. Let's use a direct LP.

    # Variables: p[t] for t in range(n_types)
    n_p = n_types
    c = np.zeros(n_p)  # Minimize 0 (just find feasible solution)

    A_ub = []
    b_ub = []

    # IR: p(t) <= sum_g V[t,g]*allocations[t,g]
    # => p(t) - sum_g V[t,g]*allocations[t,g] <= 0
    for t in range(n_types):
        row = np.zeros(n_p)
        row[t] = 1.0
        rhs = sum(V[t, g] * allocations[t, g] for g in range(2))
        A_ub.append(row)
        b_ub.append(rhs)

    # IC: p(t) - p(t_rep) <= sum_g V[t,g]*(allocations[t,g] - allocations[t_rep,g])
    for t in range(n_types):
        for t_rep in range(n_types):
            if t == t_rep:
                continue
            row = np.zeros(n_p)
            row[t] = 1.0
            row[t_rep] = -1.0
            rhs = sum(V[t, g] * (allocations[t, g] - allocations[t_rep, g]) for g in range(2))
            A_ub.append(row)
            b_ub.append(rhs)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    bounds = [(0.0, None)] * n_p

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True, 'time_limit': 3.0})
        if result.success:
            new_payments = np.maximum(result.x, 0.0)
            return allocations.copy(), new_payments
    except Exception:
        pass

    return None
# EVOLVE-BLOCK-END
