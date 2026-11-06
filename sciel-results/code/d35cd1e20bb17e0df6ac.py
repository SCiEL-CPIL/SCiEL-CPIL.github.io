"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    
    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(n_types) / n_types
    
    # Build type index mapping
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Extract support if provided
    if support is not None:
        if isinstance(support, dict):
            # Assume support maps type to allocation
            pass
        elif isinstance(support, (list, tuple)):
            # Assume support is list of allowed reports
            pass
    
    # Extract constraints
    # constraints might contain regret bound, etc.
    regret_bound = 1e-8
    if isinstance(constraints, dict):
        if 'regret' in constraints:
            regret_bound = constraints['regret']
    
    # LP-based mechanism design
    # Variables: x[i,j,g] for i,j in types, g in {0,1} (allocation probability of good g to type i when reporting j)
    #           p[i,j] for i,j in types (payment to type i when reporting j)
    # 
    # Objective: Maximize expected revenue = sum_{i,j} probs[i] * (p[i,j] - sum_g x[i,j,g]*v_i_g)
    #           Wait, revenue is what the seller gets. Revenue = sum_{i,j} probs[i] * (p[i,j] - transfer_to_bidders)
    #           But in auction, payment is from bidder to seller. So Revenue = sum probs[i] * p[i,j]
    #           However, we need to satisfy IC and IR.
    #
    # Let's use the standard LP formulation for deterministic or randomized mechanisms.
    # For a single additive bidder, the optimal mechanism is a direct revelation mechanism.
    # Let x_i be the allocation (probability of winning each good) when bidder has type i.
    # Let p_i be the payment when bidder has type i.
    #
    # IC: u(i, i) >= u(i, j) for all i, j
    #     sum_g x_{i,g} * v_{i,g} - p_i >= sum_g x_{j,g} * v_{i,g} - p_j
    # IR: u(i, i) >= 0
    #     sum_g x_{i,g} * v_{i,g} - p_i >= 0
    #
    # Revenue = sum_i probs[i] * p_i
    #
    # To maximize revenue, we can use the fact that for a single bidder, the optimal mechanism
    # is to set p_i = sum_g x_{i,g} * v_{i,g} - sum_{k<i} (x_{k+1} - x_k) * v_k (for monotone allocations)
    # But let's just solve the LP directly.
    #
    # Variables:
    # x[i,g] for i in 0..15, g in 0..1 (16*2 = 32 variables)
    # p[i] for i in 0..15 (16 variables)
    # Total variables: 48
    #
    # Constraints:
    # 1. 0 <= x[i,g] <= 1
    # 2. p[i] >= 0
    # 3. IC: sum_g x[i,g]*v[i,g] - p[i] >= sum_g x[j,g]*v[i,g] - p[j] for all i,j
    #    => p[j] - p[i] >= sum_g (x[j,g] - x[i,g])*v[i,g]
    # 4. IR: sum_g x[i,g]*v[i,g] - p[i] >= 0 => p[i] <= sum_g x[i,g]*v[i,g]
    #
    # Objective: Maximize sum_i probs[i] * p[i]
    # This is a linear program.
    
    n_goods = 2
    n_vars = n_types * n_goods + n_types  # x and p variables
    # x vars: 0 to 31, p vars: 32 to 47
    
    # Objective: Maximize sum_i probs[i] * p[i]
    # linprog minimizes, so we minimize -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[n_types * n_goods + i] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_ub = []
    b_ub = []
    
    # IC constraints: p[j] - p[i] - sum_g (x[j,g] - x[i,g])*v[i,g] <= 0
    # => -x[j,0]*v[i,0] + x[i,0]*v[i,0] - x[j,1]*v[i,1] + x[i,1]*v[i,1] + p[j] - p[i] <= 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            vi = np.array(types[i], dtype=float)
            # IC: u(i,i) >= u(i,j) => p_i - p_j + v_i . (x_j - x_i) <= 0
            # Coeff of x_i: -v_i
            row[i * n_goods + 0] = -vi[0]
            row[i * n_goods + 1] = -vi[1]
            # Coeff of x_j: +v_i
            row[j * n_goods + 0] = vi[0]
            row[j * n_goods + 1] = vi[1]
            # Coeff of p_i: +1
            row[n_types * n_goods + i] = 1
            # Coeff of p_j: -1
            row[n_types * n_goods + j] = -1
            A_ub.append(row)
            b_ub.append(0)
    
    # IR constraints: p[i] - sum_g x[i,g]*v[i,g] <= 0
    # => -x[i,0]*v[i,0] - x[i,1]*v[i,1] + p[i] <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        vi = np.array(types[i], dtype=float)
        row[i * n_goods + 0] = -vi[0]
        row[i * n_goods + 1] = -vi[1]
        row[n_types * n_goods + i] = 1
        A_ub.append(row)
        b_ub.append(0)
    
    # Bounds
    bounds = []
    for i in range(n_types * n_goods):  # x variables
        bounds.append((0, 1))
    for i in range(n_types):  # p variables
        bounds.append((0, None))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x_flat = res.x[:n_types * n_goods]
            p_flat = res.x[n_types * n_goods:]
            
            # Reshape x to (16, 2)
            allocations = x_flat.reshape(n_types, n_goods)
            payments = p_flat
            
            # Ensure non-negative
            allocations = np.maximum(allocations, 0)
            payments = np.maximum(payments, 0)
            
            # Check IR and IC violations
            # IR: sum_g x[i,g]*v[i,g] - p[i] >= 0
            ir_violation = 0
            for i in range(n_types):
                vi = np.array(types[i], dtype=float)
                util = np.dot(allocations[i], vi) - payments[i]
                if util < -1e-8:
                    ir_violation = max(ir_violation, -util)
            
            # IC: u(i,i) >= u(i,j)
            ic_violation = 0
            for i in range(n_types):
                vi = np.array(types[i], dtype=float)
                ui_i = np.dot(allocations[i], vi) - payments[i]
                for j in range(n_types):
                    ui_j = np.dot(allocations[j], vi) - payments[j]
                    if ui_i < ui_j - 1e-8:
                        ic_violation = max(ic_violation, ui_j - ui_i)
            
            if ir_violation < 1e-8 and ic_violation < 1e-8:
                return {'allocations': allocations, 'payments': payments}
            else:
                # If LP solution is infeasible due to numerical issues, return zero mechanism
                return {'allocations': np.zeros((n_types, n_goods)), 'payments': np.zeros(n_types)}
        else:
            # LP failed, return zero mechanism
            return {'allocations': np.zeros((n_types, n_goods)), 'payments': np.zeros(n_types)}
    except Exception:
        # Fallback to zero mechanism
        return {'allocations': np.zeros((n_types, n_goods)), 'payments': np.zeros(n_types)}

# EVOLVE-BLOCK-END
