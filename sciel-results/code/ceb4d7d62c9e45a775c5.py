import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # Parse problem structure
    values = [0, 1, 2, 3]

    # Get distribution
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None

    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
        # Ensure no negative probs
        probs = np.maximum(probs, 0)
        if probs.sum() == 0:
            probs = np.ones(16) / 16.0
        else:
            probs = probs / probs.sum()

    # Determine types
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]

    n_types = len(types)
    if n_types != 16:
        # Fallback if support parsing is weird, use standard grid
        types = [(i, j) for i in values for j in values]
        n_types = 16

    type_to_vals = {idx: (v1, v2) for idx, (v1, v2) in enumerate(types)}
    type_vals = np.array([type_to_vals[i] for i in range(16)], dtype=float)
    v1_all = type_vals[:, 0]
    v2_all = type_vals[:, 1]

    # Strategy 1: Myerson-based optimal mechanism for single-bidder case
    # This decouples into two independent one-dimensional auctions
    # using virtual valuations (ironing)
    
    def compute_myerson_mechanism(marginal_probs, values):
        """
        Compute optimal Myerson mechanism for one good with given marginal distribution.
        marginal_probs: probabilities for each value level
        values: list of possible values [0, 1, 2, 3]
        Returns: allocation probabilities for each value, payment for each value
        """
        n_vals = len(values)
        vals = np.array(values, dtype=float)
        
        # Sort by value
        order = np.argsort(vals)
        sorted_vals = vals[order]
        sorted_probs = marginal_probs[order]
        
        # Compute CDF
        cdf = np.cumsum(sorted_probs)
        
        # Compute virtual values: phi(v) = v - (1 - F(v))/f(v)
        # For discrete: phi(v_i) = v_i - (1 - F(v_i)) / f(v_i)
        virtual_vals = np.zeros(n_vals)
        for i in range(n_vals):
            f_i = sorted_probs[i]
            F_i = cdf[i]
            if f_i > 1e-15:
                virtual_vals[i] = sorted_vals[i] - (1 - F_i) / f_i
            else:
                virtual_vals[i] = -np.inf
        
        # Ironing: find optimal allocation using generalized ironing
        # For discrete case, we need to find the optimal step function
        # that maximizes expected virtual surplus
        
        # Simple approach: for each possible threshold, compute the allocation
        # and find the best one
        
        best_alloc = np.zeros(n_vals)
        best_virtual_surplus = -np.inf
        
        # Try all possible monotone allocation patterns
        # Allocation is monotone non-decreasing in value
        # For 4 values, possible patterns are limited
        
        # Generate all monotone non-decreasing allocation patterns in [0,1]
        # For simplicity, try allocations that are 0 or 1 (full allocation)
        # and some intermediate values
        
        # Actually, for discrete Myerson, the optimal allocation is:
        # x(v) = 1 if virtual value >= 0, 0 otherwise
        # But with ironing, we may need to adjust
        
        # Compute optimal allocation using ironing
        # The ironed virtual value is the concave envelope of the virtual value function
        
        # For discrete case, ironing means: if virtual values are not concave,
        # we replace them with their concave envelope
        
        # Concave envelope: for each point, take the minimum of lines connecting
        # pairs of points that lie above
        
        ironed_vv = virtual_vals.copy()
        
        # Check for non-concavity and iron
        # For 4 points, check all pairs
        for i in range(n_vals - 1):
            for j in range(i + 1, n_vals):
                # Line from (sorted_vals[i], virtual_vals[i]) to (sorted_vals[j], virtual_vals[j])
                slope = (virtual_vals[j] - virtual_vals[i]) / (sorted_vals[j] - sorted_vals[i])
                for k in range(i + 1, j):
                    line_val = virtual_vals[i] + slope * (sorted_vals[k] - sorted_vals[i])
                    if virtual_vals[k] > line_val + 1e-12:
                        # Non-concave, need to iron
                        # Replace virtual values in [i, j] with the line
                        for m in range(i, j + 1):
                            ironed_vv[m] = virtual_vals[i] + slope * (sorted_vals[m] - sorted_vals[i])
        
        # Allocation: x(v) = 1 if ironed virtual value >= 0, 0 otherwise
        # But we need to handle the case where we want partial allocation
        
        # For discrete Myerson, the optimal allocation is:
        # x(v_i) = 1 if ironed_vv[i] >= 0, 0 otherwise
        # But this may not be optimal if we can do better with partial allocation
        
        # Let's try all possible monotone allocations
        # For 4 values, the allocation must be non-decreasing: x[0] <= x[1] <= x[2] <= x[3]
        
        # Generate candidate allocations
        candidates = []
        
        # Binary allocations (0 or 1)
        for mask in range(1 << n_vals):
            alloc = np.zeros(n_vals)
            for i in range(n_vals):
                if mask & (1 << i):
                    alloc[i] = 1.0
            # Check monotonicity
            if all(alloc[i] <= alloc[i+1] + 1e-12 for i in range(n_vals-1)):
                candidates.append(alloc)
        
        # Also try some intermediate values
        for x0 in [0, 0.5, 1]:
            for x1 in [0, 0.5, 1]:
                if x1 < x0 - 1e-12:
                    continue
                for x2 in [0, 0.5, 1]:
                    if x2 < x1 - 1e-12:
                        continue
                    for x3 in [0, 0.5, 1]:
                        if x3 < x2 - 1e-12:
                            continue
                        alloc = np.array([x0, x1, x2, x3])
                        candidates.append(alloc)
        
        # Find the best allocation
        for alloc in candidates:
            # Compute expected virtual surplus
            virtual_surplus = np.sum(alloc * ironed_vv * sorted_probs)
            if virtual_surplus > best_virtual_surplus + 1e-12:
                best_virtual_surplus = virtual_surplus
                best_alloc = alloc.copy()
        
        # Compute payments using Myerson payment identity
        # p(v) = v * x(v) - integral from 0 to v of x(t) dt
        # For discrete: p(v_i) = v_i * x(v_i) - sum_{j: v_j < v_i} (v_i - v_j) * x(v_j) * f(v_j)
        # Actually, the correct formula is:
        # p(v_i) = v_i * x(v_i) - sum_{j < i} (v_i - v_j) * x(v_j) * f(v_j) - v_i * sum_{j < i} x(v_j) * f(v_j) + ...
        
        # Simpler: use the fact that for single bidder,
        # p(v) = v * x(v) - E[v' * x(v') | v' <= v] * P(v' <= v) / P(v' <= v) ... 
        # No, the correct formula is:
        # p(v) = v * x(v) - integral_0^v x(t) dt
        # For discrete: p(v_i) = v_i * x(v_i) - sum_{j < i} x(v_j) * (v_i - v_j) * f(v_j) - v_i * sum_{j < i} x(v_j) * f(v_j)
        # Wait, let me derive it properly.
        
        # The payment identity is: p(v) = v * x(v) - E[x(v') | v' <= v] * v - E[v' * (1 - x(v')) | v' <= v]
        # This is getting complicated. Let me use a different approach.
        
        # For a single bidder with one good, the payment is:
        # p(v) = v * x(v) - integral_0^v x(t) dt
        # For discrete values: p(v_i) = v_i * x(v_i) - sum_{j < i} x(v_j) * (v_i - v_j) * f(v_j)
        # No, that's not right either.
        
        # The correct formula: p(v) = v * x(v) - E[v * x(v) | v' <= v] ... 
        # Let me just compute it directly using the IC constraint.
        
        # For a single good, the IC constraint is:
        # u(v) >= u(v') + (v - v') * x(v') for all v, v'
        # where u(v) = v * x(v) - p(v)
        
        # The minimal payment that satisfies IC and IR is:
        # p(v) = v * x(v) - max_{v' <= v} [v' * x(v') - p(v') + (v - v') * x(v')]
        # With p(v_min) = 0 (IR at the lowest type)
        
        # Let me compute payments recursively
        payments = np.zeros(n_vals)
        
        # Start from the lowest value
        # u(v_min) = 0 (IR, and since x is monotone, this is the binding constraint)
        # p(v_min) = v_min * x(v_min)
        
        # Actually, IR says p(v) <= v * x(v), and IC says:
        # v * x(v) - p(v) >= v' * x(v') - p(v') + (v - v') * x(v') for all v' < v
        # => p(v) <= v * x(v) - (v' * x(v') - p(v') + (v - v') * x(v'))
        # => p(v) <= v * x(v) - v' * x(v') + p(v') - (v - v') * x(v')
        # => p(v) <= v * x(v) - v' * x(v') + p(v') - v * x(v') + v' * x(v')
        # => p(v) <= v * x(v) - v * x(v') + p(v')
        # => p(v) <= v * (x(v) - x(v')) + p(v')
        
        # So: p(v) = min_{v' < v} [v * (x(v) - x(v')) + p(v')]
        # And also p(v) <= v * x(v) (IR)
        
        # Let me compute this recursively
        for i in range(n_vals):
            if i == 0:
                payments[i] = sorted_vals[i] * best_alloc[i]
            else:
                # p(v_i) = min over j < i of [v_i * (x(v_i) - x(v_j)) + p(v_j)]
                min_payment = np.inf
                for j in range(i):
                    payment_candidate = sorted_vals[i] * (best_alloc[i] - best_alloc[j]) + payments[j]
                    min_payment = min(min_payment, payment_candidate)
                # Also IR: p(v_i) <= v_i * x(v_i)
                ir_payment = sorted_vals[i] * best_alloc[i]
                payments[i] = min(min_payment, ir_payment)
                # Ensure non-negative
                payments[i] = max(0, payments[i])
        
        # Map back to original order
        alloc_original = np.zeros(n_vals)
        pay_original = np.zeros(n_vals)
        for i in range(n_vals):
            orig_idx = order[i]
            alloc_original[orig_idx] = best_alloc[i]
            pay_original[orig_idx] = payments[i]
        
        return alloc_original, pay_original

    # Compute marginal distributions for each good
    # For good 1: marginal probability of v1 = v is sum over v2 of probs[(v1, v2)]
    # We need to map (v1, v2) to index
    
    # Build index mapping
    val_to_idx = {}
    for idx, (v1, v2) in enumerate(types):
        val_to_idx[(v1, v2)] = idx
    
    # Compute marginal for good 1
    marginal1 = np.zeros(4)
    for idx, (v1, v2) in enumerate(types):
        marginal1[v1] += probs[idx]
    
    # Compute marginal for good 2
    marginal2 = np.zeros(4)
    for idx, (v1, v2) in enumerate(types):
        marginal2[v2] += probs[idx]
    
    # Compute Myerson mechanisms for each good
    alloc1, pay1 = compute_myerson_mechanism(marginal1, values)
    alloc2, pay2 = compute_myerson_mechanism(marginal2, values)
    
    # Construct full mechanism
    allocations_myerson = np.zeros((16, 2))
    payments_myerson = np.zeros(16)
    
    for idx, (v1, v2) in enumerate(types):
        allocations_myerson[idx, 0] = alloc1[v1]
        allocations_myerson[idx, 1] = alloc2[v2]
        payments_myerson[idx] = pay1[v1] + pay2[v2]
    
    # Verify the Myerson mechanism
    max_regret_myerson = 0.0
    for t in range(16):
        v1_t, v2_t = type_to_vals[t]
        u_t = v1_t * allocations_myerson[t, 0] + v2_t * allocations_myerson[t, 1] - payments_myerson[t]
        for t2 in range(16):
            if t == t2:
                continue
            u_t2_as_t = v1_t * allocations_myerson[t2, 0] + v2_t * allocations_myerson[t2, 1] - payments_myerson[t2]
            if u_t < u_t2_as_t:
                viol = u_t2_as_t - u_t
                if viol > max_regret_myerson:
                    max_regret_myerson = viol
        # IR check
        if u_t < 0:
            viol = -u_t
            if viol > max_regret_myerson:
                max_regret_myerson = viol
    
    # Strategy 2: LP-based mechanism
    # Build LP
    n_vars = 48
    A_rows = []
    b_rows = []
    
    # IC constraints
    for t in range(16):
        for t2 in range(16):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            row[t] = -v1_all[t]
            row[t2] = v1_all[t]
            row[16 + t] = -v2_all[t]
            row[16 + t2] = v2_all[t]
            row[32 + t] = 1.0
            row[32 + t2] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR constraints
    for t in range(16):
        row = np.zeros(n_vars)
        row[t] = -v1_all[t]
        row[16 + t] = -v2_all[t]
        row[32 + t] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Objective: maximize expected revenue
    c = np.zeros(n_vars)
    c[32:48] = -probs
    
    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16
    
    # Solve LP
    result = None
    try:
        options = {
            'presolve': True,
            'dual_feasibility_tolerance': 1e-14,
            'primal_feasibility_tolerance': 1e-14,
            'time_limit': 40
        }
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception:
        pass
    
    allocations_lp = None
    payments_lp = None
    max_regret_lp = np.inf
    
    if result is not None and result.success:
        x = result.x
        x1 = np.clip(x[0:16], 0, 1)
        x2 = np.clip(x[16:32], 0, 1)
        p = np.maximum(x[32:48], 0)
        allocations_lp = np.column_stack((x1, x2))
        payments_lp = p
        
        # Verify
        max_regret_lp = 0.0
        for t in range(16):
            v1_t, v2_t = type_to_vals[t]
            u_t = v1_t * allocations_lp[t, 0] + v2_t * allocations_lp[t, 1] - payments_lp[t]
            for t2 in range(16):
                if t == t2:
                    continue
                u_t2_as_t = v1_t * allocations_lp[t2, 0] + v2_t * allocations_lp[t2, 1] - payments_lp[t2]
                if u_t < u_t2_as_t:
                    viol = u_t2_as_t - u_t
                    if viol > max_regret_lp:
                        max_regret_lp = viol
            if u_t < 0:
                viol = -u_t
                if viol > max_regret_lp:
                    max_regret_lp = viol
    
    # Choose the best mechanism
    best_allocations = None
    best_payments = None
    best_revenue = -np.inf
    
    # Check Myerson mechanism
    if max_regret_myerson <= 1e-8:
        revenue_myerson = np.sum(payments_myerson * probs)
        if revenue_myerson > best_revenue:
            best_revenue = revenue_myerson
            best_allocations = allocations_myerson.copy()
            best_payments = payments_myerson.copy()
    
    # Check LP mechanism
    if allocations_lp is not None and max_regret_lp <= 1e-8:
        revenue_lp = np.sum(payments_lp * probs)
        if revenue_lp > best_revenue:
            best_revenue = revenue_lp
            best_allocations = allocations_lp.copy()
            best_payments = payments_lp.copy()
    
    # If neither works, use zero mechanism
    if best_allocations is None:
        best_allocations = np.zeros((16, 2))
        best_payments = np.zeros(16)
    
    # High-precision polish: try to improve the solution
    # Check if we can increase payments while maintaining IC and IR
    # This is a local optimization step
    
    # For each type, try to increase payment slightly
    # and check if IC and IR are still satisfied
    
    improved = False
    for iteration in range(10):
        if time.time() > deadline - 5:
            break
        
        # Try to increase each payment by a small amount
        for t in range(16):
            if time.time() > deadline - 5:
                break
            
            # Try increasing payment
            for delta in [0.01, 0.001, 0.0001]:
                new_payments = best_payments.copy()
                new_payments[t] += delta
                
                # Check IR
                ir_ok = True
                for t2 in range(16):
                    v1_t2, v2_t2 = type_to_vals[t2]
                    u_t2 = v1_t2 * best_allocations[t2, 0] + v2_t2 * best_allocations[t2, 1] - new_payments[t2]
                    if u_t2 < -1e-10:
                        ir_ok = False
                        break
                if not ir_ok:
                    continue
                
                # Check IC
                ic_ok = True
                for t3 in range(16):
                    v1_t3, v2_t3 = type_to_vals[t3]
                    u_t3 = v1_t3 * best_allocations[t3, 0] + v2_t3 * best_allocations[t3, 1] - new_payments[t3]
                    for t4 in range(16):
                        if t3 == t4:
                            continue
                        u_t4_as_t3 = v1_t3 * best_allocations[t4, 0] + v2_t3 * best_allocations[t4, 1] - new_payments[t4]
                        if u_t3 < u_t4_as_t3 - 1e-10:
                            ic_ok = False
                            break
                    if not ic_ok:
                        break
                if not ic_ok:
                    continue
                
                # If feasible, accept
                best_payments = new_payments
                improved = True
                break
        
        if not improved:
            break
    
    return {'allocations': best_allocations, 'payments': best_payments}
