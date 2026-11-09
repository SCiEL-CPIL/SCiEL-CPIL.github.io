"""
Incentive-compatible finite auction design.

We design a direct-revelation mechanism for one additive bidder and two goods.
Each value in {0,1,2,3}. Three fixed distributions are expected.
The mechanism must be:
  - Feasible: allocation probabilities in [0,1], and for each type report, the
    vector of allocation probabilities across goods is in the unit simplex? 
    Actually, for a single bidder with two goods, the allocation is a vector in {0,1}^2.
    The mechanism specifies for each report t = (t1, t2), a distribution over allocations.
    But the contract says "allocations:(16,2)" and "payments:(16,)".
    This suggests that for each of the 16 reports, we return a deterministic allocation (x1, x2) in {0,1}^2
    and a payment p.
    Wait, "allocations:(16,2)" likely means a 16x2 array of allocation vectors, each in {0,1}^2.
    And payments is a 16-vector.
    
    Truthfulness: For all true types v and reports r, 
      u(v, v) >= u(v, r)  where u(v, r) = v.x(r) - p(r).
    This is the incentive compatibility (IC) constraint.
    
    Individual rationality (IR): u(v, v) >= 0 for all v.
    Nonnegative payments: p(r) >= 0.
    
    Objective: Maximize expected revenue normalized by an independently solved finite-support mechanism LP.
    The "independently solved finite-support mechanism LP" likely refers to the optimal revenue achievable by any IC, IR mechanism.
    So the score is (our expected revenue) / (optimal expected revenue).
    We want to maximize this ratio, i.e., achieve the optimal revenue.
    
    Since the problem is finite and small (16 types), we can solve the LP for the optimal mechanism.
    The mechanism can be randomized, but the contract asks for deterministic allocations? 
    "allocations:(16,2)" - if it's deterministic, each row is in {0,1}^2.
    But the LP might allow randomized allocations. However, the contract specifies the return format as (16,2) for allocations.
    If randomized, we'd need to return probabilities. But the contract says "allocations", which might imply deterministic.
    Let's re-read: "allocations:(16,2),payments:(16,)". This looks like deterministic.
    However, the optimal mechanism might require randomization. 
    But for a single bidder, the optimal deterministic mechanism is often optimal? Not necessarily.
    Actually, for a single bidder with multi-dimensional types, the optimal mechanism might be randomized.
    But let's check the contract again: "Maximize expected revenue normalized by an independently solved finite-support mechanism LP."
    This suggests we should solve the LP to get the optimal revenue, and then we need to return a mechanism that achieves that revenue.
    If the LP solution is randomized, we might need to return the expected allocation? But the contract says "allocations", not "probabilities".
    Let's assume the contract expects deterministic allocations. If so, we need to find the best deterministic mechanism.
    But the normalization is by the LP (which allows randomization). So if the LP optimum is higher than the best deterministic, we can't reach 1.0.
    However, the seed program scores 1.0, so it must be achieving the LP optimum. This suggests that either:
      1. The optimal mechanism is deterministic, or
      2. The contract allows returning randomized mechanisms in some form.
    
    Let's look at the seed program: it calls `auction_baseline`. We don't have that code, but it scores 1.0.
    So we need to replicate or improve upon it.
    
    Approach:
    1. Parse the distribution_spec, support, constraints.
    2. Enumerate all 16 types.
    3. Formulate the LP for the optimal mechanism (allowing randomized allocations).
    4. Solve the LP.
    5. Extract the optimal mechanism (allocations and payments).
    6. If the contract requires deterministic allocations, we might need to round or find the best deterministic.
       But since the seed scores 1.0, likely the LP solution is deterministic or the contract accepts the LP solution's expected values.
       However, the return format is (16,2) for allocations. If we return expected allocations, they might not be in {0,1}.
       Let's assume the contract expects deterministic allocations in {0,1}^2.
       Then we need to solve for the best deterministic mechanism.
       But the normalization is by the LP (randomized) optimum. So if the best deterministic is less than the LP optimum, we can't reach 1.0.
       Since the seed reaches 1.0, the best deterministic must equal the LP optimum, or the contract allows randomized.
    
    Let's assume the contract allows randomized mechanisms, and "allocations" means the expected allocation vector.
    But the contract says "allocations:(16,2)", which could be a 16x2 array of expected allocation probabilities.
    And payments are expected payments.
    
    Let's proceed with the LP approach:
    Variables:
      x_{r, a} >= 0 for each report r and allocation a in {0,1}^2.
      p_r >= 0 for each report r.
      Constraints:
        For each r: sum_a x_{r,a} = 1.
        IC: For each true type v and report r:
          sum_a x_{r,a} * v.a - p_r <= sum_a x_{v,a} * v.a - p_v.
        IR: For each v: sum_a x_{v,a} * v.a - p_v >= 0.
        p_r >= 0.
    Objective: Maximize sum_{v} Pr(v) * p_v.
    
    This is a linear program. We can solve it with scipy.optimize.linprog.
    
    Steps:
    1. Parse inputs to get the 16 types and their probabilities.
    2. Set up the LP.
    3. Solve.
    4. Extract x and p.
    5. Return allocations (expected) and payments.
    
    But the contract says "allocations:(16,2)". If we return expected allocations, each row sums to 1? No, each row is a vector of expected allocation for each good.
    For example, if for report r, the mechanism allocates good 1 with prob 0.5 and good 2 with prob 0.3, then the allocation vector is [0.5, 0.3].
    But the allocation is over two goods, and the bidder can get both. So the allocation vector is in [0,1]^2, not a simplex.
    The constraint sum_a x_{r,a} = 1 is over the 4 possible allocation vectors (0,0), (1,0), (0,1), (1,1).
    So x_{r,a} is the probability of allocation a given report r.
    Then the expected allocation for good 1 is x_{r,(1,0)} + x_{r,(1,1)}, and for good 2 is x_{r,(0,1)} + x_{r,(1,1)}.
    
    So the return "allocations" should be a 16x2 array where each row is [E[x1|r], E[x2|r]].
    And payments is a 16-vector of p_r.
    
    Let's implement this.
"""

def construct_auction(distribution_spec, support, constraints):
    import numpy as np
    from scipy.optimize import linprog
    import time
    
    start_time = time.time()
    deadline = start_time + 85  # 85 seconds, leaving 5s margin
    
    # Parse inputs
    # distribution_spec: likely a dict or list specifying the distribution over types.
    # support: the set of types, likely {(0,0), (0,1), ..., (3,3)} -> 16 types.
    # constraints: likely a dict with IR, IC, etc.
    
    # Assume support is a list of 16 tuples (v1, v2) with v1,v2 in {0,1,2,3}.
    # Assume distribution_spec gives probabilities for each type in support.
    
    # Let's assume the standard: support is sorted, and distribution_spec is a list of probabilities.
    # Or distribution_spec is a dict mapping type to prob.
    
    # To be safe, let's handle both.
    if isinstance(distribution_spec, dict):
        # Assume keys are types, values are probabilities.
        # Ensure all 16 types are present.
        all_types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
        probs = np.zeros(16)
        for i, t in enumerate(all_types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
        types = all_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        # Assume it's a list of probabilities for the 16 types in order.
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != 16:
            # Try to normalize or handle error
            if len(probs) > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(16) / 16
        types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    else:
        # Default: uniform
        probs = np.ones(16) / 16
        types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Types: 16 types, each (v1, v2)
    # Allocations: 4 possible: (0,0), (1,0), (0,1), (1,1)
    allocs = [(0,0), (1,0), (0,1), (1,1)]
    
    # Variables:
    # x_{r,a} for r in 0..15, a in 0..3 -> 16*4 = 64 variables
    # p_r for r in 0..15 -> 16 variables
    # Total: 80 variables
    
    n_types = 16
    n_allocs = 4
    n_vars = n_types * n_allocs + n_types  # 64 + 16 = 80
    
    # Objective: maximize sum_v probs[v] * p_v
    # In linprog, we minimize, so we minimize -sum_v probs[v] * p_v
    c = np.zeros(n_vars)
    for v in range(n_types):
        c[64 + v] = -probs[v]  # p_v is at index 64+v
    
    # Constraints:
    # 1. For each r: sum_a x_{r,a} = 1
    A_eq = np.zeros((n_types, n_vars))
    b_eq = np.ones(n_types)
    for r in range(n_types):
        for a in range(n_allocs):
            A_eq[r, r * n_allocs + a] = 1.0
    
    # 2. IC: For each true type v and report r:
    #    sum_a x_{r,a} * v.a - p_r <= sum_a x_{v,a} * v.a - p_v
    #    => sum_a x_{r,a} * v.a - p_r - sum_a x_{v,a} * v.a + p_v <= 0
    #    => sum_a (v.a) x_{r,a} - sum_a (v.a) x_{v,a} - p_r + p_v <= 0
    #    For v == r, this is 0 <= 0, which is trivially satisfied. So we can skip v==r.
    #    Number of IC constraints: 16*15 = 240
    n_ic = n_types * (n_types - 1)
    A_ub = np.zeros((n_ic, n_vars))
    b_ub = np.zeros(n_ic)
    
    ic_idx = 0
    for v in range(n_types):
        for r in range(n_types):
            if v == r:
                continue
            # Constraint: sum_a v.a * x_{r,a} - sum_a v.a * x_{v,a} - p_r + p_v <= 0
            v_type = types[v]
            for a in range(n_allocs):
                val = v_type[0] * allocs[a][0] + v_type[1] * allocs[a][1]
                A_ub[ic_idx, r * n_allocs + a] += val
                A_ub[ic_idx, v * n_allocs + a] -= val
            A_ub[ic_idx, 64 + r] -= 1.0  # -p_r
            A_ub[ic_idx, 64 + v] += 1.0   # +p_v
            ic_idx += 1
    
    # 3. IR: For each v: sum_a x_{v,a} * v.a - p_v >= 0
    #    => -sum_a x_{v,a} * v.a + p_v <= 0
    n_ir = n_types
    A_ir = np.zeros((n_ir, n_vars))
    b_ir = np.zeros(n_ir)
    for v in range(n_types):
        v_type = types[v]
        for a in range(n_allocs):
            val = v_type[0] * allocs[a][0] + v_type[1] * allocs[a][1]
            A_ir[v, v * n_allocs + a] -= val
        A_ir[v, 64 + v] += 1.0
    
    # Combine A_ub and A_ir
    A_ub_all = np.vstack([A_ub, A_ir])
    b_ub_all = np.concatenate([b_ub, b_ir])
    
    # Bounds:
    # x_{r,a} >= 0
    # p_r >= 0
    bounds = [(0, None)] * n_vars
    
    # Solve LP
    result = linprog(c, A_ub=A_ub_all, b_ub=b_ub_all, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if time.time() > deadline:
        # Fallback: return a simple mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    if result.status != 0:
        # Fallback
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x[:64].reshape(16, 4)
    p = result.x[64:]
    
    # Compute expected allocations
    allocations = np.zeros((16, 2))
    for r in range(16):
        for a in range(4):
            allocations[r, 0] += x[r, a] * allocs[a][0]
            allocations[r, 1] += x[r, a] * allocs[a][1]
    
    # Ensure nonnegative payments (should be from LP, but clip)
    p = np.maximum(p, 0)
    
    return {'allocations': allocations, 'payments': p}
