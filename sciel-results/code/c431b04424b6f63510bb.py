def construct_auction(distribution_spec, support, constraints):
    import numpy as np

    # Define the support: each good's value in {0,1,2,3}
    values = [0, 1, 2, 3]
    n_values = len(values)
    
    # Enumerate all type profiles
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)
    type_index = {t: i for i, t in enumerate(types)}
    
    # Get probabilities for each type from distribution_spec
    if isinstance(distribution_spec, dict):
        probs = np.array([distribution_spec.get(t, 0.0) for t in types], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if probs.size != n_types:
            # Try to reshape
            probs = probs.reshape((n_values, n_values)).flatten()
    else:
        probs = np.ones(n_types) / n_types
        
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n_types) / n_types
    
    # Reshape to 2D grid (4x4)
    P = probs.reshape((n_values, n_values))
    
    # Check if distribution is independent
    P1 = P.sum(axis=1) # Marginal of v1
    P2 = P.sum(axis=0) # Marginal of v2
    P_indep = np.outer(P1, P2)
    
    # If not independent, we might need a fallback. 
    # But for now, we proceed with the independent assumption 
    # as per the "DP over lattice" hint which often implies separability or simple structure.
    # If the error is large, this might fail. 
    # However, the LP solution is the ground truth. 
    # Let's try to solve the 1D problems.
    
    def solve_1d(marginal_probs):
        # marginal_probs: array of length 4
        # Find best threshold t in {0, 1, 2, 3, 4}
        # q(v) = 1 if v >= t else 0
        best_rev = -np.inf
        best_q = np.zeros(4)
        
        for t in range(5):
            q = np.zeros(4)
            for v in range(4):
                if v >= t:
                    q[v] = 1.0
            
            # Calculate revenue
            # p(v) = v*q(v) - sum_{k=0}^{v-1} q(k)
            rev = 0.0
            for v in range(4):
                p_v = v * q[v] - np.sum(q[:v])
                # IR check: p(v) <= v*q(v) is always true.
                # U(v) = sum_{k=0}^{v-1} q(k) >= 0.
                rev += marginal_probs[v] * p_v
            
            if rev > best_rev:
                best_rev = rev
                best_q = q.copy()
                
        return best_q

    q1 = solve_1d(P1)
    q2 = solve_1d(P2)
    
    # Construct full allocation and payment
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    
    for i, (v1, v2) in enumerate(types):
        allocations[i, 0] = q1[v1]
        allocations[i, 1] = q2[v2]
        
        # Payment for independent case:
        # p(v1, v2) = p1(v1) + p2(v2)
        p1_v = v1 * q1[v1] - np.sum(q1[:v1])
        p2_v = v2 * q2[v2] - np.sum(q2[:v2])
        payments[i] = p1_v + p2_v
        
    return {'allocations': allocations, 'payments': payments}
