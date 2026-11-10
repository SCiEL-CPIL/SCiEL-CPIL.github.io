"""Incentive-compatible finite auction design with explicit candidate API."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder, two goods,
    values in {0,1,2,3}, three fixed distributions.
    
    Returns dict with 'allocations' (16,2) and 'payments' (16,).
    """
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse the support: 16 type profiles (4 values x 4 values for 2 goods)
    # Each type is (v1, v2) where vi in {0,1,2,3}
    # We enumerate all 16 types
    values = [0, 1, 2, 3]
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    # types[i] = (v1_i, v2_i), i = 0..15
    
    # distribution_spec: need to interpret. Assume it gives probabilities for each type
    # or parameters for the three fixed distributions
    # Let's assume distribution_spec is a dict or list specifying the distribution
    # Common interpretation: distribution_spec could be the probability vector over 16 types
    
    # Parse distribution_spec - try multiple formats
    if isinstance(distribution_spec, dict):
        # Might have 'type_probs' or similar
        if 'type_probs' in distribution_spec:
            probs = np.array(distribution_spec['type_probs'], dtype=float)
        elif 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        else:
            # Try to find a 16-element array
            for k, v in distribution_spec.items():
                if isinstance(v, (list, np.ndarray)) and len(v) == 16:
                    probs = np.array(v, dtype=float)
                    break
            else:
                probs = np.ones(16) / 16.0
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == 16:
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = np.ones(16) / 16.0
    else:
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Myerson auction for single additive bidder with two goods
    # For each type i = (v1_i, v2_i):
    #   Allocation probability for good j: q_j(i) = P(v_j >= t_j) where t_j is reserve
    #   Actually, for additive values, the optimal mechanism is:
    #   - Each good is allocated independently with a reserve price
    #   - q_j(i) = 1 if v_j_i >= r_j, else 0 (for discrete case)
    #   - Payment: p(i) = sum over goods of [v_j_i * q_j(i) - integral of q_j from 0 to v_j_i]
    #   
    # For finite support {0,1,2,3}, the payment formula is:
    # p(i) = sum_j [v_j_i * q_j(i) - sum_{k=0}^{v_j_i - 1} q_j(k, other goods)]
    # But since allocation depends only on own value for each good (additive, single bidder),
    # q_j depends only on v_j.
    
    # For a single bidder with additive values over two goods, the IC mechanism is:
    # q_j(v_j) is non-decreasing in v_j
    # p(v1, v2) = v1*q1(v1) + v2*q2(v2) - sum_{k=0}^{v1-1} q1(k) - sum_{k=0}^{v2-1} q2(k)
    
    # To maximize revenue, we choose q1, q2 to maximize:
    # sum_i probs[i] * p(i) = sum_i probs[i] * [v1_i*q1(v1_i) + v2_i*q2(v2_i) - sum_{k=0}^{v1_i-1}q1(k) - sum_{k=0}^{v2_i-1}q2(k)]
    
    # This separates by good! For each good j, maximize:
    # sum_i probs[i] * [v_j_i * q_j(v_j_i) - sum_{k=0}^{v_j_i-1} q_j(k)]
    # = sum_{v=0}^{3} q_j(v) * [sum_{i: v_j_i=v} probs[i] * v - sum_{i: v_j_i > v} probs[i]]
    # = sum_{v=0}^{3} q_j(v) * [v * P(v_j=v) - P(v_j > v)]
    
    # The coefficient for q_j(v) is: v * P(v_j = v) - P(v_j > v)
    # Set q_j(v) = 1 if coefficient > 0, else 0 (since q_j in [0,1])
    
    # Compute marginal probabilities for each good
    # For good 1: P(v1 = v) for v in {0,1,2,3}
    p1 = np.zeros(4)  # P(v1 = v)
    p2 = np.zeros(4)  # P(v2 = v)
    for i in range(16):
        v1, v2 = types[i]
        p1[v1] += probs[i]
        p2[v2] += probs[i]
    
    # Solve the LP for the optimal mechanism
    # Variables: q[16,2], p[16]
    # Indexing: q(i, j) is at index i*2 + j. p(i) is at index 32 + i.
    
    n_vars = 48
    c = np.zeros(n_vars)
    # Objective: Maximize sum(pi_i * p_i) -> Minimize -sum(pi_i * p_i)
    for i in range(16):
        c[32 + i] = -probs[i]
        
    A_ub = []
    b_ub = []
    
    # IC Constraints: p_i - p_j <= v_i . (q_i - q_j)
    # p_i - p_j - sum_k v_ik(q_ik - q_jk) <= 0
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # p_i coeff: 1
            row[32 + i] = 1.0
            # p_j coeff: -1
            row[32 + j] = -1.0
            
            v_i = np.array(types[i], dtype=float)
            for k in range(2):
                # q_ik coeff: v_ik
                row[i * 2 + k] = v_i[k]
                # q_jk coeff: -v_ik
                row[j * 2 + k] = -v_i[k]
                
            A_ub.append(row)
            b_ub.append(0.0)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = []
    for i in range(16):
        for k in range(2):
            bounds.append((0.0, 1.0))
    for i in range(16):
        bounds.append((0.0, None))
        
    try:
        from scipy.optimize import linprog
        # Use high precision settings if available, but 'highs' is default and robust
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            q_flat = res.x[:32].reshape(16, 2)
            p_vec = res.x[32:]
            
            allocations = np.clip(q_flat, 0.0, 1.0)
            payments = np.maximum(p_vec, 0.0)
            
            # Small cleanup for numerical noise
            allocations[allocations < 1e-9] = 0.0
            allocations[allocations > 1.0 - 1e-9] = 1.0
            payments[payments < 1e-9] = 0.0
            
            return {'allocations': allocations, 'payments': payments}
        else:
            # Fallback if LP fails
            pass
    except Exception:
        pass
        
    # Fallback: Simple Myerson-like heuristic (original code logic)
    def compute_allocation(p):
        q = np.zeros(4)
        for v in range(4):
            if v == 3:
                coeff = 3 * p[3]
            else:
                coeff = v * p[v] - np.sum(p[v+1:4])
            q[v] = 1.0 if coeff > 1e-12 else 0.0
        return q
    
    q1 = compute_allocation(p1)
    q2 = compute_allocation(p2)
    
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    for i in range(16):
        v1, v2 = types[i]
        allocations[i, 0] = q1[v1]
        allocations[i, 1] = q2[v2]
        
        pay1 = v1 * q1[v1] - np.sum(q1[:v1]) if v1 > 0 else 0.0
        pay2 = v2 * q2[v2] - np.sum(q2[:v2]) if v2 > 0 else 0.0
        payments[i] = pay1 + pay2
    
    payments = np.maximum(payments, 0.0)
    allocations = np.clip(allocations, 0.0, 1.0)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
