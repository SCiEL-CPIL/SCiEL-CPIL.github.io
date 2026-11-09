"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: Specification of the value distribution
        support: Support of the value space
        constraints: Additional constraints
        
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # Parse the support and distribution
    # Each bidder has values in {0,1,2,3} for 2 goods
    # So there are 4*4 = 16 possible type reports
    # Each report is (v1, v2) where v1, v2 in {0,1,2,3}
    
    # Generate all possible reports
    values_per_good = [0, 1, 2, 3]
    all_reports = list(product(values_per_good, values_per_good))
    
    # Map report to index
    report_to_idx = {r: i for i, r in enumerate(all_reports)}
    
    # Parse distribution_spec to get probabilities
    # Assuming distribution_spec is a dict or array of probabilities for each report
    if isinstance(distribution_spec, dict):
        probs = np.array([distribution_spec.get(r, 0.0) for r in all_reports])
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) == 16:
            pass
        else:
            # Try to reshape or handle differently
            probs = np.array([distribution_spec.get(r, 0.0) if isinstance(distribution_spec, dict) else 0.0 for r in all_reports])
    else:
        # Default: uniform distribution
        probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    total_prob = probs.sum()
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(16) / 16.0
    
    # Check time
    if time.time() > deadline:
        # Return a simple baseline
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    # For a single additive bidder with two goods, incentive compatibility
    # requires that the allocation rule is monotone in each value component
    # and payments satisfy the payment identity
    
    # Simple approach: Allocate good i to the bidder if their reported value for good i >= threshold
    # Use a threshold mechanism
    
    # For simplicity and to ensure incentive compatibility, use a threshold-based allocation
    # where good i is allocated if v_i >= t_i for some threshold t_i
    
    # Try different thresholds and pick the one that maximizes revenue
    best_revenue = -np.inf
    best_allocations = np.zeros((16, 2))
    best_payments = np.zeros(16)
    
    thresholds = [0, 1, 2, 3]
    
    for t1 in thresholds:
        for t2 in thresholds:
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            
            for i, (v1, v2) in enumerate(all_reports):
                # Allocate good 1 if v1 >= t1
                alloc1 = 1.0 if v1 >= t1 else 0.0
                # Allocate good 2 if v2 >= t2
                alloc2 = 1.0 if v2 >= t2 else 0.0
                allocations[i, 0] = alloc1
                allocations[i, 1] = alloc2
                
                # Payment: For threshold mechanisms with binary allocation,
                # payment is the threshold price if allocated
                payment = 0.0
                if alloc1 > 0:
                    payment += t1
                if alloc2 > 0:
                    payment += t2
                payments[i] = payment
            
            # Check individual rationality: payment <= value for allocated goods
            ir_violated = False
            for i, (v1, v2) in enumerate(all_reports):
                value = v1 * allocations[i, 0] + v2 * allocations[i, 1]
                if payments[i] > value + 1e-8:
                    ir_violated = True
                    break
            
            if ir_violated:
                continue
            
            # Check nonnegative payments
            if np.any(payments < -1e-8):
                continue
            
            # Compute expected revenue
            expected_revenue = np.sum(probs * payments)
            
            if expected_revenue > best_revenue:
                best_revenue = expected_revenue
                best_allocations = allocations.copy()
                best_payments = payments.copy()
            
            # Check time
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # If no valid threshold mechanism found, use a simple fallback
    if best_revenue <= 0:
        # Allocate both goods for free (no payments)
        best_allocations = np.ones((16, 2))
        best_payments = np.zeros(16)
    
    return {'allocations': best_allocations, 'payments': best_payments}
# EVOLVE-BLOCK-END
