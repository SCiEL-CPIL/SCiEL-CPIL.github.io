"""Rewritten program for incentive-compatible finite auction design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with one additive bidder, two goods.
    Values in {0,1,2,3} for each good, so 16 possible reports.
    Returns dict with allocations:(16,2) and payments:(16,)
    """
    start_time = time.time()
    deadline = start_time + 75  # 75 second deadline with 15s safety margin
    
    # Generate all 16 reports: (v1, v2) where v1, v2 in {0,1,2,3}
    values = [0, 1, 2, 3]
    reports = [(v1, v2) for v1 in values for v2 in values]
    n_reports = 16
    
    # Parse distribution_spec to get probabilities for each report
    probs = np.ones(16) / 16.0
    try:
        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            arr = np.array(distribution_spec, dtype=float).flatten()
            if len(arr) == 16:
                probs = arr
            elif len(arr) == 4:
                # Marginal distributions for each good
                probs_v1 = arr[0:4] if len(arr) >= 4 else np.ones(4)/4
                probs_v2 = arr[0:4] if len(arr) >= 4 else np.ones(4)/4
                probs = np.array([probs_v1[v1] * probs_v2[v2] for v1, v2 in reports])
            elif len(arr) == 2:
                probs_v1 = np.ones(4)/4
                probs_v2 = np.ones(4)/4
                probs = np.array([probs_v1[v1] * probs_v2[v2] for v1, v2 in reports])
        elif isinstance(distribution_spec, dict):
            probs = np.zeros(16)
            for i, (v1, v2) in enumerate(reports):
                key = (v1, v2)
                if key in distribution_spec:
                    probs[i] = distribution_spec[key]
                else:
                    probs[i] = 1.0 / 16.0
    except Exception:
        probs = np.ones(16) / 16.0
    
    # Ensure probs are non-negative and sum to 1
    probs = np.maximum(probs, 0)
    total = probs.sum()
    if total > 1e-12:
        probs /= total
    else:
        probs = np.ones(16) / 16.0
    
    # Variables: x1[0..15], x2[0..15], p[0..15]
    # Total 48 variables
    c = np.zeros(48)
    for i in range(16):
        c[32 + i] = -probs[i]  # Maximize sum probs[i]*p[i] -> minimize negative
    
    # Build constraints
    A_ub_list = []
    b_ub_list = []
    
    # IC constraints: 16*16 = 256
    for i_true in range(16):
        v1_true, v2_true = reports[i_true]
        for i_rep in range(16):
            v1_rep, v2_rep = reports[i_rep]
            row = np.zeros(48)
            row[i_rep] = v1_rep
            row[16 + i_rep] = v2_rep
            row[32 + i_rep] = -1.0
            row[i_true] = -v1_true
            row[16 + i_true] = -v2_true
            row[32 + i_true] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(0.0)
    
    # IR constraints: 16
    for i in range(16):
        v1, v2 = reports[i]
        row = np.zeros(48)
        row[i] = -v1
        row[16 + i] = -v2
        row[32 + i] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(0.0)
    
    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    
    # Bounds
    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            x1 = result.x[0:16]
            x2 = result.x[16:32]
            p = result.x[32:48]
            allocations = np.column_stack((x1, x2))
            payments = p
        else:
            raise Exception("LP failed")
    except Exception:
        # Fallback: simple threshold mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for i, (v1, v2) in enumerate(reports):
            if v1 >= 2:
                allocations[i, 0] = 1.0
            if v2 >= 2:
                allocations[i, 1] = 1.0
            payments[i] = (2.0 if v1 >= 2 else 0.0) + (2.0 if v2 >= 2 else 0.0)
    
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
