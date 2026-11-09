import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = 60.0
    
    # 1. Parse Input
    # support is a list of (v1, v2) or similar. Let's assume standard grid.
    # The task says: "one additive bidder, two goods, each value in {0,1,2,3}"
    # So support is likely the full grid of 16 points.
    # distribution_spec likely contains the probabilities.
    
    # Let's infer the structure of distribution_spec.
    # Common formats: dict {(v1, v2): prob}, or array of shape (4,4).
    # Let's handle robustly.
    
    # Construct the full support grid
    values = [0, 1, 2, 3]
    support_grid = [(v1, v2) for v1 in values for v2 in values]
    n_states = len(support_grid)
    
    # Map state to index
    state_to_idx = {s: i for i, s in enumerate(support_grid)}
    
    # Extract probabilities
    probs = np.zeros(n_states)
    if isinstance(distribution_spec, dict):
        for s, p in distribution_spec.items():
            if s in state_to_idx:
                probs[state_to_idx[s]] = p
    elif isinstance(distribution_spec, np.ndarray):
        # Assume shape (4,4)
        if distribution_spec.shape == (4, 4):
            for i in range(4):
                for j in range(4):
                    idx = state_to_idx[(i, j)]
                    probs[idx] = distribution_spec[i, j]
        else:
            # Flatten and assume order matches support_grid
            flat = distribution_spec.flatten()
            if len(flat) == n_states:
                probs = flat
    else:
        # Fallback: Uniform
        probs = np.ones(n_states) / n_states
        
    # Normalize probabilities
    total_prob = probs.sum()
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(n_states) / n_states
        
    # 2. Set up LP
    # Variables:
    # x1[0..15], x2[0..15], p[0..15]
    # Total 48 variables.
    # Objective: Maximize sum(probs[i] * p[i])
    # Linprog minimizes, so minimize -sum(probs[i] * p[i])
    
    c = np.zeros(3 * n_states)
    for i in range(n_states):
        c[2 * n_states + i] = -probs[i]
        
    # Constraints:
    # A_ub @ x <= b_ub
    
    A_ub = []
    b_ub = []
    
    # IC Constraints:
    # For each pair (v, v'), p_v - p_v' <= v1*x1_v + v2*x2_v - v1*x1_v' - v2*x2_v'
    # p_v - p_v' - v1*x1_v - v2*x2_v + v1*x1_v' + v2*x2_v' <= 0
    # Indices:
    # x1_v is at index v
    # x2_v is at index n_states + v
    # p_v is at index 2 * n_states + v
    
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        for j in range(n_states):
            v1_j, v2_j = support_grid[j]
            row = np.zeros(3 * n_states)
            # p_i - p_j
            row[2 * n_states + i] = 1
            row[2 * n_states + j] = -1
            # - v1_i * x1_i - v2_i * x2_i
            row[i] = -v1_i
            row[n_states + i] = -v2_i
            # + v1_j * x1_j + v2_j * x2_j
            row[j] = v1_j
            row[n_states + j] = v2_j
            
            A_ub.append(row)
            b_ub.append(0.0)
            
    # IR Constraints:
    # p_i <= v1_i * x1_i + v2_i * x2_i
    # p_i - v1_i * x1_i - v2_i * x2_i <= 0
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        row = np.zeros(3 * n_states)
        row[2 * n_states + i] = 1
        row[i] = -v1_i
        row[n_states + i] = -v2_i
        A_ub.append(row)
        b_ub.append(0.0)
        
    # Non-negativity of payments is handled by bounds.
    # Allocations are in [0, 1].
    
    # Bounds:
    # x1 in [0, 1]
    # x2 in [0, 1]
    # p in [0, inf)
    
    bounds = []
    for i in range(n_states):
        bounds.append((0, 1)) # x1
    for i in range(n_states):
        bounds.append((0, 1)) # x2
    for i in range(n_states):
        bounds.append((0, None)) # p
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Check time
    if time.time() - start_time > deadline - 5:
        # Return a simple baseline
        allocations = np.zeros((n_states, 2))
        payments = np.zeros(n_states)
        return {"allocations": allocations, "payments": payments}
        
    # Solve
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception as e:
        # Fallback
        allocations = np.zeros((n_states, 2))
        payments = np.zeros(n_states)
        return {"allocations": allocations, "payments": payments}
        
    if res.status != 0:
        # Fallback
        allocations = np.zeros((n_states, 2))
        payments = np.zeros(n_states)
        return {"allocations": allocations, "payments": payments}
        
    x_opt = res.x
    x1 = x_opt[:n_states]
    x2 = x_opt[n_states:2*n_states]
    p = x_opt[2*n_states:]
    
    # Construct output arrays
    allocations = np.column_stack((x1, x2))
    payments = p
    
    # Clean up numerical noise
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Verify Regret (should be 0 for IC)
    # We trust the LP solution satisfies IC within tolerance.
    
    return {"allocations": allocations, "payments": payments}
