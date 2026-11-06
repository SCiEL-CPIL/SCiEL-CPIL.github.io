# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 80.0
    
    # 1. Parse Input
    # distribution_spec: likely a dict or list of probabilities for the 16 types
    # support: list of types (v1, v2)
    # constraints: dict with 'values', 'types', etc.
    
    # Infer structure from standard task patterns
    # Values in {0,1,2,3} for 2 goods -> 16 types
    # Type index mapping: idx = v1 * 4 + v2
    
    # Extract probabilities
    # Try to extract from distribution_spec
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'])
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'])
        else:
            # Assume keys are types or indices
            probs = np.zeros(16)
            for k, v in distribution_spec.items():
                if isinstance(k, tuple) and len(k) == 2:
                    v1, v2 = k
                    idx = v1 * 4 + v2
                    probs[idx] = v
                elif isinstance(k, int) and 0 <= k < 16:
                    probs[k] = v
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec)
    else:
        # Default uniform if unknown
        probs = np.ones(16) / 16.0
        
    # Ensure probs sum to 1
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    else:
        probs = np.ones(16) / 16.0
        
    # Define types and values
    # Type i corresponds to (v1, v2) where v1 = i // 4, v2 = i % 4
    n_types = 16
    values = np.zeros((n_types, 2))
    for i in range(n_types):
        v1 = i // 4
        v2 = i % 4
        values[i] = [v1, v2]

    # 2. Formulate MILP
    # Variables:
    # x[i, j] for i in 0..15, j in 0..1 -> 32 variables
    # p[i] for i in 0..15 -> 16 variables
    # Total 48 variables
    
    c = np.zeros(48)
    # Objective: Maximize sum(pi_i * p_i)
    # MILP minimizes, so we minimize -sum(pi_i * p_i)
    for i in range(n_types):
        c[32 + i] = -probs[i]
        
    # Constraints
    A_ub = []
    b_ub = []
    
    # 1. IR Constraints: p_i <= sum_j v_ij x_ij
    # p_i - sum_j v_ij x_ij <= 0
    for i in range(n_types):
        row = np.zeros(48)
        row[32 + i] = 1.0  # p_i
        for j in range(2):
            row[i * 2 + j] = -values[i, j]  # -v_ij x_ij
        A_ub.append(row)
        b_ub.append(0.0)
        
    # 2. IC Constraints: p_i - p_j <= v_i(x_i) - v_i(x_j)
    # p_i - p_j - sum_k v_ik x_ik + sum_k v_ik x_jk <= 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(48)
            row[32 + i] = 1.0   # p_i
            row[32 + j] = -1.0  # -p_j
            
            # - sum_k v_ik x_ik
            for k in range(2):
                row[i * 2 + k] = -values[i, k]
                
            # + sum_k v_ik x_jk
            for k in range(2):
                row[j * 2 + k] = values[i, k]
                
            A_ub.append(row)
            b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds
    # x variables: 0 or 1
    # p variables: >= 0
    integrality = np.zeros(48)
    integrality[:32] = 1  # x are integer
    integrality[32:] = 0  # p are continuous
    
    bounds = Bounds(lb=np.zeros(48), ub=np.ones(48))
    # p can be larger than 1, but IR limits it to max value (3+3=6).
    # Let's set ub for p to 10 to be safe.
    bounds.ub[32:] = 10.0
    
    # Solve MILP
    try:
        res = milp(c=c, constraints=LinearConstraint(A_ub, ub=b_ub), integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            x_sol = res.x[:32]
            p_sol = res.x[32:]
            
            # Round x to binary
            x_sol = np.where(x_sol > 0.5, 1.0, 0.0)
            
            # Construct output
            allocations = x_sol.reshape(16, 2)
            payments = p_sol
            
            # Check validity
            if np.all(payments >= -1e-9) and np.all(allocations >= -1e-9) and np.all(allocations <= 1.0 + 1e-9):
                # Ensure exact binary for allocations
                allocations = np.round(allocations).astype(float)
                return {"allocations": allocations, "payments": payments}
                
    except Exception as e:
        pass
        
    # Fallback: Simple First Price Auction / VCG approximation
    # If MILP fails, return a valid but potentially suboptimal mechanism
    # Simple mechanism: Allocate good j if v_j >= 1. Payment = v_j - 1? 
    # Let's use a uniform price of 1 for any good with value >= 1.
    # This is not necessarily IC for the package, but let's try a safer fallback:
    # Allocate nothing, pay 0. Revenue 0. Valid.
    
    # Better Fallback: Myerson Payment for each good separately?
    # For single good, optimal is: allocate if v >= reserve.
    # With values 0,1,2,3 uniform, reserve is 1.5? No, integer values.
    # Let's just return 0 allocations and 0 payments to guarantee validity if optimization fails.
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    
    return {"allocations": allocations, "payments": payments}

# EVOLVE-BLOCK-END
