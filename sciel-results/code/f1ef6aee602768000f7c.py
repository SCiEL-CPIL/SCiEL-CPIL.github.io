import numpy as np
import time
import random
import scipy.optimize as opt
import scipy.sparse as sp

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0

    # Generate APs
    ap_indices = []
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            terms = list(range(start - 1, n, step))
            if len(terms) >= 2:
                ap_indices.append(np.array(terms, dtype=np.int32))
                
    num_aps = len(ap_indices)
    
    # Build sparse constraint matrix A and bounds b
    # Variables x_j are 0 or 1.
    # Constraint: -D <= sum(signs) <= D
    # sum(signs) = 2 * sum(x_j) - len(AP)
    # -D <= 2 * sum(x_j) - len(AP) <= D
    # len(AP) - D <= 2 * sum(x_j) <= len(AP) + D
    # (len(AP) - D) / 2 <= sum(x_j) <= (len(AP) + D) / 2
    
    # Since sum(x_j) is integer, we can tighten bounds.
    # Lower bound: ceil((len - D) / 2)
    # Upper bound: floor((len + D) / 2)
    
    # We will iterate D from 1 upwards.
    
    # Precompute lengths
    lengths = np.array([len(ap) for ap in ap_indices], dtype=np.int32)
    
    # Build sparse matrix A (num_aps x n)
    # Each row corresponds to an AP, with 1s at the indices in that AP.
    rows = []
    cols = []
    data = []
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            rows.append(ap_id)
            cols.append(i)
            data.append(1)
            
    A = sp.csr_matrix((data, (rows, cols)), shape=(num_aps, n))
    
    # Variables bounds
    # x_j in [0, 1]
    bounds = [(0, 1)] * n
    
    # Integrality
    integrality = [1] * n
    
    # Objective: minimize 0
    c = np.zeros(n)
    
    best_signs = None
    best_D = n + 1
    
    # Try D from 1 to 10 (or higher if needed)
    # If D is too small, infeasible.
    # If D is feasible, we have a solution with max sum <= D.
    
    for D in range(1, 15):
        if time.time() > deadline - 5.0:
            break
            
        # Calculate lb and ub for each AP
        lb = np.ceil((lengths - D) / 2.0).astype(np.int32)
        ub = np.floor((lengths + D) / 2.0).astype(np.int32)
        
        # Ensure lb <= ub, otherwise infeasible for this D
        if np.any(lb > ub):
            continue
            
        # milp requires bounds as list of tuples or array
        # A @ x <= ub
        # A @ x >= lb  =>  -A @ x <= -lb
        
        # Combine into single inequality matrix
        # A_aug = [A; -A]
        # b_aug = [ub; -lb]
        
        A_aug = sp.vstack([A, -A], format='csr')
        b_aug = np.concatenate([ub, -lb])
        
        # Call milp
        try:
            res = opt.milp(
                c,
                constraints=opt.LinearConstraint(A_aug, ub=b_aug),
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": 5.0, "mip_rel_gap": 0.0}
            )
        except Exception:
            break
            
        if res.status == 0: # Optimal (feasible)
            x = res.x
            # Map x (0/1) to signs (-1/+1)
            signs = np.where(x > 0.5, 1, -1).astype(np.int8)
            best_signs = signs
            best_D = D
            # Since we iterate D from 1 up, the first feasible D is the minimum.
            # We can break here if we are confident D is minimal.
            # However, due to numerical issues or solver specifics, we might want to check D-1? 
            # No, we started from 1. So D=1 is the best possible.
            break
        else:
            # Infeasible or error, try next D
            continue
            
    # If no solution found with MILP, fall back to a quick local search
    if best_signs is None:
        # Simple random + greedy polish
        rng = random.Random(42)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        
        # Greedy improvement
        # Compute AP sums
        ap_sums = np.array([np.sum(signs[ap]) for ap in ap_indices], dtype=np.int32)
        current_max = int(np.max(np.abs(ap_sums)))
        
        # Precompute pos_to_aps
        pos_to_aps = [[] for _ in range(n)]
        for ap_id, idx in enumerate(ap_indices):
            for i in idx:
                pos_to_aps[i].append(ap_id)
                
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 1.0:
                    break
                old_sign = signs[i]
                new_sign = -old_sign
                delta = 2 * new_sign
                
                # Update sums
                affected = pos_to_aps[i]
                new_sums = ap_sums.copy()
                for ap_id in affected:
                    new_sums[ap_id] += delta
                
                new_max = int(np.max(np.abs(new_sums)))
                if new_max < current_max:
                    signs[i] = new_sign
                    ap_sums = new_sums
                    current_max = new_max
                    improved = True
        
        best_signs = signs
        best_D = current_max

    if best_signs is None:
        rng = random.Random(42)
        return [rng.choice([-1, 1]) for _ in range(n)]
        
    return [int(s) for s in best_signs]
