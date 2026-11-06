# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 85.0

    # Parse inputs
    if isinstance(distribution_spec, dict):
        if 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            probs = np.array([distribution_spec.get(i, 0.0) for i in range(16)], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0
        
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0

    if isinstance(support, (list, tuple, np.ndarray)):
        support_list = list(support)
    else:
        support_list = [(i // 4, i % 4) for i in range(16)]
        
    if len(support_list) != 16:
        support_list = [(i // 4, i % 4) for i in range(16)]

    V = np.zeros((16, 2))
    for i, (v1, v2) in enumerate(support_list):
        V[i, 0] = v1
        V[i, 1] = v2

    n_vars = 48
    c = np.zeros(n_vars)
    for i in range(16):
        c[32 + i] = -probs[i]

    A_ir = np.zeros((16, n_vars))
    b_ir = np.zeros(16)
    for i in range(16):
        A_ir[i, 2*i] = -V[i, 0]
        A_ir[i, 2*i+1] = -V[i, 1]
        A_ir[i, 32+i] = 1

    n_ic = 16 * 16
    A_ic = np.zeros((n_ic, n_vars))
    b_ic = np.zeros(n_ic)
    
    for i in range(16):
        for ip in range(16):
            idx = i * 16 + ip
            A_ic[idx, 2*ip] += V[i, 0]
            A_ic[idx, 2*ip+1] += V[i, 1]
            A_ic[idx, 32+ip] += -1
            A_ic[idx, 2*i] += -V[i, 0]
            A_ic[idx, 2*i+1] += -V[i, 1]
            A_ic[idx, 32+i] += 1

    A = np.vstack([A_ir, A_ic])
    b = np.concatenate([b_ir, b_ic])

    bounds = []
    for i in range(16):
        bounds.append((0, 1))
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, None))

    try:
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback to a simple valid mechanism
            return {'allocations': np.ones((16, 2)), 'payments': np.zeros(16)}
        
        x = result.x
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for i in range(16):
            allocations[i, 0] = x[2*i]
            allocations[i, 1] = x[2*i+1]
            payments[i] = x[32+i]
            
        # Repair Step
        # 1. Enforce IR
        for i in range(16):
            max_p = np.dot(V[i], allocations[i])
            if payments[i] > max_p:
                payments[i] = max_p
                
        # 2. Enforce IC
        for _ in range(100):
            max_violation = 0
            worst_i = -1
            worst_ip = -1
            for i in range(16):
                for ip in range(16):
                    util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                    util_truth = np.dot(V[i], allocations[i]) - payments[i]
                    viol = util_report - util_truth
                    if viol > max_violation:
                        max_violation = viol
                        worst_i = i
                        worst_ip = ip
                        
            if max_violation < 1e-8:
                break
                
            # Try to fix worst violation
            # Increase payment of the reported type (ip)
            # Or decrease payment of the true type (i)
            
            # Option 1: Increase p[worst_ip]
            new_p_ip = payments[worst_ip] + max_violation + 1e-9
            max_allowed_p_ip = np.dot(V[worst_ip], allocations[worst_ip])
            if new_p_ip <= max_allowed_p_ip:
                payments[worst_ip] = new_p_ip
            else:
                # Option 2: Decrease p[worst_i]
                new_p_i = payments[worst_i] - max_violation - 1e-9
                if new_p_i >= 0:
                    payments[worst_i] = new_p_i
                else:
                    # Cannot fix easily, break and use fallback
                    break
                    
        # Final Check
        ic_ok = True
        for i in range(16):
            for ip in range(16):
                util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                util_truth = np.dot(V[i], allocations[i]) - payments[i]
                if util_report - util_truth > 1e-8:
                    ic_ok = False
                    break
            if not ic_ok:
                break
                
        if not ic_ok:
            # Fallback: Conservative mechanism
            # Try a few simple mechanisms and pick the best valid one
            # Mechanism 1: All zeros
            # Mechanism 2: All ones, pay 0
            # Mechanism 3: All ones, pay 1
            # We'll just return all zeros to be safe, or a simple valid one
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            
        return {'allocations': allocations, 'payments': payments}
    
    except Exception:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
