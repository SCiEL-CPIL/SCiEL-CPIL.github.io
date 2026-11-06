# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s limit
    
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
    
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(16) / 16.0
    
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
    
    # IR constraints: -sum_j V[i,j]*x_{i,j} + p_i <= 0
    A_ir = np.zeros((16, n_vars))
    b_ir = np.zeros(16)
    for i in range(16):
        A_ir[i, 2*i] = -V[i, 0]
        A_ir[i, 2*i+1] = -V[i, 1]
        A_ir[i, 32+i] = 1
    
    # IC constraints: for all i, ip
    # sum_j V[i,j]*x_{ip,j} - p_{ip} - sum_j V[i,j]*x_{i,j} + p_i <= 0
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
    
    def verify_and_fix(allocations, payments):
        """Verify IR and IC, fix small violations, return (valid, allocations, payments)."""
        allocations = allocations.copy()
        payments = payments.copy()
        
        # Fix IR violations
        for i in range(16):
            util = np.dot(V[i], allocations[i]) - payments[i]
            if util < -1e-8:
                payments[i] = max(0.0, np.dot(V[i], allocations[i]))
        
        # Check IC and fix if needed
        max_violation = 0.0
        for i in range(16):
            for ip in range(16):
                util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                util_truth = np.dot(V[i], allocations[i]) - payments[i]
                viol = util_report - util_truth
                if viol > max_violation:
                    max_violation = viol
        
        if max_violation > 1e-8:
            # Try to fix by increasing payments on types that are being deviated to
            # For each violating pair (i, ip), we need:
            #   p_i - p_ip >= sum_j V[i,j]*(x_{ip,j} - x_{i,j})
            # We can increase p_ip or decrease p_i, but p_i >= 0 and IR must hold.
            # Simplest fix: increase p_ip for all ip that are deviated to from some i
            # This may break IR, so we iterate.
            for iteration in range(10):
                if time.time() > deadline - 5.0:
                    break
                max_violation = 0.0
                violated = False
                for i in range(16):
                    for ip in range(16):
                        util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                        util_truth = np.dot(V[i], allocations[i]) - payments[i]
                        viol = util_report - util_truth
                        if viol > 1e-8:
                            violated = True
                            # Increase p_ip by viol + small epsilon
                            payments[ip] += viol + 1e-9
                            if payments[ip] < 0:
                                payments[ip] = 0.0
                if not violated:
                    break
            
            # Re-check after fixes
            max_violation = 0.0
            for i in range(16):
                for ip in range(16):
                    util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                    util_truth = np.dot(V[i], allocations[i]) - payments[i]
                    viol = util_report - util_truth
                    if viol > max_violation:
                        max_violation = viol
            
            if max_violation > 1e-8:
                # Fix failed, return conservative mechanism
                return False, np.zeros((16, 2)), np.zeros(16)
            
            # Re-check IR after payment increases
            for i in range(16):
                util = np.dot(V[i], allocations[i]) - payments[i]
                if util < -1e-8:
                    # Reduce allocation to satisfy IR
                    # Scale down allocation proportionally
                    val = np.dot(V[i], allocations[i])
                    if val > 0:
                        scale = max(0.0, payments[i]) / val
                        if scale < 1.0:
                            allocations[i] *= scale
                    else:
                        allocations[i] = 0.0
                    payments[i] = max(0.0, np.dot(V[i], allocations[i]))
            
            # Final check
            for i in range(16):
                util = np.dot(V[i], allocations[i]) - payments[i]
                if util < -1e-8:
                    return False, np.zeros((16, 2)), np.zeros(16)
            
            for i in range(16):
                for ip in range(16):
                    util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                    util_truth = np.dot(V[i], allocations[i]) - payments[i]
                    if util_report - util_truth > 1e-8:
                        return False, np.zeros((16, 2)), np.zeros(16)
        
        return True, allocations, payments
    
    try:
        # Solve with tight tolerances
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                  'primal_feasibility_tolerance': 1e-12,
                                  'dual_edge_tolerance': 1e-12, 'primal_edge_tolerance': 1e-12})
        
        if not result.success or result.x is None:
            # Try without tight tolerances
            result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if not result.success or result.x is None:
                allocations = np.zeros((16, 2))
                payments = np.zeros(16)
                return {'allocations': allocations, 'payments': payments}
        
        x = result.x
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for i in range(16):
            allocations[i, 0] = x[2*i]
            allocations[i, 1] = x[2*i+1]
            payments[i] = x[32+i]
        
        # Clamp to valid ranges
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.maximum(payments, 0.0)
        
        valid, allocations, payments = verify_and_fix(allocations, payments)
        
        if not valid:
            # Fallback: simple valid mechanism
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
        
        return {'allocations': allocations, 'payments': payments}
    
    except Exception:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
