import math
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    var_indices = {}
    num_vars = 0
    for i, m in enumerate(moduli):
        for r in range(m):
            var_indices[(i, r)] = num_vars
            num_vars += 1
            
    c = np.zeros(num_vars)
    for i, m in enumerate(moduli):
        val = -1.0 / m
        for r in range(m):
            c[var_indices[(i, r)]] = val
    
    # Sparse constraint building
    rows = []
    cols = []
    vals = []
    b_lb_list = []
    b_ub_list = []
    
    def add_constraint(indices_values, lb, ub):
        # indices_values: list of (col_idx, value)
        for col, val in indices_values:
            rows.append(len(b_lb_list))
            cols.append(col)
            vals.append(val)
        b_lb_list.append(lb)
        b_ub_list.append(ub)
        
    # 1. Selection constraints: sum_r y_{m, r} <= 1
    for i, m in enumerate(moduli):
        inds_vals = []
        for r in range(m):
            inds_vals.append((var_indices[(i, r)], 1.0))
        add_constraint(inds_vals, 0.0, 1.0)
        
    # 2. Pairwise constraints
    for i in range(n_moduli):
        m_i = moduli[i]
        for j in range(i + 1, n_moduli):
            m_j = moduli[j]
            g = math.gcd(m_i, m_j)
            
            for c_res in range(g):
                inds_vals = []
                for r_i in range(c_res, m_i, g):
                    inds_vals.append((var_indices[(i, r_i)], 1.0))
                for r_j in range(c_res, m_j, g):
                    inds_vals.append((var_indices[(j, r_j)], 1.0))
                add_constraint(inds_vals, 0.0, 1.0)
                
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(b_lb_list), num_vars))
    b_lb = np.array(b_lb_list)
    b_ub = np.array(b_ub_list)
    
    constraints = LinearConstraint(A, b_lb, b_ub)
    integrality = np.ones(num_vars)
    bounds = Bounds(0, 1)
    
    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status != 0:
        return _fallback_greedy()
        
    classes = []
    for i, m in enumerate(moduli):
        for r in range(m):
            if res.x[var_indices[(i, r)]] > 0.5:
                classes.append([m, r])
                break
                
    return classes

def _fallback_greedy():
    best_classes = []
    all_moduli = list(range(20, 201))
    all_moduli.sort()
    classes = []
    for m in all_moduli:
        valid_res = None
        for r in range(m):
            ok = True
            for m_j, r_j in classes:
                g = math.gcd(m, m_j)
                if r % g == r_j % g:
                    ok = False
                    break
            if ok:
                valid_res = r
                break
        if valid_res is not None:
            classes.append([m, valid_res])
    return classes
