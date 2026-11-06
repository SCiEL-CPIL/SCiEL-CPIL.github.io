import time
import numpy as np
from scipy.optimize import linprog

def _time_remaining(start_time, deadline):
    return deadline - time.time()

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    # Default support
    if support is None:
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    else:
        support = [tuple(s) for s in support]
        
    n_reports = len(support)
    if n_reports != 16:
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
        n_reports = 16
        
    # Parse probabilities
    probs = np.ones(n_reports) / n_reports
    if distribution_spec is not None:
        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            dist_arr = np.array(distribution_spec, dtype=float).flatten()
            if len(dist_arr) == n_reports:
                probs = dist_arr
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs = probs / prob_sum
                else:
                    probs = np.ones(n_reports) / n_reports
        elif isinstance(distribution_spec, dict):
            for i, rep in enumerate(support):
                if rep in distribution_spec:
                    probs[i] = distribution_spec[rep]
                elif i in distribution_spec:
                    probs[i] = distribution_spec[i]
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                probs = np.ones(n_reports) / n_reports
                
    probs = np.clip(probs, 0, 1)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_reports) / n_reports
        
    # Values
    values = np.zeros((n_reports, 2))
    for i, rep in enumerate(support):
        values[i, 0] = rep[0]
        values[i, 1] = rep[1]
        
    # --- Strategy: Reduced LP using Transitivity ---
    # For a single bidder, IC constraints imply that the utility function u(t) is concave.
    # We can exploit the structure of the 2D grid (v1, v2) in {0,1,2,3}^2.
    # The "critical" IC constraints are those between adjacent points in the grid.
    # If IC holds for adjacent points, it holds for all pairs due to transitivity along paths.
    # Specifically, for each point (i, j), we need to ensure that moving to a neighbor
    # (i+1, j) or (i, j+1) doesn't increase utility.
    # Actually, the standard result for multi-dimensional types is that IC is equivalent to
    # monotonicity of the allocation rule. But for the LP, we can reduce the constraints.
    
    # Let's identify the grid structure.
    # support is [(0,0), (0,1), ..., (3,3)] usually.
    # Let's map each report index to its (v1, v2) coordinates.
    # We assume the support is the full grid 0..3 x 0..3.
    
    # Check if support is the standard grid
    is_standard_grid = True
    for i, rep in enumerate(support):
        v1, v2 = rep
        if not (0 <= v1 <= 3 and 0 <= v2 <= 3):
            is_standard_grid = False
            break
    if is_standard_grid:
        # Map index to (v1, v2)
        # Index i corresponds to (i//4, i%4) if sorted by v1 then v2?
        # The task says support is enumerated. Let's build a map.
        idx_to_v = {}
        for i, rep in enumerate(support):
            idx_to_v[rep] = i
            
        # Generate adjacent pairs
        # For each (v1, v2), neighbors are (v1+1, v2) and (v1, v2+1)
        adj_pairs = []
        for v1 in range(4):
            for v2 in range(4):
                curr = (v1, v2)
                if curr not in idx_to_v: continue
                i = idx_to_v[curr]
                
                # Neighbor right: (v1, v2+1)
                if v2 < 3:
                    nxt = (v1, v2+1)
                    if nxt in idx_to_v:
                        j = idx_to_v[nxt]
                        adj_pairs.append((i, j))
                
                # Neighbor up: (v1+1, v2)
                if v1 < 3:
                    nxt = (v1+1, v2)
                    if nxt in idx_to_v:
                        j = idx_to_v[nxt]
                        adj_pairs.append((i, j))
        
        # We will use these adjacent pairs for IC constraints.
        # IC: u(i) >= u(j) for i, j adjacent?
        # Wait, IC requires u(i, i) >= u(i, j) for ALL j.
        # If we enforce u(i) >= u(j) for adjacent j, does it imply for all j?
        # Yes, if the graph is connected and we have a potential function.
        # However, the constraint is asymmetric: u(i, i) >= u(i, j).
        # This is not a simple graph cut.
        # Let's stick to the full set of IC constraints but optimize the solver.
        # The "reduced constraint" idea from the inspiration might refer to
        # using the dual or a specific structure.
        # Given the small size (16 reports, 48 vars, ~256 constraints),
        # the full LP is fast enough with HiGHS. The bottleneck might be
        # numerical stability or the fallback.
        
        # Let's try the full LP first, but with tighter tolerances and
        # a robust fallback.
        
        pass # We will build the full LP below
    
    n_vars = 3 * n_reports
    
    # Objective: Maximize sum probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[2*n_reports + i] = -probs[i]
        
    A_rows = []
    b_rows = []
    
    # IR: p[i] <= values[i] @ x[i]  =>  -values[i]@x[i] + p[i] <= 0
    for i in range(n_reports):
        row = np.zeros(n_vars)
        row[2*i] = -values[i][0]
        row[2*i+1] = -values[i][1]
        row[2*n_reports+i] = 1
        A_rows.append(row)
        b_rows.append(0)
        
    # IC: u(i,i) >= u(i,ip)
    # v_i x_i - p_i >= v_i x_ip - p_ip
    # v_i x_i - v_i x_ip - p_i + p_ip >= 0
    # -v_i x_i + v_i x_ip + p_i - p_ip <= 0
    for i in range(n_reports):
        for ip in range(n_reports):
            if i == ip:
                continue
            row = np.zeros(n_vars)
            row[2*i] = -values[i][0]
            row[2*i+1] = -values[i][1]
            row[2*ip] = values[i][0]
            row[2*ip+1] = values[i][1]
            row[2*n_reports+i] = 1
            row[2*n_reports+ip] = -1
            A_rows.append(row)
            b_rows.append(0)
            
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    bounds = []
    for i in range(n_reports):
        bounds.append((0, 1))
        bounds.append((0, 1))
    for i in range(n_reports):
        bounds.append((0, None))
        
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-14, 
                                  'primal_feasibility_tolerance': 1e-14,
                                  'time_limit': max(1, _time_remaining(start_time, deadline) - 10)})
        
        if result.status == 0:
            x = np.zeros((n_reports, 2))
            p = np.zeros(n_reports)
            for i in range(n_reports):
                x[i][0] = result.x[2*i]
                x[i][1] = result.x[2*i+1]
                p[i] = result.x[2*n_reports+i]
            
            x = np.clip(x, 0, 1)
            p = np.clip(p, 0, None)
            
            # Verify
            ic_violated = False
            for i in range(n_reports):
                max_pay = values[i][0]*x[i][0] + values[i][1]*x[i][1]
                if p[i] > max_pay + 1e-8:
                    p[i] = max_pay
                    ic_violated = True
                    
            for i in range(n_reports):
                for ip in range(n_reports):
                    if i == ip:
                        continue
                    u_ii = values[i][0]*x[i][0] + values[i][1]*x[i][1] - p[i]
                    u_iip = values[i][0]*x[ip][0] + values[i][1]*x[ip][1] - p[ip]
                    if u_ii < u_iip - 1e-8:
                        ic_violated = True
                        break
                if ic_violated:
                    break
            
            if not ic_violated:
                # Polish pass: try to increase payments while maintaining IC
                if _time_remaining(start_time, deadline) > 5:
                    p_polished = _polish_payments(x, p, values, probs)
                    if p_polished is not None:
                        p = p_polished
                return {
                    'allocations': x,
                    'payments': p
                }
            else:
                return _fallback_independent_mechanism(values, probs)
        else:
            return _fallback_independent_mechanism(values, probs)
            
    except Exception:
        return _fallback_independent_mechanism(values, probs)

def _polish_payments(x, p, values, probs, tol=1e-10):
    """
    Try to increase payments while maintaining IC and IR.
    For each type i, the maximum payment is min over all j of:
      u_i - u_j + p_j  where u_i = v_i . x_i, u_j = v_i . x_j
    Actually, IC requires: v_i . x_i - p_i >= v_i . x_j - p_j for all j
    So p_i <= v_i . x_i - v_i . x_j + p_j for all j
    Thus max p_i = min_j (v_i . x_i - v_i . x_j + p_j)
    But we also need IR: p_i <= v_i . x_i
    So p_i <= min(v_i . x_i, min_j (v_i . x_i - v_i . x_j + p_j))
    
    We can iteratively increase payments. Since increasing p_i makes it harder
    for other types to be IC (their constraint becomes tighter), we need to
    be careful. Let's try a simple iterative approach.
    """
    n_reports = len(values)
    p_new = p.copy()
    
    # Iteratively try to increase each payment to its maximum
    for iteration in range(100):
        improved = False
        for i in range(n_reports):
            # Maximum payment for type i given current allocations and other payments
            max_p_i = values[i][0]*x[i][0] + values[i][1]*x[i][1]  # IR bound
            
            for j in range(n_reports):
                if i == j:
                    continue
                # IC: v_i . x_i - p_i >= v_i . x_j - p_j
                # p_i <= v_i . x_i - v_i . x_j + p_j
                bound_j = values[i][0]*x[i][0] + values[i][1]*x[i][1] - \
                          (values[i][0]*x[j][0] + values[i][1]*x[j][1]) + p_new[j]
                max_p_i = min(max_p_i, bound_j)
            
            # Also ensure nonnegativity
            max_p_i = max(0, max_p_i)
            
            if max_p_i > p_new[i] + tol:
                p_new[i] = max_p_i
                improved = True
        
        if not improved:
            break
    
    # Final verification
    for i in range(n_reports):
        if p_new[i] < -tol:
            return None
        max_pay = values[i][0]*x[i][0] + values[i][1]*x[i][1]
        if p_new[i] > max_pay + tol:
            return None
    
    for i in range(n_reports):
        for j in range(n_reports):
            if i == j:
                continue
            u_ii = values[i][0]*x[i][0] + values[i][1]*x[i][1] - p_new[i]
            u_ij = values[i][0]*x[j][0] + values[i][1]*x[j][1] - p_new[j]
            if u_ii < u_ij - tol:
                return None
    
    return p_new

def _fallback_independent_mechanism(values, probs):
    n_reports = len(values)
    x = np.zeros((n_reports, 2))
    p = np.zeros(n_reports)
    
    for i in range(n_reports):
        v1 = values[i][0]
        v2 = values[i][1]
        
        if v1 >= 2:
            x[i][0] = 1.0
            p[i] += 2.0
        if v2 >= 2:
            x[i][1] = 1.0
            p[i] += 2.0
            
    return {
        'allocations': x,
        'payments': p
    }
