import time
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

def construct():
    deadline = time.time() + 55
    
    N = 256
    nodes = list(range(2, N + 1))
    n_vars = len(nodes)
    
    # Weights
    # Minimize -sum(1/x)
    c = -np.array([1.0 / x for x in nodes])
    
    # Build conflict matrix
    # conflict[i][j] = True if i and j cannot be in the same subset
    # We need x_i + x_j <= 1 for each conflict
    
    rows = []
    cols = []
    vals = []
    constraint_idx = 0
    
    # Precompute gcds or just compute on fly
    # 255*255/2 = ~32k iterations. gcd is fast.
    
    for i_idx in range(n_vars):
        i = nodes[i_idx]
        for j_idx in range(i_idx + 1, n_vars):
            j = nodes[j_idx]
            # Check if lcm(i, j) <= 256
            # i * j <= 256 * gcd(i, j)
            g = gcd(i, j)
            if i * j <= 256 * g:
                # Conflict
                rows.append(constraint_idx)
                cols.append(i_idx)
                vals.append(1.0)
                
                rows.append(constraint_idx)
                cols.append(j_idx)
                vals.append(1.0)
                
                constraint_idx += 1
                
    num_constraints = constraint_idx
    
    if num_constraints == 0:
        return nodes
        
    A = sp.csr_matrix((vals, (rows, cols)), shape=(num_constraints, n_vars))
    
    lb = np.zeros(num_constraints)
    ub = np.ones(num_constraints)
    
    constraint = LinearConstraint(A, lb, ub)
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars)
    
    best_subset = []
    best_value = 0.0
    
    # First, try greedy to get a good initial solution
    greedy_set = []
    used = set()
    # Sort nodes by 1/x descending (x ascending)
    for i in nodes:
        ok = True
        for y in greedy_set:
            if i * y <= 256 * gcd(i, y):
                ok = False
                break
        if ok:
            greedy_set.append(i)
    
    if greedy_set:
        best_subset = greedy_set
        best_value = sum(1.0 / x for x in greedy_set)
    
    # Now try MILP with warm start from greedy
    try:
        # Create initial solution from greedy
        x0 = np.zeros(n_vars)
        for i in greedy_set:
            idx = nodes.index(i)
            x0[idx] = 1.0
        
        res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds,
                   x0=x0, options={"time_limit": 50})
        
        if res.status == 0:
            x = res.x
            subset = [nodes[i] for i in range(n_vars) if x[i] > 0.5]
            if subset:
                val = sum(1.0 / x for x in subset)
                if val > best_value:
                    best_subset = subset
                    best_value = val
            else:
                # Keep greedy if MILP returns empty
                pass
        else:
            # Fallback to greedy if solver fails or is suboptimal
            # Greedy: sort by weight (descending), add if compatible
            pass
            
    except Exception:
        # Fallback to greedy
        pass
    
    # Local search refinement: try to improve by swapping or adding
    if best_subset:
        best_set = set(best_subset)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try adding each element not in the set
            for i in nodes:
                if i in best_set:
                    continue
                ok = True
                for y in best_set:
                    if i * y <= 256 * gcd(i, y):
                        ok = False
                        break
                if ok:
                    best_set.add(i)
                    improved = True
                    break
            
            # Try removing one element and adding two others
            for remove in list(best_set):
                temp_set = best_set - {remove}
                for i in nodes:
                    if i in temp_set:
                        continue
                    ok1 = True
                    for y in temp_set:
                        if i * y <= 256 * gcd(i, y):
                            ok1 = False
                            break
                    if not ok1:
                        continue
                    # Check if i is compatible with remove
                    if i * remove <= 256 * gcd(i, remove):
                        continue
                    for j in nodes:
                        if j in temp_set or j == i:
                            continue
                        ok2 = True
                        for y in temp_set:
                            if j * y <= 256 * gcd(j, y):
                                ok2 = False
                                break
                        if not ok2:
                            continue
                        if i * j <= 256 * gcd(i, j):
                            continue
                        # Check new set value
                        new_val = sum(1.0 / x for x in temp_set) + 1.0/i + 1.0/j
                        old_val = best_value
                        if new_val > old_val + 1e-12:
                            best_set = temp_set | {i, j}
                            best_value = new_val
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            best_subset = list(best_set)
    
    return best_subset
