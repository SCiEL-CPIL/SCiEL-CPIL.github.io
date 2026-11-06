import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    limit = 85.0  # Safety margin for 90s limit
    
    n = 256
    weights = [0.0] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = 1.0 / i

    # Precompute adjacency masks for B&B
    adj_mask = [0] * (n + 1)
    for x in range(1, n + 1):
        mask = 0
        for y in range(1, n + 1):
            if x == y:
                continue
            if (x + y) % 2 == 0:
                b = (x + y) // 2
                if 1 <= b <= n:
                    mask |= (1 << (y - 1))
        adj_mask[x] = mask

    # 1. Run MILP to get a strong initial solution
    best_weight = 0.0
    best_set = set()
    
    # Generate AP constraints for MILP
    ap_constraints = []
    for a in range(1, n + 1):
        max_d = (n - a) // 2
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            ap_constraints.append((a - 1, b - 1, c - 1))
            
    num_constraints = len(ap_constraints)
    rows = []
    cols = []
    data = []
    for i, (a, b, c) in enumerate(ap_constraints):
        rows.extend([i, i, i])
        cols.extend([a, b, c])
        data.extend([1, 1, 1])
        
    A = csr_matrix((data, (rows, cols)), shape=(num_constraints, n))
    b_ub = np.full(num_constraints, 2.0)
    c_obj = np.array([-weights[i] for i in range(1, n + 1)])
    integrality = np.ones(n)
    lin_cons = LinearConstraint(A, lb=None, ub=b_ub)
    var_bounds = Bounds(lb=0, ub=1)
    
    try:
        # Use 20s for MILP
        res = milp(c=c_obj, constraints=lin_cons, integrality=integrality, bounds=var_bounds,
                   options={'time_limit': 20, 'mip_gap': 0.0})
        
        if res.x is not None:
            solution_mask = (res.x > 0.5).astype(int)
            selected_indices = [i + 1 for i in range(n) if solution_mask[i] == 1]
            
            if selected_indices:
                S_set = set(selected_indices)
                # Verify
                valid = True
                for a, b, c in ap_constraints:
                    count = sum(1 for x in [a+1, b+1, c+1] if x in S_set)
                    if count >= 3:
                        valid = False
                        break
                if valid:
                    score = sum(weights[x] for x in S_set)
                    best_weight = score
                    best_set = S_set
    except Exception:
        pass

    # Fallback to greedy if MILP failed
    if not best_set:
        S_greedy = set()
        mask_greedy = 0
        for x in range(1, n + 1):
            if adj_mask[x] & mask_greedy:
                continue
            S_greedy.add(x)
            mask_greedy |= (1 << (x - 1))
        best_weight = sum(weights[x] for x in S_greedy)
        best_set = S_greedy

    # 2. Run B&B to try to improve
    sys.setrecursionlimit(10000)

    def greedy_color_bound(mask):
        if mask == 0:
            return 0.0
        
        nodes = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length()
            nodes.append(idx)
            m ^= lsb
        
        color_classes = []
        color_of = [-1] * (n + 1)
        
        for v in nodes:
            used = set()
            neigh = adj_mask[v] & mask
            while neigh:
                lsb = neigh & -neigh
                u = lsb.bit_length()
                if color_of[u] != -1:
                    used.add(color_of[u])
                neigh ^= lsb
                
            c = 0
            while c in used:
                c += 1
                
            if c >= len(color_classes):
                color_classes.append(weights[v])
            else:
                if weights[v] > color_classes[c]:
                    color_classes[c] = weights[v]
                    
            color_of[v] = c
            
        return sum(color_classes)

    def solve(mask, current_weight, current_set):
        nonlocal best_weight, best_set
        
        if time.time() - start_time > limit:
            return
            
        if mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return
        
        bound = current_weight + greedy_color_bound(mask)
        if bound <= best_weight:
            return
        
        lsb = mask & -mask
        v = lsb.bit_length()
        
        # Branch 1: Include v
        new_mask = mask & ~(adj_mask[v] | (1 << (v - 1)))
        new_set = set(current_set)
        new_set.add(v)
        solve(new_mask, current_weight + weights[v], new_set)
        
        # Branch 2: Exclude v
        new_mask = mask & ~(1 << (v - 1))
        solve(new_mask, current_weight, current_set)

    all_mask = (1 << n) - 1
    solve(all_mask, 0.0, set())
    
    return list(best_set)
