import numpy as np
from math import lcm
from scipy.optimize import milp, LinearConstraint, Bounds
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    # Nodes are 2 to 256
    nodes = list(range(2, 257))
    n = len(nodes)
    
    # Map node value to index
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    
    # Find non-edges (pairs where lcm <= 256)
    non_edges = []
    for i in range(n):
        u = nodes[i]
        for j in range(i + 1, n):
            v = nodes[j]
            if lcm(u, v) <= 256:
                non_edges.append((i, j))
                
    # Objective: Maximize sum(1/x). 
    # scipy milp minimizes c @ x. So c = -1/x.
    c = np.zeros(n)
    for i, v in enumerate(nodes):
        c[i] = -1.0 / v
        
    A_list = []
    b_list = []
    
    # Add non-edge constraints
    for i, j in non_edges:
        row = np.zeros(n)
        row[i] = 1
        row[j] = 1
        A_list.append(row)
        b_list.append(1)
        
    # Add clique constraints only for "maximal" N (those not dividing any other number in [2,256])
    # If N | M, then D_N subset D_M, so constraint for M implies constraint for N.
    # We only need constraints for N where no multiple of N exists in [2, 256].
    for N in range(2, 257):
        # Check if N divides any other number in [2, 256]
        has_multiple = False
        for M in range(N + 1, 257):
            if M % N == 0:
                has_multiple = True
                break
        if has_multiple:
            continue  # Skip redundant constraint
        
        row = np.zeros(n)
        # Find divisors of N in nodes
        for d in range(2, N + 1):
            if N % d == 0:
                idx = node_to_idx[d]
                row[idx] = 1
        A_list.append(row)
        b_list.append(1)
        
    A = np.array(A_list)
    b = np.array(b_list)
    
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n)
    
    constraint = LinearConstraint(A, lb=-np.inf, ub=b)
    
    # Check time before calling MILP
    if time.time() > deadline:
        # Fallback: Greedy
        selected = []
        for x in range(2, 257):
            if all(lcm(x, y) > 256 for y in selected):
                selected.append(x)
        return selected if selected else [2]
    
    try:
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    except Exception:
        res = None
    
    best_selected = None
    best_score = 0.0
    
    if res is not None and res.success:
        x = res.x
        selected = [nodes[i] for i in range(n) if x[i] > 0.5]
        if selected:
            best_selected = selected
            best_score = sum(1.0 / v for v in selected)
    else:
        # Fallback: Greedy
        selected = []
        for x in range(2, 257):
            if all(lcm(x, y) > 256 for y in selected):
                selected.append(x)
        best_selected = selected if selected else [2]
        best_score = sum(1.0 / v for v in best_selected)
    
    # Local search polish: try to improve the solution
    if time.time() < deadline - 5:
        current_set = set(best_selected)
        current_score = best_score
        improved = True
        
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding elements
            for v in range(2, 257):
                if v in current_set:
                    continue
                if all(lcm(v, u) > 256 for u in current_set):
                    new_score = current_score + 1.0 / v
                    if new_score > current_score + 1e-15:
                        current_set.add(v)
                        current_score = new_score
                        improved = True
                        if time.time() > deadline - 5:
                            break
            # Try swapping: remove one, add one or more
            if time.time() > deadline - 5:
                break
            for v in list(current_set):
                temp_set = current_set - {v}
                temp_score = current_score - 1.0 / v
                for w in range(2, 257):
                    if w in temp_set:
                        continue
                    if all(lcm(w, u) > 256 for u in temp_set):
                        new_score = temp_score + 1.0 / w
                        if new_score > current_score + 1e-15:
                            current_set = temp_set | {w}
                            current_score = new_score
                            improved = True
                            break
                if time.time() > deadline - 5:
                    break
        
        if current_score > best_score:
            best_selected = list(current_set)
            best_score = current_score
    
    if not best_selected:
        return [2]
        
    return best_selected
