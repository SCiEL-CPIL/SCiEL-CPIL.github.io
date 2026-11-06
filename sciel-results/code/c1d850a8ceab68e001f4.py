import numpy as np
from math import lcm
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
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
    # Scale to integers to be safe with solver precision, though float is often fine.
    # 1/256 is approx 0.0039. 
    # Let's use float for simplicity, as n is small.
    c = np.zeros(n)
    for i, v in enumerate(nodes):
        c[i] = -1.0 / v
        
    # Constraints: x_i + x_j <= 1 for all non-edges
    # If no non-edges, we select all? 
    # But wait, if graph is complete, we select all.
    # If there are non-edges, we add constraints.
    
    A = np.zeros((len(non_edges), n))
    b = np.ones(len(non_edges))
    
    for k, (i, j) in enumerate(non_edges):
        A[k, i] = 1
        A[k, j] = 1
        
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integer (0 or 1)
    integrality = np.ones(n)
    
    # Linear constraints: A @ x <= b
    # scipy.optimize.milp uses LinearConstraint(A, lb, ub)
    # Here we have A @ x <= b, so lb=-inf, ub=b
    constraint = LinearConstraint(A, lb=-np.inf, ub=b)
    
    # Solve
    res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    
    if not res.success:
        # Fallback: Greedy (should not happen)
        selected = []
        for x in range(2, 257):
            if all(lcm(x, y) > 256 for y in selected):
                selected.append(x)
        return selected
        
    # Extract solution
    x = res.x
    selected = [nodes[i] for i in range(n) if x[i] > 0.5]
    
    # Ensure non-empty (problem states non-empty subset)
    if not selected:
        # The node with max weight (smallest number) is 2. 
        # 2 is always a valid singleton.
        return [2]
        
    return selected
