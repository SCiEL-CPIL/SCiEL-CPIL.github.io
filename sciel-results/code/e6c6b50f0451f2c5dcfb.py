import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    N = 256
    
    # 1. Build the conflict graph
    # We need to find all pairs (i, j) with i < j such that (i*j) % (i+j) == 0
    edges = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                edges.append((i - 1, j - 1))  # 0-indexed for MILP
    
    # 2. Formulate MILP
    # Maximize sum(x_i) subject to x_i + x_j <= 1 for all edges (i, j)
    # x_i in {0, 1}
    
    c = -np.ones(N)  # Minimize negative sum to maximize sum
    
    if edges:
        A = np.zeros((len(edges), N))
        b = np.ones(len(edges))
        for k, (u, v) in enumerate(edges):
            A[k, u] = 1
            A[k, v] = 1
        constraints = LinearConstraint(A, ub=b)
    else:
        # No constraints, select all
        constraints = LinearConstraint(np.zeros((0, N)), ub=np.zeros(0))
    
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(N)  # All variables are integer
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.success:
            # Extract solution
            # res.x is the array of x values
            selected = [i + 1 for i in range(N) if res.x[i] > 0.5]
            if selected:
                return selected
    except Exception:
        pass
    
    # Fallback: Greedy Independent Set on the full graph
    # This is a robust fallback if MILP fails or times out
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    remaining = set(range(N))
    selected_nodes = []
    
    # Greedy: pick node with min degree in remaining
    while remaining:
        # Find node with min degree
        min_deg_node = min(remaining, key=lambda x: len(adj[x] & remaining))
        selected_nodes.append(min_deg_node + 1)
        # Remove node and its neighbors from remaining
        remaining.discard(min_deg_node)
        for neighbor in adj[min_deg_node]:
            remaining.discard(neighbor)
    
    return selected_nodes
