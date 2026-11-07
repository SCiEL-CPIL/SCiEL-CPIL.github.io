"""Rewritten program: maximize edge-disjoint Hamiltonian cycles on the 21-vertex host."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _solve_for_k_cycles(n, edges, K):
    """
    Attempts to find K edge-disjoint Hamiltonian cycles using a single MILP.
    Returns a list of K cycles if found, else None.
    """
    m = len(edges)
    # Variables:
    # For each cycle c in 0..K-1:
    #   x[c, e] : 1 if edge e is in cycle c
    #   y[c, u, v] : 1 if directed arc (u,v) is in cycle c
    #   w[c, v] : position of vertex v in cycle c (MTZ)
    
    # Arcs for each edge
    arcs = []
    arc_to_edge = {}
    for i, (a, b) in enumerate(edges):
        arcs.append((a, b))
        arcs.append((b, a))
        arc_to_edge[(a, b)] = i
        arc_to_edge[(b, a)] = i
    
    A = len(arcs) # 2*m
    
    # Total variables: K * (m + A + n)
    # Indexing:
    # Var 0..K*(m+A+n)-1
    # Let's define helper indices
    def x_idx(c, e):
        return c * (m + A + n) + e
    def y_idx(c, arc):
        return c * (m + A + n) + m + arc
    def w_idx(c, v):
        return c * (m + A + n) + m + A + v
        
    total_vars = K * (m + A + n)
    
    c_obj = np.zeros(total_vars)
    integrality = np.zeros(total_vars)
    lb = np.zeros(total_vars)
    ub = np.zeros(total_vars)
    
    # Bounds and integrality for all vars
    for c in range(K):
        for e in range(m):
            idx = x_idx(c, e)
            integrality[idx] = 1
            ub[idx] = 1
        for a in range(A):
            idx = y_idx(c, a)
            integrality[idx] = 1
            ub[idx] = 1
        for v in range(n):
            idx = w_idx(c, v)
            integrality[idx] = 1
            lb[idx] = 1
            ub[idx] = n
            
    constraints = []
    
    # 1. Edge Disjointness: sum_c x[c, e] <= 1
    for e in range(m):
        row = np.zeros(total_vars)
        for c in range(K):
            row[x_idx(c, e)] = 1.0
        constraints.append(LinearConstraint(row, 0, 1))
        
    # 2. Per Cycle Constraints
    for c in range(K):
        # 2a. Degree: sum_{e incident to v} x[c, e] = 2
        for v in range(n):
            row = np.zeros(total_vars)
            for e, (a, b) in enumerate(edges):
                if a == v or b == v:
                    row[x_idx(c, e)] = 1.0
            constraints.append(LinearConstraint(row, 2, 2))
            
        # 2b. Link y to x: y[c, u, v] + y[c, v, u] = x[c, e]
        # Note: Since degree is 2 and graph is simple, this equality holds if we ensure flow conservation.
        # But we need to link them explicitly for the solver to know which edges are used.
        for e, (a, b) in enumerate(edges):
            row = np.zeros(total_vars)
            row[x_idx(c, e)] = 1.0
            # Find arc indices for (a,b) and (b,a)
            # We know arcs are added in order. 
            # For edge e, arcs are at indices 2*e and 2*e+1? 
            # Let's verify. 
            # for i, (a,b) in enumerate(edges): arcs.append((a,b)); arcs.append((b,a))
            # Yes, arc index 2*e is (a,b), 2*e+1 is (b,a).
            row[y_idx(c, 2*e)] = -1.0
            row[y_idx(c, 2*e+1)] = -1.0
            # x = y1 + y2
            # x - y1 - y2 = 0
            constraints.append(LinearConstraint(row, 0, 0))
            
        # 2c. Flow Conservation
        # For each v != 0: sum_in y - sum_out y = 0
        # For v == 0: sum_in y - sum_out y = 0 (since it's a cycle)
        for v in range(n):
            row = np.zeros(total_vars)
            for a_idx, (u, w) in enumerate(arcs):
                if w == v: # incoming
                    row[y_idx(c, a_idx)] = 1.0
                if u == v: # outgoing
                    row[y_idx(c, a_idx)] = -1.0
            constraints.append(LinearConstraint(row, 0, 0))
            
        # 2d. MTZ Subtour Elimination
        # Root = 0
        # w[c, 0] = 1
        row = np.zeros(total_vars)
        row[w_idx(c, 0)] = 1.0
        constraints.append(LinearConstraint(row, 1, 1))
        
        # For each arc (u, v): w[u] + 1 - w[v] <= (n-1) * (1 - y[u,v])
        # w[u] - w[v] + 1 - (n-1)*y <= n-2
        # w[u] - w[v] - (n-1)*y <= n-3
        for a_idx, (u, v) in enumerate(arcs):
            row = np.zeros(total_vars)
            row[w_idx(c, u)] = 1.0
            row[w_idx(c, v)] = -1.0
            row[y_idx(c, a_idx)] = -(n-1)
            constraints.append(LinearConstraint(row, -np.inf, n-2))
            
    res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=Bounds(lb, ub))
    
    if not res.success:
        return None
        
    cycles = []
    for c in range(K):
        # Extract edges for cycle c
        chosen_edges = []
        for e in range(m):
            if res.x[x_idx(c, e)] > 0.5:
                chosen_edges.append(e)
                
        if len(chosen_edges) != n:
            return None # Should not happen if degree=2 and connected, but check
            
        # Reconstruct cycle
        adj_c = [set() for _ in range(n)]
        for e in chosen_edges:
            a, b = edges[e]
            adj_c[a].add(b)
            adj_c[b].add(a)
            
        # Walk
        start = 0
        cur = start
        prev = -1
        cyc = [start]
        for _ in range(n):
            nxts = [u for u in adj_c[cur] if u != prev]
            if not nxts:
                return None
            prev, cur = cur, nxts[0]
            if cur == start:
                break
            cyc.append(cur)
            
        if len(cyc) != n:
            return None
            
        # Verify
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i+1) % len(cyc)]
            if b not in adj_c[a]:
                return None
                
        cycles.append(cyc)
        
    return cycles

def construct():
    n = P['n']
    edges = [tuple(e) for e in P['edges']]
    adj = _build_adj(n, edges)
    
    # Try to find 3 cycles first, then 2, then 1
    for K in [3, 2, 1]:
        cycles = _solve_for_k_cycles(n, edges, K)
        if cycles:
            # Ensure types
            out = [tuple(int(v) for v in c) for c in cycles]
            return out
            
    # Fallback if something went wrong (should not happen as K=1 is always possible for this graph)
    # Find any hamiltonian cycle using simple backtracking or the old method
    # For now, return empty or raise
    return []
# EVOLVE-BLOCK-END
