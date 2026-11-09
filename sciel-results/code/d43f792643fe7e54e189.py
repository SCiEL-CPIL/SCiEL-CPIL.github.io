import time
import math
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    # Squares up to 2*N
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Filter self-conflicts
    valid_nodes = [v for v in range(1, N + 1) if 2 * v not in squares]
    M = len(valid_nodes)
    
    if M == 0:
        return [1]
    
    # Build conflict graph as adjacency matrix
    # adj[i][j] = 1 if valid_nodes[i] and valid_nodes[j] conflict
    adj = np.zeros((M, M), dtype=np.int8)
    for i in range(M):
        for j in range(i + 1, M):
            if (valid_nodes[i] + valid_nodes[j]) in squares:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # ILP formulation for Maximum Independent Set:
    # Maximize sum(x_i)
    # Subject to: x_i + x_j <= 1 for all edges (i,j)
    #             x_i in {0, 1}
    
    # Objective: maximize sum(x) => minimize -sum(x)
    c = np.ones(M)
    
    # Constraints: x_i + x_j <= 1 for each edge
    A_ub = []
    b_ub = []
    for i in range(M):
        for j in range(i + 1, M):
            if adj[i, j] == 1:
                row = np.zeros(M)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(1)
    
    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = np.zeros((1, M))
        b_ub = np.array([1])
    
    # Bounds: 0 <= x_i <= 1
    bounds = [(0, 1)] * M
    
    # Solve LP relaxation first
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0 and res.x is not None:
            # Round the LP solution
            x_lp = res.x
            # Get indices sorted by LP value descending
            order = np.argsort(-x_lp)
            
            # Greedy rounding: include nodes in order of LP value if feasible
            selected = []
            selected_set = set()
            for idx in order:
                if x_lp[idx] < 0.5:
                    continue
                # Check if this node conflicts with any already selected
                conflict = False
                for sel in selected_set:
                    if adj[idx, sel] == 1:
                        conflict = True
                        break
                if not conflict:
                    selected.append(idx)
                    selected_set.add(idx)
            
            if selected:
                result = [valid_nodes[i] for i in selected]
                return result
    
    except Exception:
        pass
    
    # Fallback: greedy independent set
    def greedy_is():
        remaining = set(range(M))
        selected = []
        
        while remaining:
            # Pick node with minimum degree in remaining
            best_node = None
            best_deg = float('inf')
            for node in remaining:
                deg = sum(adj[node, other] for other in remaining if other != node)
                if deg < best_deg:
                    best_deg = deg
                    best_node = node
            
            selected.append(best_node)
            remaining.remove(best_node)
            # Remove neighbors of best_node
            to_remove = set()
            for other in remaining:
                if adj[best_node, other] == 1:
                    to_remove.add(other)
            remaining -= to_remove
        
        return selected
    
    selected = greedy_is()
    result = [valid_nodes[i] for i in selected]
    
    # Verify the solution
    squares_check = set()
    i = 2
    while i * i <= 2 * N:
        squares_check.add(i * i)
        i += 1
    
    for a in result:
        for b in result:
            if a <= b:
                if (a + b) in squares_check:
                    # Invalid - should not happen, but fallback
                    return [1]
    
    return result
