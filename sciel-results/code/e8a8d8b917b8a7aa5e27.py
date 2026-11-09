import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    N = 256
    max_sum = 2 * N
    
    # Precompute squares
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    # Filter nodes that violate the diagonal condition (2a is square)
    # If 2a is a square, a cannot be in the set.
    valid_nodes = []
    node_to_idx = {}
    for v in range(1, N + 1):
        if 2 * v not in squares:
            node_to_idx[v] = len(valid_nodes)
            valid_nodes.append(v)
            
    M = len(valid_nodes)
    
    # Build conflict graph
    # adjacency[i] is a set of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    # We only need i < j for the constraints, but let's build full adjacency for ease of checking in local search
    adjacency = [set() for _ in range(M)]
    
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            if vi + vj in squares:
                adjacency[i].add(j)
                adjacency[j].add(i)
                
    # Try to solve using scipy.optimize.milp
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Objective: maximize sum(x_i) -> minimize -sum(x_i)
        c = np.ones(M)
        
        # Constraints: for each edge (i, j), x_i + x_j <= 1
        # This can be formulated as a matrix A where A[k, i] = 1 and A[k, j] = 1 for each edge k
        # Number of edges
        edges = []
        for i in range(M):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
                    
        num_edges = len(edges)
        
        if num_edges > 0:
            A = np.zeros((num_edges, M))
            b_ub = np.ones(num_edges)
            for k, (i, j) in enumerate(edges):
                A[k, i] = 1
                A[k, j] = 1
            constraints = [LinearConstraint(A, 0, b_ub)]
        else:
            constraints = None
            
        # Bounds: x_i in [0, 1]
        integrality = np.ones(M)
        bounds = Bounds(lb=0, ub=1)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            solution = res.x.astype(int)
            result_set = [valid_nodes[i] for i in range(M) if solution[i] == 1]
            if result_set:
                return result_set
            else:
                # Fallback if MILP returns empty (should not happen if valid nodes exist)
                return [valid_nodes[0]]
        else:
            # Fallback to heuristic if MILP fails
            pass
    except Exception:
        # Fallback to heuristic if scipy is not available or fails
        pass
        
    # Heuristic: Greedy + Local Search
    # Initialize with greedy: pick nodes with lowest degree first? Or random?
    # Let's try a greedy independent set:
    # Sort nodes by degree (ascending) and pick if no neighbor is already picked.
    # This is a standard heuristic for MIS.
    
    # 1. Greedy Initialization
    # We want to maximize size.
    # A common heuristic is to pick vertices with small degree first.
    # Let's create a list of (degree, index) and sort.
    degrees = [len(adjacency[i]) for i in range(M)]
    order = sorted(range(M), key=lambda x: degrees[x])
    
    x = np.zeros(M, dtype=int)
    for i in order:
        # Check if all neighbors are 0
        if all(x[j] == 0 for j in adjacency[i]):
            x[i] = 1
            
    # 2. Local Search: Try to flip 0s to 1s if possible, or swap 1s for better local density?
    # Since we just want max cardinality, we just need to find any 0 that can become 1.
    # This is basically finding a maximal independent set. The greedy above produces a maximal independent set.
    # To improve, we can try "swap" moves or restart with different randomizations.
    
    best_solution = x.copy()
    best_size = np.sum(best_solution)
    
    # Randomized Restart Local Search
    # Given N=256, we can do many restarts.
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        # Random greedy initialization
        random_order = list(range(M))
        random.shuffle(random_order)
        # Weight by degree? Or just random?
        # Let's try random order but prioritize low degree slightly?
        # Or just pure random greedy.
        
        x_curr = np.zeros(M, dtype=int)
        for i in random_order:
            if all(x_curr[j] == 0 for j in adjacency[i]):
                x_curr[i] = 1
                
        curr_size = np.sum(x_curr)
        
        # Local improvement: try to add any vertex that can be added
        # This is effectively making it a maximal independent set if it wasn't already.
        # But greedy already produces a maximal independent set.
        # To improve cardinality, we need to move away from the current local optimum.
        # We can do a "kick": remove a random vertex from the set, then re-greedy fill?
        # Or simply accept if curr_size > best_size.
        
        if curr_size > best_size:
            best_solution = x_curr.copy()
            best_size = curr_size
            
    # One more pass: try to add vertices to best_solution
    # This is already done by the greedy step which ensures maximality.
    # However, a maximal independent set is not necessarily maximum.
    # The random restarts help explore different maximal independent sets.
    
    result_set = [valid_nodes[i] for i in range(M) if best_solution[i] == 1]
    
    # Ensure non-empty
    if not result_set:
        return [valid_nodes[0]]
        
    return result_set
