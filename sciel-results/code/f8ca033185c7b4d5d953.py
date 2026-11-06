"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from math import lcm
    from scipy.optimize import milp, LinearConstraint, Bounds

    start_time = time.time()
    deadline = 50.0  # Safety margin
    
    nodes = list(range(2, 257))
    n = len(nodes)
    weights = np.array([1.0 / x for x in nodes])
    
    # Build conflict edges
    edges = []
    for i in range(n):
        ai = nodes[i]
        for j in range(i + 1, n):
            aj = nodes[j]
            if lcm(ai, aj) <= 256:
                edges.append((i, j))
                
    m = len(edges)
    
    # Fallback result (Greedy)
    greedy_selected = []
    # Sort by weight descending (smallest numbers first)
    # This is the seed logic, but we can try to optimize order?
    # Actually, just use the seed logic for fallback.
    for x in sorted(nodes, key=lambda v: -1.0/v):
        if all(lcm(x, y) > 256 for y in greedy_selected):
            greedy_selected.append(x)
            
    if not greedy_selected:
        greedy_selected = [256]

    # Try MILP
    try:
        if time.time() - start_time > deadline:
            return greedy_selected
            
        c = -weights
        
        # If no edges, return all
        if m == 0:
            return nodes

        A = np.zeros((m, n))
        for k, (i, j) in enumerate(edges):
            A[k, i] = 1
            A[k, j] = 1
            
        cons = LinearConstraint(A, lb=0, ub=1)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n)
        
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.success:
            x = res.x
            selected_indices = [i for i in range(n) if x[i] > 0.5]
            selected_nodes = [nodes[i] for i in selected_indices]
            
            # Verify
            valid = True
            for i in range(len(selected_nodes)):
                for j in range(i + 1, len(selected_nodes)):
                    if lcm(selected_nodes[i], selected_nodes[j]) <= 256:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and len(selected_nodes) > 0:
                # Check if it's better than greedy? 
                # MILP should be >= Greedy if it found global opt.
                # But if MILP is suboptimal (e.g. timeout in solver?), it might be worse.
                # However, HiGHS is exact.
                # Just return MILP result if valid.
                return selected_nodes
                
    except Exception:
        pass

    return greedy_selected
# EVOLVE-BLOCK-END
