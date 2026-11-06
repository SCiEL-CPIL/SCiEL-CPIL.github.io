"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import lcm
    from fractions import Fraction
    from itertools import combinations
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    start_time = time.time()
    deadline = start_time + 85.0

    # Nodes are integers 2..256
    nodes = list(range(2, 257))
    N = len(nodes)
    
    # Weights are 1/x
    weights = [Fraction(1, x) for x in nodes]
    
    # Build the complement graph adjacency: 
    # Edge (i, j) in complement graph exists if lcm(nodes[i], nodes[j]) <= 256.
    # We want an Independent Set in this complement graph.
    # This is equivalent to a Clique in the original graph (lcm > 256).
    
    # List of edges in the complement graph (conflict edges)
    conflict_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if lcm(nodes[i], nodes[j]) <= 256:
                conflict_edges.append((i, j))
                
    # If there are no conflict edges, we can pick all nodes.
    # But usually there are many.
    
    # Formulate MILP
    # Variables: x_i for i in 0..N-1
    # Maximize sum(w_i * x_i)
    # Subject to:
    #   x_i + x_j <= 1 for all (i,j) in conflict_edges
    #   x_i in {0, 1}
    
    # Objective coefficients (to be maximized, so we negate for minimization if needed, 
    # but scipy.optimize.milp minimizes by default. 
    # So we minimize -sum(w_i * x_i).
    # Note: weights are Fractions. MILP requires numeric types. 
    # Using float might lose precision for the "exact" requirement? 
    # The task says "Maximize exact reciprocal sum". 
    # However, the evaluator measures the raw value. 
    # If we use floats, we might pick a suboptimal set if two sets have very close sums.
    # Given N=255 and weights 1/x, the sums are distinct rationals.
    # The difference between any two distinct sums of reciprocals of integers <= 255 is at least 1/(255!)? No, much larger.
    # The minimum difference between two distinct subsets sums is likely > 1e-15.
    # Float64 has ~15-17 decimal digits of precision.
    # Sum of 255 terms of ~1/2 to 1/256 is ~2.
    # Precision should be sufficient to distinguish the optimal set.
    
    # To be safer, we can scale weights to integers? 
    # LCM(1..255) is huge. 
    # Let's stick to floats for the solver, but verify the final set.
    
    c = np.array([-float(w) for w in weights])
    
    # Constraints matrix A for A @ x <= b
    # Each edge (i, j) gives a row: x_i + x_j <= 1
    A = np.zeros((len(conflict_edges), N))
    b = np.ones(len(conflict_edges))
    
    for k, (i, j) in enumerate(conflict_edges):
        A[k, i] = 1.0
        A[k, j] = 1.0
        
    # Bounds: x_i in [0, 1]
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integer (binary)
    integrality = np.ones(N)
    
    # Solve
    res = milp(
        c=c,
        constraints=LinearConstraint(A, ub=b),
        integrality=integrality,
        bounds=bounds
    )
    
    if not res.success:
        # Fallback to greedy if MILP fails
        # Greedy: sort by weight descending (x ascending), pick if no conflict with selected
        # Conflict if lcm <= 256
        selected = []
        for x in nodes:
            idx = nodes.index(x)
            compatible = True
            for y in selected:
                if lcm(x, y) <= 256:
                    compatible = False
                    break
            if compatible:
                selected.append(x)
        return selected

    # Extract solution
    x = res.x
    # x should be close to 0 or 1
    selected_indices = [i for i in range(N) if x[i] > 0.5]
    
    if not selected_indices:
        # Fallback to single best node (smallest x)
        return [2]
        
    selected_nodes = [nodes[i] for i in selected_indices]
    
    # Verify the solution is valid (just in case)
    for i in range(len(selected_nodes)):
        for j in range(i+1, len(selected_nodes)):
            if lcm(selected_nodes[i], selected_nodes[j]) <= 256:
                # Invalid solution from solver? Should not happen.
                # Fallback to greedy
                selected = []
                for x in nodes:
                    compatible = True
                    for y in selected:
                        if lcm(x, y) <= 256:
                            compatible = False
                            break
                    if compatible:
                        selected.append(x)
                return selected
                
    return selected_nodes
# EVOLVE-BLOCK-END
