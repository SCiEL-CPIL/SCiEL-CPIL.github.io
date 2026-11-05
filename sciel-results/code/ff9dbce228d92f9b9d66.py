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
    
    # Build adjacency sets for fast lookup
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

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
    
    best_solution = None
    best_size = 0

    # Try MILP first
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.success:
            selected = [i + 1 for i in range(N) if res.x[i] > 0.5]
            if selected:
                best_solution = selected
                best_size = len(selected)
    except Exception:
        pass

    # Fallback / Improvement: Greedy + Local Search
    # This is significantly better than simple greedy.
    
    def greedy_mis(adj, N):
        # Min-degree greedy
        remaining = set(range(N))
        selected = set()
        # Create a copy of degrees to update
        # Note: updating degrees dynamically in a set is tricky, so we recompute or use a list
        # For N=256, recomputing is fine.
        
        # Better greedy: sort by degree initially? No, dynamic min-degree is better.
        while remaining:
            # Find node with min degree in remaining
            min_deg_node = min(remaining, key=lambda x: len(adj[x] & remaining))
            selected.add(min_deg_node)
            # Remove node and its neighbors
            remaining.discard(min_deg_node)
            for neighbor in adj[min_deg_node]:
                if neighbor in remaining:
                    remaining.discard(neighbor)
        return selected

    def local_search(adj, N, current_set):
        # Try to improve the independent set
        # 1-move: Add a vertex not in set, remove its neighbors in set.
        # Gain = 1 - (number of neighbors in set). If gain > 0, accept.
        # 2-move: Swap one in for one out? Or remove one in, add two out?
        # Let's stick to 1-move additions for simplicity and speed.
        
        improved = True
        current_set = set(current_set)
        while improved:
            improved = False
            # Candidate vertices not in current_set
            candidates = [v for v in range(N) if v not in current_set]
            # Sort candidates by potential gain (fewest neighbors in current_set)
            # Gain = 1 - len(neighbors(v) & current_set)
            # We only want gain > 0, so len(neighbors(v) & current_set) == 0 is best (gain 1)
            # Actually, if we add v, we must remove all neighbors of v that are in current_set.
            # New size = |current_set| - |N(v) ∩ current_set| + 1
            # Improvement if |N(v) ∩ current_set| < 1, i.e., 0.
            # So we can only add isolated vertices (w.r.t current set).
            
            # This simple 1-move local search only adds isolated nodes.
            # It won't find improvements that require swapping.
            
            # Let's try a different local search: 
            # Iterate through all pairs (u in set, v not in set).
            # If we remove u and add v, is it better?
            # Change = 1 - 1 + (neighbors of u in set \ {u}) ... this is complex.
            
            # Simpler strong heuristic: 
            # Re-run greedy with different ordering?
            # Or use "Maximum Cardinality Search" based greedy?
            
            # Let's implement a simple "Shake" or "Restart" greedy.
            pass
        return current_set

    # The simple local search above is weak. 
    # Let's use a better heuristic: 
    # 1. Greedy Min-Degree.
    # 2. Try to add any vertex that has 0 neighbors in the set. (This is already done by greedy if it's maximal, but maybe order matters).
    
    # Actually, let's just use a better Greedy: 
    # "Maximal Independent Set" via randomization and keep the best of several runs.
    import random
    best_heuristic_size = 0
    best_heuristic_set = None
    
    # Run greedy multiple times with different random tie-breaking or orderings
    for _ in range(50): # 50 iterations is fast
        # Shuffle nodes to break ties randomly
        nodes = list(range(N))
        random.shuffle(nodes)
        
        remaining = set(nodes)
        selected = set()
        while remaining:
            # Pick a node from remaining. 
            # To mimic min-degree, we can pick the one with min degree in remaining.
            # But to introduce diversity, we can pick randomly from the top 10% lowest degrees?
            # Or just standard min-degree but with random tie-breaking.
            
            # Calculate degrees in remaining
            degs = {v: len(adj[v] & remaining) for v in remaining}
            min_deg = min(degs.values())
            # Get all nodes with min degree
            candidates = [v for v, d in degs.items() if d == min_deg]
            # Pick one randomly
            v = random.choice(candidates)
            
            selected.add(v)
            remaining.discard(v)
            for neighbor in adj[v]:
                if neighbor in remaining:
                    remaining.discard(neighbor)
        
        if len(selected) > best_heuristic_size:
            best_heuristic_size = len(selected)
            best_heuristic_set = selected

    # Compare with MILP result
    if best_heuristic_set is not None and best_heuristic_size > best_size:
        best_solution = list(best_heuristic_set)
        best_size = best_heuristic_size

    if best_solution:
        return best_solution
    
    # Absolute fallback: return [1]
    return [1]
