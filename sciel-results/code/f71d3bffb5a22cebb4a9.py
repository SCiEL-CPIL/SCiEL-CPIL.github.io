import numpy as np
import time

def _compute_core_numbers(adjacency):
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(int)
    # Copy degrees for the peeling process
    current_degrees = degrees.copy()
    core_numbers = np.zeros(n, dtype=int)
    
    # We need to efficiently find min degree vertices.
    # Since N=200, a simple loop is fine.
    # But we can do better.
    
    # Standard k-core algorithm
    # 1. Sort vertices by degree
    # 2. Iterate
    
    # Simpler approach for small N:
    # Repeatedly find vertex with min current_degree, assign core number = min(current_degree, ...)?
    # No, the core number of a vertex is the max k such that it is in the k-core.
    # This is equivalent to the "peeling order".
    # The core number of a vertex is the minimum value of the maximum degree 
    # encountered in the peeling process? 
    # Actually, the core number of a vertex $v$ is the largest $k$ such that $v$ 
    # is in the $k$-core.
    # In the peeling process, when we remove a vertex $v$ with current degree $d$,
    # its core number is $d$? 
    # No. If we remove a vertex with degree 1 in a graph that has a 3-core, 
    # that vertex has core number 1.
    # The vertices remaining in the graph at step $k$ form the $k$-core.
    # So, if a vertex is removed when the current minimum degree in the graph is $k$, 
    # its core number is $k$?
    # Let's trace:
    # Graph: Path 1-2-3. Degs 1, 2, 1.
    # Min deg is 1. Remove 1. Core(1)=1.
    # Remaining: 2-3. Degs 1, 1.
    # Min deg is 1. Remove 2. Core(2)=1.
    # Remaining: 3. Deg 0.
    # Remove 3. Core(3)=0? Or 1?
    # Usually, isolated vertices have core number 0 or 1 depending on definition.
    # In MIS, isolated vertices are always picked.
    # Let's use the definition: Core number is the value of $k$ when the vertex is peeled.
    
    # Algorithm:
    # 1. Compute initial degrees.
    # 2. Create a list of vertices.
    # 3. While list not empty:
    #    a. Find vertex with min current_degree.
    #    b. Its core number is its current_degree.
    #    c. Remove it from graph (decrement neighbors' degrees).
    
    # This is $O(N^2)$ which is fine for N=200.
    
    active = np.ones(n, dtype=bool)
    current_degrees = degrees.copy()
    core_numbers = np.zeros(n, dtype=int)
    
    for _ in range(n):
        if not np.any(active):
            break
        # Find active vertex with min degree
        # Mask out inactive vertices
        masked_degrees = np.where(active, current_degrees, np.inf)
        v = np.argmin(masked_degrees)
        core_numbers[v] = current_degrees[v]
        active[v] = False
        # Update neighbors
        neighbors = np.where(adjacency[v])[0]
        for u in neighbors:
            if active[u]:
                current_degrees[u] -= 1
                
    return core_numbers

def priority(adjacency):
    # Ensure numpy array
    adj = np.asarray(adjacency)
    if adj.dtype != bool:
        adj = adj.astype(bool)
        
    n = adj.shape[0]
    if n == 0:
        return np.array([])
        
    degrees = adj.sum(axis=1).astype(float)
    core_numbers = _compute_core_numbers(adj)
    
    # Score: Lower core number is better. Lower degree is better.
    # We want HIGH score to be picked first.
    # So we invert.
    # Scale core number heavily to dominate degree.
    max_degree = max(1.0, degrees.max())
    # Use a large constant
    score = -(core_numbers.astype(float) * 1000.0 + degrees)
    
    return score
