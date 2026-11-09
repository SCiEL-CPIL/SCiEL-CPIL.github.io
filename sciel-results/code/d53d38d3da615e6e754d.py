import numpy as np
import time

def _randomized_greedy_is(adj, n):
    """
    Constructs an Independent Set using a randomized greedy approach.
    At each step, picks a vertex with minimum degree among remaining,
    breaking ties randomly.
    """
    remaining = np.ones(n, dtype=bool)
    selected = np.zeros(n, dtype=bool)
    
    # Precompute adjacency list for faster neighbor lookup?
    # For n=200, boolean array operations are fine.
    
    while remaining.any():
        # Calculate degrees of remaining vertices
        # deg[i] = sum of adj[i, remaining]
        degs = adj[remaining].sum(axis=1)
        
        # Find minimum degree
        min_deg = np.min(degs)
        
        # Find indices of vertices with min degree
        # We need to map back to original indices
        # remaining is a boolean mask
        # np.where(remaining)[0] gives original indices
        # Let's get the indices of remaining vertices
        rem_indices = np.where(remaining)[0]
        
        # degs corresponds to rem_indices
        # Find which ones have min_deg
        min_mask = (degs == min_deg)
        candidates = rem_indices[min_mask]
        
        # Pick one randomly
        v = np.random.choice(candidates)
        
        selected[v] = True
        remaining[v] = False
        
        # Remove neighbors
        neighbors = np.where(adj[v] > 0)[0]
        remaining[neighbors] = False
        
    return selected

def priority(adjacency):
    """
    Computes a priority score for each vertex to maximize the size of the
    independent set found by a greedy decoder.
    
    Strategy:
    Explicitly searches for a large Independent Set (IS) using multiple
    randomized greedy runs. Returns a priority vector that forces the
    greedy decoder to select this specific IS.
    """
    adj = np.array(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Set a deadline to ensure we return before the 120s limit.
    # We use 100s to be safe.
    start_time = time.time()
    deadline = start_time + 100.0
    
    best_size = -1
    best_set = None
    
    # Run randomized greedy IS construction
    # We can run this many times.
    # Each run is O(n^2) ~ 40k ops. 100s allows millions of runs.
    # We limit to a reasonable number or time.
    
    # To improve quality, we can also try a "slightly randomized" degree:
    # Pick min of (degree + small_random_noise)
    
    while time.time() < deadline:
        # Method 1: Pure Random Tie-Break Min-Degree
        selected = _randomized_greedy_is(adj, n)
        size = np.sum(selected)
        if size > best_size:
            best_size = size
            best_set = selected.copy()
            
        # Optional: Method 2: Randomized Degree
        # Add small noise to degrees to vary the "min"
        # This is effectively the same as random tie-breaking if noise is small,
        # but if noise is larger, it explores more.
        # Let's stick to the robust one for now, but maybe add a second loop
        # if time permits?
        
        # To speed up, we can break early if we find a perfect IS?
        # Max IS size is unknown. But if size == n, break.
        if size == n:
            break

    if best_set is None:
        # Fallback to seed behavior
        return -adj.sum(axis=1).astype(float)
    
    # Construct scores
    # We want the greedy decoder to pick best_set.
    # Assign high score to best_set, low to others.
    # Since best_set is a Maximal Independent Set, the decoder will pick
    # all vertices in best_set first, and then skip all others (as they are neighbors).
    
    scores = np.where(best_set, 1.0, -1.0)
    
    return scores
