import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    Strategy: Combine low Core Number, low Degree, and spectral peripherality (Fiedler vector magnitude).
    Vertices with low core number and low degree are in sparse parts of the graph.
    The Fiedler vector (second eigenvector of Laplacian) identifies vertices that are
    less central to connectivity; low magnitude in the Fiedler vector indicates
    vertices that are "peripheral" and safer to include in an independent set.
    """
    start_time = time.time()
    # Safety margin: 150s limit, keep 10s buffer
    deadline = start_time + 140.0
    
    # Convert to numpy array if not already
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=bool)
    
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Calculate degrees
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Calculate Core Numbers
    max_deg = np.max(degrees).astype(int) if n > 0 else 0
    
    # Initialize core numbers to 0
    core_numbers = np.zeros(n, dtype=int)
    
    # Current degrees (will be modified)
    cur_degrees = degrees.copy()
    
    # Adjacency list for faster neighbor lookup
    adj_list = [np.where(adjacency[i])[0] for i in range(n)]
    
    # Active flags
    active = np.ones(n, dtype=bool)
    
    k = 1
    while np.any(active):
        # Check time
        if time.time() > deadline:
            # Fallback to simple degree if core calc takes too long (unlikely for N=200)
            break
        
        # Find active vertices with cur_degree < k
        mask = active & (cur_degrees < k)
        
        if not np.any(mask):
            # All remaining active vertices have degree >= k
            k += 1
            continue
        
        # Assign core number k-1 to these vertices
        core_numbers[mask] = k - 1
        
        # Remove them
        active[mask] = False
        
        # Update degrees of neighbors
        removed_indices = np.where(mask)[0]
        for v in removed_indices:
            neighbors = adj_list[v]
            active_neighbors = neighbors[active[neighbors]]
            if len(active_neighbors) > 0:
                cur_degrees[active_neighbors] -= 1
    
    # Compute Fiedler vector magnitude
    # Laplacian L = D - A
    L = np.diag(degrees) - adjacency.astype(float)
    
    # Compute eigenvalues and eigenvectors
    # For N=200, full eigendecomposition is fast
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # The second smallest eigenvalue corresponds to index 1
        # (index 0 is the smallest, which is 0 for connected graphs)
        fiedler_vector = eigenvectors[:, 1]
        # Use absolute value as a measure of peripherality
        # Lower magnitude -> more peripheral -> higher priority
        fiedler_mag = np.abs(fiedler_vector)
    except:
        # Fallback if eigendecomposition fails
        fiedler_mag = np.zeros(n)
    
    # Normalize features to [0, 1] for weighted combination
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val > min_val:
            return (arr - min_val) / (max_val - min_val)
        return np.zeros_like(arr)
    
    core_norm = normalize(core_numbers.astype(float))
    deg_norm = normalize(degrees)
    fiedler_norm = normalize(fiedler_mag)
    
    # Weighted combination
    # Lower core number, lower degree, and lower Fiedler magnitude should get higher score
    # So we use negative weights for these features
    w_core = 0.4
    w_deg = 0.3
    w_fiedler = 0.3
    
    score = -(w_core * core_norm + w_deg * deg_norm + w_fiedler * fiedler_norm)
    
    return score
