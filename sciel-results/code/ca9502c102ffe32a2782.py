"""Rewritten program with improved MIS heuristic scoring."""
import numpy as np
import time

_DEADLINE = 120.0  # seconds
_START_TIME = None


def _core_numbers(adjacency: np.ndarray) -> np.ndarray:
    """Compute core number for each vertex using peeling algorithm."""
    n = adjacency.shape[0]
    deg = adjacency.sum(axis=1).astype(np.int64)
    core = np.zeros(n, dtype=np.int64)
    remaining = np.ones(n, dtype=bool)
    
    # Bucket vertices by degree
    max_deg = int(deg.max()) if n > 0 else 0
    if max_deg < 0:
        return core
    
    # Use a simple iterative peeling
    deg_work = deg.copy()
    for k in range(max_deg + 1):
        # Find vertices with current degree == k
        candidates = np.where((deg_work == k) & remaining)[0]
        for v in candidates:
            if not remaining[v]:
                continue
            core[v] = k
            remaining[v] = False
            # Update neighbors
            neighbors = np.where(adjacency[v] & remaining)[0]
            for nb in neighbors:
                deg_work[nb] -= 1
    
    # Assign core number as the max k such that vertex was in k-core
    # Re-do with proper core number definition
    core = np.zeros(n, dtype=np.int64)
    remaining = np.ones(n, dtype=bool)
    deg_work = deg.copy()
    
    for k in range(max_deg + 1):
        changed = True
        while changed:
            changed = False
            candidates = np.where((deg_work == k) & remaining)[0]
            for v in candidates:
                if remaining[v]:
                    core[v] = k
                    remaining[v] = False
                    neighbors = np.where(adjacency[v] & remaining)[0]
                    for nb in neighbors:
                        deg_work[nb] -= 1
                    changed = True
    
    return core


def priority(adjacency):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5:
        # Fallback: just use negative degree
        return -adjacency.sum(axis=1).astype(float)
    
    deg = adjacency.sum(axis=1).astype(np.float64)
    
    # Feature 1: negative degree
    feat_degree = -deg
    
    # Feature 2: negative sum of neighbor degrees (lower is better)
    # sum of neighbor degrees for vertex i = row i of adjacency @ deg
    neighbor_deg_sum = adjacency.astype(np.float64) @ deg
    feat_neighbor_deg = -neighbor_deg_sum
    
    # Feature 3: core number (lower is better)
    core = _core_numbers(adjacency).astype(np.float64)
    feat_core = -core
    
    # Feature 4: "independent set potential" - number of vertices not adjacent to i
    # This is n - 1 - deg[i]
    feat_non_neighbors = (n - 1 - deg)
    
    # Normalize features to zero mean, unit variance for stable combination
    def normalize(f):
        std = f.std()
        if std < 1e-12:
            return np.zeros_like(f)
        return (f - f.mean()) / std
    
    f1 = normalize(feat_degree)
    f2 = normalize(feat_neighbor_deg)
    f3 = normalize(feat_core)
    f4 = normalize(feat_non_neighbors)
    
    # Weighted combination - empirically, degree and core number are most important
    score = 1.0 * f1 + 0.8 * f2 + 0.6 * f3 + 0.4 * f4
    
    # Check deadline again
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 2:
        return score
    
    return score
