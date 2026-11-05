"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_MASKS_CACHE = {}

def _get_masks(n, size):
    """Precompute masks for subsets of a specific size."""
    key = (n, size)
    if key not in _MASKS_CACHE:
        masks = []
        for combo in combinations(range(n), size):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            masks.append(mask)
        _MASKS_CACHE[key] = masks
    return _MASKS_CACHE[key]

def _is_regular_induced_fast(adj, mask):
    """Check if induced subgraph on mask is regular."""
    count = mask.bit_count()
    if count < 6:
        return False
    m = mask
    first_deg = -1
    while m:
        lb = m & (-m)
        idx = lb.bit_length() - 1
        deg = (adj[idx] & mask).bit_count()
        if first_deg == -1:
            first_deg = deg
        elif deg != first_deg:
            return False
        m ^= lb
    return True

def _find_one_regular_subgraph(adj, n, time_budget=0.5):
    """Find any regular induced subgraph of size >= 6."""
    start = time.time()
    # Check sizes in increasing order
    for size in range(6, n + 1):
        if time.time() - start > time_budget:
            break
        masks = _get_masks(n, size)
        for mask in masks:
            if time.time() - start > time_budget:
                break
            if _is_regular_induced_fast(adj, mask):
                return mask
    return None

def _extract_vertices(mask):
    """Extract list of vertex indices from mask."""
    verts = []
    m = mask
    while m:
        lb = m & (-m)
        idx = lb.bit_length() - 1
        verts.append(idx)
        m ^= lb
    return verts

def _count_regular_subgraphs(adj, n, max_check=200):
    """Count how many regular subgraphs of size 6 exist (for fitness)."""
    count = 0
    masks = _get_masks(n, 6)
    checked = 0
    for mask in masks:
        if checked >= max_check:
            break
        if _is_regular_induced_fast(adj, mask):
            count += 1
        checked += 1
    return count

def _local_search(adj, n, deadline, rng):
    """Local search: flip edges to eliminate regular induced subgraphs."""
    adj = adj[:]  # copy
    max_steps = 500
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Find a regular subgraph
        mask = _find_one_regular_subgraph(adj, n, time_budget=0.3)
        
        if mask is None:
            # Verify it's truly valid (no regular subgraph of any size >= 6)
            if _find_one_regular_subgraph(adj, n, time_budget=1.0) is None:
                return adj, True
            # Otherwise continue searching
            continue
        
        # Found a regular subgraph. Flip an edge to break regularity.
        verts = _extract_vertices(mask)
        if len(verts) < 2:
            continue
            
        # Try multiple random edge flips and pick the best one
        best_adj = None
        best_score = float('inf')
        
        for _ in range(5):
            u_idx, v_idx = random.sample(range(len(verts)), 2)
            u = verts[u_idx]
            v = verts[v_idx]
            
            # Flip edge (u, v)
            if (adj[u] >> v) & 1:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Measure fitness: count regular subgraphs of size 6
            score = _count_regular_subgraphs(adj, n, max_check=100)
            
            if score < best_score:
                best_score = score
                best_adj = adj[:]
            
            # Undo flip
            if (adj[u] >> v) & 1:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
        
        # Apply the best flip
        if best_adj is not None:
            adj = best_adj
            
    # Final check
    if _find_one_regular_subgraph(adj, n, time_budget=1.0) is None:
        return adj, True
    return adj, False

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_n = 5
    best_edges = []
    rng = random.Random(42)
    
    # Try to find a valid graph for n=20 down to 15
    for target_n in range(20, 14, -1):
        if time.time() > deadline - 15:
            break
            
        n = target_n
        
        # Number of restarts depends on remaining time
        remaining = deadline - time.time()
        max_restarts = max(3, int(remaining / 15))
        
        for r in range(max_restarts):
            if time.time() > deadline - 15:
                break
                
            # Random initialization
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Local Search
            adj, found_valid = _local_search(adj, n, deadline - 15, rng)
            
            if found_valid:
                # Extract edges
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                break
        
        if best_n == target_n:
            break
    
    # Fallback if no graph > 5 found
    if best_n <= 5:
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
