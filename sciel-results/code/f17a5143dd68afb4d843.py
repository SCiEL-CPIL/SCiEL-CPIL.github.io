"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
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
    """Check if induced subgraph on mask is regular using bitmask operations."""
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

def _find_one_regular_subgraph_full(adj, n):
    """Find any regular induced subgraph of size >= 6 (exhaustive)."""
    for size in range(6, n + 1):
        masks = _get_masks(n, size)
        for mask in masks:
            if _is_regular_induced_fast(adj, mask):
                return mask
    return None

def _find_one_regular_subgraph_sampled(adj, n, num_samples=30):
    """Find a regular induced subgraph of size 6 using sampling."""
    all_masks_6 = _get_masks(n, 6)
    for _ in range(num_samples):
        mask = random.choice(all_masks_6)
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

def _structured_init(n, rng):
    """Create a structured graph using threshold on random permutation distances.
    
    This creates degree heterogeneity: vertices with extreme positions in the
    permutation tend to have very different degrees, making regular induced
    subgraphs less likely.
    """
    # Random permutation defines a total order
    perm = list(range(n))
    rng.shuffle(perm)
    pos = [0] * n
    for i, v in enumerate(perm):
        pos[v] = i
    
    # Edge if distance in permutation exceeds threshold
    # Threshold around 0.5 * n gives ~50% edge density
    threshold = n * 0.5
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(pos[i] - pos[j])
            if dist > threshold:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _random_init(n, rng, p=0.5):
    """Standard random graph initialization."""
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _local_search(adj, n, deadline, rng, max_steps=800):
    """Local search to eliminate regular induced subgraphs."""
    found_valid = False
    steps_since_full_check = 0
    max_steps_between_full_checks = 40
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Periodic full check
        if steps_since_full_check >= max_steps_between_full_checks:
            mask = _find_one_regular_subgraph_full(adj, n)
            steps_since_full_check = 0
            if mask is None:
                found_valid = True
                break
            else:
                verts = _extract_vertices(mask)
                if len(verts) >= 2:
                    u_idx, v_idx = rng.sample(range(len(verts)), 2)
                    u, v = verts[u_idx], verts[v_idx]
                    if (adj[u] >> v) & 1:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        else:
            # Sampled search for speed
            mask = _find_one_regular_subgraph_sampled(adj, n, num_samples=15)
            steps_since_full_check += 1
            
            if mask is not None:
                verts = _extract_vertices(mask)
                if len(verts) >= 2:
                    u_idx, v_idx = rng.sample(range(len(verts)), 2)
                    u, v = verts[u_idx], verts[v_idx]
                    if (adj[u] >> v) & 1:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
    
    return adj, found_valid

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    best_n = 5
    best_edges = []
    
    # Try to find valid graph for n=22 down to 15
    for target_n in range(22, 14, -1):
        if time.time() > deadline - 8:
            break
        
        n = target_n
        found_valid = False
        
        # More restarts for larger n, mix structured and random inits
        max_restarts = 12 if n >= 21 else 20
        
        for r in range(max_restarts):
            if time.time() > deadline - 8:
                break
            
            # Alternate between structured and random initialization
            if r % 3 == 0:
                adj = _structured_init(n, rng)
            elif r % 3 == 1:
                adj = _random_init(n, rng, p=0.5)
            else:
                # Structured with different threshold
                perm = list(range(n))
                rng.shuffle(perm)
                pos = [0] * n
                for i, v in enumerate(perm):
                    pos[v] = i
                threshold = n * (0.4 + 0.2 * (r % 3))
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = abs(pos[i] - pos[j])
                        if dist > threshold:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            
            # Run local search
            adj, valid = _local_search(adj, n, deadline - 5, rng, max_steps=600)
            
            if valid:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found_valid = True
                break
        
        if found_valid:
            break
    
    # Fallback
    if best_n <= 5:
        return {'n': 5, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
