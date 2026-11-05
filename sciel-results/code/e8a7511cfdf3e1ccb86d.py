"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from math import comb

def _adj_to_bitmask(adj, n):
    """Convert adjacency matrix to a list of bitmask rows."""
    bits = []
    for u in range(n):
        mask = 0
        for v in range(n):
            if adj[u, v]:
                mask |= (1 << v)
        bits.append(mask)
    return bits

def _bitmask_is_regular(bits, subset):
    """Check if induced subgraph on subset is regular using bitmasks."""
    sz = len(subset)
    if sz < 2:
        return True
    # Build local adjacency
    local_adj = []
    for i, u in enumerate(subset):
        row = bits[u]
        # Count edges from u to other vertices in subset
        deg = 0
        for j, v in enumerate(subset):
            if i != j and (row >> v) & 1:
                deg += 1
        local_adj.append(deg)
    return len(set(local_adj)) == 1

def _vectorized_check_6(n, adj):
    """Check all 6-subsets for regularity using vectorized numpy."""
    for combo in combinations(range(n), 6):
        idx = np.array(combo)
        sub = adj[np.ix_(idx, idx)]
        degs = sub.sum(axis=1)
        if degs.min() == degs.max():
            return False
    return True

def _vectorized_check_7(n, adj):
    """Check all 7-subsets for regularity."""
    for combo in combinations(range(n), 7):
        idx = np.array(combo)
        sub = adj[np.ix_(idx, idx)]
        degs = sub.sum(axis=1)
        if degs.min() == degs.max():
            return False
    return True

def _check_larger(n, adj, min_sz=8, max_samples=200):
    """Check subsets of size >= 8 with sampling."""
    for sz in range(min_sz, n + 1):
        total = comb(n, sz)
        samples = min(total, max_samples)
        for _ in range(samples):
            idx = np.array(random.sample(range(n), sz))
            sub = adj[np.ix_(idx, idx)]
            degs = sub.sum(axis=1)
            if degs.min() == degs.max():
                return False
    return True

def _full_verify(n, adj):
    """Full verification: all 6-sets, all 7-sets, sampled 8+."""
    if not _vectorized_check_6(n, adj):
        return False
    if not _vectorized_check_7(n, adj):
        return False
    if not _check_larger(n, adj, min_sz=8, max_samples=200):
        return False
    return True

def _make_permutation_graph(n, seed):
    """
    Constructive approach: use a random transitive permutation group.
    Generate edges based on orbit structure to limit symmetry.
    """
    rng = random.Random(seed)
    adj = np.zeros((n, n), dtype=np.uint8)
    
    # Start with a random base graph
    density = 0.3 + 0.4 * rng.random()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < density:
                adj[u, v] = 1
                adj[v, u] = 1
    
    # Apply a random permutation to break symmetries
    perm = list(range(n))
    rng.shuffle(perm)
    new_adj = np.zeros((n, n), dtype=np.uint8)
    for u in range(n):
        for v in range(n):
            if u != v:
                new_adj[perm[u], perm[v]] = adj[u, v]
    adj = new_adj
    
    # Add asymmetric perturbations
    for _ in range(n * 2):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v:
            if rng.random() < 0.5:
                adj[u, v] = 1
                adj[v, u] = 1
            else:
                adj[u, v] = 0
                adj[v, u] = 0
    
    return adj

def _local_search_fix(n, adj, max_steps=500):
    """
    Targeted local search: find violating subsets and flip edges to break regularity.
    """
    rng = random.Random()
    
    for step in range(max_steps):
        if time.time() > _deadline - 15:
            break
        
        # Try to find a violating 6-subset
        found_violation = False
        for combo in combinations(range(n), 6):
            idx = np.array(combo)
            sub = adj[np.ix_(idx, idx)]
            degs = sub.sum(axis=1)
            if degs.min() == degs.max():
                # Found a regular 6-subset; try to break it
                # Pick a random edge within this subset and flip it
                i = rng.randint(0, 5)
                j = rng.randint(0, 5)
                if i != j:
                    u, v = combo[i], combo[j]
                    adj[u, v] = 1 - adj[u, v]
                    adj[v, u] = adj[u, v]
                    found_violation = True
                break
        
        if not found_violation:
            # No 6-violation found; check 7
            found_violation = False
            for combo in combinations(range(n), 7):
                idx = np.array(combo)
                sub = adj[np.ix_(idx, idx)]
                degs = sub.sum(axis=1)
                if degs.min() == degs.max():
                    i = rng.randint(0, 6)
                    j = rng.randint(0, 6)
                    if i != j:
                        u, v = combo[i], combo[j]
                        adj[u, v] = 1 - adj[u, v]
                        adj[v, u] = adj[u, v]
                        found_violation = True
                    break
        
        if not found_violation:
            # No small violations; do a random perturbation
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u != v:
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = adj[u, v]
    
    return adj

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 100  # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Try to build up from smaller n to larger n
    for n in range(20, 5, -1):  # Start with n=20 and work down
        if time.time() > _deadline - 15:
            break
        
        found = False
        
        # Strategy 1: Random graphs with targeted local search
        num_attempts = 30 if n <= 18 else 10
        for attempt in range(num_attempts):
            if time.time() > _deadline - 15:
                break
            
            # Generate random graph
            density = 0.25 + 0.5 * random.random()
            adj = np.zeros((n, n), dtype=np.uint8)
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < density:
                        adj[u, v] = 1
                        adj[v, u] = 1
            
            # Quick sample check
            has_violation = False
            for _ in range(30):
                idx = np.array(random.sample(range(n), 6))
                sub = adj[np.ix_(idx, idx)]
                degs = sub.sum(axis=1)
                if degs.min() == degs.max():
                    has_violation = True
                    break
            
            if has_violation:
                # Apply local search to fix
                adj = _local_search_fix(n, adj, max_steps=100)
            
            # Full verification
            if _full_verify(n, adj):
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u, v]:
                            edges.append([u, v])
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            continue
        
        # Strategy 2: Permutation-based constructive approach
        if time.time() > _deadline - 15:
            break
        
        for seed in range(20):
            if time.time() > _deadline - 15:
                break
            
            adj = _make_permutation_graph(n, seed)
            
            # Apply local search
            adj = _local_search_fix(n, adj, max_steps=150)
            
            if _full_verify(n, adj):
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u, v]:
                            edges.append([u, v])
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            continue
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
