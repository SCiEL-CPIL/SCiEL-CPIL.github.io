"""K3,3-free dense graphs on 20 vertices - optimized with bitmasks."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 20
ALL_MASK = (1 << N) - 1

def _neighbors_from_adj(adj):
    """Convert adjacency matrix to list of bitmasks."""
    nbs = [0] * N
    for i in range(N):
        m = 0
        row = adj[i]
        for j in range(N):
            if row[j]:
                m |= (1 << j)
        nbs[i] = m
    return nbs

def _has_k33(nbs):
    """Check if graph contains K3,3 using bitmask operations."""
    for a in range(N):
        na = nbs[a]
        for b in range(a+1, N):
            ab = na & nbs[b]
            if ab & (1 << b):
                ab &= ~(1 << b)
            # For each c in common neighbors of a,b
            c = ab
            while c:
                lb = c & (-c)
                c &= c - 1
                cc = lb - 1
                common = na & nbs[b] & nbs[cc]
                # Exclude a, b, c
                common &= ~(1 << a)
                common &= ~(1 << b)
                common &= lb
                if (common & (common - 1)) != 0:
                    # at least 2 bits set in common (plus we need >=3 total)
                    # Actually we need >=3 common neighbors outside {a,b,c}
                    cnt = bin(common).count('1')
                    if cnt >= 3:
                        return True
    return False

def _has_k33_fast(nbs):
    """Faster K3,3 check: for each triple (a,b,c), check common neighbors >= 3."""
    for a in range(N):
        na = nbs[a]
        for b in range(a+1, N):
            ab = na & nbs[b]
            ab &= ~(1 << b)
            c = ab
            while c:
                lb = c & (-c)
                c &= c - 1
                cc = lb - 1
                # common neighbors of a, b, c
                common = na & nbs[b] & nbs[cc]
                common &= ~(1 << a)
                common &= ~(1 << b)
                common &= ~lb
                # count bits
                if common & (common - 1) & (common - 2):
                    # at least 3 bits set (rough check)
                    x = common
                    cnt = 0
                    while x:
                        x &= x - 1
                        cnt += 1
                        if cnt >= 3:
                            return True
    return False

def _find_violation(nbs):
    """Find a K3,3 violation. Returns (a,b,c,common_mask) or None."""
    for a in range(N):
        na = nbs[a]
        for b in range(a+1, N):
            ab = na & nbs[b]
            ab &= ~(1 << b)
            c = ab
            while c:
                lb = c & (-c)
                c &= c - 1
                cc = lb - 1
                common = na & nbs[b] & nbs[cc]
                common &= ~(1 << a)
                common &= ~(1 << b)
                common &= ~lb
                if common & (common - 1) & (common - 2):
                    x = common
                    cnt = 0
                    while x:
                        x &= x - 1
                        cnt += 1
                        if cnt >= 3:
                            return (a, b, cc, common)
    return None

def _try_add_edge(nbs, u, v):
    """Try to add edge (u,v). Returns True if K3,3-free after addition."""
    nbs[u] |= (1 << v)
    nbs[v] |= (1 << u)
    if _has_k33_fast(nbs):
        nbs[u] &= ~(1 << v)
        nbs[v] &= ~(1 << u)
        return False
    return True

def _greedy_fill(nbs, deadline, max_checks=20000):
    """Greedily add edges in random order."""
    checks = 0
    for _ in range(50):
        if time.time() > deadline or checks > max_checks:
            break
        remaining = [(i, j) for i in range(N) for j in range(i+1, N) 
                     if not (nbs[i] >> j) & 1]
        if not remaining:
            break
        random.shuffle(remaining)
        added_any = False
        for u, v in remaining:
            if time.time() > deadline or checks > max_checks:
                break
            checks += 1
            if _try_add_edge(nbs, u, v):
                added_any = True
        if not added_any:
            break
    return nbs

def _local_search(nbs, deadline, max_iter=3000):
    """Local search with edge swaps."""
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        # Try to add any missing edge
        remaining = [(i, j) for i in range(N) for j in range(i+1, N) 
                     if not (nbs[i] >> j) & 1]
        if not remaining:
            break
        random.shuffle(remaining)
        added = False
        for u, v in remaining:
            if time.time() > deadline:
                break
            if _try_add_edge(nbs, u, v):
                added = True
                break
        if not added:
            # Stuck: try 1-edge removal + re-add
            existing = [(i, j) for i in range(N) for j in range(i+1, N) 
                       if (nbs[i] >> j) & 1]
            if not existing:
                break
            random.shuffle(existing)
            for u, v in existing[:8]:
                if time.time() > deadline:
                    break
                # Remove edge
                nbs[u] &= ~(1 << v)
                nbs[v] &= ~(1 << u)
                # Try to add more edges
                remaining2 = [(i, j) for i in range(N) for j in range(i+1, N) 
                             if not (nbs[i] >> j) & 1]
                random.shuffle(remaining2)
                added2 = 0
                for u2, v2 in remaining2:
                    if time.time() > deadline:
                        break
                    if _try_add_edge(nbs, u2, v2):
                        added2 += 1
                        if added2 >= 2:
                            break
                if added2 < 1:
                    # Restore
                    nbs[u] |= (1 << v)
                    nbs[v] |= (1 << u)
    return nbs

def _edge_count(nbs):
    """Count edges from neighbor masks."""
    cnt = 0
    for i in range(N):
        cnt += bin(nbs[i]).count('1')
    return cnt // 2

def _nbs_to_edges(nbs):
    """Convert neighbor masks to edge list."""
    edges = []
    for i in range(N):
        m = nbs[i]
        j = i + 1
        while j < N:
            if (m >> j) & 1:
                edges.append([i, j])
            j += 1
    return edges

def _seed_k2_18():
    """K_{2,18} seed: 2 vertices connected to 18 others."""
    nbs = [0] * N
    for a in range(2):
        for b in range(2, N):
            nbs[a] |= (1 << b)
            nbs[b] |= (1 << a)
    return nbs

def _seed_bipartite(k):
    """K_{k, 20-k} bipartite seed."""
    nbs = [0] * N
    for a in range(k):
        for b in range(k, N):
            nbs[a] |= (1 << b)
            nbs[b] |= (1 << a)
    return nbs

def _random_start(density=0.4):
    """Random graph with given edge probability."""
    nbs = [0] * N
    for i in range(N):
        for j in range(i+1, N):
            if random.random() < density:
                nbs[i] |= (1 << j)
                nbs[j] |= (1 << i)
    return nbs

def _remove_violations(nbs, deadline):
    """Remove edges until K3,3-free."""
    max_iters = 500
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        viol = _find_violation(nbs)
        if viol is None:
            break
        a, b, c, common = viol
        # Remove a random edge from the K3,3
        common_list = []
        m = common
        while m:
            lb = m & (-m)
            m &= m - 1
            common_list.append(lb - 1)
        side1 = random.choice([a, b, c])
        side2 = random.choice(common_list)
        nbs[side1] &= ~(1 << side2)
        nbs[side2] &= ~(1 << side1)
    return nbs

def construct():
    deadline = time.time() + 115  # safety margin
    
    best_nbs = None
    best_count = 0
    
    # Try multiple seeds
    seeds = []
    # K_{2,18}
    seeds.append(_seed_k2_18())
    # K_{3,17}
    seeds.append(_seed_bipartite(3))
    # K_{4,16}
    seeds.append(_seed_bipartite(4))
    # K_{5,15}
    seeds.append(_seed_bipartite(5))
    
    # Random starts with various densities
    for p in [0.2, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for _ in range(3):
            seeds.append(_random_start(p))
    
    # Process each seed
    for seed_nbs in seeds:
        if time.time() > deadline:
            break
        
        # Copy
        nbs = seed_nbs[:]
        
        # Remove violations if any
        nbs = _remove_violations(nbs, deadline)
        
        # Greedy fill
        nbs = _greedy_fill(nbs, deadline, max_checks=15000)
        
        # Local search
        nbs = _local_search(nbs, deadline, max_iter=2000)
        
        cnt = _edge_count(nbs)
        if cnt > best_count:
            best_count = cnt
            best_nbs = nbs[:]
    
    # Additional random restarts if time permits
    restarts_done = len(seeds)
    max_restarts = 50
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
        p = random.uniform(0.25, 0.5)
        nbs = _random_start(p)
        nbs = _remove_violations(nbs, deadline)
        nbs = _greedy_fill(nbs, deadline, max_checks=10000)
        nbs = _local_search(nbs, deadline, max_iter=1500)
        cnt = _edge_count(nbs)
        if cnt > best_count:
            best_count = cnt
            best_nbs = nbs[:]
    
    # Final improvement on best
    if best_nbs is not None and time.time() < deadline:
        nbs = best_nbs[:]
        nbs = _greedy_fill(nbs, deadline, max_checks=20000)
        nbs = _local_search(nbs, deadline, max_iter=3000)
        cnt = _edge_count(nbs)
        if cnt > best_count:
            best_count = cnt
            best_nbs = nbs[:]
    
    if best_nbs is None:
        # Fallback: K_{2,18}
        best_nbs = _seed_k2_18()
    
    return _nbs_to_edges(best_nbs)
# EVOLVE-BLOCK-END
