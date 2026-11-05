import numpy as np
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 145

    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        required_min_deg = n - 39

        # Try structured constructions first (fast, deterministic)
        for init_type in ['polarity', 'incidence', 'random_dense', 'random_sparse']:
            if time.time() > deadline:
                break
            for seed in range(30):
                if time.time() > deadline:
                    break
                result = _try_construct(n, required_min_deg, deadline, seed, init_type)
                if result is not None:
                    return {'n': n, 'edges': result}

        # Local search / edge swaps on best partial
        result = _local_search(n, required_min_deg, deadline)
        if result is not None:
            return {'n': n, 'edges': result}

    # Fallback
    return {'n': 39, 'edges': []}


def _init_polarity(n, seed):
    """
    Polarity graph construction: partition into groups, connect across groups.
    This is C4-free if done carefully (complete multipartite with no C4).
    Actually complete multipartite HAS C4s. Instead use a specific C4-free dense graph.
    
    Better: use a polarity graph from a finite geometry or a known C4-free construction.
    For simplicity, use a "blow-up" of a small C4-free graph.
    """
    adj = np.zeros((n, n), dtype=bool)
    rng = np.random.RandomState(seed)
    
    # Strategy: partition vertices into k groups of roughly equal size.
    # Connect all vertices in group i to all vertices in group j if (i,j) is an edge
    # in a base C4-free graph on k vertices.
    # Base: use a random C4-free graph on k vertices (small, easy to verify)
    
    k = max(3, n // 8)  # number of groups
    # Build a random C4-free graph on k vertices
    base_adj = np.zeros((k, k), dtype=bool)
    for i in range(k):
        for j in range(i + 1, k):
            if rng.random() < 0.5:
                # Check C4 in base
                if not _check_c4_small(base_adj, i, j):
                    base_adj[i, j] = True
                    base_adj[j, i] = True
    
    # Blow up
    group_size = n // k
    rem = n % k
    groups = []
    idx = 0
    for g in range(k):
        size = group_size + (1 if g < rem else 0)
        groups.append(list(range(idx, idx + size)))
        idx += size
    
    for i in range(k):
        for j in range(k):
            if i != j and base_adj[i, j]:
                for u in groups[i]:
                    for v in groups[j]:
                        adj[u, v] = True
                        adj[v, u] = True
    
    return adj


def _init_incidence(n, seed):
    """
    Incidence graph of a projective plane or similar.
    For n vertices, use a bipartite-like structure.
    """
    adj = np.zeros((n, n), dtype=bool)
    rng = np.random.RandomState(seed)
    
    # Divide into two halves
    half = n // 2
    left = list(range(half))
    right = list(range(half, n))
    
    # Connect each left vertex to a random subset of right vertices
    # Ensure no C4: for any two left vertices, they share at most one right neighbor
    # This is like a partial linear space
    
    # Simple approach: each left vertex connects to ~n/3 right vertices
    # Use a random matching-like structure
    for u in left:
        # Pick a random subset of right vertices
        targets = rng.choice(len(right), size=min(len(right), n // 3), replace=False)
        for t in targets:
            v = right[t]
            # Check if adding this edge creates C4
            if not _check_c4_numpy(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True
    
    # Also add some edges within left and within right if safe
    for u in left:
        for v in left:
            if u < v and not adj[u, v]:
                if not _check_c4_numpy(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
    
    for u in right:
        for v in right:
            if u < v and not adj[u, v]:
                if not _check_c4_numpy(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
    
    return adj


def _init_random(n, seed, density=0.3):
    """Random initial edges with C4 check."""
    adj = np.zeros((n, n), dtype=bool)
    rng = np.random.RandomState(seed)
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if rng.random() < density:
            if not _check_c4_numpy(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True
    
    return adj


def _check_c4_small(adj, u, v):
    """Check C4 for small adjacency matrix."""
    k = len(adj)
    nu = [i for i in range(k) if adj[u, i] and i != v]
    nv = [i for i in range(k) if adj[v, i] and i != u]
    for x in nu:
        for y in nv:
            if x != y and adj[x, y]:
                return True
    return False


def _check_c4_numpy(adj, u, v):
    """
    Fast C4 check using numpy.
    Adding edge (u,v) creates C4 iff there exists x,y with
    adj[u,x] and adj[x,y] and adj[y,v], where x != v, y != u, x != y.
    """
    # Get neighbors of u (excluding v) and v (excluding u)
    nu = np.where(adj[u])[0]
    nv = np.where(adj[v])[0]
    
    # Filter
    nu = nu[nu != v]
    nv = nv[nv != u]
    
    if len(nu) == 0 or len(nv) == 0:
        return False
    
    # For each x in nu, check if N(x) \ {u,v} intersects nv
    # Vectorized: for all x in nu, get adj[x, :] and check intersection with nv
    # adj[nu, :] gives a 2D array. We need adj[nu, y] for y in nv.
    # So: adj[np.ix_(nu, nv)] gives a |nu| x |nv| boolean matrix.
    # We need: exists (x, y) with x != y and adj[x, y] = True.
    # Also need x != u and y != v, but x is from nu (already excludes v) and y from nv (excludes u).
    # We also need x != y.
    
    sub = adj[np.ix_(nu, nv)]
    # sub[i, j] = adj[nu[i], nv[j]]
    # We need sub[i, j] = True and nu[i] != nv[j]
    
    # Create mask where nu[i] != nv[j]
    # nu[:, None] != nv[None, :]
    mask = nu[:, None] != nv[None, :]
    
    if np.any(sub & mask):
        return True
    
    return False


def _try_construct(n, required_min_deg, deadline, seed, init_type):
    """
    Greedy construction: initialize, then add edges to boost min degree.
    """
    random.seed(seed)
    
    if init_type == 'polarity':
        adj = _init_polarity(n, seed)
    elif init_type == 'incidence':
        adj = _init_incidence(n, seed)
    elif init_type == 'random_dense':
        adj = _init_random(n, seed, density=0.4)
    else:  # random_sparse
        adj = _init_random(n, seed, density=0.2)
    
    # Greedy edge addition to boost min degree
    max_iters = 5000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        degrees = adj.sum(axis=1)
        min_deg = int(degrees.min())
        
        if min_deg >= required_min_deg:
            break
        
        # Find vertices below required degree
        deficit_vertices = np.where(degrees < required_min_deg)[0]
        if len(deficit_vertices) == 0:
            break
        
        # Sort deficit vertices by how far below required they are
        deficit_sorted = deficit_vertices[np.argsort(degrees[deficit_vertices])]
        
        added = False
        for u in deficit_sorted:
            if time.time() > deadline:
                break
            
            # Candidates: non-neighbors of u
            candidates = np.where(~adj[u] & (np.arange(n) != u))[0]
            if len(candidates) == 0:
                continue
            
            # Sort by degree (prefer lower degree vertices to balance)
            candidates = candidates[np.argsort(degrees[candidates])]
            
            for v in candidates:
                if time.time() > deadline:
                    break
                v = int(v)
                if adj[u, v]:
                    continue
                if not _check_c4_numpy(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
                    added = True
                    break
            
            if added:
                break
        
        if not added:
            # Stuck: try random restart within this attempt
            # Perturb: remove a random edge, try again
            if it > 100:
                # Give up on this seed
                break
    
    degrees = adj.sum(axis=1)
    if degrees.min() >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges
    return None


def _local_search(n, required_min_deg, deadline):
    """
    Local search: start from best greedy result, try edge swaps.
    """
    # Get a starting point
    best_adj = None
    best_min_deg = -1
    
    for seed in range(20):
        if time.time() > deadline:
            break
        for init_type in ['polarity', 'incidence', 'random_dense']:
            if time.time() > deadline:
                break
            adj = None
            if init_type == 'polarity':
                adj = _init_polarity(n, seed)
            elif init_type == 'incidence':
                adj = _init_incidence(n, seed)
            else:
                adj = _init_random(n, seed, density=0.35)
            
            # Quick greedy fill
            for _ in range(500):
                if time.time() > deadline:
                    break
                degrees = adj.sum(axis=1)
                if degrees.min() >= required_min_deg:
                    break
                deficit = np.where(degrees < required_min_deg)[0]
                if len(deficit) == 0:
                    break
                u = int(deficit[0])
                candidates = np.where(~adj[u] & (np.arange(n) != u))[0]
                if len(candidates) == 0:
                    break
                random.shuffle(candidates.tolist())
                found = False
                for v in candidates:
                    v = int(v)
                    if not _check_c4_numpy(adj, u, v):
                        adj[u, v] = True
                        adj[v, u] = True
                        found = True
                        break
                if not found:
                    break
            
            degrees = adj.sum(axis=1)
            md = int(degrees.min())
            if md > best_min_deg:
                best_min_deg = md
                best_adj = adj.copy()
    
    if best_adj is None:
        return None
    
    # Now do local search: try edge swaps to improve min degree
    # An edge swap: remove (a,b), add (c,d) if both safe
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        degrees = best_adj.sum(axis=1)
        min_deg = int(degrees.min())
        
        if min_deg >= required_min_deg:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if best_adj[u, v]:
                        edges.append([u, v])
            return edges
        
        # Find the most deficit vertex
        min_vertices = np.where(degrees == min_deg)[0]
        u = int(min_vertices[0])
        
        # Try to find an edge to add to u
        candidates = np.where(~best_adj[u] & (np.arange(n) != u))[0]
        random.shuffle(candidates.tolist())
        
        found_add = False
        for v in candidates:
            v = int(v)
            if not _check_c4_numpy(best_adj, u, v):
                best_adj[u, v] = True
                best_adj[v, u] = True
                found_add = True
                break
        
        if found_add:
            continue
        
        # If can't add, try removing an edge from u and adding two others
        # (edge swap: remove (u, w), add (u, a) and (u, b) if possible)
        neighbors = np.where(best_adj[u])[0]
        if len(neighbors) == 0:
            break
        
        random.shuffle(neighbors.tolist())
        for w in neighbors:
            w = int(w)
            # Try removing (u, w)
            best_adj[u, w] = False
            best_adj[w, u] = False
            
            # Now try to add two edges from u
            added_count = 0
            candidates2 = np.where(~best_adj[u] & (np.arange(n) != u))[0]
            random.shuffle(candidates2.tolist())
            for v in candidates2:
                v = int(v)
                if not _check_c4_numpy(best_adj, u, v):
                    best_adj[u, v] = True
                    best_adj[v, u] = True
                    added_count += 1
                    if added_count >= 2:
                        break
            
            if added_count >= 1:
                # Keep the change if it helps
                new_degrees = best_adj.sum(axis=1)
                if new_degrees.min() >= min_deg - 1:
                    continue
                else:
                    # Revert
                    best_adj[u, w] = True
                    best_adj[w, u] = True
                    # Undo added edges
                    for v in candidates2:
                        v = int(v)
                        if best_adj[u, v] and v != w:
                            best_adj[u, v] = False
                            best_adj[v, u] = False
                    continue
            else:
                # Revert
                best_adj[u, w] = True
                best_adj[w, u] = True
    
    # Final check
    degrees = best_adj.sum(axis=1)
    if degrees.min() >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if best_adj[u, v]:
                    edges.append([u, v])
        return edges
    
    return None

# EVOLVE-BLOCK-END
