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
    deadline = time.time() + 145  # 145s out of 150s budget
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple strategies
        # Strategy 1: Greedy with fast C4 check
        for attempt in range(50):
            if time.time() > deadline:
                break
            result = _try_construct_greedy(n, required_min_deg, deadline, attempt)
            if result is not None:
                n_found, edges_found = result
                return {'n': n_found, 'edges': edges_found}
        
        # Strategy 2: Start from known C4-free structures
        result = _try_construct_from_structure(n, required_min_deg, deadline)
        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}
        
        # Strategy 3: Local search with edge swaps
        result = _local_search(n, required_min_deg, deadline)
        if result is not None:
            n_found, edges_found = result
            return {'n': n_found, 'edges': edges_found}
    
    # Fallback: return a valid smaller graph
    return {'n': 39, 'edges': []}


def _check_c4(adj, u, v):
    """
    Check if adding edge (u,v) creates a C4.
    Fast version using numpy.
    """
    # A C4 is created iff there exists x != u,v such that:
    # x is adjacent to u, and x has a common neighbor y with v (y != u, y != v)
    # i.e., (N(x) ∩ N(v)) \ {u, v} is non-empty for some x in N(u) \ {v}
    
    nu = np.where(adj[u])[0]
    
    for x in nu:
        if x == v:
            continue
        # Check if N(x) ∩ N(v) has any element other than u and v
        nx = np.where(adj[x])[0]
        nv = np.where(adj[v])[0]
        # Find intersection
        # Use set-like operation with numpy
        common = nx[np.isin(nx, nv)]
        # Filter out u and v
        common = common[(common != u) & (common != v)]
        if len(common) > 0:
            return True
    
    return False


def _check_c4_fast(adj, u, v):
    """
    Even faster C4 check using precomputed neighbor sets.
    """
    # Precompute: for each vertex, store its neighbors as a set for O(1) lookup
    # But we don't have that, so use numpy
    
    nu = np.where(adj[u])[0]
    nv = np.where(adj[v])[0]
    
    for x in nu:
        if x == v:
            continue
        nx = np.where(adj[x])[0]
        # Check if any element in nx is in nv, excluding u and v
        # adj[nx, v] tells us which neighbors of x are also neighbors of v
        mask = adj[nx, v]
        # Exclude u and v from consideration
        mask[(nx == u) | (nx == v)] = False
        if np.any(mask):
            return True
    
    return False


def _try_construct_greedy(n, required_min_deg, deadline, seed):
    """
    Greedy construction with efficient C4 checking.
    """
    random.seed(seed)
    
    adj = np.zeros((n, n), dtype=bool)
    
    # Track degrees
    degrees = np.zeros(n, dtype=int)
    
    # Precompute all edges in random order
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    
    # First pass: add all edges that don't create C4
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if not adj[u, v] and not _check_c4_fast(adj, u, v):
            adj[u, v] = True
            adj[v, u] = True
            degrees[u] += 1
            degrees[v] += 1
    
    # Second pass: try to increase min degree by adding more edges
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        min_deg = degrees.min()
        if min_deg >= required_min_deg:
            break
        
        # Find vertices with minimum degree
        min_vertices = np.where(degrees == min_deg)[0]
        random.shuffle(min_vertices.tolist())
        
        added = False
        for u in min_vertices:
            if time.time() > deadline:
                break
            
            # Try adding edges from u to other vertices
            candidates = [v for v in range(n) if v != u and not adj[u, v]]
            # Sort by degree (prefer lower degree vertices)
            candidates.sort(key=lambda v: degrees[v])
            
            for v in candidates:
                if time.time() > deadline:
                    break
                if not _check_c4_fast(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
                    degrees[u] += 1
                    degrees[v] += 1
                    added = True
                    break
            
            if added:
                break
        
        if not added:
            break
    
    if degrees.min() >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return (n, edges)
    return None


def _try_construct_from_structure(n, required_min_deg, deadline):
    """
    Try constructing from known C4-free graph structures.
    For example, use a polarity graph or incidence graph of a projective plane.
    """
    # For n around 46-47, we can try various structured approaches
    
    # Approach: Use a random regular graph and verify/fix C4s
    # Or: Start with a known C4-free graph and extend it
    
    # Let's try: start with a high-degree random graph, remove C4s, then add back
    for seed in range(20):
        if time.time() > deadline:
            break
        
        random.seed(seed + 10000)
        adj = np.zeros((n, n), dtype=bool)
        degrees = np.zeros(n, dtype=int)
        
        # Start with a random graph with edge probability ~0.5
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u, v] = True
                    adj[v, u] = True
                    degrees[u] += 1
                    degrees[v] += 1
        
        # Remove edges that are part of C4s
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Find all C4s and remove one edge from each
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u + 1, n):
                    if time.time() > deadline:
                        break
                    if adj[u, v]:
                        # Check if this edge is part of a C4
                        # Count paths of length 3 from u to v
                        count = 0
                        for x in range(n):
                            if x != u and x != v and adj[u, x]:
                                for y in range(n):
                                    if y != u and y != v and y != x and adj[x, y] and adj[y, v]:
                                        count += 1
                        if count > 0:
                            # Remove this edge
                            adj[u, v] = False
                            adj[v, u] = False
                            degrees[u] -= 1
                            degrees[v] -= 1
                            changed = True
        
        # Now try to add edges back to increase min degree
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if not adj[u, v] and not _check_c4_fast(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True
                degrees[u] += 1
                degrees[v] += 1
        
        if degrees.min() >= required_min_deg:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        edges.append([u, v])
            return (n, edges)
    
    return None


def _local_search(n, required_min_deg, deadline):
    """
    Local search: start with a greedy solution, then try edge swaps to improve min degree.
    """
    # Get an initial solution
    best_adj = None
    best_min_deg = -1
    
    for seed in range(20):
        if time.time() > deadline:
            break
        result = _try_construct_greedy(n, required_min_deg, deadline, seed)
        if result is not None:
            n_found, edges_found = result
            # Reconstruct adjacency
            adj = np.zeros((n, n), dtype=bool)
            for u, v in edges_found:
                adj[u, v] = True
                adj[v, u] = True
            min_deg = adj.sum(axis=1).min()
            if min_deg > best_min_deg:
                best_min_deg = min_deg
                best_adj = adj.copy()
            if min_deg >= required_min_deg:
                return (n, edges_found)
    
    if best_adj is None:
        return None
    
    adj = best_adj
    degrees = adj.sum(axis=1).copy()
    
    # Local search: try edge swaps
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        min_deg = degrees.min()
        if min_deg >= required_min_deg:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        edges.append([u, v])
            return (n, edges)
        
        # Find a vertex with minimum degree
        min_vertices = np.where(degrees == min_deg)[0]
        u = random.choice(min_vertices.tolist())
        
        # Try to add an edge from u to a vertex with low degree
        candidates = [v for v in range(n) if v != u and not adj[u, v]]
        candidates.sort(key=lambda v: degrees[v])
        
        found_swap = False
        for v in candidates[:10]:  # Try top 10 candidates
            if time.time() > deadline:
                break
            if not _check_c4_fast(adj, u, v):
                adj[u, v] = True
                adj[v, u] = True
                degrees[u] += 1
                degrees[v] += 1
                found_swap = True
                break
        
        if not found_swap:
            # Try removing an edge from a high-degree vertex and adding it elsewhere
            high_vertices = np.where(degrees == degrees.max())[0]
            w = random.choice(high_vertices.tolist())
            # Find a neighbor of w to remove
            neighbors = np.where(adj[w])[0]
            if len(neighbors) > 0:
                z = random.choice(neighbors.tolist())
                # Try to add edge (w, some other vertex) instead
                other_candidates = [x for x in range(n) if x != w and x != z and not adj[w, x]]
                other_candidates.sort(key=lambda x: degrees[x])
                for x in other_candidates[:5]:
                    if time.time() > deadline:
                        break
                    if not _check_c4_fast(adj, w, x):
                        # Remove (w,z), add (w,x)
                        adj[w, z] = False
                        adj[z, w] = False
                        adj[w, x] = True
                        adj[x, w] = True
                        degrees[w] = degrees[w]  # stays same
                        degrees[z] -= 1
                        degrees[x] += 1
                        found_swap = True
                        break
    
    # Check final result
    degrees = adj.sum(axis=1)
    if degrees.min() >= required_min_deg:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return (n, edges)
    
    return None

# EVOLVE-BLOCK-END
