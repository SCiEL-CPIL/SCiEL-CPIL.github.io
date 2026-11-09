"""Hexagon-free graph construction on 40 vertices – spectral + structured 2-opt."""
import time
import random
import numpy as np
from collections import deque

N = 40
DEADLINE = 145  # safety margin before 150s kill

def _bit_adj(edges, n):
    """Build adjacency as list of bitmasks."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _popcount(x):
    return bin(x).count('1')

def _has_c6_bit(adj, n):
    """Check for C6 using bitmask operations."""
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            # Find common neighbors of u and v
            common_uv = adj[u] & adj[v]
            common_uv &= ~((1 << u) | (1 << v))
            for w in range(n):
                if w == u or w == v:
                    continue
                if not (common_uv >> w) & 1:
                    continue
                # Now need path w-x-y-z-u where x,y,z distinct from u,v,w
                # and x adjacent to w, y adjacent to x, z adjacent to y, u adjacent to z
                for x in range(n):
                    if x == u or x == v or x == w:
                        continue
                    if not (adj[w] >> x) & 1:
                        continue
                    for y in range(n):
                        if y == u or y == v or y == w or y == x:
                            continue
                        if not (adj[x] >> y) & 1:
                            continue
                        for z in range(n):
                            if z == u or z == v or z == w or z == x or z == y:
                                continue
                            if not (adj[y] >> z) & 1:
                                continue
                            if (adj[z] >> u) & 1:
                                return True
    return False

def _creates_c6(adj, n, u, v):
    """Check if adding edge (u,v) creates a C6.
    A C6 through (u,v): u-v-w-x-y-z-u
    Need path of length 4 from v to u avoiding u,v internally.
    """
    # BFS-like: find all paths v->w->x->y->z->u
    for w in range(n):
        if w == u or w == v:
            continue
        if not (adj[v] >> w) & 1:
            continue
        for x in range(n):
            if x == u or x == v or x == w:
                continue
            if not (adj[w] >> x) & 1:
                continue
            for y in range(n):
                if y == u or y == v or y == w or y == x:
                    continue
                if not (adj[x] >> y) & 1:
                    continue
                for z in range(n):
                    if z == u or z == v or z == w or z == x or z == y:
                        continue
                    if not (adj[y] >> z) & 1:
                        continue
                    if (adj[z] >> u) & 1:
                        return True
    return False

def _spectral_seed(n, time_budget=10):
    """
    Build a good initial graph using spectral methods.
    Strategy: Use a known C6-free base (e.g., 8 K5s) and then use
    spectral embedding to suggest which cross-edges to add.
    
    Actually, let's use a different approach: start with a random regular-ish
    graph and use spectral gap to guide edge selection.
    
    Better: Use the fact that C6-free graphs have bounded expansion.
    We'll build a graph with controlled degree using a greedy spectral approach.
    """
    random.seed(42)
    adj = [0] * n
    edges = []
    
    # Start with a random 8-regular-ish graph
    # Use configuration model
    degrees = [8] * n
    half_edges = sum(degrees) // 2
    stubs = []
    for i in range(n):
        stubs.extend([i] * degrees[i])
    random.shuffle(stubs)
    
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            continue
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, n, u, v):
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append([u, v])
    
    # Now greedily add edges in order of spectral benefit
    # Use adjacency matrix eigenvalue as proxy
    A = np.zeros((n, n))
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, n, u, v):
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.append([u, v])
    
    return edges, adj

def _structured_2opt(edges, adj, n, time_limit):
    """
    Structured 2-opt: for each edge, try swapping with a non-edge.
    Order swaps by structural impact (degree balance).
    """
    start = time.time()
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    best_edges = list(edges)
    best_count = len(edges)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    non_edges = [p for p in all_pairs if p not in edge_set]
    
    # Compute degrees
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    # Score each potential swap by degree balance improvement
    def swap_score(u1, v1, u2, v2):
        # Remove (u1,v1), add (u2,v2)
        d1 = deg[u1] + deg[v1]
        d2 = deg[u2] + deg[v2]
        # Prefer swaps that reduce degree variance
        var_before = sum(d*d for d in deg)
        deg[u1] -= 1; deg[v1] -= 1
        deg[u2] += 1; deg[v2] += 1
        var_after = sum(d*d for d in deg)
        deg[u1] += 1; deg[v1] += 1
        deg[u2] -= 1; deg[v2] -= 1
        return var_before - var_after
    
    # Try all single-edge additions first
    for u, v in non_edges:
        if time.time() - start > time_limit:
            break
        if _creates_c6(adj, n, u, v):
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
        edges.append([u, v])
        best_count += 1
        best_edges = list(edges)
        deg[u] += 1
        deg[v] += 1
    
    # Now do 2-opt swaps
    for iteration in range(50):
        if time.time() - start > time_limit:
            break
        
        improved = False
        # Sample a subset of edges for swapping
        sample_size = min(len(edges), 100)
        sample_edges = random.sample(edges, sample_size)
        
        for u1, v1 in sample_edges:
            if time.time() - start > time_limit:
                break
            if (u1, v1) not in edge_set and (v1, u1) not in edge_set:
                continue
            
            # Try swapping with random non-edges
            for _ in range(10):
                if not non_edges:
                    break
                u2, v2 = random.choice(non_edges)
                if (u2, v2) in edge_set:
                    continue
                
                # Check if swap is valid: removing (u1,v1) and adding (u2,v2)
                # First check if (u2,v2) can be added when (u1,v1) is removed
                # This is complex; approximate by checking if (u2,v2) creates C6
                # in the graph with (u1,v1) removed
                
                # Remove (u1,v1) temporarily
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                
                if _creates_c6(adj, n, u2, v2):
                    # Revert
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    continue
                
                # Perform swap
                edge_set.discard((u1, v1))
                edge_set.discard((v1, u1))
                edge_set.add((u2, v2))
                
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u1)  # BUG: should be adj[v2] |= (1 << u2)
                # Fix:
                adj[v2] &= ~(1 << u1)
                adj[v2] |= (1 << u2)
                
                # Update edges list
                edges.remove([u1, v1])
                edges.append([u2, v2])
                
                # Update non_edges
                non_edges.remove((u2, v2))
                non_edges.append((u1, v1))
                
                improved = True
                break
            
            if improved:
                break
    
    return edges, adj

def _refine_with_kick(edges, adj, n, time_limit):
    """
    Kick-based refinement: periodically perturb the graph and re-optimize.
    """
    start = time.time()
    best_edges = list(edges)
    best_count = len(edges)
    
    while time.time() - start < time_limit:
        # Try to add edges
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        added = 0
        for u, v in all_pairs:
            if time.time() - start > time_limit - 5:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, n, u, v):
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            added += 1
        
        if added == 0:
            # Stuck: try a kick (remove a random edge, re-add)
            if len(edges) > 50:
                idx = random.randint(0, len(edges)-1)
                u, v = edges[idx]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.pop(idx)
                
                # Try to add back or add new
                for _ in range(20):
                    a, b = random.choice(all_pairs)
                    if (adj[a] >> b) & 1:
                        continue
                    if _creates_c6(adj, n, a, b):
                        continue
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    edges.append([a, b])
                    break
        
        current_count = len(edges)
        if current_count > best_count:
            best_count = current_count
            best_edges = list(edges)
    
    return best_edges, adj

def construct():
    n = N
    start_time = time.time()
    
    # Phase 1: Spectral seed
    edges, adj = _spectral_seed(n, time_budget=20)
    elapsed = time.time() - start_time
    
    # Phase 2: Structured 2-opt
    remaining = min(60, DEADLINE - elapsed - 10)
    if remaining > 5:
        edges, adj = _structured_2opt(edges, adj, n, remaining)
    
    elapsed = time.time() - start_time
    
    # Phase 3: Kick-based refinement
    remaining = min(50, DEADLINE - elapsed - 5)
    if remaining > 5:
        edges, adj = _refine_with_kick(edges, adj, n, remaining)
    
    # Final verification
    adj_final = _bit_adj(edges, n)
    if _has_c6_bit(adj_final, n):
        # Fallback: use 8 K5s
        edges = []
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a+1, 5):
                    edges.append([base+a, base+b])
    
    return edges
