import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    """
    Build a C4-free graph on n vertices with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 110  # Safety margin
    
    best_result = None
    best_n = 0
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple random restarts with local search
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            result = _local_search_c4_free(n, required_min_deg, deadline, attempt)
            if result is not None:
                if n > best_n:
                    best_n = n
                    best_result = result
                    # If we found a solution for the current n, we can return it 
                    # unless we want to try higher n (but we are iterating down).
                    # So returning immediately is optimal if we are maximizing n.
                    return result
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a valid small graph
    return {'n': 1, 'edges': []}


def _check_c4_add(adj_sets, n, u, v):
    """
    Check if adding edge (u,v) creates a C4.
    Returns True if adding (u,v) creates a C4, False otherwise.
    """
    # A C4 is formed if there exists a in N(u) \ {v} such that 
    # N(a) intersects N(v) \ {u}.
    # Let A = N(u) - {v}
    # Let B = N(v) - {u}
    # We need to check if any a in A has a neighbor in B.
    
    # Optimization: Iterate over the smaller set
    nu = adj_sets[u]
    nv = adj_sets[v]
    
    # We want to know if there is a in nu such that a != v and (adj_sets[a] & (nv - {u})) is not empty.
    
    # Let's iterate over neighbors of u
    for a in nu:
        if a == v:
            continue
        # Check if a has any common neighbor with v, excluding u.
        # Common neighbors of a and v are adj_sets[a] & nv.
        # If u is in this intersection, we must ignore it because the cycle would be u-a-u-v-u which is not a simple cycle (length 3 or degenerate).
        # Actually, if u~a and u~v and a~v, that's a triangle. 
        # The C4 is u-a-b-v-u. 
        # So we need b in N(a) and b in N(v). b != u. b != a (no self loop). b != v (no self loop).
        # So we check if (adj_sets[a] & nv) contains any element other than u.
        
        # To be fast, we can do:
        common = adj_sets[a] & nv
        if common:
            # If common contains only u, then no C4 is formed via this a.
            # If common contains anything else, C4 is formed.
            if u in common:
                common.discard(u)
            if common:
                return True
    return False


def _init_greedy(n, rng):
    """
    Initialize with a greedy C4-free graph.
    """
    adj_sets = [set() for _ in range(n)]
    
    # Generate all potential edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if not _check_c4_add(adj_sets, n, u, v):
            adj_sets[u].add(v)
            adj_sets[v].add(u)
            
    return adj_sets


def _local_search_c4_free(n, min_deg_req, deadline, seed_offset):
    """Construct a C4-free graph on n vertices with min degree >= min_deg_req using local search."""
    rng = random.Random(seed_offset * 7919 + 31)
    
    # Use greedy initialization for better starting point (dense C4-free)
    adj_sets = _init_greedy(n, rng)
    
    # Phase 2: Local search to improve minimum degree
    max_iters = 5000
    
    for _iter in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate degrees
        degrees = [len(s) for s in adj_sets]
        min_deg = min(degrees)
        
        if min_deg >= min_deg_req:
            # Convert to edge list
            edges = []
            for u in range(n):
                for v in adj_sets[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        # Find vertices with minimum degree
        min_verts = [i for i, d in enumerate(degrees) if d == min_deg]
        v = rng.choice(min_verts)
        
        # Strategy 1: Try to add an edge from v to a non-neighbor
        candidates = [u for u in range(n) if u != v and u not in adj_sets[v]]
        rng.shuffle(candidates)
        
        added = False
        for u in candidates:
            if time.time() > deadline:
                break
            if not _check_c4_add(adj_sets, n, u, v):
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                added = True
                break
        
        if added:
            continue
        
        # Strategy 2: Edge swap - remove an edge (a,b) and add (v,u) for some u
        # We want to free up "C4 capacity" for v.
        # If v cannot connect to u, it's because there is a C4 path u-a-b-v.
        # Removing (a,b) might break this.
        
        # Let's try to find a blocking edge.
        # For each candidate u that v cannot connect to, find the blocking structure.
        # This is expensive. Instead, let's try random edge swaps that involve v.
        
        # Select a random edge (a, b) from the graph.
        all_current_edges = []
        for u in range(n):
            for w in adj_sets[u]:
                if u < w:
                    all_current_edges.append((u, w))
                        
        if not all_current_edges:
            break
        
        found_swap = False
        for _ in range(10): # Try more swaps per iteration
            if time.time() > deadline:
                break
            
            a, b = rng.choice(all_current_edges)
            
            # Remove edge (a, b)
            adj_sets[a].remove(b)
            adj_sets[b].remove(a)
            
            # Try to add edges to v
            # We want to increase degree of v.
            # Try all candidates for v again? Or just random ones?
            # Let's try to connect v to some u.
            candidates = [u for u in range(n) if u != v and u not in adj_sets[v]]
            rng.shuffle(candidates)
            
            added_edges = []
            for u in candidates:
                if time.time() > deadline:
                    break
                if not _check_c4_add(adj_sets, n, u, v):
                    adj_sets[u].add(v)
                    adj_sets[v].add(u)
                    added_edges.append(u)
                    # We removed 1 edge. We added 1. Net change 0 in edge count, but deg(v) increased by 1.
                    # This is good if deg(v) was the minimum.
                    break # Just add one edge to keep it simple and fast
                    
            new_degrees = [len(s) for s in adj_sets]
            new_min = min(new_degrees)
            
            if new_min >= min_deg:
                # Accept the swap
                found_swap = True
                break
            else:
                # Revert
                adj_sets[a].add(b)
                adj_sets[b].add(a)
                for u in added_edges:
                    adj_sets[u].remove(v)
                    adj_sets[v].remove(u)
        
        if not found_swap:
            # If no improvement, break to restart? Or continue?
            # Continuing might allow other vertices to improve, which might unblock v.
            # But if we are stuck, restarting is better.
            if _iter > 1000:
                break

    # Final check
    degrees = [len(s) for s in adj_sets]
    if min(degrees) >= min_deg_req:
        edges = []
        for u in range(n):
            for v in adj_sets[u]:
                if u < v:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    return None

# EVOLVE-BLOCK-END
