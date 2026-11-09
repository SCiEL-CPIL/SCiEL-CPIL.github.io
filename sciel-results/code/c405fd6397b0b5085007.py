"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
import heapq

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycle_ear_decomp(adj, n, deadline):
    """Find a cycle using a modified approach: find any cycle, then extend if possible."""
    # First, find any cycle using DFS
    visited = {}
    parent = {}
    for start in range(n):
        if start in visited:
            continue
        stack = [(start, iter(sorted(adj[start])))]
        visited[start] = True
        parent[start] = -1
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                if not visited.get(nxt, False):
                    visited[nxt] = True
                    parent[nxt] = node
                    stack.append((nxt, iter(sorted(adj[nxt]))))
                    advanced = True
                    break
                elif nxt != parent.get(node, -1):
                    # Found a back edge, extract cycle
                    cycle = [node]
                    cur = node
                    while cur != nxt:
                        cur = parent[cur]
                        cycle.append(cur)
                    cycle.reverse()
                    return cycle
            if not advanced:
                stack.pop()
    return None

def _extend_cycle(cycle, adj, n, deadline):
    """Try to extend a cycle by finding paths between non-adjacent vertices in the cycle."""
    if not cycle:
        return cycle
    
    best_cycle = cycle[:]
    # Try multiple times to extend
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        # Try to find a path between two non-adjacent vertices in the cycle
        cycle_set = set(best_cycle)
        # Create a list of vertices in cycle order
        for i in range(len(best_cycle)):
            u = best_cycle[i]
            # Find next vertex in cycle
            v = best_cycle[(i + 1) % len(best_cycle)]
            # Try to find an alternative path from u to v that doesn't use edge (u,v)
            # and doesn't use other cycle edges
            # We'll do a BFS/DFS avoiding the direct edge and other cycle edges
            
            # For simplicity, try to find a path from u to v of length >= 2
            # that avoids the direct edge (u,v)
            # But we need to be careful not to use other cycle edges
            
            # Actually, let's try a different approach: find a vertex not in the cycle
            # that has edges to two non-adjacent vertices in the cycle
            for w in range(n):
                if w in cycle_set:
                    continue
                # Check if w has edges to at least two non-adjacent vertices in the cycle
                neighbors_in_cycle = []
                for j, vertex in enumerate(best_cycle):
                    if (min(w, vertex), max(w, vertex)) in adj.get(w, set()):
                        neighbors_in_cycle.append(j)
                
                if len(neighbors_in_cycle) >= 2:
                    # Check if any two are non-adjacent in the cycle
                    for a_idx in range(len(neighbors_in_cycle)):
                        for b_idx in range(a_idx + 1, len(neighbors_in_cycle)):
                            i1 = neighbors_in_cycle[a_idx]
                            i2 = neighbors_in_cycle[b_idx]
                            # Check if i1 and i2 are non-adjacent in the cycle
                            diff = abs(i1 - i2)
                            if diff > 1 and diff < len(best_cycle) - 1:
                                # We can extend the cycle
                                # Reconstruct: go from i1 to w to i2, then follow cycle from i2 to i1
                                new_cycle = []
                                # Start from i1, go to w, then to i2, then follow cycle from i2 back to i1
                                new_cycle.append(best_cycle[i1])
                                new_cycle.append(w)
                                new_cycle.append(best_cycle[i2])
                                # Now follow cycle from i2 to i1 (the long way)
                                idx = (i2 + 1) % len(best_cycle)
                                while idx != i1:
                                    new_cycle.append(best_cycle[idx])
                                    idx = (idx + 1) % len(best_cycle)
                                
                                if len(new_cycle) > len(best_cycle):
                                    best_cycle = new_cycle
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
            if improved:
                break
            # Also try to find a path of length 2 between non-adjacent cycle vertices
            # (already covered above with w)
            # Try to find a path of length 3+
            # For now, just the length-2 extension
            if not improved:
                break
    
    return best_cycle

def _decompose(edges, n):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    pieces = []
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = {i: set() for i in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    start_time = time.time()
    deadline = start_time + 40  # Per-graph deadline
    
    # Iteratively find and extract cycles
    max_iterations = 1000
    iters = 0
    while remaining and iters < max_iterations:
        if time.time() > deadline:
            break
        
        # Find a cycle
        cycle = _find_cycle_ear_decomp(adj, n, deadline)
        if not cycle or len(cycle) < 3:
            break
        
        # Try to extend the cycle
        extended = _extend_cycle(cycle, adj, n, deadline)
        if extended and len(extended) > len(cycle):
            cycle = extended
        
        # Verify the cycle is valid (all edges exist in remaining)
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = (min(u, v), max(u, v))
            if edge not in remaining:
                valid = False
                break
        
        if not valid:
            break
        
        # Extract this cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = (min(u, v), max(u, v))
            remaining.discard(edge)
            adj[u].discard(v)
            adj[v].discard(u)
        
        pieces.append(cycle[:])
        iters += 1
    
    # Remaining edges become single-edge pieces
    for u, v in remaining:
        pieces.append([u, v])
    
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # Return what we have so far, fill rest with singletons
            while len(results) < 3:
                s = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results)]
                results.append([list(e) for e in _instance(*s)])
            break
        
        edges = _instance(*spec)
        n = spec[0]
        pieces = _decompose(edges, n)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
