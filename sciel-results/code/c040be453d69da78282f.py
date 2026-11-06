# EVOLVE-BLOCK-START
import time
from random import Random

def _build_graph(n, p, seed):
    rng = Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _greedy_clique_partition_v3(adj, deadline):
    """
    Improved greedy clique partition with multi-start candidate selection.
    Uses bitmask operations for speed and tries multiple starting vertices
    to find larger cliques at each step.
    """
    n = len(adj)
    remaining = set()
    for u in range(n):
        for v in adj[u]:
            if v > u:
                remaining.add((u, v))
    
    if not remaining:
        return []
    
    cliques = []
    
    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    for u in range(n):
        for v in adj[u]:
            adj_mask[u] |= (1 << v)
    
    # Track remaining edges as a set of (u, v) tuples where u < v
    remaining_set = set(remaining)
    
    # Active vertices: those with at least one remaining edge
    active_vertices = set()
    for u, v in remaining_set:
        active_vertices.add(u)
        active_vertices.add(v)
    
    # Precompute degree in remaining subgraph for active vertices
    def get_remaining_degree(v):
        count = 0
        for u in active_vertices:
            if u != v and ((min(u, v), max(u, v)) in remaining_set):
                count += 1
        return count
    
    while remaining_set and time.time() < deadline:
        if not active_vertices:
            break
        
        # Build subgraph adjacency for active vertices from remaining edges
        sub_adj = {v: set() for v in active_vertices}
        for u, v in remaining_set:
            if u in active_vertices and v in active_vertices:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
        
        # Multi-start: try several candidate starting vertices
        # Sort active vertices by degree in remaining subgraph
        sorted_vertices = sorted(active_vertices, key=lambda v: len(sub_adj[v]), reverse=True)
        
        # Try top candidates (max-degree, 2nd, 3rd, plus a randomized one)
        candidates = sorted_vertices[:3]
        if len(sorted_vertices) > 3:
            # Add a random candidate from the middle
            mid_idx = len(sorted_vertices) // 2
            candidates.append(sorted_vertices[mid_idx])
        
        best_clique_found = None
        best_clique_size = 0
        
        for start_vertex in candidates:
            if time.time() > deadline:
                break
            
            # Find a large clique containing start_vertex using greedy extension
            # Candidates: neighbors of start_vertex in the remaining subgraph
            cand = set(sub_adj[start_vertex])
            
            if not cand:
                continue
            
            current_clique = {start_vertex}
            
            while cand and time.time() < deadline:
                # Pick vertex with most connections to current_clique in remaining subgraph
                best_v = None
                best_count = -1
                for v in cand:
                    # Count how many vertices in current_clique are connected to v in remaining subgraph
                    count = 0
                    for u in current_clique:
                        if u != v and ((min(u, v), max(u, v)) in remaining_set):
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_v = v
                
                if best_v is None or best_count < len(current_clique):
                    break
                
                current_clique.add(best_v)
                # Update candidates: only vertices connected to all in current_clique
                new_cand = set()
                for v in cand:
                    if v != best_v:
                        valid = True
                        for u in current_clique:
                            if u != v and ((min(u, v), max(u, v)) not in remaining_set):
                                valid = False
                                break
                        if valid:
                            new_cand.add(v)
                cand = new_cand
            
            # Verify all edges in current_clique are in remaining_set
            valid = True
            clique_list = sorted(current_clique)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    if (min(a, b), max(a, b)) not in remaining_set:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and len(current_clique) > best_clique_size:
                best_clique_size = len(current_clique)
                best_clique_found = clique_list
        
        if best_clique_found and len(best_clique_found) >= 2:
            # Remove all edges in this clique from remaining_set
            for i in range(len(best_clique_found)):
                for j in range(i + 1, len(best_clique_found)):
                    a, b = best_clique_found[i], best_clique_found[j]
                    remaining_set.discard((min(a, b), max(a, b)))
            cliques.append(best_clique_found)
            
            # Update active_vertices
            for v in best_clique_found:
                has_remaining = False
                for u in active_vertices:
                    if u != v and ((min(v, u), max(v, u)) in remaining_set):
                        has_remaining = True
                        break
                if not has_remaining:
                    active_vertices.discard(v)
        else:
            # Fallback: take a single edge
            if remaining_set:
                edge = next(iter(remaining_set))
                u, v = edge
                cliques.append([u, v])
                remaining_set.discard(edge)
                # Update active
                for vertex in (u, v):
                    has_remaining = False
                    for w in active_vertices:
                        if w != vertex and ((min(vertex, w), max(vertex, w)) in remaining_set):
                            has_remaining = True
                            break
                    if not has_remaining:
                        active_vertices.discard(vertex)
    
    # Cover any remaining edges as pairs
    for edge in list(remaining_set):
        if time.time() > deadline:
            break
        u, v = edge
        if (u, v) in remaining_set:
            cliques.append([u, v])
            remaining_set.discard((u, v))
    
    return cliques

def construct():
    start = time.time()
    deadline = start + 140  # Safety margin before 150s limit
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Fallback: trivial partition
            adj = _build_graph(n, p, seed)
            cliques = []
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        cliques.append([u, v])
            results.append(cliques)
            continue
        
        adj = _build_graph(n, p, seed)
        cliques = _greedy_clique_partition_v3(adj, deadline)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
