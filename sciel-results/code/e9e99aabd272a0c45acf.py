"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score_graph_fast(n, edges):
    """Compute the maximum degree sum over all triangles using bitmasks for speed."""
    # Build adjacency as bitmasks
    adj_mask = [0] * n
    deg = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
    
    # Find all triangles and compute max degree sum
    max_sum = 0
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_mask[u] >> v) & 1:
                # Common neighbors of u and v
                common = adj_mask[u] & adj_mask[v]
                # Only consider w > v to avoid duplicates
                common &= ~((1 << (v + 1)) - 1)
                while common:
                    w_bit = common & (-common)
                    w = w_bit.bit_length() - 1
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
                    common ^= w_bit
    return max_sum

def _build_adj_deg(n, edges):
    """Build adjacency bitmask and degree list from edge list."""
    adj_mask = [0] * n
    deg = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
    return adj_mask, deg

def _score_with_adj_deg(n, adj_mask, deg):
    """Compute max triangle degree sum given precomputed adj_mask and deg."""
    max_sum = 0
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_mask[u] >> v) & 1:
                common = adj_mask[u] & adj_mask[v]
                common &= ~((1 << (v + 1)) - 1)
                while common:
                    w_bit = common & (-common)
                    w = w_bit.bit_length() - 1
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
                    common ^= w_bit
    return max_sum

def _get_worst_triangles(n, adj_mask, deg, top_k=5):
    """Return the top_k worst triangles (by degree sum) as list of (sum, (u,v,w))."""
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_mask[u] >> v) & 1:
                common = adj_mask[u] & adj_mask[v]
                common &= ~((1 << (v + 1)) - 1)
                while common:
                    w_bit = common & (-common)
                    w = w_bit.bit_length() - 1
                    s = deg[u] + deg[v] + deg[w]
                    triangles.append((s, (u, v, w)))
                    common ^= w_bit
    triangles.sort(reverse=True)
    return triangles[:top_k]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n = 16
    target_edges = 65
    
    random.seed(42)
    
    # Multiple initial configurations to try
    initial_configs = []
    
    # Config 1: K_{8,8} plus one edge inside part A
    edges1 = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    initial_configs.append(edges1)
    
    # Config 2: K_{8,8} plus one edge inside part B
    edges2 = [[u, v] for u in range(8) for v in range(8, 16)] + [[8, 9]]
    initial_configs.append(edges2)
    
    # Config 3: Slightly different partition - K_{7,9} plus edges
    # 7*9 = 63, need 2 more edges
    edges3 = [[u, v] for u in range(7) for v in range(7, 16)]
    # Add 2 edges within the parts to reach 65
    edges3.append([0, 1])
    edges3.append([8, 9])
    initial_configs.append(edges3)
    
    # Config 4: K_{8,8} with the extra edge connecting a low-degree vertex pattern
    # Try different single edges within parts
    for extra in [(0,2), (1,3), (0,7), (8,15)]:
        edges_extra = [[u, v] for u in range(8) for v in range(8, 16)] + [list(extra)]
        initial_configs.append(edges_extra)
    
    # Evaluate all initial configs and pick the best
    best_edges = None
    best_score = float('inf')
    for config in initial_configs:
        # Validate: exactly 65 edges, no duplicates
        edge_set = set((min(u,v), max(u,v)) for u, v in config)
        if len(edge_set) != target_edges:
            continue
        score = _score_graph_fast(n, config)
        if score < best_score:
            best_score = score
            best_edges = list(config)
    
    if best_edges is None:
        best_edges = initial_configs[0]
        best_score = _score_graph_fast(n, best_edges)
    
    # Build structures for local search
    adj_mask, deg = _build_adj_deg(n, best_edges)
    best_edge_set = set((min(u,v), max(u,v)) for u, v in best_edges)
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    non_edges = [e for e in all_edges if e not in best_edge_set]
    
    # Local search with targeted improvements
    # Focus on edges that participate in the worst triangles
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Get worst triangles to guide the search
        worst_tris = _get_worst_triangles(n, adj_mask, deg, top_k=3)
        
        # Generate candidate swaps biased toward breaking worst triangles
        candidates = []
        
        # For each worst triangle, try removing one of its edges
        for _, (u, v, w) in worst_tris:
            for edge in [(u,v), (u,w), (v,w)]:
                edge_key = (min(edge[0], edge[1]), max(edge[0], edge[1]))
                if edge_key in best_edge_set:
                    candidates.append(('remove', edge_key))
        
        # Also add some random candidates for diversity
        for _ in range(5):
            e_remove_key = tuple(random.choice(best_edges))
            e_remove_key = (min(e_remove_key[0], e_remove_key[1]), max(e_remove_key[0], e_remove_key[1]))
            candidates.append(('remove', e_remove_key))
        
        # Shuffle candidates
        random.shuffle(candidates)
        
        improved = False
        for action, e_remove_key in candidates:
            if time.time() >= deadline:
                break
            
            u1, v1 = e_remove_key
            
            # Pick a random non-edge to add
            if not non_edges:
                break
            e_add = random.choice(non_edges)
            e_add_key = (min(e_add[0], e_add[1]), max(e_add[0], e_add[1]))
            if e_add_key in best_edge_set:
                continue
            
            u2, v2 = e_add_key
            
            # Perform the swap
            adj_mask[u1] &= ~(1 << v1)
            adj_mask[v1] &= ~(1 << u1)
            adj_mask[u2] |= (1 << v2)
            adj_mask[v2] |= (1 << u2)
            deg[u1] -= 1
            deg[v1] -= 1
            deg[u2] += 1
            deg[v2] += 1
            
            best_edge_set.remove(e_remove_key)
            best_edge_set.add(e_add_key)
            
            # Update best_edges list (for final return)
            # Find and replace
            for i, e in enumerate(best_edges):
                if (min(e[0], e[1]), max(e[0], e[1])) == e_remove_key:
                    best_edges[i] = list(e_add_key)
                    break
            
            # Update non_edges
            non_edges.remove(e_add)
            non_edges.append(e_remove_key)
            
            # Compute new score
            new_score = _score_with_adj_deg(n, adj_mask, deg)
            
            if new_score < best_score:
                best_score = new_score
                improved = True
                no_improve_count = 0
            else:
                # Revert
                adj_mask[u1] |= (1 << v1)
                adj_mask[v1] |= (1 << u1)
                adj_mask[u2] &= ~(1 << v2)
                adj_mask[v2] &= ~(1 << u2)
                deg[u1] += 1
                deg[v1] += 1
                deg[u2] -= 1
                deg[v2] -= 1
                
                best_edge_set.remove(e_add_key)
                best_edge_set.add(e_remove_key)
                
                for i, e in enumerate(best_edges):
                    if (min(e[0], e[1]), max(e[0], e[1])) == e_add_key:
                        best_edges[i] = list(e_remove_key)
                        break
                
                non_edges.remove(e_remove_key)
                non_edges.append(e_add)
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a random restart with a perturbation
                if time.time() < deadline - 5:
                    # Perturb: do 5 random swaps
                    for _ in range(5):
                        if not non_edges:
                            break
                        e_remove = random.choice(best_edges)
                        e_remove_key = (min(e_remove[0], e_remove[1]), max(e_remove[0], e_remove[1]))
                        e_add = random.choice(non_edges)
                        e_add_key = (min(e_add[0], e_add[1]), max(e_add[0], e_add[1]))
                        
                        u1, v1 = e_remove_key
                        u2, v2 = e_add_key
                        
                        adj_mask[u1] &= ~(1 << v1)
                        adj_mask[v1] &= ~(1 << u1)
                        adj_mask[u2] |= (1 << v2)
                        adj_mask[v2] |= (1 << u2)
                        deg[u1] -= 1
                        deg[v1] -= 1
                        deg[u2] += 1
                        deg[v2] += 1
                        
                        best_edge_set.remove(e_remove_key)
                        best_edge_set.add(e_add_key)
                        
                        for i, e in enumerate(best_edges):
                            if (min(e[0], e[1]), max(e[0], e[1])) == e_remove_key:
                                best_edges[i] = list(e_add_key)
                                break
                        
                        non_edges.remove(e_add)
                        non_edges.append(e_remove_key)
                    
                    no_improve_count = 0
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
