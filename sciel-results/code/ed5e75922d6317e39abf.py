"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    def build_adj(edges):
        """Build adjacency bitmasks and degrees from edge list."""
        adj = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def compute_score_from_adj(adj, deg):
        """Compute max triangle degree sum using bitmasks."""
        max_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                while common:
                    k = common & (-common)
                    k_idx = k.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k_idx]
                    if s > max_sum:
                        max_sum = s
                    common &= common - 1
        return max_sum
    
    def compute_score(edges):
        """Compute the maximum triangle degree sum."""
        adj, deg = build_adj(edges)
        return compute_score_from_adj(adj, deg)
    
    def is_valid(edges):
        """Check if edges form a simple graph with exactly target_edges edges."""
        if len(edges) != target_edges:
            return False
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in seen:
                return False
            seen.add(key)
        return True
    
    def get_worst_triangles(adj, deg):
        """Get the set of triangles that achieve the maximum degree sum."""
        max_sum = 0
        worst_triangles = set()
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                while common:
                    k = common & (-common)
                    k_idx = k.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k_idx]
                    if s > max_sum:
                        max_sum = s
                        worst_triangles = {(i, j, k_idx)}
                    elif s == max_sum:
                        worst_triangles.add((i, j, k_idx))
                    common &= common - 1
        return max_sum, worst_triangles
    
    def edge_affects_triangles(edge, worst_triangles):
        """Check if removing/adding an edge affects any worst triangle."""
        u, v = edge
        for tri in worst_triangles:
            if u in tri or v in tri:
                return True
        return False
    
    best_edges = None
    best_score = float('inf')
    
    # Try multiple partition sizes
    for small_size in [6, 7, 5, 8]:
        if time.time() > deadline - 15.0:
            break
        
        large_size = n - small_size
        base_edges = [[u, v] for u in range(small_size) for v in range(small_size, n)]
        num_internal = target_edges - len(base_edges)
        
        if num_internal < 0 or num_internal > large_size * (large_size - 1) // 2:
            continue
        
        large_vertices = list(range(small_size, n))
        all_internal_edges = [(a, b) for a in range(small_size, n) for b in range(a+1, n)]
        
        # Generate multiple random initial configurations
        num_restarts = 15 if small_size in [6, 7] else 8
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
            
            # Random selection of internal edges
            random.shuffle(all_internal_edges)
            random_internal = all_internal_edges[:num_internal]
            
            current_edges = base_edges + [[a, b] for a, b in random_internal]
            
            if not is_valid(current_edges):
                continue
            
            current_score = compute_score(current_edges)
            
            if current_score < best_score:
                best_score = current_score
                best_edges = current_edges[:]
            
            # Local search with targeted moves
            if time.time() > deadline - 10.0:
                continue
            
            adj, deg = build_adj(current_edges)
            current_score = compute_score_from_adj(adj, deg)
            
            internal_set = set()
            for u, v in current_edges:
                if u >= small_size and v >= small_size:
                    internal_set.add((min(u,v), max(u,v)))
            
            def rebuild_edges_from_internal(internal_set, base_edges):
                """Rebuild full edge list from base cross edges + internal set."""
                edges = base_edges[:]
                for u, v in internal_set:
                    edges.append([u, v])
                return edges
            
            base_edges_for_partition = [[u, v] for u in range(small_size) for v in range(small_size, n)]
            
            improved = True
            max_ls_iters = 12000 if small_size in [6, 7] else 6000
            iters = 0
            while improved and iters < max_ls_iters:
                improved = False
                iters += 1
                if time.time() > deadline - 5.0:
                    break
                
                # Get worst triangles to guide search
                _, worst_triangles = get_worst_triangles(adj, deg)
                
                internal_list = list(internal_set)
                random.shuffle(internal_list)
                
                # --- Single-edge swap ---
                for u1, v1 in internal_list:
                    if time.time() > deadline - 5.0:
                        break
                    
                    attempts = 8 if edge_affects_triangles((u1, v1), worst_triangles) else 4
                    for _ in range(attempts):
                        u2 = random.randint(small_size, n-1)
                        v2 = random.randint(small_size, n-1)
                        if u2 == v2:
                            continue
                        u2, v2 = min(u2, v2), max(u2, v2)
                        if (u2, v2) == (u1, v1) or (u2, v2) in internal_set:
                            continue
                        if (adj[u2] >> v2) & 1:
                            continue
                        
                        new_adj = adj[:]
                        new_deg = deg[:]
                        
                        new_adj[u1] &= ~(1 << v1)
                        new_adj[v1] &= ~(1 << u1)
                        new_deg[u1] -= 1
                        new_deg[v1] -= 1
                        
                        new_adj[u2] |= (1 << v2)
                        new_adj[v2] |= (1 << u2)
                        new_deg[u2] += 1
                        new_deg[v2] += 1
                        
                        new_score = compute_score_from_adj(new_adj, new_deg)
                        
                        if new_score < current_score:
                            adj = new_adj
                            deg = new_deg
                            current_score = new_score
                            internal_set.remove((u1, v1))
                            internal_set.add((u2, v2))
                            current_edges = rebuild_edges_from_internal(internal_set, base_edges_for_partition)
                            improved = True
                            break
                    if improved:
                        break
                
                # --- Two-edge swap (2-opt) ---
                if not improved and len(internal_list) >= 2:
                    if time.time() > deadline - 5.0:
                        pass
                    else:
                        # Pick two random internal edges to remove
                        two = random.sample(internal_list, 2)
                        (u1, v1), (u3, v3) = two
                        
                        # Try random replacements for both
                        found = False
                        for _ in range(12):
                            u2 = random.randint(small_size, n-1)
                            v2 = random.randint(small_size, n-1)
                            if u2 == v2:
                                continue
                            u2, v2 = min(u2, v2), max(u2, v2)
                            if (u2, v2) in internal_set or (u2, v2) == (u1, v1) or (u2, v2) == (u3, v3):
                                continue
                            if (adj[u2] >> v2) & 1:
                                continue
                            
                            u4 = random.randint(small_size, n-1)
                            v4 = random.randint(small_size, n-1)
                            if u4 == v4:
                                continue
                            u4, v4 = min(u4, v4), max(u4, v4)
                            if (u4, v4) in internal_set or (u4, v4) == (u1, v1) or (u4, v4) == (u3, v3) or (u4, v4) == (u2, v2):
                                continue
                            if (adj[u4] >> v4) & 1:
                                continue
                            # Also check that the two new edges don't share vertices with each other in a way that creates issues
                            # (they're both internal, so fine as long as distinct)
                            
                            new_adj = adj[:]
                            new_deg = deg[:]
                            
                            new_adj[u1] &= ~(1 << v1)
                            new_adj[v1] &= ~(1 << u1)
                            new_deg[u1] -= 1
                            new_deg[v1] -= 1
                            
                            new_adj[u3] &= ~(1 << v3)
                            new_adj[v3] &= ~(1 << u3)
                            new_deg[u3] -= 1
                            new_deg[v3] -= 1
                            
                            new_adj[u2] |= (1 << v2)
                            new_adj[v2] |= (1 << u2)
                            new_deg[u2] += 1
                            new_deg[v2] += 1
                            
                            new_adj[u4] |= (1 << v4)
                            new_adj[v4] |= (1 << u4)
                            new_deg[u4] += 1
                            new_deg[v4] += 1
                            
                            new_score = compute_score_from_adj(new_adj, new_deg)
                            
                            if new_score < current_score:
                                adj = new_adj
                                deg = new_deg
                                current_score = new_score
                                internal_set.discard((u1, v1))
                                internal_set.discard((u3, v3))
                                internal_set.add((u2, v2))
                                internal_set.add((u4, v4))
                                current_edges = rebuild_edges_from_internal(internal_set, base_edges_for_partition)
                                improved = True
                                found = True
                                break
                        
                if current_score < best_score:
                    best_score = current_score
                    best_edges = current_edges[:]
    
    # Final high-precision polish pass: try every possible single internal edge
    # replacement exhaustively on the best solution found so far
    if best_edges is not None and is_valid(best_edges) and time.time() < deadline - 3.0:
        # Determine which partition was used by checking the structure
        # Try to infer small_size from the best solution
        best_adj, best_deg = build_adj(best_edges)
        best_score_now = compute_score_from_adj(best_adj, best_deg)
        
        # Try each small_size to see which matches
        for ss in [6, 7, 5, 8]:
            base_e = [[u, v] for u in range(ss) for v in range(ss, n)]
            if len(base_e) != 65 - len([e for e in best_edges if e[0] >= ss and e[1] >= ss]):
                continue
            # Check all cross edges are present
            cross_set = set((min(u,v), max(u,v)) for u, v in base_e)
            best_set = set((min(u,v), max(u,v)) for u, v in best_edges)
            if not cross_set.issubset(best_set):
                continue
            
            # This is likely the partition used
            internal_set = set()
            for u, v in best_edges:
                if u >= ss and v >= ss:
                    internal_set.add((min(u,v), max(u,v)))
            
            base_edges_for_polish = base_e[:]
            
            # Exhaustive single-edge replacement
            all_internal_candidates = [(a, b) for a in range(ss, n) for b in range(a+1, n)]
            improved_polish = True
            while improved_polish and time.time() < deadline - 1.0:
                improved_polish = False
                adj_p, deg_p = build_adj(best_edges)
                score_p = compute_score_from_adj(adj_p, deg_p)
                
                internal_list_p = list(internal_set)
                for u1, v1 in internal_list_p:
                    if time.time() > deadline - 1.0:
                        break
                    for u2, v2 in all_internal_candidates:
                        if time.time() > deadline - 1.0:
                            break
                        if (u2, v2) == (u1, v1) or (u2, v2) in internal_set:
                            continue
                        if (adj_p[u2] >> v2) & 1:
                            continue
                        
                        new_adj = adj_p[:]
                        new_deg = deg_p[:]
                        new_adj[u1] &= ~(1 << v1)
                        new_adj[v1] &= ~(1 << u1)
                        new_deg[u1] -= 1
                        new_deg[v1] -= 1
                        new_adj[u2] |= (1 << v2)
                        new_adj[v2] |= (1 << u2)
                        new_deg[u2] += 1
                        new_deg[v2] += 1
                        
                        new_score = compute_score_from_adj(new_adj, new_deg)
                        if new_score < score_p:
                            adj_p = new_adj
                            deg_p = new_deg
                            score_p = new_score
                            internal_set.discard((u1, v1))
                            internal_set.add((u2, v2))
                            best_edges = rebuild_edges_from_internal(internal_set, base_edges_for_polish)
                            improved_polish = True
                            break
                    if improved_polish:
                        break
                
                if score_p < best_score:
                    best_score = score_p
                    best_edges = rebuild_edges_from_internal(internal_set, base_edges_for_polish)
            
            break  # Found the matching partition
    
    # Final verification
    if best_edges is None or not is_valid(best_edges):
        base_edges = [[u, v] for u in range(6) for v in range(6, 16)]
        best_edges = base_edges + [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
