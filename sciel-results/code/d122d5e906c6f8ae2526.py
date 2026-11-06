"""Rewritten program using degree-ordered edge-replacement heuristic."""
import itertools

def construct():
    n = 16
    target_edges = 65  # floor(16^2/4) + 1 = 64 + 1
    
    # Start with K_{8,8} plus one intra-part edge (same as seed)
    # Vertices 0..7 in part A, 8..15 in part B
    edges_set = set()
    for u in range(8):
        for v in range(8, 16):
            edges_set.add((u, v))
    edges_set.add((0, 1))
    
    # Helper: get adjacency set
    def get_adj(edge_set, n):
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def get_degrees(adj):
        return [len(adj[i]) for i in range(len(adj))]
    
    def find_worst_triangle_deg_sum(adj, deg):
        """Find the triangle with the maximum degree sum. Return (max_sum, triangle)."""
        max_sum = -1
        worst_tri = None
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[v]:
                    if w <= v:
                        continue
                    if w in adj[u]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                            worst_tri = (u, v, w)
        return max_sum, worst_tri
    
    def try_improve(edge_set, adj, deg, max_iter=5000):
        """
        Iteratively try to reduce the maximum triangle degree sum.
        For the worst triangle, try swapping one of its edges with a non-edge
        that reduces the degree of the highest-degree vertex in the triangle.
        """
        for _ in range(max_iter):
            max_sum, tri = find_worst_triangle_deg_sum(adj, deg)
            if tri is None:
                break  # No triangles
            
            u, v, w = tri
            # Sort by degree descending to prioritize reducing highest degree vertex
            tri_vertices = sorted([u, v, w], key=lambda x: -deg[x])
            hi, mid, lo = tri_vertices[0], tri_vertices[1], tri_vertices[2]
            
            improved = False
            # Try removing each edge of the triangle and adding a non-edge
            # to keep edge count constant, prioritizing edges incident to hi
            # Edges in triangle: (u,v), (v,w), (u,w)
            tri_edges = [(u, v), (v, w), (u, w)]
            
            # Sort triangle edges by whether they involve hi
            def edge_priority(e):
                a, b = e
                if a == hi or b == hi:
                    return 0
                if a == mid or b == mid:
                    return 1
                return 2
            
            tri_edges_sorted = sorted(tri_edges, key=edge_priority)
            
            for rem_u, rem_v in tri_edges_sorted:
                if (rem_u, rem_v) not in edge_set and (rem_v, rem_u) not in edge_set:
                    continue
                # Remove this edge
                if (rem_u, rem_v) in edge_set:
                    edge_set.discard((rem_u, rem_v))
                else:
                    edge_set.discard((rem_v, rem_u))
                
                # Find candidate non-edges to add
                # We want to add an edge that doesn't create worse triangles
                # and ideally reduces the degree sum
                best_add = None
                best_score = max_sum + 1  # Must be strictly better
                
                # Generate candidate non-edges
                # Focus on edges that don't involve hi if possible, or that connect
                # low-degree vertices
                candidates = []
                for a in range(n):
                    for b in range(a + 1, n):
                        if (a, b) in edge_set or (b, a) in edge_set:
                            continue
                        # Skip if this would be the same edge we just removed
                        if (a == rem_u and b == rem_v) or (a == rem_v and b == rem_u):
                            continue
                        candidates.append((a, b))
                
                # Limit candidates for speed: prioritize edges not involving hi
                # and connecting lower-degree vertices
                def add_priority(cand):
                    a, b = cand
                    if a == hi or b == hi:
                        return 2
                    if a == mid or b == mid:
                        return 1
                    return 0
                
                candidates.sort(key=lambda c: (add_priority(c), deg[c[0]] + deg[c[1]]))
                
                # Try top candidates
                for add_u, add_v in candidates[:50]:  # Limit to 50 for speed
                    # Add the edge
                    edge_set.add((add_u, add_v))
                    
                    # Recompute adj and deg
                    new_adj = get_adj(edge_set, n)
                    new_deg = get_degrees(new_adj)
                    
                    # Check if this creates any triangle with higher sum
                    new_max_sum, _ = find_worst_triangle_deg_sum(new_adj, new_deg)
                    
                    if new_max_sum < max_sum:
                        best_add = (add_u, add_v)
                        best_score = new_max_sum
                        break
                    elif new_max_sum == max_sum:
                        # Tie: prefer if hi's degree decreased
                        if new_deg[hi] < deg[hi]:
                            best_add = (add_u, add_v)
                            best_score = max_sum
                            # Don't break, keep looking for strict improvement
                    else:
                        # Worse, undo
                        edge_set.discard((add_u, add_v))
                
                if best_add is not None:
                    # Keep the best addition
                    if best_score < max_sum:
                        # Strict improvement: keep
                        # Remove the trial edge if we added it during search
                        # Actually we need to re-apply the best_add
                        # First, make sure edge_set is in state after removing tri edge
                        # and then add best_add
                        if (best_add[0], best_add[1]) not in edge_set and (best_add[1], best_add[0]) not in edge_set:
                            edge_set.add((best_add[0], best_add[1]))
                        # Update adj and deg
                        adj = get_adj(edge_set, n)
                        deg = get_degrees(adj)
                        improved = True
                        break
                    else:
                        # Tie: only accept if hi degree decreased
                        if (best_add[0], best_add[1]) not in edge_set and (best_add[1], best_add[0]) not in edge_set:
                            edge_set.add((best_add[0], best_add[1]))
                        adj = get_adj(edge_set, n)
                        deg = get_degrees(adj)
                        if deg[hi] < deg[tri_vertices[0]]:
                            improved = True
                            break
                        else:
                            # Undo
                            edge_set.discard((best_add[0], best_add[1]))
                
                # Undo the removal of tri edge if we didn't improve
                if not improved:
                    edge_set.add((rem_u, rem_v))
            
            if not improved:
                break
        
        return edge_set, adj, deg, max_sum
    
    # Run the heuristic
    edge_set, adj, deg, max_sum = try_improve(edge_set, get_adj(edge_set, n), get_degrees(get_adj(edge_set, n)), max_iter=10000)
    
    # Convert to list of lists
    edges_list = [[u, v] for (u, v) in edge_set]
    
    return {'n': n, 'edges': edges_list}
