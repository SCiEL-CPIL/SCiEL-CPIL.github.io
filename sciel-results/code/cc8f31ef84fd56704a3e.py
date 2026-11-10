import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    best_score = 20
    best_points = None
    
    # Helper: Check if 4 points are collinear
    def has_four_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                count = 2 # i and j
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    # Cross product (x3-x1, y3-y1) x (dx, dy)
                    if (x3 - x1) * dy == (y3 - y1) * dx:
                        count += 1
                        if count >= 4:
                            return True
        return False

    # Helper: Count 3-collinear triples
    def count_triples(points):
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # To avoid double counting, we just count lines with >=3 points
                # But for objective, we want to maximize the number of distinct triples?
                # Or just the number of lines?
                # Maximizing triples is a good proxy.
                pts_on_line = []
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if (x3 - x1) * dy == (y3 - y1) * dx:
                        pts_on_line.append(k)
                if len(pts_on_line) >= 1:
                    # Number of triples involving i, j and one of pts_on_line
                    # This counts each triple exactly once? 
                    # Triple (a,b,c) is counted when processing pair (a,b) with c, 
                    # (a,c) with b, (b,c) with a.
                    # So this counts each triple 3 times.
                    # Consistent metric.
                    count += len(pts_on_line)
        return count

    # Helper: Exact MIS for 3-uniform hypergraph
    def calc_mis(points):
        n = len(points)
        # Build list of edges (triples)
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    if (x3 - x1) * dy == (y3 - y1) * dx:
                        edges.append((i, j, k))
        
        if not edges:
            return n
            
        # Map vertex to set of edges it belongs to
        vertex_edges = [set() for _ in range(n)]
        for idx, e in enumerate(edges):
            for v in e:
                vertex_edges[v].add(idx)
                
        # Branch and Bound
        best = 0
        # Order vertices by degree (descending) to prune faster?
        # Simple B&B:
        def dfs(available_mask, current_size, current_set):
            nonlocal best
            if current_size + bin(available_mask).count('1') <= best:
                return
            
            if available_mask == 0:
                if current_size > best:
                    best = current_size
                return
                
            # Pick a vertex with highest degree in the remaining graph?
            # Or just the first available?
            # Let's pick the one with the most edges to prune.
            candidates = []
            for v in range(n):
                if available_mask & (1 << v):
                    # Count active edges incident to v
                    deg = 0
                    for e_idx in vertex_edges[v]:
                        e = edges[e_idx]
                        if (available_mask & (1 << e[0])) and (available_mask & (1 << e[1])) and (available_mask & (1 << e[2])):
                            deg += 1
                    candidates.append((deg, v))
                    
            if not candidates:
                # No active edges, take all remaining
                best = max(best, current_size + bin(available_mask).count('1'))
                return
                
            candidates.sort(reverse=True)
            _, v = candidates[0]
            
            # Branch 1: Include v
            # If we include v, we must remove all vertices that form an edge with v
            # i.e., if (v, a, b) is an edge, we cannot include a and b together?
            # No, if we include v, we cannot include ANY pair (a,b) such that (v,a,b) is an edge.
            # This means if v is in the set, the set of remaining vertices must be an independent set
            # in the graph where edges are pairs (a,b) from triples containing v.
            # Actually, simpler:
            # If v is selected, we must remove from `available` all vertices that are in any edge with v?
            # No. If v is selected, we cannot select any two vertices a,b such that (v,a,b) is an edge.
            # This is complex to mask directly.
            
            # Alternative B&B for 3-uniform:
            # State: Set of available vertices.
            # If we pick v, we remove v. And we must ensure no other edge is completed.
            # An edge is completed if all 3 are picked.
            # If we pick v, then for any edge (v, a, b), we cannot pick BOTH a and b.
            # This doesn't force removing a or b immediately.
            
            # Let's use a simpler recursive step:
            # 1. Pick a vertex v from available.
            # 2. Option A: Don't pick v. Remove v from available.
            # 3. Option B: Pick v. 
            #    If we pick v, we must ensure that no edge containing v is fully selected.
            #    This means for every edge (v, a, b) in the remaining graph, we must NOT select both a and b.
            #    This is equivalent to finding an independent set in the "shadow" graph?
            
            # Given n=20, a simpler exhaustive search with pruning might work.
            # Let's try a different approach: 
            # The score is the max size of a subset with NO 3 collinear.
            # This is the independence number of the 3-uniform hypergraph.
            
            # Standard B&B for Hypergraph MIS:
            # 1. If no edges, take all.
            # 2. Pick an edge (a,b,c).
            #    At least one of a,b,c must be excluded.
            #    Branch 1: Exclude a. (Remove a from available)
            #    Branch 2: Exclude b. (Remove b from available)
            #    Branch 3: Exclude c. (Remove c from available)
            #    Wait, this is for Vertex Cover (hitting set).
            #    MIS in hypergraph = N - Min Vertex Cover?
            #    Yes! The size of the Maximum Independent Set in a hypergraph is N - Size of Minimum Vertex Cover.
            #    A Vertex Cover in a hypergraph is a set of vertices that intersects every edge.
            #    So we want to MINIMIZE the Vertex Cover size.
            #    This is easier? Min VC in 3-uniform hypergraph is NP-hard, but n=20 is small.
            
            # Let's solve Min Vertex Cover.
            # B&B for Min VC:
            # 1. If no edges, cover size = 0.
            # 2. Pick an edge (a,b,c).
            #    We must pick at least one of a,b,c.
            #    Branch 1: Pick a. Add a to cover. Remove all edges containing a.
            #    Branch 2: Pick b. Add b to cover. Remove all edges containing b.
            #    Branch 3: Pick c. Add c to cover. Remove all edges containing c.
            #    (Symmetric branches can be pruned if a,b,c have same degrees etc)
            
            pass

        # Implement Min VC B&B
        best_vc = n
        def dfs_vc(remaining_edges_mask, current_cover_size):
            nonlocal best_vc
            if current_cover_size >= best_vc:
                return
            if remaining_edges_mask == 0:
                if current_cover_size < best_vc:
                    best_vc = current_cover_size
                return
            
            # Pick the first set bit in remaining_edges_mask
            # This corresponds to a specific edge
            # We need to map bit index to edge tuple
            # To make this fast, we should precompute edge indices
            
            # Get index of first edge
            idx = (remaining_edges_mask & -remaining_edges_mask).bit_length() - 1
            e = edges[idx]
            a, b, c = e
            
            # Branch on a
            dfs_vc(remaining_edges_mask & ~vertex_edge_mask[a], current_cover_size + 1)
            # Branch on b
            dfs_vc(remaining_edges_mask & ~vertex_edge_mask[b], current_cover_size + 1)
            # Branch on c
            dfs_vc(remaining_edges_mask & ~vertex_edge_mask[c], current_cover_size + 1)
            
        # Precompute vertex_edge_mask: bitmask of edges incident to vertex v
        vertex_edge_mask = [0] * n
        for idx, e in enumerate(edges):
            for v in e:
                vertex_edge_mask[v] |= (1 << idx)
                
        dfs_vc((1 << len(edges)) - 1, 0)
        
        return n - best_vc

    # Main Search Loop
    for _ in range(100000):
        if time.time() > deadline:
            break
        
        # Generate candidate
        # Try structured start: 5x4 grid perturbed
        if random.random() < 0.5:
            base = [(i, j) for i in range(5) for j in range(4)]
            random.shuffle(base)
            pts = base[:20]
            # Perturb
            for i in range(20):
                if random.random() < 0.5:
                    pts[i] = [pts[i][0] + random.randint(-2, 2), pts[i][1] + random.randint(-2, 2)]
        else:
            pts = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(20)]
            
        # Ensure distinct
        # Simple check
        if len(set(tuple(p) for p in pts)) < 20:
            continue
            
        # Repair 4-collinear
        # Local Search to fix 4-collinear and optimize
        current_pts = [p[:] for p in pts]
        
        # Optimization Phase
        for step in range(200):
            if time.time() > deadline:
                break
            if has_four_collinear(current_pts):
                # Find a 4-collinear set and move one
                # Simplified: just move a random point
                idx = random.randint(0, 19)
                old_x, old_y = current_pts[idx]
                # Try random new pos
                for _ in range(10):
                    new_x = random.randint(-100, 100)
                    new_y = random.randint(-100, 100)
                    current_pts[idx] = [new_x, new_y]
                    if not has_four_collinear(current_pts) and len(set(tuple(p) for p in current_pts)) == 20:
                        break
            else:
                # Valid. Try to improve (increase triples)
                old_triples = count_triples(current_pts)
                idx = random.randint(0, 19)
                old_x, old_y = current_pts[idx]
                improved = False
                for _ in range(10):
                    new_x = old_x + random.randint(-3, 3)
                    new_y = old_y + random.randint(-3, 3)
                    if abs(new_x) > 100 or abs(new_y) > 100:
                        continue
                    if (new_x, new_y) == (old_x, old_y):
                        continue
                    # Check distinct
                    if any(p[0] == new_x and p[1] == new_y for p in current_pts if p is not current_pts[idx]):
                        continue
                        
                    current_pts[idx] = [new_x, new_y]
                    if not has_four_collinear(current_pts) and len(set(tuple(p) for p in current_pts)) == 20:
                        new_triples = count_triples(current_pts)
                        if new_triples > old_triples:
                            improved = True
                            break
                        else:
                            # Revert
                            current_pts[idx] = [old_x, old_y]
                if not improved:
                    # Random kick
                    current_pts[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
                    
        # Final Evaluation
        if not has_four_collinear(current_pts) and len(set(tuple(p) for p in current_pts)) == 20:
            score = calc_mis(current_pts)
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in current_pts]
                
    if best_points is None:
        # Fallback: Parabola
        best_points = [[x, x * x] for x in range(-10, 10)]
        
    return best_points
