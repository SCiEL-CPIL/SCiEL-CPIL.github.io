import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def build_polarity_graph(q):
        # Projective plane of order q
        # Points: (x, y, 1) for x,y in F_q, plus (x, 1, 0) for x in F_q, plus (0, 1, 0)
        # Total q^2 + q + 1 points
        # Polarity: (x1, y1, z1) perp (x2, y2, z2) if x1*x2 + y1*y2 + z1*z2 == 0 ?
        # Standard polarity for P^2(q) (q odd): x1*x2 + y1*y2 + z1*z2 = 0 is not a polarity (symmetric bilinear form).
        # We need a symmetric bilinear form.
        # Let's use the standard construction for C4-free graphs from finite fields.
        # Vertices: (x, y) in F_q x F_q.
        # Edge if y - y' = x * x' is NOT symmetric.
        
        # Let's use the graph G where V = F_q x F_q.
        # Edge (x,y) ~ (x',y') if y - y' = x * x' + x * x' ? No.
        
        # Correct C4-free graph:
        # V = F_q x F_q.
        # Edge (x,y) ~ (x',y') if y - y' = x * x' for x != x'?
        
        # Actually, the "Polarity" graph of the affine plane is:
        # V = F_q x F_q.
        # Edge (x,y) ~ (x',y') if y - y' = x * x' is not symmetric.
        
        # Let's use the graph defined by:
        # V = F_q x F_q.
        # Edge (x,y) ~ (x',y') if y - y' = x * x' is replaced by a symmetric relation.
        # How about y * x' - y' * x = 1? (Points on a hyperbola)
        
        # Let's stick to the known result:
        # The graph with vertices F_q x F_q and edges (x,y)-(x',y') if y - y' = x * x' is bipartite? No.
        
        # Let's use the construction from "C4-free graphs with high minimum degree".
        # One such graph is the "Erdos-Renyi" graph?
        
        # Let's try the graph on q^2 vertices:
        # V = {0, ..., q^2-1}.
        # Edge i-j if (i//q, i%q) and (j//q, j%q) satisfy y - y' = x * x' mod q?
        # This is not symmetric.
        
        # Symmetric version:
        # Edge if y - y' = x * x' AND y' - y = x' * x ?
        # This implies 2(y-y') = x x' + x' x = 2 x x'.
        # So y - y' = x x'.
        # This is the same condition. It is symmetric only if x x' = -x' x, i.e., 2 x x' = 0.
        
        # For q odd, we can define edges as:
        # (x,y) ~ (x',y') if y - y' = x * x' is FALSE.
        
        # Let's use the Polarity Graph of the Projective Plane.
        # n = q^2 + q + 1.
        # Min degree q+1.
        # For q=7, n=57, min degree 8.
        
        # Construction of Polarity Graph of P^2(q):
        # Points are 1-dim subspaces of F_q^3.
        # Polarity: p perp l if dot(p, l) = 0.
        # We need a self-polarity (symmetric).
        # Form: x1*x2 + y1*y2 + z1*z2 = 0 is not a polarity for q odd?
        # It is a symmetric bilinear form.
        # The graph has vertices = points of P^2(q).
        # Edge between p and q if p is on the polar line of q.
        # This graph is C4-free.
        # Min degree is q+1.
        
        # Let's implement this.
        # Represent points as (x, y, z) normalized.
        # Generate all points.
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        for x in range(q):
            points.append((x, 1, 0))
        points.append((0, 1, 0)) # Wait, (0,1,0) is already in the second loop if x=0?
        # Second loop: x in 0..q-1, (x, 1, 0).
        # So (0,1,0) is included.
        # Total q^2 + q points?
        # P^2(q) has q^2 + q + 1 points.
        # Standard basis:
        # (x, y, 1) for x,y in F_q -> q^2 points.
        # (x, 1, 0) for x in F_q -> q points.
        # (0, 1, 0) is included in the second set (x=0).
        # We are missing (0, 0, 1)? No, (0,0,1) is in the first set.
        # We are missing points with z=0, y=0? (1,0,0).
        # (1,0,0) is not in (x,y,1) nor (x,1,0).
        # So add (1,0,0).
        # Total: q^2 + q + 1.
        
        # Let's verify the list.
        # Set 1: z=1. (x,y,1). q^2 points.
        # Set 2: z=0, y=1. (x,1,0). q points.
        # Set 3: z=0, y=0. (1,0,0). 1 point.
        # Total q^2+q+1.
        
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        for x in range(q):
            points.append((x, 1, 0))
        points.append((1, 0, 0))
        
        n_pts = len(points)
        adj = np.zeros((n_pts, n_pts), dtype=bool)
        
        # Polarity: p1 perp p2 if dot(p1, p2) == 0?
        # For q odd, the form x1x2 + y1y2 + z1z2 is non-degenerate.
        # However, we need a SYMMETRIC polarity.
        # The form is symmetric.
        # Edge if dot(p, q) == 0.
        
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                p1 = points[i]
                p2 = points[j]
                dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
                if dot == 0:
                    adj[i][j] = adj[j][i] = True
        
        return adj, points

    def solve(n_target, seed):
        rng = random.Random(seed)
        
        # Start with q=7 polarity graph (n=57)
        q = 7
        adj_full, _ = build_polarity_graph(q)
        n_full = len(adj_full)
        
        # Remove vertices to reach n_target
        # We want to keep the "best" vertices?
        # Just remove random ones for now, then fix degrees.
        vertices_to_keep = list(range(n_full))
        rng.shuffle(vertices_to_keep)
        keep_set = set(vertices_to_keep[:n_target])
        
        # Build subgraph
        adj = np.zeros((n_target, n_target), dtype=bool)
        # Map old indices to new indices
        idx_map = {old: new for new, old in enumerate(sorted(keep_set))}
        
        for old_i, new_i in idx_map.items():
            for old_j, new_j in idx_map.items():
                if adj_full[old_i][old_j]:
                    adj[new_i][new_j] = True
        
        # Compute common neighbors
        common = np.zeros((n_target, n_target), dtype=np.int32)
        for i in range(n_target):
            for j in range(n_target):
                if i != j:
                    # Count common neighbors
                    # Use bitsets for speed?
                    # n=47 is small, loop is fine.
                    count = 0
                    for k in range(n_target):
                        if adj[i][k] and adj[j][k]:
                            count += 1
                    common[i][j] = count
        
        def can_add(u, v):
            if u == v or adj[u][v]:
                return False
            # Check 1: common(u, v) <= 1
            if common[u][v] > 1:
                return False
            # Check 2: For all w in N(v), common(u, w) == 0
            for w in range(n_target):
                if w == u or w == v:
                    continue
                if adj[v][w]:
                    if common[u][w] > 0:
                        return False
            # Check 3: For all w in N(u), common(v, w) == 0
            for w in range(n_target):
                if w == u or w == v:
                    continue
                if adj[u][w]:
                    if common[v][w] > 0:
                        return False
            return True

        def add_edge(u, v):
            adj[u][v] = adj[v][u] = True
            # Update common counts
            # common[u][v] is already computed, but we need to update others
            # common[u][w] += adj[v][w]
            # common[v][w] += adj[u][w]
            # common[w][u] += adj[v][w]
            # common[w][v] += adj[u][w]
            for w in range(n_target):
                if w == u or w == v:
                    continue
                common[u][w] += adj[v][w]
                common[w][u] += adj[v][w]
                common[v][w] += adj[u][w]
                common[w][v] += adj[u][w]
            # common[u][v] remains the same?
            # No, common[u][v] is |N(u) \cap N(v)|.
            # Adding edge (u,v) does NOT change N(u) \cap N(v) because u is not in N(u) and v is not in N(v).
            # So common[u][v] is unchanged.
            pass

        # Local Search
        # Goal: min degree >= n_target - 39
        target_deg = n_target - 39
        
        # Iterate
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            # Find min degree vertex
            degs = [int(np.sum(adj[i])) for i in range(n_target)]
            min_deg = min(degs)
            if min_deg >= target_deg:
                break
            
            # Try to add edges to boost degrees
            # Sort vertices by degree
            verts = sorted(range(n_target), key=lambda x: degs[x])
            improved = False
            for u in verts:
                if degs[u] >= target_deg:
                    continue
                # Try to add edge from u to v
                # Prefer v with high degree? Or random?
                candidates = [v for v in range(n_target) if v != u and not adj[u][v]]
                rng.shuffle(candidates)
                for v in candidates:
                    if can_add(u, v):
                        add_edge(u, v)
                        degs[u] += 1
                        degs[v] += 1
                        improved = True
                        break
                if degs[u] >= target_deg:
                    break
            
            if not improved:
                # Stuck? Try random restart or swap?
                # For now, just break if no improvement
                break
        
        # Verify
        degs = [int(np.sum(adj[i])) for i in range(n_target)]
        if min(degs) >= target_deg:
            edges = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n_target, 'edges': edges}
        
        return None

    best_result = None
    for n in [47, 46, 45]:
        if time.time() > deadline - 5:
            break
        for seed in range(50):
            if time.time() > deadline - 5:
                break
            res = solve(n, seed)
            if res:
                best_result = res
                break
        if best_result:
            break
    
    if not best_result:
        # Fallback to previous logic or simple graph
        # Return a valid small graph
        best_result = {'n': 39, 'edges': []}
        
    return best_result
