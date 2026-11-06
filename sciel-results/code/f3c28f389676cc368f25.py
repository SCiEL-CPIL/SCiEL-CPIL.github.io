import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds safety margin for 120s limit
    
    n = 20
    # Use a larger grid to allow more diversity, but keep coordinates small for integer precision
    # The constraint is |x|, |y| <= 100.
    # Let's stick to a reasonable range for random search, e.g., [-50, 50]
    coords = list(range(-50, 51))
    
    def is_valid(pts):
        """Check if no 4 points are collinear."""
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                # Count points on the line
                count = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) - dy * (x3 - x1) == 0:
                        count += 1
                if count >= 2:
                    return False
        return True

    def get_triples(pts):
        """Get list of all 3-collinear triples as tuples of indices."""
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = pts[k]
                    if dx * (y3 - y1) - dy * (x3 - x1) == 0:
                        triples.append((i, j, k))
        return triples

    def min_vertex_cover(triples):
        """
        Find minimum vertex cover of 3-uniform hypergraph using bitmask B&B.
        """
        if not triples:
            return 0
        
        m = len(triples)
        # Map edge index to bitmask of vertices
        edge_verts = [0] * m
        # Map vertex to bitmask of edges
        vert_edges = [0] * n
        
        for idx, t in enumerate(triples):
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            edge_verts[idx] = mask
            vert_edges[t[0]] |= (1 << idx)
            vert_edges[t[1]] |= (1 << idx)
            vert_edges[t[2]] |= (1 << idx)
            
        # Initial active edges mask
        initial_active = (1 << m) - 1
        
        best_cover = n # Upper bound
        
        # Precompute popcount for small integers? Python 3.8+ has int.bit_count()
        # We can use a simple lower bound: number of active edges / 2 (since each vertex covers at most 2 edges? No, 3 edges).
        # Actually, a vertex covers at most 3 edges in this hypergraph.
        # So LB = ceil(active_edge_count / 3).
        
        def popcount(x):
            return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count("1")

        def bnb(active_mask, depth):
            nonlocal best_cover
            
            if active_mask == 0:
                if depth < best_cover:
                    best_cover = depth
                return
                
            # Lower bound: ceil(popcount(active_mask) / 3)
            # Actually, a vertex can cover multiple edges. Max 3.
            # So we need at least ceil(num_active_edges / 3) more vertices.
            lb = (popcount(active_mask) + 2) // 3
            if depth + lb >= best_cover:
                return
                
            # Heuristic: Pick an edge that is "hard" to cover?
            # Or just pick the first set bit.
            # Let's pick an edge index
            # Find lowest set bit
            # active_mask & -active_mask gives lowest set bit
            lsb = active_mask & -active_mask
            e_idx = lsb.bit_length() - 1
            
            v_mask = edge_verts[e_idx]
            
            # Branch on each vertex in this edge
            # v_mask has 3 bits set
            # Iterate through them
            while v_mask:
                v_lsb = v_mask & -v_mask
                v_idx = v_lsb.bit_length() - 1
                
                # New active mask: remove all edges covered by v_idx
                new_active = active_mask & ~vert_edges[v_idx]
                
                bnb(new_active, depth + 1)
                
                v_mask ^= v_lsb
                
        bnb(initial_active, 0)
        return best_cover

    def calculate_score(pts):
        triples = get_triples(pts)
        cover_size = min_vertex_cover(triples)
        return n - cover_size

    def random_points():
        pts = []
        while len(pts) < n:
            x = random.choice(coords)
            y = random.choice(coords)
            p = (x, y)
            if p not in pts:
                pts.append(p)
        return pts

    best_score = 21
    best_pts = None
    
    # Initial random search
    # We can do more iterations now if B&B is fast
    for _ in range(100):
        if time.time() > deadline:
            break
        pts = random_points()
        if is_valid(pts):
            score = calculate_score(pts)
            if score < best_score:
                best_score = score
                best_pts = pts[:]
                if best_score <= 12:
                    break

    if best_pts is None:
        # Fallback: parabola
        # Check if parabola is valid (no 4 collinear)
        # y = x^2. Any 3 points on a parabola are not collinear?
        # Actually, a line intersects a parabola in at most 2 points.
        # So no 3 are collinear. Thus no 4 are collinear.
        # Score: Largest subset with no 3 collinear.
        # Since NO 3 are collinear in the whole set, the largest subset with no 3 collinear is 20.
        # Wait, the score is the LARGEST subset with NO 3 collinear.
        # If no 3 are collinear, then the whole set has no 3 collinear.
        # So score = 20.
        # This is a BAD score (we want to minimize).
        # We want MANY collinear triples to force the vertex cover to be large?
        # No, Score = N - MinVertexCover(Triples).
        # To minimize Score, we need to MAXIMIZE MinVertexCover.
        # MinVertexCover is the min number of points to remove to break all collinear triples.
        # If we have many triples, the cover is likely larger.
        # If we have NO triples, Cover = 0, Score = 20.
        # So we want a configuration with MANY collinear triples.
        # The parabola has 0 triples, so Score = 20.
        # A grid has many triples.
        
        # Let's construct a better fallback: a grid-like structure that avoids 4 collinear.
        # e.g. points (i, j) for i in 0..4, j in 0..3. 5x4=20 points.
        # Check 4 collinear:
        # Horizontal: 5 points in a row -> 4 collinear! Invalid.
        # We need to perturb.
        
        # Let's just return the parabola if we fail to find anything better, 
        # but it's a bad score. 
        # The random search should find something better than 20 quickly.
        return [[x, x*x] for x in range(-10, 10)]

    current_pts = best_pts[:]
    current_score = best_score
    
    # Local Search
    while time.time() < deadline:
        improved = False
        # Try moving each point
        for i in range(n):
            if time.time() > deadline:
                break
            original_p = current_pts[i]
            # Try a few random moves for this point
            for _ in range(20): # Increased moves per point
                new_x = random.choice(coords)
                new_y = random.choice(coords)
                new_p = (new_x, new_y)
                if new_p == original_p:
                    continue
                if new_p in current_pts:
                    continue
                
                # Temporarily replace
                old_p = current_pts[i]
                current_pts[i] = new_p
                
                if is_valid(current_pts):
                    score = calculate_score(current_pts)
                    if score < current_score:
                        current_score = score
                        best_score = score
                        best_pts = current_pts[:]
                        improved = True
                    else:
                        current_pts[i] = old_p
                else:
                    current_pts[i] = old_p
            
            if improved:
                break # Restart loop from improved state
        
        if not improved:
            # Stuck. Try a random restart from best? Or just break.
            # Let's try a few random restarts from the best point
            restarts = 0
            while not improved and restarts < 5 and time.time() < deadline:
                # Perturb best_pts
                perturbed = best_pts[:]
                for _ in range(3):
                    idx = random.randint(0, n-1)
                    new_x = random.choice(coords)
                    new_y = random.choice(coords)
                    new_p = (new_x, new_y)
                    if new_p not in perturbed:
                        perturbed[idx] = new_p
                if is_valid(perturbed):
                    score = calculate_score(perturbed)
                    if score < best_score:
                        best_score = score
                        best_pts = perturbed[:]
                        current_pts = perturbed[:]
                        current_score = score
                        improved = True
                restarts += 1
            if not improved:
                break

    return [list(p) for p in best_pts]
