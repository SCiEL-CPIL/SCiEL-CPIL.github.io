import random
import time
import sys
import numpy as np

N = 64
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Precompute all coordinates as numpy arrays
    xs = np.arange(N, dtype=np.float64)
    ys = np.arange(N, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)
    coords = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Precompute all pairwise squared distances
    # dist[i, j] = squared distance between point i and point j
    # Using broadcasting: dist[i,j] = |xi - xj|^2 + |yi - yj|^2
    dx = coords[:, 0:1] - coords[1:, 0]  # shape (N, N-1) for pairs (i, j) where j > i
    # Actually let's do full matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 2)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # We need to find all isosceles triangles.
    # A triangle (a, b, c) is isosceles if any two of its sides are equal.
    # For each triple (a, b, c), check if d(a,b)==d(a,c) or d(a,b)==d(b,c) or d(a,c)==d(b,c)
    
    # Strategy: For each point a, group other points by their squared distance to a.
    # If two points b, c are at the same distance from a, then (a, b, c) is isosceles with apex a.
    # Also, we need to check if b is the apex: d(b,a)==d(b,c), etc.
    
    # Let's compute the "degree" of each point: how many isosceles triangles it participates in.
    # For point a to be apex: count pairs (b,c) where d(a,b)==d(a,c)
    # For point b to be apex with a fixed a: count c where d(b,a)==d(b,c)
    
    # Vectorized approach:
    # For each point i, compute histogram of distances to all other points.
    # The number of isosceles triangles where i is the apex = sum over each distance value of C(count, 2)
    
    # But we also need triangles where i is NOT the apex.
    # Triangle (a,b,c) is isosceles if:
    #   d(a,b) == d(a,c)  [a is apex]
    #   d(a,b) == d(b,c)  [b is apex]  
    #   d(a,c) == d(b,c)  [c is apex]
    
    # Total isosceles triangles = sum over all triples of indicator that it's isosceles.
    # Each point's "degree" = number of isosceles triangles containing that point.
    
    # Let's compute for each point i, how many isosceles triangles contain i.
    # This is more complex. Let me think differently.
    
    # Alternative: Build a graph where edge (i,j) exists if there's some k such that (i,j,k) is isosceles.
    # No, that's not quite right either.
    
    # Let me just directly compute: for each point, count how many pairs of other points form an isosceles triangle with it.
    
    # For point i:
    # Case 1: i is apex. Count pairs (j,k) where d(i,j) == d(i,k).
    # Case 2: i is a base vertex. For some j (apex), d(j,i) == d(j,k) for some k != i, j.
    #          This means: for each j, count how many k have d(j,k) == d(j,i), then sum over j.
    #          But we need to be careful not to double count.
    
    # Actually, let's just compute the total number of isosceles triangles and track which points are involved.
    
    # Simpler approach: For each pair (i,j), find all k such that (i,j,k) is isosceles.
    # (i,j,k) is isosceles if:
    #   d(i,j) == d(i,k)  -> k is on circle centered at i with radius d(i,j)
    #   d(i,j) == d(j,k)  -> k is on circle centered at j with radius d(i,j)  
    #   d(i,k) == d(j,k)  -> k is on perpendicular bisector of ij
    
    # This is getting complex. Let me use a different strategy.
    
    # For each point i, compute the multiset of distances to all other points.
    # Then for each distance value d that appears m times, there are C(m,2) pairs (j,k) 
    # where d(i,j)=d(i,k)=d, forming isosceles triangles with apex i.
    
    # For the "base" cases, I'll compute separately.
    
    # Let me just compute the degree of each point in the hypergraph.
    # degree[i] = number of isosceles triangles containing point i.
    
    # Method:
    # 1. For each point i as apex: for each distance d, if m points are at distance d from i,
    #    add C(m,2) to the triangle count, and each of those m points gets +C(m-1,1) = (m-1) to their degree?
    #    No, each triangle (i,j,k) with apex i contributes 1 to degree of i, j, and k.
    
    # Let me just iterate through all isosceles triangles and count.
    
    # Precompute: for each point i, and each possible squared distance d, 
    # how many points are at distance d from i?
    
    # Max squared distance = (N-1)^2 + (N-1)^2 = 2*(N-1)^2 = 2*63^2 = 7938
    MAX_DIST = 2 * (N - 1) ** 2
    
    # dist_count[i, d] = number of points j != i such that dist_sq[i,j] == d
    # Shape: (TOTAL_POINTS, MAX_DIST + 1)
    dist_count = np.zeros((TOTAL_POINTS, MAX_DIST + 1), dtype=np.int32)
    
    for i in range(TOTAL_POINTS):
        # Get all distances from i to other points
        d = dist_sq[i].copy()
        d[i] = 0  # exclude self
        # Count occurrences of each distance
        for dist_val in np.unique(d):
            if dist_val > 0 and dist_val <= MAX_DIST:
                count = np.sum(d == dist_val)
                dist_count[i, int(dist_val)] = count
    
    # Now compute degree[i] = number of isosceles triangles containing point i
    degree = np.zeros(TOTAL_POINTS, dtype=np.int64)
    
    # Case 1: i is the apex
    # For point i, for each distance d with count m, there are C(m,2) triangles (i,j,k) with apex i.
    # Each such triangle contributes 1 to degree[i] and 1 to degree[j] and 1 to degree[k].
    # So degree[i] gets C(m,2) from this case.
    # For j and k: each pair (j,k) at distance d from i forms a triangle, so j gets 1 for each such pair.
    # Number of pairs containing j where both j and k are at distance d from i = (m-1) for each such j.
    
    # Let me compute apex contributions first
    for i in range(TOTAL_POINTS):
        for d in range(1, MAX_DIST + 1):
            m = dist_count[i, d]
            if m >= 2:
                c = m * (m - 1) // 2
                degree[i] += c
    
    # Now for the base cases:
    # Triangle (i,j,k) is isosceles with apex j if d(j,i) == d(j,k).
    # This is already counted when we process j as apex!
    # Wait, no. When I say "i is apex", I mean d(i,j) == d(i,k).
    # When j is apex: d(j,i) == d(j,k).
    # These are different conditions.
    
    # So I need to also count:
    # Case 2: For triangle (i,j,k), j is apex means d(j,i) == d(j,k).
    # This is symmetric to case 1 but from j's perspective.
    
    # Actually, I think I've only been counting the case where the first point is the apex.
    # Let me reconsider.
    
    # An isosceles triangle (a,b,c) satisfies one of:
    #   d(a,b) == d(a,c)  [a is apex]
    #   d(a,b) == d(b,c)  [b is apex]
    #   d(a,c) == d(b,c)  [c is apex]
    
    # When I iterate over "i as apex", I'm counting triangles where i is the apex,
    # i.e., d(i,j) == d(i,k). This covers all triangles where i is the apex.
    
    # But a triangle can have i as a base vertex, not the apex.
    # For example, triangle (a,b,c) where b is apex: d(b,a)==d(b,c).
    # In this case, a is a base vertex. When I process a as "apex", I check d(a,b)==d(a,c),
    # which is a different condition.
    
    # So I need to count all three cases. But by symmetry, if I count "i as apex" for all i,
    # I'm counting each isosceles triangle exactly once (the apex is unique unless it's equilateral,
    # but equilateral triangles aren't possible on integer grids except trivially).
    
    # Wait, is that true? An isosceles triangle has exactly one apex (the vertex between the two equal sides),
    # unless it's equilateral. On a grid, equilateral triangles are very rare.
    
    # Actually, an isosceles triangle could have two equal sides in different ways only if it's equilateral.
    # So each non-equilateral isosceles triangle has a unique apex.
    
    # Therefore, by counting "i as apex" for all i, I count each isosceles triangle exactly once.
    # But I need the degree of each vertex, not just the total count.
    
    # Let me redo: for each point i, count how many isosceles triangles have i as a vertex.
    
    # degree[i] = (triangles where i is apex) + (triangles where i is a base vertex)
    
    # Triangles where i is apex: already computed as sum over d of C(dist_count[i,d], 2)
    
    # Triangles where i is a base vertex: 
    # For each other point j (potential apex), count how many k != i,j such that d(j,i) == d(j,k).
    # This is: for each j, let d_ji = dist_sq[i,j]. Count points k (k != i, k != j) with dist_sq[j,k] == d_ji.
    # That count is dist_count[j, d_ji] - 1 (excluding i itself, since dist_sq[j,i] = d_ji).
    # But we also need k != j, which is already excluded since dist_count excludes self.
    # Wait, dist_count[j, d] counts points k != j with dist_sq[j,k] == d.
    # So dist_count[j, d_ji] includes i (since dist_sq[j,i] = d_ji).
    # So the number of valid k is dist_count[j, d_ji] - 1.
    
    # So base_contribution[i] = sum over j != i of max(0, dist_count[j, dist_sq[i,j]] - 1)
    
    # But wait, this counts each triangle (i,j,k) with apex j as contributing 1 to degree[i].
    # And triangle (i,j,k) has apex j, so it's counted once in base_contribution[i] for vertex i.
    # Good.
    
    # But I also need to make sure I'm not double-counting. Each isosceles triangle (a,b,c) with apex b:
    # - Contributes 1 to degree[a] via base_contribution[a] (j=b, d=dist_sq[a,b])
    # - Contributes 1 to degree[c] via base_contribution[c] (j=b, d=dist_sq[c,b])
    # - Contributes 1 to degree[b] via apex_count[b]
    # Total: 3, which is correct (the triangle contains 3 vertices).
    
    # Let me compute base contributions efficiently.
    # base_contribution[i] = sum over j != i of (dist_count[j, dist_sq[i,j]] - 1) if dist_sq[i,j] > 0
    
    # This is O(N^3) if done naively. Let me vectorize.
    
    # For each j, and for each distance d, the points i at distance d from j get contribution (dist_count[j,d] - 1).
    # So I can do:
    # for j in range(TOTAL_POINTS):
    #     for d in range(1, MAX_DIST+1):
    #         m = dist_count[j, d]
    #         if m >= 1:
    #             # All points i at distance d from j get (m-1) added to their base contribution
    #             # Find those points
    #             mask = (dist_sq[j] == d) & (dist_sq[j] > 0)
    #             indices = np.where(mask)[0]
    #             base_contrib[indices] += (m - 1)
    
    # This is still O(N^2 * MAX_DIST) which is too slow.
    
    # Alternative: For each j, create an array where entry i = dist_count[j, dist_sq[i,j]] - 1 if dist_sq[i,j]>0 else 0
    # Then sum over i for each i? No, I need per-i sums.
    
    # Let me think... 
    # base_contribution[i] = sum_j f(j, i) where f(j,i) = dist_count[j, dist_sq[i,j]] - 1 if dist_sq[i,j] > 0
    
    # I can compute this as: for each j, add (dist_count[j, dist_sq[i,j]] - 1) to base_contribution[i] for all i.
    
    # Using numpy:
    # For fixed j, dist_sq[j, :] gives distances from j to all points.
    # I want to look up dist_count[j, dist_sq[j, i]] for each i.
    # dist_count[j] is an array indexed by distance value.
    # So I can do: lookup = dist_count[j, dist_sq[j, :]]  # This indexes dist_count[j] with each distance value
    # But dist_sq[j, j] = 0, and dist_count[j, 0] might be 0 or undefined.
    
    # Let me handle this carefully.
    
    base_contrib = np.zeros(TOTAL_POINTS, dtype=np.int64)
    
    for j in range(TOTAL_POINTS):
        # Get distances from j to all points
        d_from_j = dist_sq[j, :].copy()
        d_from_j[j] = -1  # Mark self
        
        # For valid distances (> 0), look up dist_count[j, d]
        valid = d_from_j > 0
        if np.any(valid):
            # Create lookup array
            lookup = np.zeros(TOTAL_POINTS, dtype=np.int64)
            valid_indices = np.where(valid)[0]
            valid_dists = d_from_j[valid_indices]
            # Look up dist_count[j, valid_dists]
            counts = dist_count[j, valid_dists.astype(np.int32)]
            lookup[valid_indices] = counts - 1  # Subtract 1 to exclude j itself... 
            # Wait, dist_count[j, d] counts points k != j with dist_sq[j,k] == d.
            # For point i at distance d from j, the number of other points k (k != i, k != j) at distance d from j is dist_count[j,d] - 1.
            # Yes, that's correct.
            
            base_contrib += lookup
    
    # Now degree[i] = apex_degree[i] + base_contrib[i]
    # But wait, I already computed degree[i] as apex contribution. Let me rename.
    
    apex_degree = np.zeros(TOTAL_POINTS, dtype=np.int64)
    for i in range(TOTAL_POINTS):
        for d in range(1, MAX_DIST + 1):
            m = dist_count[i, d]
            if m >= 2:
                c = m * (m - 1) // 2
                apex_degree[i] += c
    
    degree = apex_degree + base_contrib
    
    # Now do greedy deletion: repeatedly remove the point with the highest degree.
    # After removing a point, we need to update the degrees of remaining points.
    # This is complex. Let me use a simpler approach: just remove points in order of initial degree,
    # and check validity after each removal.
    
    # Actually, for the greedy deletion to work properly, I need to recompute degrees after each removal.
    # That's expensive. Let me try a different approach.
    
    # Alternative: Start with all points, and repeatedly find and remove one point from each isosceles triangle.
    # Greedy: remove the point that is in the most isosceles triangles.
    
    # Let me implement a proper greedy deletion with incremental updates.
    
    # For simplicity and speed, let me just sort points by initial degree and remove them in that order,
    # checking if the remaining set is valid.
    
    # Actually, let me try a different structural approach: use the distance histogram to quickly check
    # if a point can be added, and do a greedy addition with a different ordering.
    
    # Let me try: sort points by some heuristic (e.g., number of "conflicts") and greedily add.
    
    # For now, let me just do the greedy deletion properly.
    
    # Track which points are still active
    active = np.ones(TOTAL_POINTS, dtype=bool)
    
    # Current degree of each active point
    current_degree = degree.copy()
    
    # We'll iteratively remove the point with the highest current degree.
    # After removing point p, we need to update degrees of other points.
    # When p is removed, any isosceles triangle containing p is "solved".
    # For each other point q, its degree decreases by the number of isosceles triangles 
    # that contained both p and q.
    
    # This is getting complicated. Let me just do a simple version:
    # Remove points in order of decreasing initial degree, and return the remaining set.
    
    # Sort by degree descending
    order = np.argsort(-current_degree)
    
    # Remove points until no isosceles triangles remain
    # For efficiency, let me just remove a fraction and check.
    
    # Actually, let me try a completely different approach that might be faster:
    # Use the structure of the grid to find a good set directly.
    
    # Points on a line: if I take points on a single row, no three form an isosceles triangle? 
    # No, three collinear points don't form a triangle at all, so they're fine.
    # But I want to maximize points.
    
    # Let me try: take every other point in a checkerboard pattern.
    # Or take points where (x+y) % k == 0 for some k.
    
    # Let me try several structured sets and see which is best.
    
    best_set = []
    best_size = 0
    
    # Try various structured sets
    # 1. All points on one row
    # 2. Checkerboard
    # 3. Every k-th point in x and y
    
    # Let me try a greedy approach with a better ordering.
    # Order points by their "isolation" - points that are far from others.
    
    # For each point, compute the minimum distance to any other point.
    # Points with larger minimum distance are "more isolated" and easier to include.
    
    min_dist = np.full(TOTAL_POINTS, np.inf)
    for i in range(TOTAL_POINTS):
        d = dist_sq[i].copy()
        d[i] = np.inf
        min_dist[i] = np.sqrt(np.min(d))
    
    # Sort by min_dist descending (most isolated first)
    order2 = np.argsort(-min_dist)
    
    # Greedy addition with this ordering
    selected = []
    dist_sets = {}
    
    def sq_dist_fast(i, j):
        return int(dist_sq[i, j])
    
    for p in order2:
        if time.time() > deadline - 10:
            break
        valid = True
        dists_to_p = []
        for s in selected:
            d = sq_dist_fast(p, s)
            dists_to_p.append((s, d))
        
        seen_dists = set()
        for s, d in dists_to_p:
            if d in seen_dists:
                valid = False
                break
            seen_dists.add(d)
        
        if not valid:
            continue
        
        for s, d in dists_to_p:
            if d in dist_sets[s]:
                valid = False
                break
        
        if valid:
            selected.append(p)
            dist_sets[p] = set(d for _, d in dists_to_p)
            for s, d in dists_to_p:
                dist_sets[s].add(d)
    
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = selected
    
    # Also try the original random greedy as fallback
    if time.time() < deadline - 30:
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            random.seed(seed)
            candidates = list(range(TOTAL_POINTS))
            random.shuffle(candidates)
            
            sel = []
            dsets = {}
            
            for p in candidates:
                valid = True
                dists = []
                for s in sel:
                    d = sq_dist_fast(p, s)
                    dists.append((s, d))
                
                seen = set()
                for s, d in dists:
                    if d in seen:
                        valid = False
                        break
                    seen.add(d)
                
                if not valid:
                    continue
                
                for s, d in dists:
                    if d in dsets[s]:
                        valid = False
                        break
                
                if valid:
                    sel.append(p)
                    dsets[p] = set(d for _, d in dists)
                    for s, d in dists:
                        dsets[s].add(d)
            
            if len(sel) > best_size:
                best_size = len(sel)
                best_set = sel
    
    return [tuple(coords[i].astype(int)) for i in best_set]
