"""Construct 32 lattice points on triangular lattice maximizing unit-distance edges."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _dist2(a1, b1, a2, b2):
    """Squared distance on triangular lattice: (a1-a2)^2 + (a1-a2)(b1-b2) + (b1-b2)^2."""
    da = a1 - a2
    db = b1 - b2
    return da*da + da*db + db*db

def _count_edges(points):
    """Count unit-distance pairs among given (a,b) points."""
    point_set = set(tuple(p) for p in points)
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    edge_count = 0
    for a, b in point_set:
        for da, db in neighbors:
            if (a+da, b+db) in point_set:
                edge_count += 1
    return edge_count // 2

def _hexagon_points(r):
    """All points with hex distance <= r from origin: max(|a|,|b|,|a+b|) <= r."""
    pts = []
    for a in range(-r, r+1):
        for b in range(-r, r+1):
            if abs(a+b) <= r:
                pts.append((a, b))
    return pts

def _spiral_points(n):
    """Generate n points in a triangular lattice spiral starting from origin."""
    if n <= 0:
        return []
    points = [(0, 0)]
    if n == 1:
        return points
    
    # The spiral goes in rings. Ring k (1-indexed) has 6*k points.
    # Ring 1: 6 points, Ring 2: 12 points, etc.
    # Total points after ring k: 1 + 6*(1+2+...+k) = 1 + 3k(k+1)
    
    # Determine which ring we're in
    k = 1
    while 1 + 3*k*(k+1) < n:
        k += 1
    
    # We need points from rings 1 through k, but only partial ring k if needed
    # Actually, let's just generate ring by ring
    
    points = [(0, 0)]
    ring = 1
    while len(points) < n:
        # Generate ring 'ring'
        # Ring r consists of 6 segments of length r
        # Starting point for ring r: (r, 0)
        # Directions: (1,0), (0,-1), (-1,-1), (-1,0), (0,1), (1,1)
        
        start = (ring, 0)
        directions = [(1,0), (0,-1), (-1,-1), (-1,0), (0,1), (1,1)]
        
        for d_idx, (da, db) in enumerate(directions):
            for step in range(ring):
                if len(points) >= n:
                    break
                # Position: start + step * direction
                # But we need to be careful: the start of each segment is the end of the previous
                # Let's compute positions properly
                
                # Actually, let me think about this differently.
                # Ring r starts at (r, 0) and goes:
                # Segment 0: (r,0), (r+1,0), ..., wait no.
                
                # Let me reconsider. The hexagonal spiral:
                # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
                # Ring 2: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
                
                # Let me just generate all ring points and take what we need.
                pass
        
        # Simpler approach: generate all points in ring r
        ring_points = []
        # The six corners of ring r are:
        # (r, 0), (0, -r), (-r, -r), (-r, 0), (0, r), (r, r)
        # Wait, let me verify. Hex distance of (a,b) is max(|a|,|b|,|a+b|)
        # (r,0): max(r,0,r) = r ✓
        # (0,-r): max(0,r,r) = r ✓  
        # (-r,-r): max(r,r,0) = r ✓
        # (-r,0): max(r,0,r) = r ✓
        # (0,r): max(0,r,r) = r ✓
        # (r,r): max(r,r,2r) = 2r ✗ -- that's wrong
        
        # Let me recalculate. For hex distance = r:
        # max(|a|, |b|, |a+b|) = r
        # The six "corner" directions are:
        # (r, 0): max(r, 0, r) = r ✓
        # (r, -r): max(r, r, 0) = r ✓
        # (0, -r): max(0, r, r) = r ✓
        # (-r, 0): max(r, 0, r) = r ✓
        # (-r, r): max(r, r, 0) = r ✓
        # (0, r): max(0, r, r) = r ✓
        
        # So the six corners are: (r,0), (r,-r), (0,-r), (-r,0), (-r,r), (0,r)
        # Going around: start at (r,0), then move to (r,-r) [direction (0,-1), r steps]
        # then to (0,-r) [direction (-1,0), r steps]
        # then to (-r,0) [direction (0,1), r steps]
        # then to (-r,r) [direction (1,1), r steps]
        # then to (0,r) [direction (1,0), r steps]
        # then back to (r,0) [direction (0,-1)... wait]
        
        # Let me trace ring 1: corners (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        # (1,0) to (1,-1): step (0,-1), 1 step ✓
        # (1,-1) to (0,-1): step (-1,0), 1 step ✓
        # (0,-1) to (-1,0): step (-1,1), 1 step ✓
        # (-1,0) to (-1,1): step (0,1), 1 step ✓
        # (-1,1) to (0,1): step (1,0), 1 step ✓
        # (0,1) to (1,0): step (1,-1), 1 step ✓
        
        # So directions between corners: (0,-1), (-1,0), (-1,1), (0,1), (1,0), (1,-1)
        # Each segment has r steps (r points excluding the starting corner, or r-1 intermediate + 1 corner?)
        # Ring 1 has 6 points total. 6 segments of 1 point each.
        # Ring 2 has 12 points. 6 segments of 2 points each.
        
        corners = [(r,0), (r,-r), (0,-r), (-r,0), (-r,r), (0,r)]
        directions = [(0,-1), (-1,0), (-1,1), (0,1), (1,0), (1,-1)]
        
        for i in range(6):
            ca, cb = corners[i]
            da, db = directions[i]
            for s in range(1, r+1):  # s=1 to r, giving r points
                pa = ca + s*da
                pb = cb + s*db
                if len(points) < n:
                    points.append((pa, pb))
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        ring += 1
    
    return points[:n]

def construct():
    """Return exactly 32 distinct integer pairs [a, b] with |a|, |b| <= 64
    maximizing unit-distance edges (target: 76)."""
    deadline = time.time() + 120.0
    
    # Strategy 1: Use the hexagonal region of radius 2 (37 points) and remove 5 points
    # to leave 32 points with maximum edges.
    
    hex_pts = _hexagon_points(2)
    # Should be 1 + 6 + 12 + 18 = 37 points
    assert len(hex_pts) == 37, f"Expected 37 hex points, got {len(hex_pts)}"
    
    # Count edges in the full hexagon
    full_edges = _count_edges(hex_pts)
    
    # We need to remove 5 points to minimize edge loss.
    # Each point removal loses its degree (number of unit-distance neighbors in the current set).
    # But removing a point also affects the degrees of its neighbors.
    # We need to find the 5-point subset whose removal minimizes total edge loss.
    
    # Total edges in hex(2): let's compute
    # Each interior point has 6 neighbors, boundary points have fewer.
    # Let's just compute it.
    
    # Enumerate all 5-combinations of 37 points: C(37,5) = 435,897
    # For each, compute edges in the remaining 32 points.
    
    # To speed up: precompute adjacency
    point_set = set(hex_pts)
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    # Build adjacency list
    adj = {}
    for p in point_set:
        a, b = p
        adj[p] = set()
        for da, db in neighbors:
            np = (a+da, b+db)
            if np in point_set:
                adj[p].add(np)
    
    # Count edges: sum of degrees / 2
    total_edges = sum(len(adj[p]) for p in point_set) // 2
    
    # For removing a set S, edges lost = edges within S + edges between S and complement
    # = sum over p in S of deg(p) - edges within S (each edge within S counted twice in sum of degs)
    # Actually: edges lost = edges with at least one endpoint in S
    # = sum_{p in S} deg(p) - edges_within_S (since each edge within S is counted twice in the sum)
    
    # So edges remaining = total_edges - (sum_{p in S} deg(p) - edges_within_S)
    # = total_edges - sum_{p in S} deg(p) + edges_within_S
    
    # To maximize edges remaining, we want to minimize: sum_{p in S} deg(p) - edges_within_S
    
    # Let's enumerate. C(37,5) = 435897. For each, compute sum of degs and edges within S.
    # We can precompute deg for each point.
    
    deg = {p: len(adj[p]) for p in point_set}
    pts_list = list(point_set)
    
    best_edges_remaining = -1
    best_remove = None
    
    # Precompute edge set for quick lookup
    edge_set = set()
    for p in point_set:
        for q in adj[p]:
            if p < q:
                edge_set.add((p, q))
    
    idx = {p: i for i, p in enumerate(pts_list)}
    
    # Enumerate 5-combinations
    from itertools import combinations as comb
    
    # For efficiency, precompute for each pair whether they're adjacent
    # adj_matrix[i][j] = 1 if pts_list[i] and pts_list[j] are adjacent
    
    n_pts = len(pts_list)
    is_adj = [[False]*n_pts for _ in range(n_pts)]
    for i in range(n_pts):
        pi = pts_list[i]
        for q in adj[pi]:
            j = idx[q]
            is_adj[i][j] = True
            is_adj[j][i] = True
    
    deg_list = [deg[p] for p in pts_list]
    
    # Enumerate all 5-combinations
    for combo in comb(range(n_pts), 5):
        # Sum of degrees
        s = sum(deg_list[i] for i in combo)
        # Edges within combo
        ew = 0
        for ii in range(5):
            for jj in range(ii+1, 5):
                if is_adj[combo[ii]][combo[jj]]:
                    ew += 1
        # Edges remaining = total_edges - s + ew
        er = total_edges - s + ew
        if er > best_edges_remaining:
            best_edges_remaining = er
            best_remove = combo
    
    # Build the result
    if best_remove is not None:
        remove_set = set(best_remove)
        result = [list(pts_list[i]) for i in range(n_pts) if i not in remove_set]
    else:
        # Fallback: just take first 32
        result = [list(p) for p in pts_list[:32]]
    
    # Verify
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for a, b in result:
        assert abs(a) <= 64 and abs(b) <= 64
    
    edges = _count_edges(result)
    
    # If we didn't achieve 76, try the spiral approach as fallback
    if edges < 76:
        spiral = _spiral_points(32)
        spiral_edges = _count_edges(spiral)
        if spiral_edges > edges:
            result = [list(p) for p in spiral]
            edges = spiral_edges
    
    return result
# EVOLVE-BLOCK-END
