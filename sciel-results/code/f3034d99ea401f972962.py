"""Exact maximum-edge subgraph extraction for the 32-point triangular lattice problem."""
# EVOLVE-BLOCK-START
import time
import math

def _count_edges(points):
    """Count the number of unit-distance pairs among given points on triangular lattice."""
    point_set = set(points)
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    edge_count = 0
    for a, b in points:
        for da, db in neighbors:
            if (a+da, b+db) in point_set:
                edge_count += 1
    return edge_count // 2

def _harborth_bound(n):
    """Return the Harborth bound floor(3n - sqrt(12n - 3))."""
    return int(math.floor(3*n - math.sqrt(12*n - 3)))

def _build_lattice_graph(n=32, box=3):
    """Build the unit-distance graph over a bounded triangular lattice region.
    
    Uses axial coordinates (a, b) for point a + b*omega.
    The region is { (a,b) : |a|<=box, |b|<=box, |a+b|<=box } which is a hexagon.
    """
    # Generate all lattice points in the hexagonal region
    points = []
    for a in range(-box, box+1):
        for b in range(-box, box+1):
            if abs(a+b) <= box:
                points.append((a, b))
    
    # Map point -> index
    point_to_idx = {p: i for i, p in enumerate(points)}
    npts = len(points)
    
    # Build adjacency using only "forward" edges to avoid duplicates
    # Forward directions: (1,0), (0,1), (1,-1)
    forward_dirs = [(1,0), (0,1), (1,-1)]
    
    adj = [set() for _ in range(npts)]
    edge_list = []
    
    for i, (a, b) in enumerate(points):
        for da, db in forward_dirs:
            j_point = (a+da, b+db)
            if j_point in point_to_idx:
                j = point_to_idx[j_point]
                adj[i].add(j)
                adj[j].add(i)
                edge_list.append((i, j))
    
    return points, point_to_idx, adj, edge_list, npts

def _solve_max_edge_subgraph(adj, npts, k=32, target=76, time_limit=110.0):
    """Find a k-vertex induced subgraph with at least `target` edges using branch-and-bound.
    
    Returns the set of selected vertex indices, or None if not found within time.
    """
    start_time = time.time()
    best_solution = [None]
    found_target = [False]
    
    # Precompute degrees
    degrees = [len(adj[i]) for i in range(npts)]
    
    # Order vertices by degree descending (higher degree vertices are more valuable)
    order = sorted(range(npts), key=lambda i: -degrees[i])
    
    # For efficiency, work with the ordered list
    # selected: set of vertex indices currently selected
    # available: set of vertex indices not yet decided (can still be added)
    
    # We use a recursive approach with inclusion/exclusion
    # But for k=32 and npts~61, we need a smart approach.
    
    # Strategy: Start with a greedy solution, then try to improve via local search.
    # Actually, let's do a proper branch and bound.
    
    # For the hexagonal region with box=3, npts = 1 + 6 + 12 + 18 = 37
    # Wait, let me recalculate. For box=3:
    # a from -3 to 3, b from -3 to 3, |a+b| <= 3
    # This gives a hexagon with 37 points.
    # We need to choose 32 from 37, so we're excluding 5.
    
    # Better approach: find the 5 vertices to REMOVE to maximize remaining edges.
    # Total edges in full graph minus edges lost by removal.
    # Edges lost = edges incident to removed vertices (counting edges between removed vertices once)
    
    total_edges = sum(degrees) // 2
    
    # We want to remove (npts - k) = 5 vertices to minimize edges lost.
    # Edges lost when removing set R = |E(R, V\R)| + |E(R, R)|
    # = sum_{v in R} degree(v) - |E(R, R)|  ... wait no
    # Edges lost = number of edges with at least one endpoint in R
    # = sum_{v in R} degree(v) - |E(R, R)|  ... this isn't right either
    
    # Actually: edges lost = |{(u,v) in E : u in R or v in R}|
    # = sum_{v in R} degree(v) - |E(R, R)|
    # No wait. Each edge in E(R,R) is counted twice in sum_{v in R} degree(v),
    # but it's only one edge. Each edge in E(R, V\R) is counted once.
    # So edges lost = sum_{v in R} degree(v) - |E(R, R)|
    
    # We want to MINIMIZE edges lost = sum_{v in R} degree(v) - |E(R,R)|
    # which is equivalent to MAXIMIZING |E(R,R)| - sum_{v in R} degree(v) ... no
    # MINIMIZE (sum_{v in R} degree(v) - |E(R,R)|)
    # = MAXIMIZE (|E(R,R)| - sum_{v in R} degree(v)) ... no that's the negative
    
    # Let's just enumerate. We need to choose 5 vertices from 37.
    # C(37,5) = 435897. That's very manageable!
    
    n_remove = npts - k  # 5
    
    if n_remove <= 0:
        # Take all vertices
        return set(range(npts))
    
    if n_remove > 10:
        # Too many to enumerate, fall back to heuristic
        return None
    
    # Enumerate all combinations of n_remove vertices to remove
    from itertools import combinations
    
    best_removed = None
    min_edges_lost = float('inf')
    
    # Precompute edge lookup for fast checking
    edge_set = set()
    for i in range(npts):
        for j in adj[i]:
            if i < j:
                edge_set.add((i, j))
    
    count = 0
    for combo in combinations(range(npts), n_remove):
        if time.time() - start_time > time_limit:
            break
        
        count += 1
        removed = set(combo)
        
        # Calculate edges lost
        edges_lost = 0
        for v in combo:
            for u in adj[v]:
                if u not in removed:
                    edges_lost += 1
                elif u > v:  # Count edges within removed set once
                    edges_lost += 1
        
        if edges_lost < min_edges_lost:
            min_edges_lost = edges_lost
            best_removed = removed
            
            if total_edges - edges_lost >= target:
                # We found a solution that meets the target
                best_solution[0] = removed
                found_target[0] = True
                # We can continue to see if we can do better, but 76 is the Harborth bound
                # so we can't do better. Let's stop.
                break
    
    if found_target[0] and best_solution[0] is not None:
        removed = best_solution[0]
        selected = set(range(npts)) - removed
        return selected
    
    # If we didn't find a target-meeting solution, return the best we found
    if best_removed is not None:
        selected = set(range(npts)) - best_removed
        return selected
    
    return None

def _construct_hexagonal_ring(n=32):
    """Construct points in hexagonal ring pattern on triangular lattice."""
    points = []
    
    # Ring 0: center
    points.append((0, 0))
    
    # Ring 1: 6 points
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    points.extend(ring1)
    
    # Ring 2: 12 points
    ring2 = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            if max(abs(a), abs(b), abs(a+b)) == 2:
                ring2.append((a, b))
    points.extend(ring2)
    
    # Ring 3: need 13 points
    ring3_full = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) == 3:
                ring3_full.append((a, b))
    
    existing = set(points)
    
    def count_connections(p):
        a, b = p
        count = 0
        for da, db in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
            if (a+da, b+db) in existing:
                count += 1
        return count
    
    ring3_sorted = sorted(ring3_full, key=lambda p: (-count_connections(p), p[0], p[1]))
    
    for p in ring3_sorted[:13]:
        points.append(p)
    
    return points

def construct():
    """Construct 32 distinct lattice points maximizing unit-distance pairs."""
    deadline = time.time() + 115
    
    n = 32
    target = _harborth_bound(n)  # 76
    
    # Strategy: Exact subgraph extraction
    # Build the lattice graph and find the optimal 32-vertex subgraph
    
    # Use a hexagonal region large enough to contain 32 points
    # With box=3, we get 37 points in the hexagon
    # We need to remove 5 to get 32
    
    points_list, point_to_idx, adj, edge_list, npts = _build_lattice_graph(n=32, box=3)
    
    # Verify we have enough points
    if npts < n:
        # Expand the box
        box = 4
        points_list, point_to_idx, adj, edge_list, npts = _build_lattice_graph(n=32, box=box)
    
    # Solve the exact problem
    selected_indices = _solve_max_edge_subgraph(adj, npts, k=n, target=target, time_limit=100.0)
    
    if selected_indices is not None and len(selected_indices) == n:
        result_points = [points_list[i] for i in selected_indices]
        edges = _count_edges(result_points)
        
        # Validate
        result_set = set(result_points)
        if len(result_set) == n and all(abs(a) <= 64 and abs(b) <= 64 for a, b in result_set):
            if edges >= target:
                return [[a, b] for a, b in result_set]
    
    # Fallback: hexagonal ring construction
    if time.time() < deadline - 5:
        points = _construct_hexagonal_ring(n)
        edges = _count_edges(points)
        
        if edges >= target:
            result_set = set(points)
            if len(result_set) == n:
                return [[a, b] for a, b in result_set]
    
    # Final fallback: rectangular strip
    result = [[a, b] for a in range(16) for b in range(2)]
    return result
# EVOLVE-BLOCK-END
