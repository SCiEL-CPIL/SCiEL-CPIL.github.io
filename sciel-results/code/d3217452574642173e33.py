"""Rewritten program: discrete candidate-site search with local refinement."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 200
DEADLINE = 120.0  # seconds
SAFETY_MARGIN = 10.0  # seconds to return before kill

def _compute_radius(pts):
    """Compute the packing radius for given centres in [-0.5, 0.5]^2."""
    if len(pts) < 2:
        return 0.0
    # Boundary distance: min over all points and all 4 sides
    # distance to each side: 0.5 - |x|, 0.5 - |y|
    bx = 0.5 - np.abs(pts[:, 0])
    by = 0.5 - np.abs(pts[:, 1])
    min_boundary = np.minimum(bx.min(), by.min())
    
    # Pairwise distances
    tree = cKDTree(pts)
    pairs = tree.query_pairs(1.0)  # all pairs within distance 1.0 (safe upper bound)
    if not pairs:
        min_pair = float('inf')
    else:
        d = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in pairs])
        min_pair = d.min()
    
    r = min(min_boundary, min_pair / 2.0)
    return r

def _build_hex_grid(spacing, n_iter=50):
    """Build a fine hexagonal grid of candidate sites within [-0.5, 0.5]^2."""
    sites = []
    # Hex grid with horizontal spacing = spacing, vertical spacing = spacing * sqrt(3)/2
    dx = spacing
    dy = spacing * np.sqrt(3) / 2
    x_min, x_max = -0.5, 0.5
    y_min, y_max = -0.5, 0.5
    
    row = 0
    y = y_min
    while y <= y_max + 1e-9:
        offset = (dx / 2.0) if (row % 2 == 1) else 0.0
        x = x_min + offset
        while x <= x_max + 1e-9:
            if x_min - 1e-9 <= x <= x_max + 1e-9 and y_min - 1e-9 <= y <= y_max + 1e-9:
                sites.append((x, y))
            x += dx
        y += dy
        row += 1
    
    if not sites:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        sites = [(x, y) for y in ticks for x in ticks]
    
    return np.array(sites)

def _greedy_select(sites, n_select):
    """
    Greedy best-fit: repeatedly pick the site that maximizes the minimum
    distance to already-selected sites AND to the square boundary.
    """
    n = len(sites)
    if n_select >= n:
        return sites.copy()
    
    selected = []
    selected_set = set()
    
    # Precompute boundary distances for each site
    bx = 0.5 - np.abs(sites[:, 0])
    by = 0.5 - np.abs(sites[:, 1])
    boundary_dist = np.minimum(bx, by)
    
    # For each site, track the min distance to selected sites
    # Start: min_dist_to_selected = inf for all
    min_dist = np.full(n, np.inf)
    
    # We'll use a simple approach: at each step, find the site that maximizes
    # min(boundary_dist[i], min_dist[i])
    # To speed up, we can use a priority queue or just scan (n is moderate)
    
    for step in range(n_select):
        # score[i] = min(boundary_dist[i], min_dist[i])
        score = np.minimum(boundary_dist, min_dist)
        # Pick the site with maximum score
        best_idx = np.argmax(score)
        selected.append(sites[best_idx])
        selected_set.add(best_idx)
        
        # Update min_dist for all other sites
        if step < n_select - 1:
            d = np.linalg.norm(sites - sites[best_idx], axis=1)
            min_dist = np.minimum(min_dist, d)
            # Set selected sites to -inf so they're not re-picked
            min_dist[list(selected_set)] = -np.inf
    
    return np.array(selected)

def _local_refine(pts, sites, max_iter=200, time_limit=5.0):
    """
    Local refinement: try swapping each selected point with a nearby unselected
    site if it improves the radius.
    """
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    n_sites = len(sites)
    
    # Build a tree for sites for fast neighbor queries
    site_tree = cKDTree(sites)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute current radius
        current_r = _compute_radius(pts)
        
        improved = False
        # Try swapping each point with a nearby site
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            # Find nearby sites (within 2 * current_r * 1.5 to be safe)
            query_r = max(current_r * 2.0, 0.05)
            nearby_indices = site_tree.query_ball_point(pts[i], query_r)
            
            for j in nearby_indices:
                if j in set(np.arange(n)):  # skip if it's already selected (approx)
                    continue
                
                # Try swapping
                old_pt = pts[i].copy()
                pts[i] = sites[j]
                new_r = _compute_radius(pts)
                
                if new_r > current_r + 1e-12:
                    current_r = new_r
                    improved = True
                else:
                    pts[i] = old_pt
        
        if not improved:
            break
    
    return pts

def construct():
    start_time = time.time()
    
    # Build a fine hexagonal grid
    # For N=200, ideal spacing ~ 1/sqrt(200) ~ 0.07
    # Use a finer grid: spacing ~ 0.02 for good candidate resolution
    spacing = 0.02
    sites = _build_hex_grid(spacing)
    
    # Greedy selection
    pts = _greedy_select(sites, N)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        return pts
    
    # Local refinement
    pts = _local_refine(pts, sites, max_iter=100, time_limit=min(30.0, DEADLINE - SAFETY_MARGIN - elapsed))
    
    # Check time again
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        return pts
    
    # Try a second pass with slightly different initial grid offset
    # Shift the grid by a small amount and redo
    spacing2 = 0.025
    sites2 = _build_hex_grid(spacing2)
    pts2 = _greedy_select(sites2, N)
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN - 10.0:
        pts2 = _local_refine(pts2, sites2, max_iter=50, time_limit=min(15.0, DEADLINE - SAFETY_MARGIN - elapsed))
    
    # Pick the better solution
    r1 = _compute_radius(pts)
    r2 = _compute_radius(pts2)
    
    if r2 > r1:
        pts = pts2
    
    # Final time check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        return pts
    
    # One more refinement pass on the best solution
    pts = _local_refine(pts, sites, max_iter=50, time_limit=min(10.0, DEADLINE - SAFETY_MARGIN - elapsed))
    
    return pts
