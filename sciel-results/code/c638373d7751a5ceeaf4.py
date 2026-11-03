"""Improved program for the 200-circle packing task using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import random
import time


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    if len(centres) < 2:
        min_dist_to_border = np.min(0.5 - np.abs(centres))
        return float(min_dist_to_border)
    
    # Compute all pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Compute minimum distance to border
    min_dist_to_border = np.min(0.5 - np.abs(centres))
    
    return float(min(min_pair_dist / 2.0, min_dist_to_border))


def _get_delaunay_edges(tri):
    """Extract unique edges from a Delaunay triangulation as a set of tuples."""
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                edges.add(edge)
    return edges


def _find_shared_edge_triangles(tri, edge):
    """Find the two triangles that share the given edge."""
    i, j = edge
    triangles = []
    for simplex in tri.simplices:
        if i in simplex and j in simplex:
            triangles.append(simplex)
    return triangles


def _is_valid_flip(centres, edge_a, edge_b):
    """
    Check if flipping edge_a to edge_b is valid.
    The new edge must not intersect existing edges and must be inside the convex hull.
    """
    i, j = edge_a
    k, l = edge_b
    
    # Check if the new edge (k, l) would intersect any existing edge
    # For simplicity, we'll just check if the points are in a valid configuration
    # A proper check would require checking against all other edges
    
    # Simple heuristic: check if the quadrilateral i-j-k-l is convex
    # and the diagonals i-k and j-l would be the new edges
    p1, p2, p3, p4 = centres[i], centres[j], centres[k], centres[l]
    
    # Check if the four points form a convex quadrilateral
    # by checking if the cross products have consistent signs
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Check if k is inside triangle i-j-l or l is inside triangle i-j-k
    # If either is true, the flip is not valid
    def point_in_triangle(p, a, b, c):
        d1 = cross(p, a, b)
        d2 = cross(p, b, c)
        d3 = cross(p, c, a)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)
    
    # The flip is valid if neither k is inside triangle i-j-l nor l is inside triangle i-j-k
    if point_in_triangle(centres[k], centres[i], centres[j], centres[l]):
        return False
    if point_in_triangle(centres[l], centres[i], centres[j], centres[k]):
        return False
    
    return True


def _perform_edge_flip(centres, edge_a, edge_b, target_radius):
    """
    Attempt to perform an edge flip to improve the minimum distance.
    Returns (success, new_centres) where success indicates if the flip improved things.
    """
    i, j = edge_a
    k, l = edge_b
    
    # Check if the flip is valid
    if not _is_valid_flip(centres, edge_a, edge_b):
        return False, centres
    
    # The flip changes the topology: instead of triangles (i,j,k) and (i,j,l),
    # we get triangles (i,k,l) and (j,k,l)
    # We need to adjust the positions to maintain or improve the minimum distance
    
    # Strategy: move i and j apart along the direction that increases the minimum edge length
    # The key insight is that after the flip, the edges (i,k), (i,l), (j,k), (j,l) become important
    
    # Compute the current minimum edge length
    current_min = np.inf
    for a, b in [(i, k), (i, l), (j, k), (j, l)]:
        d = np.linalg.norm(centres[a] - centres[b])
        current_min = min(current_min, d)
    
    # Try to optimize by moving i and j to equalize the distances to k and l
    # This is a local optimization problem
    
    new_centres = centres.copy()
    
    # Use a simple gradient step to improve the minimum of the four edges
    for _ in range(20):
        # Compute gradients for edges (i,k), (i,l), (j,k), (j,l)
        edges_to_optimize = [(i, k), (i, l), (j, k), (j, l)]
        
        # Find the minimum edge
        min_edge = None
        min_dist = np.inf
        for a, b in edges_to_optimize:
            d = np.linalg.norm(new_centres[a] - new_centres[b])
            if d < min_dist:
                min_dist = d
                min_edge = (a, b)
        
        if min_edge is None:
            break
        
        a, b = min_edge
        
        # Move a and b apart
        direction = new_centres[a] - new_centres[b]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            break
        direction = direction / norm
        
        # Small step
        step = 0.001
        new_centres[a] += direction * step
        new_centres[b] -= direction * step
        
        # Keep within bounds
        new_centres = np.clip(new_centres, -0.5, 0.5)
    
    # Check if the flip improved the overall minimum distance
    old_radius = _compute_radius(centres)
    new_radius = _compute_radius(new_centres)
    
    if new_radius > old_radius + 1e-10:
        return True, new_centres
    else:
        return False, centres


def _delaunay_flip_optimization(centres, max_iterations=100):
    """
    Optimize the packing using Delaunay edge flips.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for iteration in range(max_iterations):
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(best_centres)
        except Exception:
            break
        
        # Get all edges
        edges = _get_delaunay_edges(tri)
        
        # Find the shortest edge
        min_edge = None
        min_dist = np.inf
        for i, j in edges:
            d = np.linalg.norm(best_centres[i] - best_centres[j])
            if d < min_dist:
                min_dist = d
                min_edge = (i, j)
        
        if min_edge is None:
            break
        
        # Find the triangles sharing this edge
        triangles = _find_shared_edge_triangles(tri, min_edge)
        
        if len(triangles) != 2:
            # Edge is on the boundary, can't flip
            continue
        
        # Get the third vertices
        i, j = min_edge
        tri1, tri2 = triangles
        third1 = [v for v in tri1 if v != i and v != j][0]
        third2 = [v for v in tri2 if v != i and v != j][0]
        
        # Try to flip
        new_edge = (min(third1, third2), max(third1, third2))
        success, new_centres = _perform_edge_flip(best_centres, min_edge, new_edge, best_radius)
        
        if success:
            best_centres = new_centres
            best_radius = _compute_radius(best_centres)
    
    return best_centres, best_radius


def _hybrid_optimization(centres, time_budget=60.0):
    """
    Combine Delaunay flips with local gradient optimization.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    start_time = time.time()
    
    # Phase 1: Delaunay edge flips
    while time.time() - start_time < time_budget * 0.4:
        optimised, radius = _delaunay_flip_optimization(best_centres, max_iterations=50)
        if radius > best_radius + 1e-10:
            best_radius = radius
            best_centres = optimised.copy()
        else:
            # No improvement, try a random perturbation
            best_centres = best_centres + np.random.randn(N, 2) * 0.002
            best_centres = np.clip(best_centres, -0.5, 0.5)
    
    # Phase 2: Local gradient optimization
    while time.time() - start_time < time_budget:
        # Find the bottleneck
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        min_i, min_j = np.unravel_index(np.argmin(dists), dists.shape)
        
        dist_to_border = 0.5 - np.abs(best_centres)
        min_border_dist = np.min(dist_to_border)
        min_border_i, min_border_k = np.unravel_index(np.argmin(dist_to_border), dist_to_border.shape)
        
        improved = False
        
        if min_dist / 2.0 < min_border_dist:
            # Pair constraint is active
            direction = best_centres[min_i] - best_centres[min_j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            step = 0.003 * (1.0 + 0.5 * np.random.rand())
            new_centres = best_centres.copy()
            new_centres[min_i] += direction * step
            new_centres[min_j] -= direction * step
            new_centres = np.clip(new_centres, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-10:
                best_radius = new_radius
                best_centres = new_centres.copy()
                improved = True
        else:
            # Border constraint is active
            pos = best_centres[min_border_i]
            direction = np.zeros(2)
            if pos[0] > 0:
                direction[0] = -1.0
            elif pos[0] < 0:
                direction[0] = 1.0
            
            if pos[1] > 0:
                direction[1] = -1.0
            elif pos[1] < 0:
                direction[1] = 1.0
            
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            step = 0.003 * (1.0 + 0.5 * np.random.rand())
            new_centres = best_centres.copy()
            new_centres[min_border_i] += direction * step
            new_centres = np.clip(new_centres, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-10:
                best_radius = new_radius
                best_centres = new_centres.copy()
                improved = True
        
        if not improved:
            # Try a small random perturbation
            idx = np.random.randint(N)
            perturbation = np.random.randn(2) * 0.001
            new_centres = best_centres.copy()
            new_centres[idx] += perturbation
            new_centres = np.clip(new_centres, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-10:
                best_radius = new_radius
                best_centres = new_centres.copy()
            else:
                # No improvement, break
                break
    
    return best_centres, best_radius


def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius using Delaunay-based edge flips.
    """
    # Start with a grid arrangement
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Run multiple restarts with hybrid optimization
    start_time = time.time()
    for restart in range(5):
        if time.time() - start_time > 100:
            break
        
        # Add small random perturbation for diversity
        if restart > 0:
            centres = best_centres + np.random.randn(N, 2) * 0.003
            centres = np.clip(centres, -0.5, 0.5)
        
        # Hybrid optimization
        optimised, radius = _hybrid_optimization(centres, time_budget=15.0)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
