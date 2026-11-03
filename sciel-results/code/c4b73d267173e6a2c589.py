"""Improved program for the 200-circle packing task using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import random
import copy


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    if len(centres) < 2:
        # Edge case: single circle
        min_dist_to_border = np.min(0.5 - np.abs(centres))
        return float(min_dist_to_border)
    
    # Compute all pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Compute minimum distance to border
    min_dist_to_border = np.min(0.5 - np.abs(centres))
    
    return float(min(min_pair_dist / 2.0, min_dist_to_border))


def _delaunay_edges(tri):
    """Extract unique edges from a Delaunay triangulation."""
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                edges.add(edge)
    return list(edges)


def _get_adjacent_triangles(tri, edge_i, edge_j):
    """Find the two triangles sharing the edge (edge_i, edge_j)."""
    tri_a = None
    tri_b = None
    for simplex in tri.simplices:
        if edge_i in simplex and edge_j in simplex:
            if tri_a is None:
                tri_a = simplex
            else:
                tri_b = simplex
                break
    return tri_a, tri_b


def _is_valid_flip(centres, tri_a, tri_b, edge_i, edge_j):
    """Check if flipping the edge is valid (new edge doesn't cross other edges)."""
    third_a = [v for v in tri_a if v != edge_i and v != edge_j][0]
    third_b = [v for v in tri_b if v != edge_i and v != edge_j][0]
    
    # Check if the new edge (third_a, third_b) intersects with any other edge
    # For simplicity, we check if the midpoint of the new edge is inside the convex hull
    # of the four points
    p1 = centres[edge_i]
    p2 = centres[edge_j]
    p3 = centres[third_a]
    p4 = centres[third_b]
    
    # Check if the quadrilateral is convex
    # Compute cross products to check orientation
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # The flip is valid if the quadrilateral is convex
    # Check if all four points are on the same side of each edge
    # For a convex quadrilateral, the new diagonal should be inside
    
    # A simpler check: the new edge should not intersect with the old edge
    # They share endpoints, so they can't intersect in the interior
    # The main concern is whether the new triangulation is valid
    
    # For now, assume it's valid if the four points form a convex quadrilateral
    # We can check this by ensuring that p3 and p4 are on opposite sides of the line p1-p2
    c1 = cross(p1, p2, p3)
    c2 = cross(p1, p2, p4)
    
    # If they're on opposite sides, the quadrilateral is convex (or at least the flip is valid)
    return c1 * c2 < 0


def _try_flip(tri, centres, edge_i, edge_j):
    """
    Attempt to flip edges between two adjacent triangles sharing an edge.
    Returns (new_centres, improved) if the flip improves the radius, otherwise (None, False).
    """
    tri_a, tri_b = _get_adjacent_triangles(tri, edge_i, edge_j)
    
    if tri_a is None or tri_b is None:
        return None, False
    
    # The shared edge is (edge_i, edge_j), the other vertices are the third vertices
    third_a = [v for v in tri_a if v != edge_i and v != edge_j][0]
    third_b = [v for v in tri_b if v != edge_i and v != edge_j][0]
    
    # Check if flipping is valid
    if not _is_valid_flip(centres, tri_a, tri_b, edge_i, edge_j):
        return None, False
    
    # For a combinatorial approach, we don't change the centres directly
    # Instead, we would need to recompute the optimal positions given the new topology
    # For now, let's just return None as we need a more sophisticated approach
    
    return None, False


def _local_optimization(centres, max_iterations=500, step_size=0.01):
    """
    Local optimization using gradient-like steps to maximize minimum distance.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for iteration in range(max_iterations):
        improved = False
        
        # Compute current radius
        current_radius = _compute_radius(centres)
        
        # Find the bottleneck: the pair of circles (or circle-border) that limits the radius
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        min_i, min_j = np.unravel_index(np.argmin(dists), dists.shape)
        
        dist_to_border = 0.5 - np.abs(centres)
        min_border_dist = np.min(dist_to_border)
        min_border_i, min_border_k = np.unravel_index(np.argmin(dist_to_border), dist_to_border.shape)
        
        # Determine which constraint is active
        if min_dist / 2.0 < min_border_dist:
            # Pair constraint is active
            direction = centres[min_i] - centres[min_j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            # Move the two circles apart
            step = step_size * (1.0 + 0.5 * np.random.rand())
            new_centres = centres.copy()
            new_centres[min_i] += direction * step
            new_centres[min_j] -= direction * step
            
            # Keep within bounds
            new_centres = np.clip(new_centres, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-10:
                best_radius = new_radius
                best_centres = new_centres.copy()
                improved = True
        else:
            # Border constraint is active
            # Move the circle away from the border
            pos = centres[min_border_i]
            if pos[0] > 0:
                direction = np.array([-1.0, 0.0])
            elif pos[0] < 0:
                direction = np.array([1.0, 0.0])
            else:
                direction = np.array([0.0, 0.0])
            
            if pos[1] > 0:
                direction[1] = -1.0
            elif pos[1] < 0:
                direction[1] = 1.0
            else:
                direction[1] = 0.0
            
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            step = step_size * (1.0 + 0.5 * np.random.rand())
            new_centres = centres.copy()
            new_centres[min_border_i] += direction * step
            new_centres = np.clip(new_centres, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-10:
                best_radius = new_radius
                best_centres = new_centres.copy()
                improved = True
        
        if not improved:
            # Try a small random perturbation to escape local minima
            if iteration > max_iterations // 2:
                idx = np.random.randint(len(centres))
                perturbation = np.random.randn(2) * step_size * 0.5
                new_centres = centres.copy()
                new_centres[idx] += perturbation
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-10:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    improved = True
        
        if not improved and iteration > max_iterations // 3:
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return best_centres, best_radius


def _combinatorial_flip_optimization(centres, max_iterations=200):
    """
    Combinatorial edge-flip optimization to improve the packing.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for iteration in range(max_iterations):
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(best_centres)
        except Exception:
            break
        
        edges = _delaunay_edges(tri)
        
        improved = False
        
        # Try to find an edge flip that improves the minimum distance
        for edge_i, edge_j in edges:
            new_centres, improved = _try_flip(tri, best_centres, edge_i, edge_j)
            if improved:
                best_centres = new_centres
                best_radius = _compute_radius(best_centres)
                break
        
        if not improved:
            # No more flips improve the radius
            break
    
    return best_centres, best_radius


def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius.
    """
    # Start with a grid arrangement
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Run multiple iterations of local optimization with different random restarts
    for restart in range(3):
        # Add small random perturbation
        if restart > 0:
            centres = best_centres + np.random.randn(N, 2) * 0.005
            centres = np.clip(centres, -0.5, 0.5)
        
        # Local optimization
        optimised, radius = _local_optimization(centres, max_iterations=300, step_size=0.008)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    # Try combinatorial flip optimization
    flipped, flip_radius = _combinatorial_flip_optimization(best_centres, max_iterations=100)
    if flip_radius > best_radius:
        best_radius = flip_radius
        best_centres = flipped.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
