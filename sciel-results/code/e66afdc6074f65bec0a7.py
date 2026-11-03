"""Delaunay edge-flip optimization for 200-circle packing in unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import cdist
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
    """Extract unique edges from a Delaunay triangulation."""
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                edges.add(edge)
    return list(edges)


def _find_adjacent_triangles(tri, edge_i, edge_j):
    """Find the two triangles sharing edge (edge_i, edge_j)."""
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
    """Check if flipping the edge is geometrically valid (no self-intersection)."""
    third_a = [v for v in tri_a if v != edge_i and v != edge_j][0]
    third_b = [v for v in tri_b if v != edge_i and v != edge_j][0]
    
    # Check if the new edge (third_a, third_b) would cause self-intersection
    # by checking if the quadrilateral is convex
    points = np.array([centres[edge_i], centres[edge_j], centres[third_b], centres[third_a]])
    
    # Check convexity using cross products
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    c1 = cross(points[0], points[1], points[2])
    c2 = cross(points[1], points[2], points[3])
    c3 = cross(points[2], points[3], points[0])
    c4 = cross(points[3], points[0], points[1])
    
    # Convex if all cross products have the same sign
    all_positive = c1 > 0 and c2 > 0 and c3 > 0 and c4 > 0
    all_negative = c1 < 0 and c2 < 0 and c3 < 0 and c4 < 0
    
    return all_positive or all_negative


def _perform_edge_flip(centres, edge_i, edge_j, tri_a, tri_b):
    """
    Perform an edge flip: move vertices to equalize edge lengths.
    Returns new centres if the flip improves the minimum edge length.
    """
    third_a = [v for v in tri_a if v != edge_i and v != edge_j][0]
    third_b = [v for v in tri_b if v != edge_i and v != edge_j][0]
    
    # Current edge lengths
    len_ij = np.linalg.norm(centres[edge_i] - centres[edge_j])
    len_ia = np.linalg.norm(centres[edge_i] - centres[third_a])
    len_ja = np.linalg.norm(centres[edge_j] - centres[third_a])
    len_ib = np.linalg.norm(centres[edge_i] - centres[third_b])
    len_jb = np.linalg.norm(centres[edge_j] - centres[third_b])
    
    # Target: make all edges in the two triangles equal to the average
    target_len = (len_ij + len_ia + len_ja + len_ib + len_jb) / 5.0
    
    # Move vertices to achieve equal edge lengths
    # This is a simplified approach: move edge_i and edge_j apart along their connecting line
    direction = centres[edge_i] - centres[edge_j]
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return None
    
    direction = direction / norm
    
    # Calculate how much to move to increase the minimum edge
    min_edge = min(len_ij, len_ia, len_ja, len_ib, len_jb)
    needed_increase = target_len - min_edge
    
    if needed_increase <= 0:
        return None
    
    # Move edge_i and edge_j apart by half the needed increase each
    step = needed_increase / 2.0
    
    new_centres = centres.copy()
    new_centres[edge_i] += direction * step
    new_centres[edge_j] -= direction * step
    
    # Keep within bounds
    new_centres = np.clip(new_centres, -0.5, 0.5)
    
    return new_centres


def _delaunay_optimization(centres, max_flips=1000, time_limit=60.0):
    """
    Optimize centres using Delaunay edge flips to maximize minimum edge length.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    start_time = time.time()
    
    for flip_iter in range(max_flips):
        if time.time() - start_time > time_limit:
            break
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(centres)
        except Exception:
            break
        
        # Get all edges
        edges = _get_delaunay_edges(tri)
        
        # Find the shortest edge
        min_edge_len = np.inf
        min_edge = None
        for i, j in edges:
            d = np.linalg.norm(centres[i] - centres[j])
            if d < min_edge_len:
                min_edge_len = d
                min_edge = (i, j)
        
        if min_edge is None:
            break
        
        i, j = min_edge
        
        # Find adjacent triangles
        tri_a, tri_b = _find_adjacent_triangles(tri, i, j)
        
        if tri_a is None or tri_b is None:
            # Edge is on the boundary, can't flip
            # Try moving the vertex inward
            if i == j:
                break
            # Move the vertex with the smaller index toward the center
            idx = i if np.linalg.norm(centres[i]) < np.linalg.norm(centres[j]) else j
            direction = -centres[idx]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
                new_centres = centres.copy()
                new_centres[idx] += direction * 0.001
                new_centres = np.clip(new_centres, -0.5, 0.5)
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-12:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
            continue
        
        # Check if flip is valid
        if not _is_valid_flip(centres, tri_a, tri_b, i, j):
            continue
        
        # Perform the flip
        new_centres = _perform_edge_flip(centres, i, j, tri_a, tri_b)
        if new_centres is None:
            continue
        
        # Check if the flip improves the radius
        new_radius = _compute_radius(new_centres)
        if new_radius > best_radius + 1e-12:
            best_radius = new_radius
            best_centres = new_centres.copy()
            centres = new_centres
    
    return best_centres, best_radius


def _grid_initialization():
    """Create a grid initialization for N circles."""
    # Use a slightly irregular grid to avoid symmetry issues
    m = int(np.ceil(np.sqrt(N)))
    
    # Create grid points
    points = []
    for i in range(m):
        for j in range(m):
            if len(points) >= N:
                break
            # Add small random perturbation to break symmetry
            x = (i + 0.5) / m - 0.5 + np.random.randn() * 0.001
            y = (j + 0.5) / m - 0.5 + np.random.randn() * 0.001
            points.append([x, y])
    
    return np.array(points[:N])


def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius using Delaunay edge-flip optimization.
    """
    np.random.seed(42)
    
    # Start with grid initialization
    centres = _grid_initialization()
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Run Delaunay optimization with multiple restarts
    for restart in range(5):
        if restart > 0:
            # Add random perturbation for restart
            centres = best_centres + np.random.randn(N, 2) * 0.002
            centres = np.clip(centres, -0.5, 0.5)
        
        # Run Delaunay edge-flip optimization
        optimised, radius = _delaunay_optimization(centres, max_flips=500, time_limit=20.0)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
