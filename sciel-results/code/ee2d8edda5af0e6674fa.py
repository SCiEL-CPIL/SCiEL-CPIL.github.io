"""Improved program for the 200-circle packing task using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import random
import copy
import time


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    if len(centres) < 2:
        min_dist_to_border = np.min(0.5 - np.abs(centres))
        return float(min_dist_to_border)
    
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
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


def _build_adjacency(tri, n):
    """Build adjacency list from Delaunay triangulation."""
    adj = {i: [] for i in range(n)}
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                adj[a].append(b)
                adj[b].append(a)
    return adj


def _flip_edge(tri, centres, edge, max_iterations=100):
    """
    Attempt to improve packing by flipping the given edge in the Delaunay triangulation.
    Returns improved centres if the flip helps, otherwise None.
    """
    n = len(centres)
    i, j = edge
    
    # Find the two triangles sharing this edge
    tri_a = None
    tri_b = None
    for simplex in tri.simplices:
        if i in simplex and j in simplex:
            if tri_a is None:
                tri_a = simplex
            else:
                tri_b = simplex
                break
    
    if tri_a is None or tri_b is None:
        return None
    
    # Get the third vertices
    third_a = [v for v in tri_a if v != i and v != j][0]
    third_b = [v for v in tri_b if v != i and v != j][0]
    
    # The flip would replace edge (i,j) with edge (third_a, third_b)
    # We simulate this by moving points i and j apart slightly
    # and points third_a and third_b toward each other slightly
    
    # Current positions
    p_i = centres[i].copy()
    p_j = centres[j].copy()
    p_a = centres[third_a].copy()
    p_b = centres[third_b].copy()
    
    # Direction from i to j
    dir_ij = p_j - p_i
    norm_ij = np.linalg.norm(dir_ij)
    if norm_ij < 1e-10:
        return None
    dir_ij = dir_ij / norm_ij
    
    # Direction from a to b
    dir_ab = p_b - p_a
    norm_ab = np.linalg.norm(dir_ab)
    if norm_ab < 1e-10:
        return None
    dir_ab = dir_ab / norm_ab
    
    # Try small perturbations
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for step_size in [0.002, 0.001, 0.0005, 0.0002]:
        # Move i and j apart
        new_centres = centres.copy()
        new_centres[i] = p_i - dir_ij * step_size
        new_centres[j] = p_j + dir_ij * step_size
        
        # Move a and b toward each other (to create room for the new edge)
        new_centres[third_a] = p_a + dir_ab * step_size * 0.5
        new_centres[third_b] = p_b - dir_ab * step_size * 0.5
        
        # Keep within bounds
        new_centres = np.clip(new_centres, -0.499, 0.499)
        
        new_radius = _compute_radius(new_centres)
        if new_radius > best_radius + 1e-12:
            best_radius = new_radius
            best_centres = new_centres.copy()
    
    if best_radius > _compute_radius(centres) + 1e-12:
        return best_centres
    return None


def _equalize_edges(tri, centres, n, max_iterations=50):
    """
    Iteratively equalize edge lengths in the Delaunay triangulation.
    Focus on the shortest edges and try to increase them.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for iteration in range(max_iterations):
        # Compute all pairwise distances
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        
        # Get Delaunay edges
        edges = _delaunay_edges(tri)
        
        if len(edges) == 0:
            break
        
        # Find the shortest Delaunay edge
        min_edge_dist = np.inf
        min_edge = None
        for i, j in edges:
            d = dists[i, j]
            if d < min_edge_dist:
                min_edge_dist = d
                min_edge = (i, j)
        
        if min_edge is None:
            break
        
        # Try to flip/improve this edge
        improved = _flip_edge(tri, centres, min_edge)
        
        if improved is not None:
            new_radius = _compute_radius(improved)
            if new_radius > best_radius + 1e-12:
                best_radius = new_radius
                best_centres = improved.copy()
                # Recompute Delaunay
                try:
                    tri = Delaunay(centres)
                except:
                    tri = Delaunay(best_centres)
        
        # Also try moving the shortest-edge endpoints apart directly
        i, j = min_edge
        dir_ij = centres[j] - centres[i]
        norm_ij = np.linalg.norm(dir_ij)
        if norm_ij > 1e-10:
            dir_ij = dir_ij / norm_ij
            for step in [0.001, 0.0005, 0.0002]:
                new_centres = centres.copy()
                new_centres[i] -= dir_ij * step
                new_centres[j] += dir_ij * step
                new_centres = np.clip(new_centres, -0.499, 0.499)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-12:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    try:
                        tri = Delaunay(best_centres)
                    except:
                        pass
    
    return best_centres, best_radius


def _local_optimization(centres, max_iterations=300, step_size=0.01):
    """
    Local optimization combining edge-flip and direct perturbation.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Initial Delaunay
    try:
        tri = Delaunay(centres)
    except:
        return best_centres, best_radius
    
    for iteration in range(max_iterations):
        # Equalize edges
        equalized, eq_radius = _equalize_edges(tri, centres, N, max_iterations=20)
        
        if eq_radius > best_radius + 1e-12:
            best_radius = eq_radius
            best_centres = equalized.copy()
            try:
                tri = Delaunay(best_centres)
            except:
                pass
        
        # Direct perturbation of the most constrained circle
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        min_i, min_j = np.unravel_index(np.argmin(dists), dists.shape)
        
        dist_to_border = 0.5 - np.abs(centres)
        min_border_dist = np.min(dist_to_border)
        
        if min_dist / 2.0 < min_border_dist:
            # Pair constraint active
            direction = centres[min_i] - centres[min_j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            step = step_size * (1.0 + 0.5 * np.random.rand())
            new_centres = centres.copy()
            new_centres[min_i] += direction * step
            new_centres[min_j] -= direction * step
            new_centres = np.clip(new_centres, -0.499, 0.499)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-12:
                best_radius = new_radius
                best_centres = new_centres.copy()
                try:
                    tri = Delaunay(best_centres)
                except:
                    pass
        else:
            # Border constraint active
            min_border_i, min_border_k = np.unravel_index(np.argmin(dist_to_border), dist_to_border.shape)
            pos = centres[min_border_i]
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
            
            step = step_size * (1.0 + 0.5 * np.random.rand())
            new_centres = centres.copy()
            new_centres[min_border_i] += direction * step
            new_centres = np.clip(new_centres, -0.499, 0.499)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-12:
                best_radius = new_radius
                best_centres = new_centres.copy()
                try:
                    tri = Delaunay(best_centres)
                except:
                    pass
        
        # Random perturbation to escape local minima
        if iteration > max_iterations // 2:
            idx = np.random.randint(N)
            perturbation = np.random.randn(2) * step_size * 0.5
            new_centres = best_centres.copy()
            new_centres[idx] += perturbation
            new_centres = np.clip(new_centres, -0.499, 0.499)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-12:
                best_radius = new_radius
                best_centres = new_centres.copy()
                try:
                    tri = Delaunay(best_centres)
                except:
                    pass
        
        # Reduce step size
        if iteration > max_iterations // 3:
            step_size *= 0.95
            if step_size < 1e-6:
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
    
    # Run multiple iterations with different random restarts
    for restart in range(5):
        if restart > 0:
            # Add random perturbation for diversity
            centres = best_centres + np.random.randn(N, 2) * 0.003
            centres = np.clip(centres, -0.499, 0.499)
        
        # Local optimization with edge-flip
        optimised, radius = _local_optimization(centres, max_iterations=200, step_size=0.006)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
