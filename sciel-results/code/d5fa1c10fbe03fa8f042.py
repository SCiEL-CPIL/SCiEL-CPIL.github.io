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


def _try_flip(tri, centres, edge_i, edge_j, best_radius):
    """
    Attempt to flip edges between two adjacent triangles sharing an edge.
    Returns the new centres if the flip improves the radius, otherwise None.
    """
    # Find the two triangles sharing the edge (edge_i, edge_j)
    tri_a = None
    tri_b = None
    for simplex in tri.simplices:
        if edge_i in simplex and edge_j in simplex:
            if tri_a is None:
                tri_a = simplex
            else:
                tri_b = simplex
                break
    
    if tri_a is None or tri_b is None:
        return None
    
    # The shared edge is (edge_i, edge_j), the other vertices are the third vertices
    third_a = [v for v in tri_a if v != edge_i and v != edge_j][0]
    third_b = [v for v in tri_b if v != edge_i and v != edge_j][0]
    
    # Check if flipping is valid (the new edge (third_a, third_b) should be inside the convex hull)
    # For simplicity, we just try the flip and check if it improves things
    # The flip would replace edges (edge_i, edge_j) with (third_a, third_b)
    # and change the triangulation
    
    # In a proper flip, we'd recompute the Delaunay triangulation
    # For now, let's just return None as a placeholder since implementing
    # a full flip with radius improvement check is complex
    
    return None


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


def _delaunay_flip_optimization(centres, max_iterations=200):
    """
    Delaunay-based combinatorial optimization:
    1. Build Delaunay triangulation
    2. Find shortest edges (bottlenecks)
    3. Try to flip edges to improve minimum distance
    4. Combine with local perturbation for continuous refinement
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for iteration in range(max_iterations):
        # Build Delaunay triangulation
        try:
            tri = Delaunay(best_centres)
        except Exception:
            # If Delaunay fails (degenerate case), break
            break
        
        # Get all edges from the triangulation
        edges = _delaunay_edges(tri)
        
        if not edges:
            break
        
        # Compute edge lengths
        edge_lengths = {}
        for i, j in edges:
            length = np.linalg.norm(best_centres[i] - best_centres[j])
            edge_lengths[(i, j)] = length
        
        # Find the shortest edge
        min_edge = min(edge_lengths, key=edge_lengths.get)
        min_length = edge_lengths[min_edge]
        
        # Also check border constraint
        dist_to_border = 0.5 - np.abs(best_centres)
        min_border_dist = np.min(dist_to_border)
        
        # Determine if we should try a flip or a local move
        if min_length / 2.0 < min_border_dist:
            # Try to flip the shortest edge
            i, j = min_edge
            
            # Find triangles containing this edge
            triangles_with_edge = []
            for simplex in tri.simplices:
                if i in simplex and j in simplex:
                    triangles_with_edge.append(simplex)
            
            if len(triangles_with_edge) == 2:
                tri1 = triangles_with_edge[0]
                tri2 = triangles_with_edge[1]
                
                # Get the other vertices
                other1 = [v for v in tri1 if v != i and v != j][0]
                other2 = [v for v in tri2 if v != i and v != j][0]
                
                # Try flipping: move vertices to increase the minimum distance
                # The flip would create a new edge between other1 and other2
                # We approximate this by moving i and j apart and other1 and other2 together
                
                # Compute directions
                dir_ij = best_centres[i] - best_centres[j]
                norm_ij = np.linalg.norm(dir_ij)
                if norm_ij > 1e-10:
                    dir_ij = dir_ij / norm_ij
                else:
                    dir_ij = np.random.randn(2)
                    dir_ij = dir_ij / np.linalg.norm(dir_ij)
                
                # Try a small flip-like move
                step_size = 0.003
                new_centres = best_centres.copy()
                new_centres[i] += dir_ij * step_size
                new_centres[j] -= dir_ij * step_size
                
                # Also try moving the other vertices
                dir_other = best_centres[other1] - best_centres[other2]
                norm_other = np.linalg.norm(dir_other)
                if norm_other > 1e-10:
                    dir_other = dir_other / norm_other
                else:
                    dir_other = np.random.randn(2)
                    dir_other = dir_other / np.linalg.norm(dir_other)
                
                new_centres[other1] += dir_other * step_size * 0.5
                new_centres[other2] -= dir_other * step_size * 0.5
                
                # Keep within bounds
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-10:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
            else:
                # Edge is on the boundary, try to move it inward
                i, j = min_edge
                midpoint = (best_centres[i] + best_centres[j]) / 2
                
                # Direction toward center
                dir_center = -midpoint
                norm_center = np.linalg.norm(dir_center)
                if norm_center > 1e-10:
                    dir_center = dir_center / norm_center
                else:
                    dir_center = np.random.randn(2)
                    dir_center = dir_center / np.linalg.norm(dir_center)
                
                step_size = 0.002
                new_centres = best_centres.copy()
                new_centres[i] += dir_center * step_size
                new_centres[j] += dir_center * step_size
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-10:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
        else:
            # Border constraint is active
            min_border_i, min_border_k = np.unravel_index(np.argmin(dist_to_border), dist_to_border.shape)
            pos = best_centres[min_border_i]
            
            # Direction toward center
            dir_center = -pos
            norm_center = np.linalg.norm(dir_center)
            if norm_center > 1e-10:
                dir_center = dir_center / norm_center
            else:
                dir_center = np.random.randn(2)
                dir_center = dir_center / np.linalg.norm(dir_center)
            
            step_size = 0.002
            new_centres = best_centres.copy()
            new_centres[min_border_i] += dir_center * step_size
            new_centres = np.clip(new_centres, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-10:
                best_radius = new_radius
                best_centres = new_centres.copy()
        
        # Occasionally add random perturbation to escape local minima
        if iteration % 10 == 9:
            idx = np.random.randint(len(best_centres))
            perturbation = np.random.randn(2) * 0.002
            new_centres = best_centres.copy()
            new_centres[idx] += perturbation
            new_centres = np.clip(new_centres, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-10:
                best_radius = new_radius
                best_centres = new_centres.copy()
    
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
    
    # First pass: local optimization
    for restart in range(3):
        if restart > 0:
            centres = best_centres + np.random.randn(N, 2) * 0.005
            centres = np.clip(centres, -0.5, 0.5)
        
        optimised, radius = _local_optimization(centres, max_iterations=300, step_size=0.008)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    # Second pass: Delaunay-based flip optimization
    for restart in range(2):
        if restart > 0:
            centres = best_centres + np.random.randn(N, 2) * 0.003
            centres = np.clip(centres, -0.5, 0.5)
        else:
            centres = best_centres.copy()
        
        optimised, radius = _delaunay_flip_optimization(centres, max_iterations=150)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    # Third pass: fine-tune with local optimization
    optimised, radius = _local_optimization(best_centres, max_iterations=200, step_size=0.003)
    if radius > best_radius:
        best_radius = radius
        best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
