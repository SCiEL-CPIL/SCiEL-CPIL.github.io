"""Voronoi-based circle packing in perimeter-4 rectangle."""
import numpy as np
import time
from scipy.spatial import Voronoi, Delaunay
import sys

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi, Delaunay

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0
    
    def compute_voronoi_inradii(points, W, H):
        """
        Compute the inradius for each point in a Voronoi diagram constrained to [0,W]x[0,H].
        points: (N, 2) array of sites.
        Returns: (N,) array of inradii.
        """
        n = len(points)
        if n < 3:
            # Degenerate case, just distance to walls
            rs = np.zeros(n)
            for i in range(n):
                rs[i] = min(points[i,0], W-points[i,0], points[i,1], H-points[i,1])
            return rs
        
        try:
            vor = Voronoi(points)
        except Exception:
            # Fallback to simple distance if Voronoi fails
            rs = np.zeros(n)
            for i in range(n):
                min_d = min(points[i,0], W-points[i,0], points[i,1], H-points[i,1])
                for j in range(n):
                    if i != j:
                        d = np.linalg.norm(points[i] - points[j])
                        min_d = min(min_d, d/2.0)
                rs[i] = min_d
            return rs

        # Get the Voronoi vertices and region indices
        # Voronoi.vertices and Voronoi.pointregion
        # We need to handle the infinite regions by clipping to the rectangle.
        
        inradii = np.zeros(n)
        
        # Helper to check if a point is inside the rectangle
        def is_inside(p):
            return -1e-9 <= p[0] <= W + 1e-9 and -1e-9 <= p[1] <= H + 1e-9
        
        # The vertices of the Voronoi diagram are in vor.vertices
        # The regions are defined by vor.regions[i] which contains indices into vor.vertices
        # Some vertices are -1, indicating the region is unbounded (infinite).
        # We need to intersect the Voronoi cells with the bounding box.
        
        # A robust way to find the inradius of a clipped Voronoi cell:
        # The inradius is the minimum distance from the site to the boundary of its cell.
        # The boundary consists of segments between Voronoi vertices (clipped to the box).
        # Alternatively, we can sample the boundary or use the fact that the inradius
        # is the radius of the largest circle centered at the site that fits in the cell.
        
        # A simpler, robust approximation for inradius:
        # The inradius is the minimum distance from the site to any edge of its Voronoi cell.
        # The edges are formed by the perpendicular bisectors of the Delaunay triangles.
        # However, clipping makes this tricky.
        
        # Let's use a more direct method:
        # For each site i, the inradius is the minimum distance to the boundary of its cell.
        # The boundary of the cell is a polygon.
        # We can reconstruct the polygon for each cell by clipping the Voronoi edges.
        
        # Actually, there's a simpler geometric property:
        # The inradius of the Voronoi cell of site i is the minimum of:
        # 1. Distance to the walls (if the cell touches a wall)
        # 2. Half the distance to the nearest neighbor site (if the cell is bounded by a Voronoi edge)
        # This is NOT strictly true for all cells, but it's a good lower bound.
        # Actually, the inradius is exactly the minimum distance from the site to the boundary of its cell.
        # The boundary is composed of line segments.
        
        # Let's implement a robust clipping-based inradius calculation.
        # We'll find the polygon for each cell by clipping the Voronoi edges against the rectangle.
        
        # For efficiency, we'll use the fact that the inradius is the minimum distance to the cell boundary.
        # We can compute this by finding the closest point on the cell boundary to the site.
        
        # A practical approach:
        # 1. Compute the Voronoi diagram.
        # 2. For each site, find its Voronoi region.
        # 3. Clip the region to the rectangle [0,W]x[0,H].
        # 4. The inradius is the minimum distance from the site to the boundary of the clipped polygon.
        
        # To avoid complex polygon clipping, we can use the following:
        # The inradius is the minimum distance from the site to any of the following:
        # - The walls of the rectangle
        # - The Voronoi edges (perpendicular bisectors)
        
        # For each site i, the inradius r_i is:
        # r_i = min(
        #     distance to walls,
        #     min over neighbors j of (distance(i, j) / 2)  # This is only valid if the bisector is the limiting factor
        # )
        # This is an approximation. For a more accurate result, we need to check if the point on the bisector
        # that is closest to site i is actually within the Voronoi cell.
        
        # Let's use a more accurate method:
        # For each site i, consider all other sites j.
        # The Voronoi edge between i and j is the perpendicular bisector.
        # The distance from site i to this bisector is dist(i,j)/2.
        # However, this distance is only relevant if the projection of site i onto the bisector lies within the Voronoi cell.
        # A simpler and robust method is to use the fact that the inradius is the minimum distance to the boundary.
        # We can approximate the boundary by sampling points on the Voronoi edges.
        
        # For now, let's use the minimum of distance to walls and half-distance to nearest neighbors.
        # This is a valid lower bound and often close to the true inradius.
        # To improve accuracy, we can refine this by checking the actual Voronoi cell boundaries.
        
        for i in range(n):
            # Distance to walls
            d_wall = min(points[i,0], W-points[i,0], points[i,1], H-points[i,1])
            min_d = d_wall
            
            # Distance to other sites
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(points[i] - points[j])
                # The distance to the bisector is d/2
                # But we need to check if this is the limiting factor.
                # For simplicity, we take the minimum of d/2 and d_wall.
                min_d = min(min_d, d/2.0)
            
            inradii[i] = min_d
            
        return inradii

    def compute_gradients(points, W, H, inradii):
        """
        Compute the gradient of the sum of inradii with respect to the points.
        This is an approximation. The gradient of the inradius of site i with respect to site i's position
        is in the direction away from the closest boundary point.
        """
        n = len(points)
        grad = np.zeros_like(points)
        
        for i in range(n):
            # Find the closest boundary point for site i
            # This is the point on the Voronoi cell boundary that is closest to site i.
            # We approximate this by finding the closest point among:
            # - Wall segments
            # - Voronoi edges (bisectors)
            
            min_dist = inradii[i]
            closest_point = None
            
            # Check walls
            # Left wall: x=0
            p = np.array([0, points[i,1]])
            d = np.linalg.norm(points[i] - p)
            if d < min_dist:
                min_dist = d
                closest_point = p
            # Right wall: x=W
            p = np.array([W, points[i,1]])
            d = np.linalg.norm(points[i] - p)
            if d < min_dist:
                min_dist = d
                closest_point = p
            # Bottom wall: y=0
            p = np.array([points[i,0], 0])
            d = np.linalg.norm(points[i] - p)
            if d < min_dist:
                min_dist = d
                closest_point = p
            # Top wall: y=H
            p = np.array([points[i,0], H])
            d = np.linalg.norm(points[i] - p)
            if d < min_dist:
                min_dist = d
                closest_point = p
            
            # Check Voronoi edges (bisectors)
            for j in range(n):
                if i == j:
                    continue
                # Midpoint
                mid = (points[i] + points[j]) / 2.0
                # Vector from i to j
                v = points[j] - points[i]
                # The bisector is the line through mid perpendicular to v.
                # The closest point on the bisector to site i is mid.
                # The distance from i to mid is dist(i,j)/2.
                d = np.linalg.norm(points[i] - mid)
                if d < min_dist:
                    # Check if mid is within the Voronoi cell of i.
                    # This is complex, so we assume it is for now.
                    min_dist = d
                    closest_point = mid
            
            if closest_point is not None:
                # Gradient is in the direction from closest_point to site i
                # But we want to maximize the inradius, so we move away from the boundary.
                # The gradient of the distance function is (site - closest_point) / distance
                dir = (points[i] - closest_point) / min_dist if min_dist > 1e-9 else np.zeros(2)
                grad[i] = dir
        
        return grad

    def optimize_voronoi(W, H, initial_points, iterations=5000):
        """
        Optimize the positions of the points to maximize the sum of inradii.
        Uses gradient ascent.
        """
        points = initial_points.copy()
        n = len(points)
        
        # Adaptive learning rate
        lr = 0.01
        best_sum_r = -np.inf
        best_points = points.copy()
        
        for it in range(iterations):
            if time.time() > deadline - 5:
                break
            
            # Compute inradii
            inradii = compute_voronoi_inradii(points, W, H)
            sum_r = np.sum(inradii)
            
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_points = points.copy()
            
            # Compute gradients
            grad = compute_gradients(points, W, H, inradii)
            
            # Update points
            points += lr * grad
            
            # Project to bounds
            points[:, 0] = np.clip(points[:, 0], 1e-6, W - 1e-6)
            points[:, 1] = np.clip(points[:, 1], 1e-6, H - 1e-6)
            
            # Reduce learning rate
            lr *= 0.999
        
        return best_points, best_sum_r

    # Main loop: try different aspect ratios and random initializations
    np.random.seed(42)
    
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    num_restarts = 3
    
    for W_ratio in aspect_ratios:
        if time.time() > deadline - 30:
            break
        
        W = W_ratio
        H = 2.0 - W_ratio
        if W <= 0 or H <= 0:
            continue
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + restart * 1000 + int(W_ratio * 100))
            
            # Initialize points in a grid-like pattern with jitter
            side = int(np.ceil(np.sqrt(N)))
            init_points = np.zeros((N, 2))
            for i in range(N):
                row = i // side
                col = i % side
                init_points[i, 0] = (col + 0.5) * W / side + np.random.uniform(-0.1, 0.1)
                init_points[i, 1] = (row + 0.5) * H / side + np.random.uniform(-0.1, 0.1)
            
            # Optimize
            optimized_points, sum_r = optimize_voronoi(W, H, init_points, iterations=2000)
            
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                # Compute final inradii for the optimized points
                inradii = compute_voronoi_inradii(optimized_points, W, H)
                # Create the circle configuration
                config = np.column_stack([optimized_points, inradii])
                best_config = config
                best_W = W
                best_H = H
    
    if best_config is not None:
        return best_config
    
    # Fallback: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
