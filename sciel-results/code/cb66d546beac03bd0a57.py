"""Improved program for Heilbronn triangles task using lattice search + local polish."""
import numpy as np
from itertools import product

N = 11


def construct():
    """
    Place 11 points on or inside a unit-area triangle and maximize the smallest
    area of a triangle formed by three points.
    
    Approach:
    1. Use a 1D lattice search over barycentric coordinates to find a good initial
       configuration that maximizes minimum pairwise distance (packing proxy).
    2. Perform local polish using gradient-free optimization to directly maximize
       the minimum triangle area.
    """
    # The reference triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Area of this triangle is sqrt(3)/4
    
    def barycentric_to_cartesian(b):
        """Convert barycentric coordinates (b0, b1, b2) with b0+b1+b2=1 to cartesian."""
        # b = (b0, b1, b2), point = b0*(0,0) + b1*(1,0) + b2*(0.5, sqrt(3)/2)
        x = b[1] + 0.5 * b[2]
        y = (3 ** 0.5) / 2 * b[2]
        return np.array([x, y])
    
    def triangle_area(p1, p2, p3):
        """Compute area of triangle formed by three points."""
        return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    
    def min_triangle_area(points):
        """Compute the minimum area among all triangles formed by triples of points."""
        min_area = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    area = triangle_area(points[i], points[j], points[k])
                    if area < min_area:
                        min_area = area
        return min_area
    
    def min_pairwise_distance(points):
        """Compute minimum pairwise distance."""
        min_dist = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_dist:
                    min_dist = d
        return min_dist
    
    def points_in_triangle(points):
        """Check if all points are inside or on the boundary of the reference triangle."""
        for p in points:
            x, y = p
            if x < -1e-10 or y < -1e-10 or x > 1 + 1e-10:
                return False
            # Check y <= sqrt(3)/2 * x (for x <= 0.5) and y <= sqrt(3)/2 * (1-x) (for x >= 0.5)
            if x <= 0.5:
                if y > (3 ** 0.5) / 2 * x + 1e-10:
                    return False
            else:
                if y > (3 ** 0.5) / 2 * (1 - x) + 1e-10:
                    return False
        return True
    
    def clamp_to_triangle(points):
        """Clamp points to be inside the triangle."""
        clamped = []
        for p in points:
            x, y = p
            # Clamp to bounds
            x = max(0.0, min(1.0, x))
            y = max(0.0, y)
            if x <= 0.5:
                max_y = (3 ** 0.5) / 2 * x
            else:
                max_y = (3 ** 0.5) / 2 * (1 - x)
            y = min(y, max_y)
            clamped.append([x, y])
        return np.array(clamped)
    
    # Step 1: Lattice search in barycentric coordinates
    # We'll search over a grid of barycentric coordinates
    # For N=11 points, we use a structured approach
    
    # Use a known good starting configuration based on known optimal arrangements
    # For 11 points, the known best arrangement has a specific structure
    
    # Start with a grid-based initialization in barycentric space
    # The barycentric coordinates must sum to 1 and each >= 0
    
    # Let's use a simple approach: place points on a hexagonal-like grid
    # inside the triangle
    
    # Known good configuration for 11 points (from literature):
    # The optimal configuration for n=11 has a minimum area of approximately 0.0116
    # Let's try to find a good configuration through optimization
    
    # Initialize with points spread across the triangle
    rng = np.random.RandomState(42)
    
    # Start with a regular grid in barycentric coordinates
    # Divide the triangle into regions
    
    # Use a simple initialization: place points at vertices, midpoints, and center
    # plus additional points
    
    # Better: use a structured initialization
    # Place points at:
    # - 3 vertices
    # - 3 edge midpoints  
    # - 1 center
    # - 4 more points distributed
    
    # Actually, let's use a more systematic approach
    # Start with points on a triangular lattice
    
    # Triangular lattice points inside the triangle
    # For a triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    # Lattice points at (i/n + j/(2n), j*sqrt(3)/(2n)) for appropriate i,j
    
    def generate_lattice_points(n_div):
        """Generate lattice points inside the triangle."""
        points = []
        for j in range(n_div + 1):
            for i in range(n_div - j + 1):
                # Barycentric coordinates: (1 - i/n_div - j/n_div, i/n_div, j/n_div)
                b0 = 1 - i / n_div - j / n_div
                b1 = i / n_div
                b2 = j / n_div
                if b0 >= -1e-10 and b1 >= -1e-10 and b2 >= -1e-10:
                    p = barycentric_to_cartesian([b0, b1, b2])
                    points.append(p)
        return np.array(points)
    
    # Generate a set of candidate lattice points
    all_lattice = generate_lattice_points(10)
    
    # Select 11 points from the lattice that maximize minimum pairwise distance
    # Use a greedy approach
    
    def greedy_select(points, n_select):
        """Greedy select n points from candidates to maximize min pairwise distance."""
        if len(points) <= n_select:
            return points.copy()
        
        selected = [points[0]]
        remaining = list(range(1, len(points)))
        
        for _ in range(n_select - 1):
            best_idx = -1
            best_min_dist = -1
            for idx in remaining:
                p = points[idx]
                min_d = min(np.linalg.norm(p - s) for s in selected)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            selected.append(points[best_idx])
            remaining.remove(best_idx)
        
        return np.array(selected)
    
    # Try different lattice resolutions and pick the best
    best_points = None
    best_score = -1
    
    for n_div in [3, 4, 5, 6, 7, 8]:
        lattice = generate_lattice_points(n_div)
        if len(lattice) >= N:
            selected = greedy_select(lattice, N)
            score = min_pairwise_distance(selected)
            if score > best_score:
                best_score = score
                best_points = selected
    
    # Also try random initialization and keep the best
    for trial in range(200):
        # Random points in barycentric coordinates
        bary = rng.dirichlet(np.ones(N), size=N)
        pts = np.array([barycentric_to_cartesian(b) for b in bary])
        pts = clamp_to_triangle(pts)
        score = min_pairwise_distance(pts)
        if score > best_score:
            best_score = score
            best_points = pts
    
    if best_points is None:
        # Fallback: simple grid
        best_points = greedy_select(generate_lattice_points(5), N)
    
    # Step 2: Local polish to maximize minimum triangle area
    # Use a simple hill-climbing approach
    
    def local_polish(points, max_iters=5000, step_size=0.01):
        """Local polish using coordinate ascent."""
        points = points.copy()
        current_min_area = min_triangle_area(points)
        
        for it in range(max_iters):
            improved = False
            for i in range(N):
                # Try small perturbations
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        new_points = points.copy()
                        new_points[i] = points[i] + np.array([dx, dy])
                        new_points = clamp_to_triangle(new_points)
                        new_min_area = min_triangle_area(new_points)
                        if new_min_area > current_min_area:
                            points = new_points
                            current_min_area = new_min_area
                            improved = True
            if not improved:
                # Reduce step size
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        return points, current_min_area
    
    # Perform local polish
    polished_points, final_min_area = local_polish(best_points, max_iters=3000, step_size=0.05)
    
    # Final check
    if not points_in_triangle(polished_points):
        polished_points = clamp_to_triangle(polished_points)
    
    return polished_points
