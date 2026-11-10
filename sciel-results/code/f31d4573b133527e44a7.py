# EVOLVE-BLOCK-START
import numpy as np
import time

def erdos_szekeres_problem7() -> np.ndarray:
    """
    Place 33 points in the unit square to minimize convex 7-gons.
    Uses a hierarchical arc-based construction.
    """
    n = 33
    
    # Use a construction based on nested arcs/clusters
    # Strategy: place points on concentric arcs with limited angular spread
    # to prevent 7 points from forming a convex polygon
    
    # Construction: Use a "spiral" or "nested cluster" approach
    # Points are placed in groups on arcs, with each group having limited angular range
    
    # Approach: Place points along a logarithmic spiral in the unit square
    # This creates a structure where convex chains are limited
    
    # Better approach: Use the recursive ES construction
    # For k=7, n=33: recursively build using cups and caps
    
    # Let's use a practical approach: place points on a few arcs
    # with points clustered to limit convex 7-gon formation
    
    # Use a grid of clusters: 3x3 clusters with 11-12 points each
    # arranged so that the overall structure limits convex 7-gons
    
    # Actually, let's use a proven approach: place points on a parabola
    # y = x^2, which is known to minimize convex polygons
    
    # For 33 points on a parabola in [0,1]x[0,1]:
    # x values evenly spaced, y = x^2
    # This gives a "cup" structure - all points are on the convex hull
    # but in a way that limits 7-gons... actually this maximizes them.
    
    # Better: Use a "cap" structure - points on a concave curve
    # or use the ES recursive construction
    
    # The ES construction for 0 convex k-gons uses:
    # S(k) = 2*S(k-1) + 1, with S(3) = 1
    # S(3)=1, S(4)=3, S(5)=7, S(6)=15, S(7)=31... but we need 33
    # Actually the bound is 2^(k-2)+1, so for k=7: 2^5+1 = 33
    
    # The construction: recursively, S(k) consists of two copies of S(k-1)
    # with one extra point. The two copies are placed so that no convex k-gon
    # can be formed.
    
    # Let me implement the recursive ES construction:
    # Base: S(3) = 1 point
    # S(k) = S(k-1) in lower-left, S(k-1) in upper-right, + 1 point
    
    # But we need to fit in unit square and ensure general position
    
    def build_es_points(k, x0, y0, x1, y1, seed_offset=0):
        """Recursively build ES construction for k-gon avoidance."""
        if k == 3:
            # Base case: 1 point in the center
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            # Add small perturbation for general position
            np.random.seed(42 + seed_offset)
            dx = (np.random.rand() - 0.5) * 0.001
            dy = (np.random.rand() - 0.5) * 0.001
            return np.array([[cx + dx, cy + dy]])
        
        # Recursive case: two copies of S(k-1) + 1 point
        # Lower-left quadrant and upper-right quadrant
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        
        # Slightly offset the quadrants to avoid alignment
        eps = 0.001
        
        points_lower = build_es_points(k-1, x0, y0, mx - eps, my - eps, seed_offset + 1)
        points_upper = build_es_points(k-1, mx + eps, my + eps, x1, y1, seed_offset + 2)
        
        # Extra point in the center-ish area
        np.random.seed(42 + seed_offset + 3)
        cx = (x0 + x1) / 2 + (np.random.rand() - 0.5) * 0.01
        cy = (y0 + y1) / 2 + (np.random.rand() - 0.5) * 0.01
        extra = np.array([[cx, cy]])
        
        return np.vstack([points_lower, points_upper, extra])
    
    # Build for k=7
    np.random.seed(12345)
    points = build_es_points(7, 0.01, 0.01, 0.99, 0.99, 0)
    
    # Verify we have 33 points
    assert points.shape[0] == 33, f"Expected 33 points, got {points.shape[0]}"
    
    # Clip to [0, 1] with small margin
    points = np.clip(points, 1e-4, 1 - 1e-4)
    
    # Ensure no three points are collinear by adding tiny perturbations
    # Check and fix if needed
    for i in range(33):
        for j in range(i+1, 33):
            for l in range(j+1, 33):
                p1, p2, p3 = points[i], points[j], points[l]
                cross = p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])
                if abs(cross) < 1e-9:
                    # Perturb p3 slightly
                    points[l, 1] += 1e-6 * (1 if cross >= 0 else -1)
    
    # Final validation: ensure all in [0,1]
    points = np.clip(points, 1e-4, 1 - 1e-4)
    
    return points

# EVOLVE-BLOCK-END
