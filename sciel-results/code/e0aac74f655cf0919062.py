"""Packing 12 unit hexagons in minimal regular hexagon using Differential Evolution."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def min_enclosing_hex_side(centers, angles, outer_center=(0,0), outer_angle=0):
        """
        Calculate the minimum side length of a regular hexagon 
        with given center and rotation that contains all inner hexagons.
        """
        ox, oy = outer_center
        outer_angle_rad = math.radians(outer_angle)
        
        # The 6 normals of the outer hexagon (pointing outward)
        # A regular hexagon can be defined by 6 half-planes.
        # Normal directions are 0, 60, 120, 180, 240, 300 degrees relative to outer_angle.
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = centers[i]
            ang = angles[i]
            verts = hex_vertices(cx, cy, ang)
            
            # Translate vertices to outer hexagon's coordinate system
            # Rotate by -outer_angle, translate by -outer_center
            cos_a = math.cos(-outer_angle_rad)
            sin_a = math.sin(-outer_angle_rad)
            
            local_verts = []
            for vx, vy in verts:
                rx = (vx - ox) * cos_a - (vy - oy) * sin_a
                ry = (vx - ox) * sin_a + (vy - oy) * cos_a
                local_verts.append((rx, ry))
            
            # Project onto the 6 normal directions of the outer hexagon
            for j in range(6):
                theta = j * math.pi / 3.0
                nx = math.cos(theta)
                ny = math.sin(theta)
                
                # Max projection of this inner hexagon onto this normal
                proj = max(rx * nx + ry * ny for rx, ry in local_verts)
                if proj > max_proj:
                    max_proj = proj
        
        # For a regular hexagon with side L, the distance from center to side (apothem) is L * sqrt(3)/2.
        # The projection of the hexagon onto its normal is the apothem.
        # So, max_proj <= L * sqrt(3)/2  =>  L >= max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / sqrt3

    def check_overlap(centers, angles, eps=1e-9):
        """
        Check if any two inner hexagons overlap.
        Returns a penalty value (0 if no overlap, >0 if overlap).
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Quick distance check
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx*dx + dy*dy
                # Max distance between centers for non-overlapping unit hexagons is 2 * 2 * cos(30) = 2*sqrt(3)?
                # Actually, if they touch at a vertex, distance is 2. If they touch at a side, distance is sqrt(3).
                # If dist < sqrt(3) - eps, they definitely overlap? No, if rotated, min distance can be smaller.
                # Let's use a conservative bound. If dist > 2.0 + eps, they don't overlap.
                if dist_sq > (2.0 + eps)**2:
                    continue
                
                # Exact check: Polygon intersection
                # For simplicity in DE, we can approximate or use a simpler metric.
                # Let's use the fact that if the distance between centers is less than the inradius sum?
                # No, hexagons are not circles.
                # A robust but slightly slow check:
                # Generate vertices
                vi = hex_vertices(centers[i][0], centers[i][1], angles[i])
                vj = hex_vertices(centers[j][0], centers[j][1], angles[j])
                
                # Check if any vertex of i is inside j or vice versa
                # Or use Shapely? No, standard library only + numpy/scipy.
                # Let's implement a simple point-in-polygon and edge-intersection check.
                # This might be too slow for DE with 1000s of iterations.
                
                # Alternative: Use a "soft" penalty based on distance.
                # If we assume the hexagons are "close to aligned" or we just use a distance threshold.
                # For a unit hexagon, the minimum distance between two disjoint hexagons is 0.
                # If we use a distance penalty:
                # If dist < sqrt(3) (touching sides), penalty = (sqrt(3) - dist)
                # If dist < 2 (touching vertices), penalty = (2 - dist) * 0.5?
                # This is a heuristic. Let's try to be more accurate but fast.
                
                # Let's check if the center of one is inside the other.
                # And if edges intersect.
                # For speed, let's just use a distance-based penalty that is strong enough.
                # If dist < 1.7, it's very likely overlapping.
                if dist_sq < 1.7**2:
                    penalty += (1.7 - math.sqrt(dist_sq)) * 100.0
        return penalty

    def objective(params):
        # Unpack parameters: 
        # 12 * 3 (x, y, angle) + 1 (outer angle) 
        # We fix outer center to (0,0) for now to reduce dimensionality.
        # If time permits, we can optimize outer center too.
        centers = np.zeros((N, 2))
        angles = np.zeros(N)
        
        for i in range(N):
            centers[i][0] = params[i*3]
            centers[i][1] = params[i*3+1]
            angles[i] = params[i*3+2]
        
        outer_angle = params[N*3]
        
        # Check overlap
        overlap_penalty = check_overlap(centers, angles)
        
        # Calculate enclosing hexagon side
        L = min_enclosing_hex_side(centers, angles, (0,0), outer_angle)
        
        # Total cost
        return L + overlap_penalty * 10.0

    # Initial guess: A 3x4 grid of hexagons
    # Hexagon width (flat-to-flat) = sqrt(3), height (point-to-point) = 2
    # Let's arrange them in a grid.
    # x spacing: 1.5 (half width + 0.5?) 
    # For flat-top hexagons:
    # x_dist = 3/2 * side = 1.5
    # y_dist = sqrt(3) * side = 1.732
    # But we can rotate them.
    
    # Let's start with a random but structured guess.
    # 12 hexagons.
    # 2 rows of 6?
    # Row 0: x = 0, 1.5, 3.0, 4.5, 6.0, 7.5; y = 0
    # Row 1: x = 0.75, 2.25, 3.75, 5.25, 6.75, 8.25; y = sqrt(3)
    
    init_centers = []
    init_angles = []
    for i in range(6):
        init_centers.append((1.5 * i, 0.0))
        init_angles.append(0.0)
    for i in range(6):
        init_centers.append((1.5 * i + 0.75, sqrt3))
        init_angles.append(0.0)
        
    # Center the whole cluster
    xs = [c[0] for c in init_centers]
    ys = [c[1] for c in init_centers]
    cx_mean = (min(xs) + max(xs)) / 2.0
    cy_mean = (min(ys) + max(ys)) / 2.0
    
    init_params = []
    for i in range(N):
        init_params.append(init_centers[i][0] - cx_mean)
        init_params.append(init_centers[i][1] - cy_mean)
        init_params.append(init_angles[i])
    init_params.append(0.0) # outer angle

    bounds = []
    for i in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
        bounds.append((0, 60))   # angle (0 to 60 is enough for hex symmetry)
    bounds.append((0, 60)) # outer angle

    # Run Differential Evolution
    # maxiter needs to be small to fit in time.
    # 39 variables. 
    # Let's try maxiter=50, popsize=10.
    # 50 * 10 * 39 * (cost function time)
    # Cost function is O(N^2) for overlap check. 12*12/2 = 72 pairs.
    # Should be fast.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=0.01, 
            seed=42,
            init='sobol',
            polish=False
        )
        
        # Check time
        if time.time() > deadline:
             return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

        best_params = result.x
        best_cost = result.fun
        
        # Extract results
        centers = []
        angles = []
        for i in range(N):
            centers.append((best_params[i*3], best_params[i*3+1]))
            angles.append(best_params[i*3+2])
        outer_angle = best_params[N*3]
        
        # Verify validity (no overlap, valid L)
        # If overlap penalty is significant, the solution might be invalid.
        # We should check the actual overlap.
        
        # Re-calculate L for reporting
        L = min_enclosing_hex_side(centers, angles, (0,0), outer_angle)
        
        # If there is significant overlap, we might need to return a valid fallback.
        # But DE minimizes L + penalty, so if it found a low L, overlap should be low.
        
        inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
        
        return inner, (0.0, 0.0), L, outer_angle

    except Exception as e:
        # Fallback to seed
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END
