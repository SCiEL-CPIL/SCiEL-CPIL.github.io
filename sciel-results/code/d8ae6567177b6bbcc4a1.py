import time
import math
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = math.sqrt(3.0)
    
    # Unit hexagon vertices (flat-top, side=1)
    unit_verts = np.array([
        [math.cos(math.radians(i*60)), math.sin(math.radians(i*60))] 
        for i in range(6)
    ])
    
    # Outer hexagon edge normals (for a hexagon with side L, the inradius is L*sqrt(3)/2)
    # Edge normals at 30, 90, 150, 210, 270, 330 degrees
    edge_normals = np.array([
        [math.cos(math.radians(30 + i*60)), math.sin(math.radians(30 + i*60))] 
        for i in range(6)
    ])
    
    # Generate hexagonal lattice points (axial coordinates)
    # For a hexagon of "radius" R (side length in lattice units), centers are at
    # (q, r) where |q| <= R, |r| <= R, |q+r| <= R
    # Cartesian: x = 1.5*q, y = sqrt(3)*(r + q/2)
    
    def generate_lattice(R):
        pts = []
        for q in range(-R, R+1):
            for r in range(-R, R+1):
                if abs(q + r) <= R:
                    x = 1.5 * q
                    y = sqrt3 * (r + q / 2.0)
                    pts.append((x, y))
        return pts
    
    # For R=3, we get 19 points
    lattice_R3 = generate_lattice(3)
    lattice_R4 = generate_lattice(4)
    
    def compute_min_L(centers, outer_center=(0.0, 0.0), outer_angle=0.0):
        """Compute minimum side length L of outer hexagon enclosing all unit hexagons."""
        cx, cy = outer_center
        ca, sa = math.cos(outer_angle), math.sin(outer_angle)
        
        # Rotated edge normals
        normals = np.zeros((6, 2))
        for k in range(6):
            theta = outer_angle + math.radians(30 + k*60)
            normals[k] = [math.cos(theta), math.sin(theta)]
        
        max_proj = 0.0
        for px, py in centers:
            # For each edge normal, compute max projection of hexagon vertices
            for k in range(6):
                nx, ny = normals[k]
                # Support: max over vertices of dot(R(0)*v + (px,py), n) - dot((cx,cy), n)
                # = max_j dot(unit_verts[j], n) + px*nx + py*ny - cx*nx - cy*ny
                dots = unit_verts @ np.array([nx, ny])
                max_dot = dots.max()
                proj = px * nx + py * ny - (cx * nx + cy * ny) + max_dot
                if proj > max_proj:
                    max_proj = proj
        
        # L such that inradius = L*sqrt(3)/2 = max_proj
        # L = 2*max_proj/sqrt(3)
        return 2.0 * max_proj / sqrt3
    
    # Check if two unit hexagons at p1, p2 (both aligned) overlap
    def hexagons_overlap(p1, p2, eps=1e-10):
        """SAT check for two aligned unit hexagons."""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 1e-12:
            return True
        
        # For aligned hexagons, check 6 edge normals
        for k in range(6):
            nx, ny = edge_normals[k]
            # Project hex 1
            dots1 = unit_verts @ np.array([nx, ny])
            proj1_min = p1[0]*nx + p1[1]*ny + dots1.min()
            proj1_max = p1[0]*nx + p1[1]*ny + dots1.max()
            # Project hex 2
            dots2 = unit_verts @ np.array([nx, ny])
            proj2_min = p2[0]*nx + p2[1]*ny + dots2.min()
            proj2_max = p2[0]*nx + p2[1]*ny + dots2.max()
            
            if proj1_max + eps < proj2_min or proj2_max + eps < proj1_min:
                return False
        return True
    
    # Check if a set of centers are all non-overlapping
    def all_non_overlapping(centers):
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                if hexagons_overlap(centers[i], centers[j]):
                    return False
        return True
    
    best_L = float('inf')
    best_config = None
    
    # Step 1: Try all 12-subsets of the 19-point lattice (R=3)
    n_pts = len(lattice_R3)
    if n_pts >= 12:
        for subset in combinations(range(n_pts), 12):
            if time.time() > deadline - 20:
                break
            centers = [lattice_R3[i] for i in subset]
            if all_non_overlapping(centers):
                L = compute_min_L(centers)
                if L < best_L:
                    best_L = L
                    best_config = centers[:]
    
    # Step 2: Try subsets from larger lattice (R=4, 37 points) - sample
    n_pts4 = len(lattice_R4)
    if n_pts4 >= 12:
        # Try center + ring patterns
        # Center is at index where q=0, r=0
        center_idx = None
        for i, (x, y) in enumerate(lattice_R4):
            if abs(x) < 1e-10 and abs(y) < 1e-10:
                center_idx = i
                break
        
        if center_idx is not None:
            # Try: center + 6 inner ring + 5 outer
            # Inner ring: 6 points at distance sqrt(3)
            # Outer ring: 12 points at distance 2*sqrt(3) or 3
            
            # For now, just try a few structured subsets
            # Center + 6 neighbors + 5 more
            for subset in combinations(range(n_pts4), 12):
                if time.time() > deadline - 15:
                    break
                centers = [lattice_R4[i] for i in subset]
                if all_non_overlapping(centers):
                    L = compute_min_L(centers)
                    if L < best_L:
                        best_L = L
                        best_config = centers[:]
    
    # Step 3: Continuous refinement
    if best_config is not None and time.time() < deadline - 5:
        def objective(params):
            if time.time() > deadline:
                return 1e10
            centers = [(params[i*2], params[i*2+1]) for i in range(12)]
            oc = (params[24], params[25])
            oa = params[26]
            
            L = compute_min_L(centers, oc, oa)
            
            # Penalty for overlap
            penalty = 0.0
            for i in range(12):
                for j in range(i+1, 12):
                    if hexagons_overlap(centers[i], centers[j]):
                        dx = centers[i][0] - centers[j][0]
                        dy = centers[i][1] - centers[j][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        min_dist = sqrt3 * 0.99
                        if dist < min_dist:
                            penalty += (min_dist - dist) ** 2
            return L + 50.0 * penalty
        
        x0 = np.array([c[0] for c in best_config] + [c[1] for c in best_config] + [0.0, 0.0, 0.0])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_L and res.fun < 1e6:
                centers_ref = [(res.x[i*2], res.x[i*2+1]) for i in range(12)]
                L_ref = compute_min_L(centers_ref, (res.x[24], res.x[25]), res.x[26])
                if all_non_overlapping(centers_ref):
                    best_L = L_ref
                    best_config = centers_ref
        except Exception:
            pass
    
    if best_config is None:
        # Fallback: 12 points from R=3 lattice
        best_config = [lattice_R3[i] for i in range(12)]
        best_L = compute_min_L(best_config)
    
    return (
        [[best_config[i][0], best_config[i][1], 0.0] for i in range(12)],
        (0.0, 0.0),
        best_L,
        0.0
    )
