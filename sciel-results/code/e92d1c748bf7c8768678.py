"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import combinations
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    sqrt3 = np.sqrt(3.0)
    
    # Generate hexagonal cluster points
    # Ring 1: 6 points at distance sqrt3
    ring1_angles = [0, 60, 120, 180, 240, 300]
    ring1 = [(sqrt3 * np.cos(np.radians(a)), sqrt3 * np.sin(np.radians(a))) for a in ring1_angles]
    
    # Ring 2 options: two orientations
    ring2_orient_a = [30, 90, 150, 210, 270, 330]
    ring2_orient_b = [0, 60, 120, 180, 240, 300]
    
    ring2_a = [(2 * sqrt3 * np.cos(np.radians(a)), 2 * sqrt3 * np.sin(np.radians(a))) for a in ring2_orient_a]
    ring2_b = [(2 * sqrt3 * np.cos(np.radians(a)), 2 * sqrt3 * np.sin(np.radians(a))) for a in ring2_orient_b]
    
    base = [(0.0, 0.0)] + ring1  # 7 points
    
    best_side = float('inf')
    best_points = None
    best_center = None
    best_rot = 0.0
    
    # Coarse search over rotation and ring2 subsets
    candidate_configs = []
    for ring2_set in [ring2_a, ring2_b]:
        for subset in combinations(range(6), 4):
            pts = base + [ring2_set[i] for i in subset]
            candidate_configs.append(pts)
    
    # Also add some non-hexagonal configs for robustness
    # Linear-ish arrangements won't be optimal but let's focus on hexagonal
    
    for points in candidate_configs:
        if time.time() > deadline:
            break
        points = np.array(points)
        
        # Coarse rotation search
        for rot_deg in np.arange(0, 60, 1.0):
            if time.time() > deadline:
                break
            
            rot = np.radians(rot_deg)
            edge_normals = [rot + np.radians(30 + k * 60) for k in range(6)]
            
            # Compute optimal center via least squares
            A = np.array([[np.cos(na), np.sin(na)] for na in edge_normals])
            b = np.zeros(6)
            for i, na in enumerate(edge_normals):
                n = np.array([np.cos(na), np.sin(na)])
                projs = points @ n
                b[i] = (np.max(projs) + np.min(projs)) / 2.0
            
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Compute required apothem
            points_shifted = points - c
            max_proj = 0
            for na in edge_normals:
                n = np.array([np.cos(na), np.sin(na)])
                projs = points_shifted @ n
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            
            side = 2 * max_proj / sqrt3
            
            if side < best_side:
                best_side = side
                best_points = points.copy()
                best_center = c.copy()
                best_rot = rot_deg
    
    if best_points is None:
        inner = [[0.0, 0.0, 0.0]]
        return inner, (0.0, 0.0), 10.0, 0.0
    
    # Fine rotation search around best_rot
    if time.time() < deadline - 10:
        for rot_deg in np.arange(best_rot - 2.0, best_rot + 2.0, 0.1):
            if time.time() > deadline:
                break
            if rot_deg < 0 or rot_deg >= 60:
                continue
            
            rot = np.radians(rot_deg)
            edge_normals = [rot + np.radians(30 + k * 60) for k in range(6)]
            
            A = np.array([[np.cos(na), np.sin(na)] for na in edge_normals])
            b = np.zeros(6)
            for i, na in enumerate(edge_normals):
                n = np.array([np.cos(na), np.sin(na)])
                projs = best_points @ n
                b[i] = (np.max(projs) + np.min(projs)) / 2.0
            
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            points_shifted = best_points - c
            max_proj = 0
            for na in edge_normals:
                n = np.array([np.cos(na), np.sin(na)])
                projs = points_shifted @ n
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            
            side = 2 * max_proj / sqrt3
            
            if side < best_side:
                best_side = side
                best_center = c.copy()
                best_rot = rot_deg
    
    # Very fine rotation search
    if time.time() < deadline - 5:
        for rot_deg in np.arange(best_rot - 0.2, best_rot + 0.2, 0.01):
            if time.time() > deadline:
                break
            if rot_deg < 0 or rot_deg >= 60:
                continue
            
            rot = np.radians(rot_deg)
            edge_normals = [rot + np.radians(30 + k * 60) for k in range(6)]
            
            A = np.array([[np.cos(na), np.sin(na)] for na in edge_normals])
            b = np.zeros(6)
            for i, na in enumerate(edge_normals):
                n = np.array([np.cos(na), np.sin(na)])
                projs = best_points @ n
                b[i] = (np.max(projs) + np.min(projs)) / 2.0
            
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            points_shifted = best_points - c
            max_proj = 0
            for na in edge_normals:
                n = np.array([np.cos(na), np.sin(na)])
                projs = points_shifted @ n
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            
            side = 2 * max_proj / sqrt3
            
            if side < best_side:
                best_side = side
                best_center = c.copy()
                best_rot = rot_deg
    
    # Format output
    inner = [[float(p[0]), float(p[1]), 0.0] for p in best_points]
    
    return inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_rot)
# EVOLVE-BLOCK-END
