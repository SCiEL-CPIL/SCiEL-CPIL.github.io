import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    eps = 1e-12
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    # Iterate all 15 combinations of 4 outer hexagons from 6 possible positions
    all_angles = [0, 60, 120, 180, 240, 300]
    combinations = list(itertools.combinations(all_angles, 4))
    
    best_score = float('inf')
    best_config = None
    best_theta_rel = 0.0
    best_centers = None
    
    # Pre-compute centers for all combinations to avoid rebuilding in loop if possible?
    # Or just build in loop.
    
    for combo in combinations:
        # Build centers
        centers = np.array([[0.0, 0.0]])
        
        # Inner ring (6) at 30, 90, ...
        r1 = np.sqrt(3) + eps
        for k in range(6):
            angle = np.deg2rad(30 + k * 60)
            centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        
        # Outer ring (4) at chosen angles
        # Using radius 3.0 for the "corner" hexagons of the second shell
        r2 = 3.0 + 2 * eps
        for angle_deg in combo:
            angle = np.deg2rad(angle_deg)
            centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
        def compute_side_rigid(theta_rel, c=centers):
            theta_out = theta_rel
            normals = np.array([
                [np.cos(np.deg2rad(30 + k * 60) + theta_out),
                 np.sin(np.deg2rad(30 + k * 60) + theta_out)]
                for k in range(6)
            ])
            
            max_proj = 0.0
            for i in range(len(c)):
                verts = base_vertices + c[i]
                dots = verts @ normals.T
                max_proj = max(max_proj, np.max(np.abs(dots)))
            
            return 2.0 * max_proj / np.sqrt(3)

        # High-resolution grid search (Structural change: avoid minimize_scalar noise)
        thetas = np.linspace(0, np.pi/6, 1000)
        local_best_score = float('inf')
        local_best_theta = 0.0
        
        for t in thetas:
            if time.time() > deadline - 5:
                break
            s = compute_side_rigid(t)
            if s < local_best_score:
                local_best_score = s
                local_best_theta = t
        
        # Local refinement around the best grid point
        if time.time() < deadline - 2 and local_best_score < float('inf'):
            step = (np.pi/6) / 1000
            bounds = (max(0, local_best_theta - 5*step), min(np.pi/6, local_best_theta + 5*step))
            # Fine grid in the local neighborhood
            local_thetas = np.linspace(bounds[0], bounds[1], 100)
            for t in local_thetas:
                s = compute_side_rigid(t)
                if s < local_best_score:
                    local_best_score = s
                    local_best_theta = t

        if local_best_score < best_score:
            best_score = local_best_score
            best_config = combo
            best_theta_rel = local_best_theta
            best_centers = centers

    # Fallback if best_centers is still None (should not happen now)
    if best_centers is None:
        # Use the first combination as a valid fallback
        combo = combinations[0]
        centers = np.array([[0.0, 0.0]])
        r1 = np.sqrt(3) + eps
        for k in range(6):
            angle = np.deg2rad(30 + k * 60)
            centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        r2 = 3.0 + 2 * eps
        for angle_deg in combo:
            angle = np.deg2rad(angle_deg)
            centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
        def compute_side_rigid(theta_rel, c=centers):
            theta_out = theta_rel
            normals = np.array([
                [np.cos(np.deg2rad(30 + k * 60) + theta_out),
                 np.sin(np.deg2rad(30 + k * 60) + theta_out)]
                for k in range(6)
            ])
            max_proj = 0.0
            for i in range(len(c)):
                verts = base_vertices + c[i]
                dots = verts @ normals.T
                max_proj = max(max_proj, np.max(np.abs(dots)))
            return 2.0 * max_proj / np.sqrt(3)
            
        best_score = compute_side_rigid(0.0)
        best_theta_rel = 0.0
        best_centers = centers
        best_config = combo

    inner = []
    for i in range(len(best_centers)):
        inner.append([best_centers[i][0], best_centers[i][1], 0.0])

    side = best_score
    theta_out_deg = np.rad2deg(best_theta_rel)

    return inner, (0.0, 0.0), side, theta_out_deg

if __name__ == "__main__":
    construct()
