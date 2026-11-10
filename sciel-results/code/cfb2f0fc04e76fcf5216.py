# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Unit hexagon vertices (angle 0)
    # Vertices at 0, 60, ... 300 degrees
    hex_verts = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])
    
    # --- 1. Generate Candidate Clusters (Topologies) ---
    
    # Helper: Axial to Cartesian for hex grid
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        return np.array([x, y])
    
    # Cluster A: "Flower" (7) + 4 extensions in symmetric directions
    # Center (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: 4 specific neighbors to keep it compact
    # Let's try a few distinct topologies
    
    topologies = []
    
    # Topology 0: The seed's best candidate (likely [0, 3, 6, 9] in ring2 indices)
    # Ring 1 vectors
    d = np.sqrt(3)
    v = []
    for i in range(6):
        angle = np.deg2rad(60 * i)
        v.append([d * np.cos(angle), d * np.sin(angle)])
    v = np.array(v)
    
    # Ring 2 vectors
    ring2 = []
    for i in range(6):
        ring2.append(2 * v[i])
        ring2.append(v[i] + v[(i + 1) % 6])
    ring2 = np.array(ring2)
    
    # Seed's candidates were indices into ring2
    # Let's convert the seed's best few into explicit coordinate lists
    
    # Seed candidate [0, 3, 6, 9] -> 0, 90, 180, 270 deg offsets
    # 0: 2v0 (0 deg)
    # 3: v1+v2 (90 deg)
    # 6: 2v3 (180 deg)
    # 9: v4+v5 (270 deg)
    coords_0 = [np.array([0,0])] + [v[i] for i in range(6)] + [ring2[i] for i in [0, 3, 6, 9]]
    topologies.append(np.array(coords_0))
    
    # Seed candidate [0, 2, 6, 8] -> 0, 60, 180, 240
    coords_1 = [np.array([0,0])] + [v[i] for i in range(6)] + [ring2[i] for i in [0, 2, 6, 8]]
    topologies.append(np.array(coords_1))
    
    # Seed candidate [1, 4, 7, 10] -> 30, 120, 210, 300
    coords_2 = [np.array([0,0])] + [v[i] for i in range(6)] + [ring2[i] for i in [1, 4, 7, 10]]
    topologies.append(np.array(coords_2))
    
    # New Topology: "Line" or "Bar" shape
    # Axial coords for a 3x4 block? 12 cells. Remove 1.
    # Let's try a 2x5 block minus corners?
    # Or just a straight line of 11? No, too long.
    # Let's try a "Pentomino" like shape on hex grid.
    
    # Topology 3: Compact cluster from axial coords
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (0,2), (-2,2), (-2,0) -> This is similar to seed [0, 3, 6, 9] but different mapping?
    # Let's stick to the seed's best ones and maybe add one random-ish compact one.
    
    # Center all topologies
    for i in range(len(topologies)):
        topologies[i] -= np.mean(topologies[i], axis=0)
        
    best_side = float('inf')
    best_config = None
    
    # --- 2. Optimization for each Topology ---
    
    for t_idx, cluster in enumerate(topologies):
        if time.time() > deadline - 5.0:
            break
            
        # Variables: theta (cluster rotation), phi (outer hex rotation)
        # We assume the outer hexagon is centered at the centroid of the cluster (which is 0,0 after centering).
        # Actually, for a fixed cluster orientation, the optimal center of the enclosing hexagon
        # might not be exactly the centroid, but for symmetric clusters it is.
        # Let's optimize theta and phi.
        
        def objective(params):
            theta, phi = params
            c = np.cos(theta)
            s = np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            rot_cluster = cluster @ R.T
            
            # Outer hexagon normals
            # Normals at phi, phi+60, ...
            normals = []
            for i in range(6):
                a = phi + np.deg2rad(60 * i)
                normals.append([np.cos(a), np.sin(a)])
            normals = np.array(normals)
            
            # Support function of the cluster
            # max_i (n . p_i)
            proj_centers = rot_cluster @ normals.T
            max_proj_centers = np.max(proj_centers, axis=0)
            
            # Support function of the unit hexagon (fixed orientation 0 in local frame, but cluster is rotated)
            # Wait, the unit hexagons are part of the cluster. Their orientation is also rotated by theta.
            # So we need the support function of the Minkowski sum: Cluster + UnitHexagon(theta)
            # h_total(n) = max_{p in Cluster} (n . p) + h_hex_theta(n)
            # h_hex_theta(n) = max_{v in HexVerts} (n . (R v)) = max_{v in HexVerts} ((R^T n) . v)
            
            # Let's compute h_hex_theta for each normal
            # R^T normals
            rot_normals = normals @ R  # (6, 2)
            
            # proj_hex = hex_verts @ rot_normals.T -> (6, 6)
            proj_hex = hex_verts @ rot_normals.T
            max_proj_hex = np.max(proj_hex, axis=0)
            
            h_total = max_proj_centers + max_proj_hex
            max_h = np.max(h_total)
            
            side = (2.0 / np.sqrt(3.0)) * max_h
            return side
        
        # Initial guess
        x0 = [0.0, 0.0]
        
        # Try multiple random starts for phi and theta to escape local minima
        # The objective is periodic.
        
        best_local_side = float('inf')
        best_local_params = None
        
        num_starts = 10
        for k in range(num_starts):
            if time.time() > deadline - 5.0:
                break
                
            if k == 0:
                theta0 = 0.0
                phi0 = 0.0
            else:
                theta0 = np.random.uniform(0, np.pi/3)
                phi0 = np.random.uniform(0, np.pi/3)
                
            try:
                res = minimize(objective, [theta0, phi0], method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if res.success and res.fun < best_local_side:
                    best_local_side = res.fun
                    best_local_params = res.x
                    
            except Exception:
                continue
                
        if best_local_params is not None and best_local_side < best_side:
            best_side = best_local_side
            theta_opt, phi_opt = best_local_params
            
            # Reconstruct the configuration
            c = np.cos(theta_opt)
            s = np.sin(theta_opt)
            R = np.array([[c, -s], [s, c]])
            rot_cluster = cluster @ R.T
            
            # inner hexagons: centers are rot_cluster, angle is theta_opt (in degrees)
            inner_hex_data = np.column_stack([rot_cluster, np.full(11, np.rad2deg(theta_opt))])
            
            # outer hexagon: center (0,0), angle phi_opt (in degrees)
            outer_hex_data = np.array([0, 0, np.rad2deg(phi_opt)])
            
            best_config = (inner_hex_data, outer_hex_data, best_side)
            
    if best_config is None:
        return _fallback()

    return best_config[0], best_config[1], best_config[2]

def _fallback():
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8.0
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
