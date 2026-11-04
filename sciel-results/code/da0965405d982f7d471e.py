"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Standard hexagonal lattice positions for 11 hexagons
    # Center at (0,0), 6 neighbors at distance sqrt(3), 4 more at distance 2*sqrt(3)
    lattice_positions = [(0.0, 0.0)]
    for k in range(6):
        theta = k * np.pi / 3
        cx = np.sqrt(3) * np.cos(theta)
        cy = np.sqrt(3) * np.sin(theta)
        lattice_positions.append((cx, cy))
    for k in range(4):
        theta = k * np.pi / 3
        cx = 2 * np.sqrt(3) * np.cos(theta)
        cy = 2 * np.sqrt(3) * np.sin(theta)
        lattice_positions.append((cx, cy))
    
    lattice_positions = np.array(lattice_positions)
    
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    def compute_S_for_container(vertices_list, ocx, ocy, oangle):
        """Compute the minimum side length S of the outer hexagon."""
        cos_a = np.cos(oangle)
        sin_a = np.sin(oangle)
        S = 0.0
        apothem = np.sqrt(3) / 2
        for verts in vertices_list:
            for v in verts:
                lx = v[0] - ocx
                ly = v[1] - ocy
                rx = lx * cos_a + ly * sin_a
                ry = -lx * sin_a + ly * cos_a
                for k in range(6):
                    theta_n = np.pi / 6 + k * np.pi / 3
                    proj = rx * np.cos(theta_n) + ry * np.sin(theta_n)
                    S = max(S, abs(proj))
        return S / apothem
    
    def smooth_overlap_penalty(v1, v2, softness=0.1):
        """Compute smooth overlap penalty using vertex-to-edge distances."""
        # For each edge of v1, compute distance from v2 vertices to the edge line
        # Use a smooth approximation of max(0, penetration)
        penalty = 0.0
        n1, n2 = len(v1), len(v2)
        
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i + 1) % n1]
            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-10:
                continue
            normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
            
            # Project v2 vertices onto normal
            for j in range(n2):
                d = np.dot(v2[j] - p1, normal)
                # Smooth penalty: if d < 0, we have penetration
                if d < 0:
                    # Use smooth approximation: -d + softness*log(1 + exp(-d/softness))
                    penalty += max(0, -d) ** 2 / softness
        
        # Also check v2 edges against v1 vertices
        for i in range(n2):
            p1 = v2[i]
            p2 = v2[(i + 1) % n2]
            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 1e-10:
                continue
            normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
            
            for j in range(n1):
                d = np.dot(v1[j] - p1, normal)
                if d < 0:
                    penalty += max(0, -d) ** 2 / softness
        
        return penalty
    
    def objective(params):
        """
        Parameters: 2*N displacements + N angles
        First 2*N: dx, dy for each hexagon (relative to lattice position)
        Last N: rotation angles for each hexagon
        """
        n = N
        # Extract parameters
        dx = params[0:2*n:2]
        dy = params[1:2*n:2]
        angles = params[2*n:]
        
        # Compute actual positions
        cx = lattice_positions[:, 0] + dx
        cy = lattice_positions[:, 1] + dy
        
        # Compute vertices for each hexagon
        vertices_list = []
        for i in range(n):
            verts = hex_vertices(cx[i], cy[i], angles[i])
            vertices_list.append(verts)
        
        # Compute outer hexagon size (assume centered at origin, rotation 0 for now)
        S = compute_S_for_container(vertices_list, 0.0, 0.0, 0.0)
        
        # Compute smooth overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                # Quick distance check to avoid unnecessary computation
                dist = np.linalg.norm([cx[i] - cx[j], cy[i] - cy[j]])
                if dist < 2.5:  # Only compute penalty if potentially close
                    penalty += smooth_overlap_penalty(vertices_list[i], vertices_list[j], softness=0.05)
        
        return S + 10.0 * penalty
    
    def find_best_container(vertices_list):
        """Find the minimal enclosing hexagon for a set of vertices."""
        # Get all vertices
        all_verts = np.vstack(vertices_list)
        
        # Initial guess: centroid, 0 angle
        cx, cy = all_verts.mean(axis=0)
        angle = 0.0
        
        def cont_obj(params_c):
            ocx, ocy, oangle = params_c
            return compute_S_for_container(vertices_list, ocx, ocy, oangle)
        
        # Optimize container parameters
        res = minimize(cont_obj, [cx, cy, angle], method='L-BFGS-B', 
                      options={'maxiter': 200, 'ftol': 1e-8})
        return res.x, res.fun
    
    # Initial parameters: zero displacements, zero angles
    init_params = np.zeros(3 * N)
    
    # Multi-start optimization
    np.random.seed(42)
    best_params = init_params
    best_score = objective(init_params)
    
    for start in range(8):
        if start == 0:
            p = init_params.copy()
        else:
            p = init_params + np.random.normal(0, 0.15, size=init_params.size)
        
        res = minimize(objective, p, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    
    # Extract final positions
    n = N
    dx = best_params[0:2*n:2]
    dy = best_params[1:2*n:2]
    angles = best_params[2*n:]
    
    cx = lattice_positions[:, 0] + dx
    cy = lattice_positions[:, 1] + dy
    
    inner = []
    vertices_list = []
    for i in range(n):
        inner.append([cx[i], cy[i], np.degrees(angles[i])])
        vertices_list.append(hex_vertices(cx[i], cy[i], angles[i]))
    
    # Refine the outer container
    cont_params, S = find_best_container(vertices_list)
    ocx, ocy, oangle = cont_params
    oangle_deg = np.degrees(oangle)
    
    return inner, (ocx, ocy), S, oangle_deg
# EVOLVE-BLOCK-END
