import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0  # seconds, with safety margin

    # Hexagon vertices relative to center (unit side, angle 0)
    verts_rel = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Edge normals for hexagons (6 unique directions at 30, 90, 150, 210, 270, 330 degrees)
    normals = np.array([
        [np.cos(np.radians(30)), np.sin(np.radians(30))],
        [np.cos(np.radians(90)), np.sin(np.radians(90))],
        [np.cos(np.radians(150)), np.sin(np.radians(150))],
        [np.cos(np.radians(210)), np.sin(np.radians(210))],
        [np.cos(np.radians(270)), np.sin(np.radians(270))],
        [np.cos(np.radians(330)), np.sin(np.radians(330))]
    ])
    
    def get_positions(x):
        return x.reshape(11, 2)
    
    def check_separation(coords, eps=1e-6):
        """Check if all pairs of hexagons are separated by > eps."""
        for i in range(11):
            for j in range(i+1, 11):
                d = coords[i] - coords[j]
                separated = False
                for n in normals:
                    proj = np.dot(d, n)
                    if abs(proj) > np.sqrt(3) + eps:
                        separated = True
                        break
                if not separated:
                    return False
        return True

    def compute_min_side(coords):
        """Compute the minimum side length of outer hexagon containing all inner hexagons."""
        max_abs_proj = 0
        for c in coords:
            verts = c + verts_rel
            proj = verts @ normals.T
            max_abs_proj = max(max_abs_proj, np.max(np.abs(proj)))
        S = max_abs_proj * 2 / np.sqrt(3)
        return S

    def make_initial(scale=1.01):
        x0 = np.zeros(22)
        x0[0:2] = [0, 0]
        r1 = np.sqrt(3) * scale
        angles1 = [0, 60, 120, 180, 240, 300]
        for k, ang in enumerate(angles1):
            idx = (k+1)*2
            x0[idx] = r1 * np.cos(np.radians(ang))
            x0[idx+1] = r1 * np.sin(np.radians(ang))
        r2 = 2.0 * np.sqrt(3) * scale
        angles2 = [0, 60, 120, 180]
        for k, ang in enumerate(angles2):
            idx = (k+7)*2
            x0[idx] = r2 * np.cos(np.radians(ang))
            x0[idx+1] = r2 * np.sin(np.radians(ang))
        return x0

    x0 = make_initial(1.02)
    best_coords = get_positions(x0)
    best_S = compute_min_side(best_coords)
    
    # Use a smooth surrogate objective with soft-max for the outer hexagon constraint
    # and a smooth penalty for separation
    def objective_smooth(x, beta=50.0):
        coords = get_positions(x)
        
        # Compute projections for outer hexagon constraint
        all_verts = np.zeros((11*6, 2))
        for i in range(11):
            all_verts[i*6:(i+1)*6] = coords[i] + verts_rel
        
        # Compute |proj| for all vertices and normals
        projs = all_verts @ normals.T  # shape (66, 6)
        abs_projs = np.abs(projs)  # shape (66, 6)
        
        # Soft-max over all values to approximate max_abs_proj
        # Use log-sum-exp for numerical stability
        max_val = np.max(abs_projs)
        soft_max = max_val + np.log(np.sum(np.exp(abs_projs - max_val))) / (66*6)
        
        S_approx = soft_max * 2 / np.sqrt(3)
        
        # Smooth penalty for separation violations
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                d = coords[i] - coords[j]
                for n in normals:
                    proj = np.dot(d, n)
                    sep = abs(proj) - (np.sqrt(3) + 1e-6)
                    if sep < 0:
                        # Use smooth approximation: -sep * exp(-beta * sep) approximates max(0, -sep)
                        penalty += (-sep) * np.exp(-beta * sep) * 1000
        
        return S_approx + penalty

    np.random.seed(42)
    max_restarts = 8
    
    for restart in range(max_restarts):
        if time.time() > deadline - 90:
            break
        
        if restart == 0:
            x_curr = x0.copy()
        else:
            x_curr = best_coords.flatten() + np.random.randn(22) * 0.3

        # Use Nelder-Mead with tighter tolerances for polish
        res = minimize(objective_smooth, x_curr, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                'adaptive': True})
        
        coords = get_positions(res.x)
        if check_separation(coords):
            S = compute_min_side(coords)
            if S < best_S:
                best_S = S
                best_coords = coords.copy()
        
        # Final polish pass with tighter tolerances
        if time.time() < deadline - 60:
            res2 = minimize(objective_smooth, res.x, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14,
                                    'adaptive': True})
            coords2 = get_positions(res2.x)
            if check_separation(coords2):
                S2 = compute_min_side(coords2)
                if S2 < best_S:
                    best_S = S2
                    best_coords = coords2.copy()

    # Final check
    if not check_separation(best_coords):
        x_safe = make_initial(2.0)
        best_coords = get_positions(x_safe)
        best_S = compute_min_side(best_coords)

    inner = np.zeros((11, 3))
    inner[:, :2] = best_coords
    inner[:, 2] = 0
    outer = np.array([0, 0, 0])
    return inner, outer, float(best_S)
