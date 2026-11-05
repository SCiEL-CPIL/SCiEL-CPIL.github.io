import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # ---- Fixed inner layout: "flower" pattern ----
    eps = 1e-10
    centers = np.array([[0.0, 0.0]])

    # Ring 1: 6 hexagons
    r1 = np.sqrt(3) + eps
    for k in range(6):
        a = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(a), r1 * np.sin(a)]])

    # Ring 2: 4 hexagons
    r2 = 3.0 + 2 * eps
    for k in range(4):
        a = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(a), r2 * np.sin(a)]])

    n_hex = len(centers)
    assert n_hex == 11

    # Base vertices of a unit hexagon, pointy-right (angle 0)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    base_vertices = np.array(
        [[np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)]
    )

    def get_all_vertices(phis):
        """
        Compute vertices for all inner hexagons.
        phis: (n_hex,) array of rotation angles in radians.
        Returns: (n_hex * 6, 2) array of vertices.
        """
        out = np.empty((n_hex * 6, 2))
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            R = np.array([[c, -s], [s, c]])
            out[i * 6:(i + 1) * 6] = base_vertices @ R.T + centers[i]
        return out

    def compute_outer_side_for_angle(theta_out, verts):
        """
        Compute side length of outer hexagon with rotation theta_out 
        containing all vertices in 'verts'.
        Outer hexagon is centered at origin.
        """
        # Normals for the 6 faces of the outer hexagon
        # Face normals are at 30, 90, 150, 210, 270, 330 degrees + theta_out
        # Wait, if vertices are at 0, 60..., faces are perpendicular to bisectors.
        # Bisectors are at 30, 90...
        # So normal directions are 30 + k*60 + theta_out
        angles = np.deg2rad(30) + np.arange(6) * np.deg2rad(60) + theta_out
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Project all vertices onto normals
        # verts: (M, 2), normals: (2, 6) -> dots: (M, 6)
        dots = verts @ normals.T
        max_proj = np.max(dots, axis=0) # (6,)
        
        # The side length S is related to the max projection.
        # For a regular hexagon of side S, the apothem is S * sqrt(3) / 2.
        # The max projection onto a face normal is the apothem of the enclosing hexagon.
        # So apothem = max(max_proj).
        # S = 2 * apothem / sqrt(3)
        apothem = float(np.max(max_proj))
        return 2.0 * apothem / np.sqrt(3)

    def objective_inner_only(phis):
        """
        Objective function for optimizing inner rotations only.
        Finds the best outer rotation theta_out for the given phis.
        """
        if time.time() > deadline:
            return float('inf')
            
        verts = get_all_vertices(phis)
        
        # Find best theta_out
        # We know theta_out is periodic with pi/3 (60 deg).
        # We can search in [0, pi/3)
        # Coarse grid search first to be safe and fast
        best_S = float('inf')
        best_theta = 0.0
        
        # Coarse search: 12 steps
        n_coarse = 12
        for i in range(n_coarse):
            theta = i * (np.pi / 3) / n_coarse
            S = compute_outer_side_for_angle(theta, verts)
            if S < best_S:
                best_S = S
                best_theta = theta
        
        # Fine search around best_theta using minimize_scalar
        # Boundaries: ensure we stay within the period
        lower = best_theta - (np.pi / 3) / n_coarse
        upper = best_theta + (np.pi / 3) / n_coarse
        
        # Handle wrap-around if needed, but for this symmetric case it's usually fine
        # To be robust, we can just minimize over the whole period if coarse didn't find a clear min,
        # but bounded around the coarse best is faster.
        
        def S_of_theta(theta):
            return compute_outer_side_for_angle(theta, verts)
            
        try:
            res = minimize_scalar(S_of_theta, bounds=(lower, upper), method='bounded', 
                                  options={'xatol': 1e-9, 'disp': False})
            if res.success:
                if res.fun < best_S:
                    best_S = res.fun
                    best_theta = res.x
        except Exception:
            pass
            
        return best_S

    # Initial guess: All inner hexagons rotated 30 degrees (flat-top)
    # This matches the symmetry of the flower pattern
    x0 = np.full(n_hex, np.deg2rad(30.0))
    
    best_score = objective_inner_only(x0)
    best_phis = x0.copy()
    
    # Multi-start optimization
    # The objective is non-smooth due to the max and the inner minimization.
    # Nelder-Mead is generally robust for this.
    
    starts = [
        np.full(n_hex, np.deg2rad(30.0)),
        np.zeros(n_hex),
        np.full(n_hex, np.deg2rad(15.0)),
        np.full(n_hex, np.deg2rad(45.0)),
        np.random.uniform(0, np.pi/6, n_hex),
        np.random.uniform(0, np.pi/6, n_hex),
    ]
    
    for i, x_start in enumerate(starts):
        if time.time() > deadline - 10:
            break
            
        try:
            # Use Nelder-Mead
            res = minimize(objective_inner_only, x_start, method='Nelder-Mead',
                           options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 2000, 
                                    'adaptive': True, 'disp': False})
            if res.fun < best_score:
                best_score = res.fun
                best_phis = res.x.copy()
        except Exception:
            pass
            
    # Polish with Powell if time permits
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective_inner_only, best_phis, method='Powell',
                            options={'maxiter': 1000, 'ftol': 1e-15, 'xtol': 1e-15, 'disp': False})
            if res2.fun < best_score:
                best_score = res2.fun
                best_phis = res2.x.copy()
        except Exception:
            pass

    # Final fine-tuning of inner angles with small steps
    if time.time() < deadline - 2:
        current_best = best_score
        current_phis = best_phis
        step_sizes = [1e-4, 1e-5, 1e-6]
        for step in step_sizes:
            if time.time() > deadline - 1:
                break
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                for j in range(n_hex):
                    if time.time() > deadline - 1:
                        break
                    for delta in [step, -step]:
                        test_phis = current_phis.copy()
                        test_phis[j] += delta
                        # Ensure angles are in a reasonable range, though objective handles any angle
                        try:
                            test_score = objective_inner_only(test_phis)
                            if test_score < current_best:
                                current_best = test_score
                                current_phis = test_phis
                                improved = True
                        except Exception:
                            pass
        best_score = current_best
        best_phis = current_phis

    # Find the final best outer angle
    final_verts = get_all_vertices(best_phis)
    best_theta_out = 0.0
    min_S = float('inf')
    
    # Coarse
    for i in range(12):
        theta = i * (np.pi / 3) / 12
        S = compute_outer_side_for_angle(theta, final_verts)
        if S < min_S:
            min_S = S
            best_theta_out = theta
            
    # Fine
    lower = best_theta_out - (np.pi / 3) / 12
    upper = best_theta_out + (np.pi / 3) / 12
    def S_of_theta(theta):
        return compute_outer_side_for_angle(theta, final_verts)
    try:
        res_final = minimize_scalar(S_of_theta, bounds=(lower, upper), method='bounded', 
                                    options={'xatol': 1e-10, 'disp': False})
        if res_final.success:
            if res_final.fun < min_S:
                min_S = res_final.fun
                best_theta_out = res_final.x
    except Exception:
        pass
        
    # Final check: is this better than the previous best?
    if min_S < best_score:
        best_score = min_S

    # Construct return values
    inner = []
    for i in range(n_hex):
        inner.append([float(centers[i][0]), float(centers[i][1]),
                      float(np.rad2deg(best_phis[i]))])

    # Add tiny epsilon to side length to ensure validity if on boundary
    side = best_score + 1e-9
    return inner, (0.0, 0.0), float(side), float(np.rad2deg(best_theta_out))

if __name__ == "__main__":
    construct()
