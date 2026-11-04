"""
Strategy: Direct Geometric Parameterization via Support Function Minimization.

Key Differences from Current Program:
1. Problem Framing: Abandons the rigid cluster assumption. Instead, directly parameterizes the 11 hexagons' positions and angles (33 DOF) and optimizes them to minimize the outer hexagon side length.
2. Optimization Method: Uses a global-to-local optimization strategy. A coarse random search over a broad parameter space is followed by a local Nelder-Mead refinement on the full 33-dimensional space.
3. Objective Function: Computes the minimum enclosing regular hexagon side length for a given set of inner hexagons using the support function. The objective is the maximum support over all vertices of the inner hexagons in the 6 directions defined by the outer hexagon's orientation.
4. Constraint Handling: Disjointness is enforced via a penalty term in the objective function. If any two inner hexagons overlap, a large penalty is added, guiding the optimizer away from invalid configurations.
5. Search Method: Employs a multi-start approach with diverse initial configurations. Each start uses a random perturbation of a base hexagonal lattice configuration. This helps escape local minima that might trap a single-start optimizer.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 150s kill

def _hex_vertices(center_x, center_y, angle_degrees, side=1.0):
    """Generate vertices of a regular hexagon."""
    a = np.radians(angle_degrees)
    ca, sa = np.cos(a), np.sin(a)
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    unit_verts_x = side * np.cos(base_angles)
    unit_verts_y = side * np.sin(base_angles)
    
    x = unit_verts_x
    y = unit_verts_y
    
    new_x = ca * x - sa * y + center_x
    new_y = sa * x + ca * y + center_y
    
    return np.column_stack((new_x, new_y))

def _min_outer_side_and_center(inner_hexagons, outer_angle_deg):
    """
    Given inner hexagons (list of (x, y, angle)), find the minimum side length
    and optimal center for the outer hexagon to contain them.
    
    Returns (side_length, center_x, center_y).
    """
    phi = np.radians(outer_angle_deg)
    ca, sa = np.cos(phi), np.sin(phi)
    sqrt3 = np.sqrt(3)
    
    # Collect all vertices of all inner hexagons
    all_verts = []
    for x, y, ang in inner_hexagons:
        verts = _hex_vertices(x, y, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # The outer hexagon is defined by 6 linear inequalities in its local frame.
    # For a point p, transformed to local frame q = R(-phi)(p - c):
    # The constraints are |q.x| <= S, |q.x + q.y/sqrt(3)| <= S, |q.x - q.y/sqrt(3)| <= S
    # We want to minimize S over c.
    
    def objective(c):
        cx, cy = c
        dx = all_verts[:, 0] - cx
        dy = all_verts[:, 1] - cy
        
        lx = ca * dx + sa * dy
        ly = -sa * dx + ca * dy
        
        v1 = np.abs(lx)
        v2 = np.abs(lx + ly / sqrt3)
        v3 = np.abs(lx - ly / sqrt3)
        
        return np.max(np.maximum(v1, np.maximum(v2, v3)))
    
    # Initial guess: centroid of all vertices
    c0 = np.mean(all_verts, axis=0)
    
    res = minimize(objective, c0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
    
    return res.fun, res.x[0], res.x[1]

def _check_overlap_fast(inner_hexagons, tol=1e-6):
    """
    Check if any two inner hexagons overlap.
    Returns a penalty value (0 if no overlap, large if overlap).
    """
    penalty = 0.0
    n = len(inner_hexagons)
    
    for i in range(n):
        x1, y1, a1 = inner_hexagons[i]
        for j in range(i+1, n):
            x2, y2, a2 = inner_hexagons[j]
            
            # Quick distance check
            dx = x1 - x2
            dy = y1 - y2
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Minimum distance between two non-overlapping unit hexagons is 2*sqrt(3)
            # If dist < 2*sqrt(3) - tol, they might overlap
            if dist < 2*np.sqrt(3) - tol:
                # Do a more precise check using SAT
                verts1 = _hex_vertices(x1, y1, a1)
                verts2 = _hex_vertices(x2, y2, a2)
                
                # SAT: check if any separating axis exists
                axes = []
                for v in verts1:
                    axes.append(v)
                for v in verts2:
                    axes.append(v)
                
                # For convex polygons, check edges of both polygons
                edges1 = [verts1[(i+1)%6] - verts1[i] for i in range(6)]
                edges2 = [verts2[(i+1)%6] - verts2[i] for i in range(6)]
                
                for edge in edges1 + edges2:
                    # Normal to the edge
                    n_edge = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n_edge)
                    if norm < 1e-10:
                        continue
                    n_edge = n_edge / norm
                    
                    # Project both polygons onto the normal
                    proj1 = np.dot(verts1, n_edge)
                    proj2 = np.dot(verts2, n_edge)
                    
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    # If there's a gap, they don't overlap
                    if max1 < min2 - tol or max2 < min1 - tol:
                        break
                else:
                    # No separating axis found -> overlap
                    # Penalty: how much they overlap (approximate)
                    penalty += (2*np.sqrt(3) - dist) * 100.0
    
    return penalty

def _objective(params, outer_angle_deg):
    """
    Full objective: minimize outer side length + overlap penalty.
    params: 33D vector (x1, y1, a1, x2, y2, a2, ..., x11, y11, a11)
    """
    inner_hexagons = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner_hexagons.append((x, y, a))
    
    # Check overlap
    overlap_penalty = _check_overlap_fast(inner_hexagons)
    
    # If overlap, return large penalty
    if overlap_penalty > 1e-6:
        return overlap_penalty + 10.0  # Large penalty to avoid overlap
    
    # Compute outer side length
    side, _, _ = _min_outer_side_and_center(inner_hexagons, outer_angle_deg)
    
    return side

def construct():
    global START_TIME, DEADLINE
    
    best_score = np.inf
    best_params = None
    
    # Base configuration: standard hexagonal lattice
    sqrt3 = np.sqrt(3)
    base_centers = [
        (0.0, 0.0),
        (2*sqrt3, 0.0),
        (2*sqrt3 * np.cos(np.pi/3), 2*sqrt3 * np.sin(np.pi/3)),
        (2*sqrt3 * np.cos(2*np.pi/3), 2*sqrt3 * np.sin(np.pi*2/3)),
        (2*sqrt3 * np.cos(np.pi), 2*sqrt3 * np.sin(np.pi)),
        (2*sqrt3 * np.cos(4*np.pi/3), 2*sqrt3 * np.sin(4*np.pi/3)),
        (2*sqrt3 * np.cos(5*np.pi/3), 2*sqrt3 * np.sin(5*np.pi/3)),
        (6.0 * np.cos(np.pi/6), 6.0 * np.sin(np.pi/6)),
        (6.0 * np.cos(5*np.pi/6), 6.0 * np.sin(5*np.pi/6)),
        (6.0 * np.cos(7*np.pi/6), 6.0 * np.sin(7*np.pi/6)),
        (6.0 * np.cos(11*np.pi/6), 6.0 * np.sin(11*np.pi/6)),
    ]
    
    # Try multiple random starts
    n_starts = 20
    rng = np.random.RandomState(42)
    
    for start in range(n_starts):
        if time.time() > DEADLINE - 30.0:
            break
        
        # Perturb the base configuration
        params = np.zeros(3*N)
        for i in range(N):
            x, y = base_centers[i]
            # Random perturbation
            dx = rng.uniform(-0.5, 0.5)
            dy = rng.uniform(-0.5, 0.5)
            da = rng.uniform(-15, 15)
            params[3*i] = x + dx
            params[3*i+1] = y + dy
            params[3*i+2] = da
        
        # Try a few outer angles
        for outer_angle in [0, 15, 30, 45]:
            if time.time() > DEADLINE - 30.0:
                break
            
            try:
                # Local refinement
                res = minimize(_objective, params, args=(outer_angle,), method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                
                if res.fun < best_score and res.fun < 10.0:  # Valid solution
                    best_score = res.fun
                    best_params = (res.x, outer_angle)
                    
            except Exception:
                pass
    
    # If no valid solution found, fallback to base configuration
    if best_params is None:
        # Use base configuration
        inner_hexagons = [(x, y, 0.0) for x, y in base_centers]
        side, cx, cy = _min_outer_side_and_center(inner_hexagons, 0.0)
        
        return inner_hexagons, (cx, cy), float(side), 0.0
    
    # Extract best solution
    best_params_vec, best_outer_angle = best_params
    
    inner_hexagons = []
    for i in range(N):
        x = best_params_vec[3*i]
        y = best_params_vec[3*i+1]
        a = best_params_vec[3*i+2]
        inner_hexagons.append((float(x), float(y), float(a)))
    
    # Recompute the outer hexagon for the final configuration
    side, cx, cy = _min_outer_side_and_center(inner_hexagons, best_outer_angle)
    
    return inner_hexagons, (float(cx), float(cy)), float(side), float(best_outer_angle)

if __name__ == "__main__":
    construct()
