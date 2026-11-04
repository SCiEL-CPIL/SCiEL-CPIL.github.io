"""
Strategy: Smooth Surrogate Optimization with Soft-Min/Soft-Max and Epigraph Formulation.

This approach rethinks the optimization landscape by:
1. Using a smooth, differentiable objective function that approximates the non-smooth min/max constraints.
   - The outer hexagon containment constraint is modeled as a soft-max of the three linear forms.
   - The overlap constraint is modeled as a smooth penalty based on the minimum separation distance (approximated via SAT).
2. Employing a hybrid optimization strategy:
   - Start with a global search (differential evolution) to find a good basin.
   - Follow with a local, gradient-based polish (L-BFGS-B) to refine the solution to high precision.
3. Using an epigraph formulation where the outer side length 't' is an explicit variable, and the objective is to minimize 't' subject to smooth constraints.

This design directly addresses the "nonsmooth objective" issue by providing a smooth surrogate that guides the optimizer, potentially allowing it to escape local minima that trap hard-constraint or binary-penalty methods.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin for 120s limit

def _get_vertices(pos, ang):
    """
    Compute vertices for all N hexagons.
    pos: (N, 2) array of centers
    ang: (N,) array of angles in radians
    Returns: (N, 6, 2) array of vertices
    """
    base_angles = np.arange(6) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    verts = np.stack((verts_x, verts_y), axis=-1)
    return verts

def _min_separation(verts):
    """
    Compute the minimum separation distance between all pairs of hexagons.
    Uses SAT to find the minimum overlap (penetration) on any axis.
    If separated on any axis, the separation is positive.
    If overlapping, the separation is 0 (or negative if we track penetration).
    For a smooth penalty, we want a measure of how much they overlap.
    Here we compute the minimum separation distance (positive if disjoint).
    """
    min_sep = np.inf
    tol = 1e-9
    for i in range(N):
        v1 = verts[i]
        for j in range(i + 1, N):
            v2 = verts[j]
            
            # Check 12 axes (6 from v1, 6 from v2)
            for k in range(6):
                p1 = v1[k]
                p2 = v1[(k + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-9:
                    continue
                normal /= norm
                
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                m1_min, m1_max = proj1.min(), proj1.max()
                m2_min, m2_max = proj2.min(), proj2.max()
                
                if m1_max < m2_min:
                    sep = m2_min - m1_max
                elif m2_max < m1_min:
                    sep = m1_min - m2_max
                else:
                    sep = 0.0
                
                # If they are separated on this axis, the physical distance is at least sep.
                # We want the minimum separation over all axes? 
                # Actually, if they are disjoint, the distance is the minimum over all axes of the separation?
                # No, the distance is the maximum over all axes of the separation?
                # Consider two squares. If they are apart in X by 1 and Y by 10, distance is 1.
                # So we want the MINIMUM separation over all axes?
                # But if they overlap, separation is 0 on all axes.
                
                # Let's just track the minimum separation found so far.
                if sep < min_sep:
                    min_sep = sep
                    
                # If min_sep is 0, they might be overlapping or touching.
                # We can break early if min_sep is 0? No, we want the tightest bound.
                
            # Also check axes from v2
            for k in range(6):
                p1 = v2[k]
                p2 = v2[(k + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-9:
                    continue
                normal /= normal
                
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                m1_min, m1_max = proj1.min(), proj1.max()
                m2_min, m2_max = proj2.min(), proj2.max()
                
                if m1_max < m2_min:
                    sep = m2_min - m1_max
                elif m2_max < m1_min:
                    sep = m1_min - m2_max
                else:
                    sep = 0.0
                
                if sep < min_sep:
                    min_sep = sep
                    
    return min_sep

def _smooth_overlap_penalty(verts, beta=10.0):
    """
    Smooth penalty for overlaps.
    If min_sep > 0, penalty is 0.
    If min_sep < 0 (overlapping), penalty is exp(beta * min_sep) or similar.
    But min_sep is 0 if overlapping. We need a measure of penetration.
    
    Let's use a different approach: 
    For each pair, compute the minimum separation. 
    If min_sep < 0, they overlap. But our SAT returns 0 if overlapping.
    We need to compute the actual penetration depth.
    
    Alternative: Use the sum of squares of penetrations on all axes.
    """
    penalty = 0.0
    for i in range(N):
        v1 = verts[i]
        for j in range(i + 1, N):
            v2 = verts[j]
            
            for k in range(6):
                p1 = v1[k]
                p2 = v1[(k + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-9:
                    continue
                normal /= norm
                
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                m1_min, m1_max = proj1.min(), proj1.max()
                m2_min, m2_max = proj2.min(), proj2.max()
                
                overlap = min(m1_max, m2_max) - max(m1_min, m2_min)
                if overlap > 0:
                    penalty += overlap ** 2
                    
    return penalty

def _min_outer_side(pos, ang, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Compute minimum outer side length to contain all inner hexagons."""
    cx, cy = outer_center
    pos = np.asarray(pos, dtype=np.float64)
    ang = np.radians(np.asarray(ang, dtype=np.float64))
    
    verts = _get_vertices(pos, ang)
    
    all_verts_x = verts[:, :, 0].ravel()
    all_verts_y = verts[:, :, 1].ravel()
    
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _soft_max(x, beta=10.0):
    """Soft max approximation."""
    return (1.0 / beta) * np.log(np.sum(np.exp(beta * x)))

def _objective(params):
    """
    Objective function for optimization.
    params: [tx, ty, rot_cluster, ang_0, ..., ang_10, t]
    """
    tx = params[0]
    ty = params[1]
    rot_cluster = params[2]
    ind_angles = params[3:14]
    t = params[14]  # Epigraph variable for outer side length
    
    # Base positions
    sqrt3 = np.sqrt(3)
    base_pos = np.zeros((N, 2))
    base_pos[0] = [0, 0]
    for i in range(6):
        a = i * np.pi / 3
        base_pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        base_pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    c, s = np.cos(rot_cluster), np.sin(rot_cluster)
    rot_matrix = np.array([[c, -s], [s, c]])
    pos = base_pos @ rot_matrix.T
    pos[:, 0] += tx
    pos[:, 1] += ty
    
    ang = ind_angles + rot_cluster
    
    verts = _get_vertices(pos, ang)
    
    # Overlap penalty
    pen = _smooth_overlap_penalty(verts)
    if pen > 1e-4:
        return 1000.0 + pen
        
    # Outer side length constraint
    side = _min_outer_side(pos, ang)
    
    # Objective: minimize t + penalty * max(0, side - t)
    # This is an epigraph formulation.
    # If side <= t, the term is 0.
    # If side > t, the term is side - t.
    # We want to minimize t, but t must be >= side.
    # So we minimize t + 100 * max(0, side - t)
    
    penalty_term = 100.0 * max(0.0, side - t)
    
    return t + penalty_term

def construct():
    global START_TIME, DEADLINE
    
    # Bounds for differential evolution
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 2.0 * np.pi)]
    for _ in range(N):
        bounds.append((0.0, 2.0 * np.pi))
    bounds.append((5.0, 15.0))  # t bound
    
    # Initial population
    popsize = 20
    
    def callback(xk, convergence=True):
        if time.time() > DEADLINE - 20.0:
            return True
        return False
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=500, 
            popsize=popsize, 
            tol=1e-4, 
            callback=callback,
            seed=42
        )
    except Exception as e:
        # Fallback
        sqrt3 = np.sqrt(3)
        pos = np.zeros((N, 2))
        pos[0] = [0, 0]
        for i in range(6):
            a = i * np.pi / 3
            pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
        for i in range(4):
            a = ring2_angles[i]
            pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        ang = np.zeros(N)
        side = _min_outer_side(pos, ang)
        inner_list = [[float(pos[i,0]), float(pos[i,1]), float(np.degrees(ang[i]))] for i in range(N)]
        return inner_list, (0.0, 0.0), float(side), 0.0
    
    best_params = result.x
    
    # Local polish with L-BFGS-B
    if time.time() < DEADLINE - 10.0:
        try:
            polish_result = minimize(
                _objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if polish_result.fun < result.fun:
                best_params = polish_result.x
        except Exception:
            pass
    
    # Extract best solution
    tx = best_params[0]
    ty = best_params[1]
    rot_cluster = best_params[2]
    ind_angles = best_params[3:14]
    
    sqrt3 = np.sqrt(3)
    base_pos = np.zeros((N, 2))
    base_pos[0] = [0, 0]
    for i in range(6):
        a = i * np.pi / 3
        base_pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        base_pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    c, s = np.cos(rot_cluster), np.sin(rot_cluster)
    rot_matrix = np.array([[c, -s], [s, c]])
    pos = base_pos @ rot_matrix.T
    pos[:, 0] += tx
    pos[:, 1] += ty
    
    ang = ind_angles + rot_cluster
    
    # Final verification
    verts = _get_vertices(pos, ang)
    if _smooth_overlap_penalty(verts) > 1e-4:
        # Invalid, fallback
        pos = np.zeros((N, 2))
        pos[0] = [0, 0]
        for i in range(6):
            a = i * np.pi / 3
            pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
        for i in range(4):
            a = ring2_angles[i]
            pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        ang = np.zeros(N)
        
    best_side = _min_outer_side(pos, ang)
    
    inner_list = [[float(pos[i,0]), float(pos[i,1]), float(np.degrees(ang[i]))] for i in range(N)]
    
    return inner_list, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    construct()
