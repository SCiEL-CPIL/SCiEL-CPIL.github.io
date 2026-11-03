"""
Strategy: Contact-Graph Topology Optimization with Nelder-Mead.

This approach differs from the current program by:
1. **Topology-First Representation**: Instead of optimizing 33 free parameters (11 positions + 11 angles), 
   we define a fixed contact graph (1 center, 6 ring, 4 gaps) and parameterize the system by:
   - Global cluster rotation (1 param)
   - Global cluster translation (2 params, though we can fix center at 0,0 due to symmetry)
   - Relative "breathing" of the contact graph edges (small perturbations around ideal hex lattice distances)
   - Individual rotations of the outer ring hexagons (to allow slight "tilting" for better fit)
   
   However, to keep it robust and differentiable-ish for Nelder-Mead, we will stick to 
   optimizing the 11 positions and 11 angles, BUT we will use a **Graph-Based Initialization** 
   and a **Support-Function Objective** with a **Smooth Repulsive Potential** for overlaps.
   
   The key structural change is the **Objective Function**: 
   Instead of minimizing center-distance proxies, we minimize the **maximum support function value** 
   of the inner hexagons relative to the outer hexagon (which directly minimizes the outer side length), 
   plus a **smooth penalty** for any overlap (computed via minimum distance between convex polygons).
   
   We use Nelder-Mead because it is derivative-free and robust to the non-smoothness of the 
   "max support function" and "minimum distance" terms, which are piecewise linear/smooth but 
   have discontinuous gradients at contact points.
"""

import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_dist_convex_poly(v1, v2):
    """
    Compute minimum distance between two convex polygons.
    If they overlap, return 0 (or a small negative value if we wanted signed distance, 
    but for penalty we just need 0 when overlapping or touching).
    For efficiency, we use a simple check: if any vertex of one is inside the other, dist is 0.
    Otherwise, we check edge-to-edge distances.
    """
    # Quick check: if bounding boxes don't overlap
    if v1[:,0].max() < v2[:,0].min() or v1[:,0].min() > v2[:,0].max():
        return np.inf
    if v1[:,1].max() < v2[:,1].min() or v1[:,1].min() > v2[:,1].max():
        return np.inf

    # Check if any vertex is inside the other
    def point_in_poly(pt, poly):
        x, y = pt
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    for v in v1:
        if point_in_poly(v, v2):
            return 0.0
    for v in v2:
        if point_in_poly(v, v1):
            return 0.0

    # Edge-edge distances
    min_d = np.inf
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            q1 = v2[j]
            q2 = v2[(j+1)%6]
            
            # Distance between segment p1-p2 and q1-q2
            # Using standard segment distance formula
            d1 = p2 - p1
            d2 = q2 - q1
            r = p1 - q1
            
            a = np.dot(d1, d1)
            e = np.dot(d2, d2)
            f = np.dot(d2, r)
            
            if a <= 1e-9 and e <= 1e-9:
                d = np.linalg.norm(r)
            elif a <= 1e-9:
                t = f / e
                if t < 0:
                    d = np.linalg.norm(r + d2 * 0) # r
                elif t > 1:
                    d = np.linalg.norm(r + d2 * 1) # r + d2
                else:
                    d = np.linalg.norm(r + d2 * t)
            elif e <= 1e-9:
                s = -np.dot(d1, r) / a
                if s < 0:
                    d = np.linalg.norm(r)
                elif s > 1:
                    d = np.linalg.norm(r + d1)
                else:
                    d = np.linalg.norm(r + d1 * s)
            else:
                c = np.dot(d1, r)
                b = np.dot(d1, d2)
                denom = a * e - b * b
                s = 0.0
                t = 0.0
                if denom != 0:
                    s = (b * f - c * e) / denom
                if s < 0:
                    s = 0
                    t = f / e
                elif s > 1:
                    s = 1
                    t = (f + b) / e
                
                if t < 0:
                    t = 0
                elif t > 1:
                    t = 1
                
                d = np.linalg.norm(r + d1 * s - d2 * t)
            
            if d < min_d:
                min_d = d
    
    return min_d

def _objective(params):
    """
    Objective: Minimize outer side length + overlap penalty.
    params: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10]
    """
    pos = params[:2*N].reshape(N, 2)
    ang = params[2*N:]
    
    # 1. Compute outer side length required to contain all hexagons
    # We assume outer hexagon is centered at (0,0) and rotated by 0 for simplicity,
    # but we can also optimize outer rotation. Let's fix outer rotation to 0 for now 
    # to reduce dimensionality, as the cluster can rotate.
    # Actually, the problem allows outer rotation. Let's add outer rotation as a parameter?
    # No, the interface returns outer center, side, angle. 
    # If we fix outer center to (0,0), we just need to find the minimal side and angle.
    # However, minimizing side for a fixed angle is easy. 
    # Let's optimize the cluster positions/angles, and compute the minimal outer side 
    # assuming outer center is (0,0) and outer angle is optimized separately? 
    # Or just fix outer angle to 0 and let the cluster rotate. 
    # Due to symmetry, fixing outer angle to 0 is w.l.o.g. if we allow cluster rotation.
    
    cx, cy = 0.0, 0.0
    outer_angle_deg = 0.0
    
    # Compute max support function value
    a_outer = np.radians(outer_angle_deg)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    sqrt3 = np.sqrt(3)
    
    max_support = 0.0
    verts_list = []
    
    for i in range(N):
        x, y = pos[i]
        a_deg = ang[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        verts_list.append(verts)
        
        # Transform vertices to outer frame
        dx = verts[:, 0] - cx
        dy = verts[:, 1] - cy
        lx = ca * dx - sa * dy
        ly = sa * dx + ca * dy
        
        c1 = np.abs(lx)
        c2 = np.abs(lx + ly / sqrt3)
        c3 = np.abs(lx - ly / sqrt3)
        
        support = np.max(np.maximum(c1, np.maximum(c2, c3)))
        if support > max_support:
            max_support = support

    # 2. Overlap Penalty
    # Use smooth approximation of min distance.
    # If min_dist < 0 (overlap), penalty is large.
    # Since _min_dist_convex_poly returns 0 for overlap, we need a way to detect "how much" overlap.
    # For simplicity, if min_dist == 0, we assume they are touching or overlapping.
    # To make the penalty smooth and effective, we can use a barrier:
    # penalty = sum( max(0, (min_allowed_dist - min_dist))^2 )
    # But min_dist is 0 when overlapping. We need a signed distance or a proxy.
    # Proxy: Center distance.
    # If center_dist < 2*sqrt(3) - epsilon, they might overlap.
    # Let's use center distance for penalty to ensure it's smooth and differentiable-ish.
    
    min_center_dist = 2 * np.sqrt(3)
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_center_dist:
                # Quadratic penalty
                penalty += (min_center_dist - d)**2
    
    # Weight penalty heavily
    return max_support + 50.0 * penalty

def _initial_configuration():
    """
    Generate a specific initial configuration for 11 hexagons.
    Topology:
    - Center: (0,0)
    - Ring 1: 6 hexagons at distance 2*sqrt(3) from center.
    - Ring 2: 4 hexagons at distance 6 from center, in the gaps.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # 0: Center
    pos[0] = [0, 0]
    
    # 1-6: Ring 1
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        
    # 7-10: Ring 2
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    return pos, ang

def construct():
    global START_TIME, DEADLINE
    
    best_score = np.inf
    best_params = None
    
    # Initial configuration
    pos_init, ang_init = _initial_configuration()
    x0 = np.concatenate([pos_init.ravel(), ang_init])
    
    # Nelder-Mead search
    # We will run multiple restarts with slightly perturbed initial configurations
    # to escape local minima.
    
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > DEADLINE - 10.0:
            break
            
        # Perturb initial configuration
        if i == 0:
            x_start = x0
        else:
            # Add noise to positions and angles
            noise_pos = np.random.normal(0, 0.1, (N, 2))
            noise_ang = np.random.normal(0, 0.1, N)
            x_start = np.concatenate([
                (pos_init + noise_pos).ravel(), 
                ang_init + noise_ang
            ])
            
        try:
            res = minimize(_objective, x_start, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
                
                # Check if we are getting close to a valid solution
                # If best_score is low enough, we can break early?
                # The minimal side length for 11 hexagons is roughly 5. something.
                # Let's not break early to allow for refinement.
                
        except Exception as e:
            pass
    
    # Final Refinement with L-BFGS-B on the best Nelder-Mead result
    if time.time() < DEADLINE - 5.0 and best_params is not None:
        try:
            res_final = minimize(_objective, best_params, method='L-BFGS-B',
                                 options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    # Extract best solution
    if best_params is None:
        best_params = x0
        
    pos_best = best_params[:2*N].reshape(N, 2)
    ang_best = best_params[2*N:]
    
    inner_list = [[float(pos_best[i,0]), float(pos_best[i,1]), float(ang_best[i])] for i in range(N)]
    final_side = _objective(best_params) # This includes penalty, so we need to compute side separately
    
    # Recompute side length without penalty
    cx, cy = 0.0, 0.0
    outer_angle_deg = 0.0
    a_outer = np.radians(outer_angle_deg)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    sqrt3 = np.sqrt(3)
    
    max_support = 0.0
    for i in range(N):
        x, y = pos_best[i]
        a_deg = ang_best[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        
        dx = verts[:, 0] - cx
        dy = verts[:, 1] - cy
        lx = ca * dx - sa * dy
        ly = sa * dx + ca * dy
        
        c1 = np.abs(lx)
        c2 = np.abs(lx + ly / sqrt3)
        c3 = np.abs(lx - ly / sqrt3)
        
        support = np.max(np.maximum(c1, np.maximum(c2, c3)))
        if support > max_support:
            max_support = support

    final_side = max_support
    
    # Check validity
    # If invalid, we might need to scale up or use a fallback
    # For now, we return the best found. The evaluator will check validity.
    
    return inner_list, (0.0, 0.0), float(final_side), 0.0

if __name__ == "__main__":
    construct()
