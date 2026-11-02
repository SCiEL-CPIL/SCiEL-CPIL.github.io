import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # Leave 10s buffer
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _check_overlap_and_containment(inner_hexes, outer_center, outer_side, outer_angle_deg):
    """
    Checks if inner hexagons are disjoint and inside the outer hexagon.
    inner_hexes: list of (x, y, angle_deg)
    Returns penalty score (0 if valid, >0 if invalid).
    """
    penalty = 0.0
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_angle_deg)
    # Scale outer vertices to actual size
    # _hex_vertices returns side length 1. We need side length outer_side.
    # Vertices are at distance R = side * sqrt(3)/2 from center.
    # Current vertices are at distance 1 * sqrt(3)/2.
    # So we scale by outer_side.
    outer_verts_scaled = outer_center + (outer_verts - outer_center) * outer_side
    
    # For each inner hexagon, check if its vertices are inside the outer hexagon
    # and if it overlaps with other inner hexagons.
    
    # Precompute outer hexagon edges for point-in-polygon or distance checks
    # A point p is inside a convex polygon if it is on the same side of all edges.
    # Edge from v_i to v_{i+1}. Normal pointing inward.
    # For a regular hexagon centered at c, the inward normal of edge i is 
    # pointing towards the center.
    # Actually, simpler: distance from center to edge is apothem = side * sqrt(3)/2.
    # For a point p, if max projection onto 6 normal directions > apothem, it's outside.
    
    base_angle = math.radians(outer_angle_deg + 30) # Normals
    normals = np.column_stack([np.cos(base_angle + np.arange(6)*math.pi/3), 
                               np.sin(base_angle + np.arange(6)*math.pi/3)])
    apothem = outer_side * _SQRT3 / 2
    
    # Check containment for all inner hex vertices
    for i in range(N):
        x, y, ang = inner_hexes[i]
        verts = _hex_vertices(x, y, ang)
        # Vectorized dot product
        # verts: (6, 2), normals: (6, 2)
        # We need dot(verts - outer_center, normals)
        rel_verts = verts - outer_center
        projections = rel_verts @ normals.T # (6, 6)
        # Max projection for each vertex
        max_proj = np.max(projections, axis=1)
        if np.any(max_proj > apothem + 1e-9):
            # Penalty proportional to how far outside
            excess = np.max(max_proj - apothem)
            penalty += 100.0 * excess
            
    # Check overlap between inner hexagons
    # Two convex polygons overlap if their edges intersect or one contains a vertex of the other.
    # Simpler for small N: check if any vertex of hex i is strictly inside hex j or vice versa.
    # Or check minimum distance between centers? No, rotation matters.
    # Let's use a robust method: check if any vertex of A is inside B.
    # If no vertex of A is inside B and no vertex of B is inside A, they might still overlap (edge crossing).
    # However, for regular hexagons, if they don't contain each other's vertices, they are disjoint?
    # Not necessarily. But for small perturbations, it's a good heuristic.
    # Better: Compute minimum distance between edges? Too slow.
    # Let's use the property: Two convex polygons are disjoint iff there is a separating axis.
    # The separating axes are the normals to the edges of both polygons.
    
    for i in range(N):
        for j in range(i+1, N):
            x1, y1, a1 = inner_hexes[i]
            x2, y2, a2 = inner_hexes[j]
            verts1 = _hex_vertices(x1, y1, a1)
            verts2 = _hex_vertices(x2, y2, a2)
            
            # Axes to test: normals of hex 1 and normals of hex 2
            # Normals of hex 1
            n1_base = math.radians(a1 + 30)
            n1s = np.column_stack([np.cos(n1_base + np.arange(6)*math.pi/3), 
                                   np.sin(n1_base + np.arange(6)*math.pi/3)])
            # Normals of hex 2
            n2_base = math.radians(a2 + 30)
            n2s = np.column_stack([np.cos(n2_base + np.arange(6)*math.pi/3), 
                                   np.sin(n2_base + np.arange(6)*math.pi/3)])
            
            axes = np.vstack([n1s, n2s])
            
            overlap = True
            for axis in axes:
                proj1 = np.dot(verts1, axis)
                proj2 = np.dot(verts2, axis)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # If there is a gap, they are separated
                if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                    overlap = False
                    break
            
            if overlap:
                # They overlap. Penalty.
                # Estimate overlap depth. 
                # If we found a separating axis with gap g, depth is -g.
                # If no separating axis, we can just add a fixed penalty or estimate.
                # Let's re-check the minimum gap.
                min_gap = float('inf')
                for axis in axes:
                    proj1 = np.dot(verts1, axis)
                    proj2 = np.dot(verts2, axis)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    gap = min2 - max1
                    if gap > min_gap:
                        min_gap = gap
                    gap = min1 - max2
                    if gap > min_gap:
                        min_gap = gap
                # min_gap is negative if overlapping
                penalty += 100.0 * abs(min_gap)
                
    return penalty

def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts.
    Returns (cx, cy, side) or None.
    """
    # Normals to the 6 sides of the outer hexagon
    # Vertices at outer_angle + 60k => edge normals at outer_angle + 30 + 60k
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    # v_dot_n: (M, 6)
    v_dot_n = verts @ normals.T
    
    # For a fixed center c, the required apothem R is max_i,k (v_i - c) . n_k
    # R(c) = max_k ( max_i (v_i . n_k) - c . n_k )
    # We want to minimize R(c).
    # This is a convex optimization problem.
    # For small M, we can use a heuristic or scipy minimize.
    # However, doing this for EVERY SA step is too slow.
    # Strategy: 
    # 1. Compute bounding box of verts.
    # 2. Set center to center of bounding box.
    # 3. Calculate required side length.
    # 4. This gives an upper bound.
    # 5. To optimize, we can do a few iterations of gradient descent on (cx, cy) for R(cx, cy).
    
    cx0 = np.mean(verts[:, 0])
    cy0 = np.mean(verts[:, 1])
    
    # Quick check: is the current center good?
    # Let's try to optimize (cx, cy) to minimize max projection.
    # f(cx, cy) = max_k ( max_i (v_i . n_k) - (cx*n_kx + cy*n_ky) )
    
    def objective(params):
        cx, cy = params
        c = np.array([cx, cy])
        c_dot_n = c @ normals.T # (6,)
        # max_i (v_i . n_k) is precomputed as max of v_dot_n
        max_v_dot_n = np.max(v_dot_n, axis=0) # (6,)
        # Required R for each direction k
        R_k = max_v_dot_n - c_dot_n
        return np.max(R_k)
        
    # Use Nelder-Mead for speed and robustness on this small 2D problem
    try:
        res = minimize(objective, [cx0, cy0], method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 50})
        cx, cy = res.x
        R = res.fun
        side = R / (_SQRT3/2)
        return cx, cy, side
    except Exception:
        # Fallback to center of mass
        R = objective([cx0, cy0])
        side = R / (_SQRT3/2)
        return cx0, cy0, side

def _evaluate_configuration(inner_hexes, outer_angle_deg):
    """
    Takes a list of inner hexagons, finds the minimal enclosing outer hexagon,
    and returns the side length.
    """
    # Collect all vertices
    all_vertices = []
    for x, y, ang in inner_hexes:
        verts = _hex_vertices(x, y, ang)
        all_vertices.extend(verts)
    all_vertices = np.array(all_vertices)
    
    cx, cy, side = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
    return side, (cx, cy)

def _simulate_annealing(initial_config, outer_angle_deg, steps=2000, T_start=1.0, T_end=0.01):
    """
    Simulated Annealing to minimize the side length of the enclosing hexagon.
    """
    current_hexes = [tuple(h) for h in initial_config]
    current_side, _ = _evaluate_configuration(current_hexes, outer_angle_deg)
    
    best_hexes = [tuple(h) for h in current_hexes]
    best_side = current_side
    
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / steps)
    
    for i in range(steps):
        if time.time() > _DEADLINE - 5.0:
            break
            
        # Generate neighbor
        new_hexes = [list(h) for h in current_hexes]
        idx = random.randint(0, N-1)
        dx = random.gauss(0, 0.1)
        dy = random.gauss(0, 0.1)
        da = random.gauss(0, 0.1)
        
        new_hexes[idx][0] += dx
        new_hexes[idx][1] += dy
        new_hexes[idx][2] += da
        
        new_side, _ = _evaluate_configuration(new_hexes, outer_angle_deg)
        
        delta = new_side - current_side
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_hexes = [tuple(h) for h in new_hexes]
            current_side = new_side
            
            if current_side < best_side:
                best_side = current_side
                best_hexes = [tuple(h) for h in current_hexes]
                
        T *= cooling
        
    return best_hexes, best_side

def _build_initial_configurations():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Hexagonal lattice
    u = (_SQRT3, 0.0)
    v = (_SQRT3/2, 1.5)
    target = [(0,0)]
    ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    target.extend(ring1)
    ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
    ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
    target.extend(ring2_sorted[:4])
    
    pos = []
    for ti, tj in target:
        x = ti*u[0] + tj*v[0]
        y = ti*u[1] + tj*v[1]
        pos.append((x, y, 0.0))
    configs.append(pos)
    
    # 2. Random
    random.seed(42)
    pos = [(random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0, 360)) for _ in range(N)]
    configs.append(pos)
    
    # 3. Linear row
    pos = [(i*1.7, 0, 0) for i in range(N)]
    configs.append(pos)
    
    return configs

def construct():
    global _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Emergency fallback
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    initial_configs = _build_initial_configurations()
    
    # Try a few outer angles
    outer_angles = [0.0, 15.0, 30.0]
    
    for outer_angle in outer_angles:
        if time.time() > _DEADLINE - 10.0:
            break
            
        for config in initial_configs:
            if time.time() > _DEADLINE - 10.0:
                break
            
            # Run SA
            # Adjust steps based on time remaining
            time_left = _DEADLINE - time.time()
            steps = int(500 * (time_left / 100.0))
            steps = max(100, min(steps, 2000))
            
            best_hexes, side = _simulate_annealing(config, outer_angle, steps=steps)
            
            if side < best_side:
                best_side = side
                best_inner = best_hexes
                # Re-evaluate to get center
                _, center = _evaluate_configuration(best_inner, outer_angle)
                best_center = center
                best_outer_angle = outer_angle
                
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    # Final polish: try to optimize the outer angle slightly?
    # And ensure the returned inner list is in the correct format [[x, y, angle], ...]
    final_inner = [[h[0], h[1], h[2]] for h in best_inner]
    
    return final_inner, best_center, best_side, best_outer_angle

if __name__ == "__main__":
    # Quick self-test
    start = time.time()
    inner, center, side, angle = construct()
    print(f"Time: {time.time()-start}s, Side: {side}")
