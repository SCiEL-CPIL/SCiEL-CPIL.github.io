import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)
DEADLINE = 145.0

# Pre-compute unit hexagon vertices
# Orientation 0: flat-top (vertices at 0, 60, 120, 180, 240, 300 degrees)
VERTS_FLAT = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
# Orientation 1: pointy-top (vertices at 30, 90, 150, 210, 270, 330 degrees)
VERTS_POINTY = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 390, 60)])

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def time_left():
        return deadline - time.time()

    def get_verts(orient_idx):
        """Get pre-computed vertex array for given orientation index (0=flat, 1=pointy)."""
        return VERTS_FLAT if orient_idx == 0 else VERTS_POINTY

    def calc_side_length(positions, orientations, outer_center=(0.0, 0.0), outer_angle_deg=0.0):
        """
        Calculate the minimum outer hexagon side length that contains all inner hexagons.
        positions: (N, 2) array
        orientations: (N,) array of 0 or 1
        outer_center: (x, y)
        outer_angle_deg: rotation of outer hexagon in degrees
        Returns: side length of containing hexagon
        """
        theta_rad = math.radians(outer_angle_deg)
        # Outer hexagon has 6 supporting lines with normals at theta + 30 + k*60 degrees
        # For a regular hexagon with side s centered at c, the apothem is s*sqrt(3)/2
        # The max projection of any point onto normal direction gives the distance from center
        # side = 2 * max_proj / sqrt(3)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append(np.array([math.cos(ang), math.sin(ang)]))
        normals = np.array(normals)  # (6, 2)
        
        oc = np.array(outer_center)
        max_proj = 0.0
        
        for i in range(N):
            v = get_verts(orientations[i])
            # All 6 vertices of this hexagon
            verts = v + positions[i] - oc  # (6, 2)
            projs = verts @ normals.T  # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / SQRT3

    def overlap_penalty(positions, orientations, strength=100.0):
        """
        Calculate a smooth penalty for overlaps between hexagons.
        Uses SAT (Separating Axis Theorem) for convex polygons.
        Returns a smooth approximation of the total overlap depth.
        """
        penalty = 0.0
        polys = []
        for i in range(N):
            v = get_verts(orientations[i])
            polys.append(v + positions[i])
        
        for i in range(N):
            p1 = polys[i]
            for j in range(i+1, N):
                p2 = polys[j]
                
                # Quick bounding box check
                if np.max(p1[:,0]) < np.min(p2[:,0]) or np.max(p2[:,0]) < np.min(p1[:,0]):
                    continue
                if np.max(p1[:,1]) < np.min(p2[:,1]) or np.max(p2[:,1]) < np.min(p1[:,1]):
                    continue
                
                # SAT: check all edge normals from both polygons
                min_overlap = float('inf')
                separated = False
                
                # Get axes from p1
                for k in range(6):
                    edge = p1[(k+1)%6] - p1[k]
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm < 1e-12:
                        continue
                    n = n / norm
                    proj1 = np.dot(p1, n)
                    proj2 = np.dot(p2, n)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap < 0:
                        separated = True
                        break
                    if overlap < min_overlap:
                        min_overlap = overlap
                
                if separated:
                    continue
                
                # Get axes from p2
                for k in range(6):
                    edge = p2[(k+1)%6] - p2[k]
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm < 1e-12:
                        continue
                    n = n / norm
                    proj1 = np.dot(p1, n)
                    proj2 = np.dot(p2, n)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap < 0:
                        separated = True
                        break
                    if overlap < min_overlap:
                        min_overlap = overlap
                
                if not separated and min_overlap > 0:
                    # Smooth penalty: use exponential to make it smooth and strong
                    penalty += strength * min_overlap * min_overlap
        
        return penalty

    def objective(params, orientations):
        """
        params: 2*N values for positions
        orientations: fixed (N,) array
        """
        pos = params.reshape(N, 2)
        side = calc_side_length(pos, orientations)
        pen = overlap_penalty(pos, orientations)
        return side + pen

    def objective_with_outer(params):
        """
        params: 2*N positions + 2 outer center + 1 outer angle (radians)
        """
        pos = params[:2*N].reshape(N, 2)
        oc = params[2*N:2*N+2]
        oa = params[2*N+2]
        side = calc_side_length(pos, orientations_fixed, outer_center=oc, outer_angle_deg=math.degrees(oa))
        pen = overlap_penalty(pos, orientations_fixed)
        return side + pen

    # --- Generate initial configurations ---
    inits = []
    
    # 1. Honeycomb lattice - optimal for hexagons
    # In a honeycomb, hexagons tile perfectly. For 12 hexagons, use a compact arrangement.
    # Hex grid coordinates: axial (q, r) -> Cartesian
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    # Configuration A: 4-3-4-1 or similar compact honeycomb
    # Let's use a 3x4 block of honeycomb
    honeycomb_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            x = 1.5 * q
            y = SQRT3 * (r + q/2.0)
            honeycomb_points.append((x, y))
    
    # Sort by distance from center and pick 12 closest
    honeycomb_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    for subset_start in range(0, min(10, len(honeycomb_points)-11)):
        pts = honeycomb_points[subset_start:subset_start+12]
        pos = np.array(pts)
        pos -= np.mean(pos, axis=0)
        orient = np.zeros(N, dtype=int)
        inits.append((pos, orient))
    
    # 2. Ring configuration
    for radius in [2.0, 2.5, 3.0]:
        ring_pos = []
        for i in range(N):
            a = 2 * math.pi * i / N
            ring_pos.append([radius * math.cos(a), radius * math.sin(a)])
        ring_pos = np.array(ring_pos)
        ring_pos -= np.mean(ring_pos, axis=0)
        orient = np.zeros(N, dtype=int)
        inits.append((ring_pos, orient))
    
    # 3. Two-row configuration
    two_row = []
    for i in range(6):
        x = i * 1.5 - 3.75
        y = 0
        two_row.append([x, y])
    for i in range(6):
        x = i * 1.5 - 3.75
        y = SQRT3
        two_row.append([x, y])
    two_row = np.array(two_row)
    two_row -= np.mean(two_row, axis=0)
    inits.append((two_row, np.zeros(N, dtype=int)))
    
    # 4. 3x4 grid
    grid = []
    for i in range(3):
        for j in range(4):
            x = i * 1.5
            y = j * SQRT3
            if i % 2 == 1:
                y += SQRT3 / 2
            grid.append([x, y])
    grid = np.array(grid)
    grid -= np.mean(grid, axis=0)
    inits.append((grid, np.zeros(N, dtype=int)))
    
    # 5. Random initializations
    np.random.seed(42)
    for _ in range(8):
        pos = np.random.uniform(-4, 4, (N, 2))
        pos -= np.mean(pos, axis=0)
        orient = np.zeros(N, dtype=int)
        inits.append((pos, orient))
    
    # --- Optimization ---
    best_side = float('inf')
    best_pos = None
    best_orient = None
    best_outer_center = [0.0, 0.0]
    best_outer_angle = 0.0

    # Phase 1: Optimize positions for each initial configuration
    for idx, (pos0, orient0) in enumerate(inits):
        if time_left() < 25:
            break
        
        orientations_fixed = orient0.copy()
        
        def obj(p):
            pos = p.reshape(N, 2)
            side = calc_side_length(pos, orientations_fixed)
            pen = overlap_penalty(pos, orientations_fixed)
            return side + pen
        
        try:
            res = minimize(obj, pos0.flatten(), method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
            
            pos_res = res.x.reshape(N, 2)
            pen = overlap_penalty(pos_res, orientations_fixed)
            if pen < 1e-6:
                side = calc_side_length(pos_res, orientations_fixed)
                if side < best_side:
                    best_side = side
                    best_pos = pos_res
                    best_orient = orientations_fixed.copy()
        except:
            pass
    
    # Phase 2: Try different orientation patterns
    if time_left() > 30 and best_pos is not None:
        # Try all-pointy
        orientations_fixed = np.ones(N, dtype=int)
        
        def obj_pointy(p):
            pos = p.reshape(N, 2)
            side = calc_side_length(pos, orientations_fixed)
            pen = overlap_penalty(pos, orientations_fixed)
            return side + pen
        
        try:
            res = minimize(obj_pointy, best_pos.flatten(), method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            pos_res = res.x.reshape(N, 2)
            pen = overlap_penalty(pos_res, orientations_fixed)
            if pen < 1e-6:
                side = calc_side_length(pos_res, orientations_fixed)
                if side < best_side:
                    best_side = side
                    best_pos = pos_res
                    best_orient = orientations_fixed.copy()
        except:
            pass
    
    # Phase 3: Joint optimization of positions + outer center/angle
    if time_left() > 15 and best_pos is not None:
        orientations_fixed = best_orient.copy()
        
        def obj_joint(p):
            pos = p[:2*N].reshape(N, 2)
            oc = p[2*N:2*N+2]
            oa = p[2*N+2]
            side = calc_side_length(pos, orientations_fixed, outer_center=oc, outer_angle_deg=math.degrees(oa))
            pen = overlap_penalty(pos, orientations_fixed)
            return side + pen
        
        centroid = np.mean(best_pos, axis=0)
        x0 = np.concatenate([best_pos.flatten(), [centroid[0], centroid[1], 0.0]])
        
        try:
            res = minimize(obj_joint, x0, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100})
            
            pos_res = res.x[:2*N].reshape(N, 2)
            oc_res = res.x[2*N:2*N+2]
            oa_res = res.x[2*N+2]
            pen = overlap_penalty(pos_res, orientations_fixed)
            if pen < 1e-6:
                side = calc_side_length(pos_res, orientations_fixed, outer_center=oc_res, outer_angle_deg=math.degrees(oa_res))
                if side < best_side:
                    best_side = side
                    best_pos = pos_res
                    best_orient = orientations_fixed.copy()
                    best_outer_center = oc_res.tolist()
                    best_outer_angle = math.degrees(oa_res)
        except:
            pass
    
    # Phase 4: High-precision polish
    if time_left() > 5 and best_pos is not None:
        orientations_fixed = best_orient.copy()
        
        def obj_polish(p):
            pos = p[:2*N].reshape(N, 2)
            oc = p[2*N:2*N+2]
            oa = p[2*N+2]
            side = calc_side_length(pos, orientations_fixed, outer_center=oc, outer_angle_deg=math.degrees(oa))
            pen = overlap_penalty(pos, orientations_fixed)
            return side + pen
        
        x0 = np.concatenate([best_pos.flatten(), np.array(best_outer_center), [math.radians(best_outer_angle)]])
        
        try:
            res = minimize(obj_polish, x0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200})
            
            pos_res = res.x[:2*N].reshape(N, 2)
            oc_res = res.x[2*N:2*N+2]
            oa_res = res.x[2*N+2]
            pen = overlap_penalty(pos_res, orientations_fixed)
            if pen < 1e-8:
                side = calc_side_length(pos_res, orientations_fixed, outer_center=oc_res, outer_angle_deg=math.degrees(oa_res))
                if side < best_side:
                    best_side = side
                    best_pos = pos_res
                    best_orient = orientations_fixed.copy()
                    best_outer_center = oc_res.tolist()
                    best_outer_angle = math.degrees(oa_res)
        except:
            pass

    # Fallback if optimization failed
    if best_pos is None:
        # Simple honeycomb
        honeycomb_points = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                x = 1.5 * q
                y = SQRT3 * (r + q/2.0)
                honeycomb_points.append((x, y))
        honeycomb_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        pts = honeycomb_points[:12]
        pos = np.array(pts)
        pos -= np.mean(pos, axis=0)
        best_pos = pos
        best_orient = np.zeros(N, dtype=int)
        best_side = calc_side_length(pos, best_orient)
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0

    # Construct return values
    inner = []
    for i in range(N):
        angle_deg = 0.0 if best_orient[i] == 0 else 30.0
        inner.append([float(best_pos[i][0]), float(best_pos[i][1]), angle_deg])
    
    return inner, (float(best_outer_center[0]), float(best_outer_center[1])), float(best_side), float(best_outer_angle)
