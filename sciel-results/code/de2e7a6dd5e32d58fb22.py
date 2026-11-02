"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Returns: (inner, center, side, angle)
      inner: list of [x, y, angle_degrees] for each inner hexagon
      center: (cx, cy) of outer hexagon
      side: side length of outer hexagon
      angle: rotation of outer hexagon in degrees
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Helper: vertices of a unit regular hexagon with given center, angle
    def hex_vertices(cx, cy, angle_deg=0.0, scale=1.0):
        angle_rad = math.radians(angle_deg)
        # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
        # Side length = scale, so circumradius = scale
        verts = []
        for i in range(6):
            theta = angle_rad + math.radians(60 * i)
            x = cx + scale * math.cos(theta)
            y = cy + scale * math.sin(theta)
            verts.append((x, y))
        return np.array(verts)
    
    # Helper: check if two convex polygons overlap using separating axis theorem
    def polygons_overlap(verts1, verts2):
        """SAT test for two convex polygons."""
        for i in range(len(verts1)):
            v1 = verts1[i]
            v2 = verts1[(i + 1) % len(verts1)]
            # Edge normal
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal = normal / np.linalg.norm(normal)
            
            # Project both polygons onto normal
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 or max2 < min1:
                return False
        return True
    
    # Helper: compute minimal enclosing regular hexagon (with given orientation) for a set of points
    def min_enclosing_hex_side(points, angle_deg=0.0):
        """
        Find the minimum side length of a regular hexagon (with given orientation)
        that encloses all given points. The hexagon is centered optimally.
        
        For a regular hexagon with side length s and orientation angle, the constraint
        for a point to be inside is: for each of 6 directions, the projection is bounded.
        
        A regular hexagon with side s has apothem = s * sqrt(3)/2.
        The 6 inward normals point in directions angle_deg + 60*k for k=0..5.
        For a point to be inside a hexagon centered at c with side s:
          |dot(p - c, n_k)| <= s * sqrt(3)/2 for all k=0..5
        
        So we need to find c and s such that for all points p and all k:
          |dot(p - c, n_k)| <= s * sqrt(3)/2
        
        This is equivalent to: for each k, max_p dot(p, n_k) - min_p dot(p, n_k) <= s * sqrt(3)
        (because we can choose c to center the range)
        
        Wait, that's not quite right. Let me think again.
        
        For a fixed orientation, the hexagon is defined by 6 half-planes.
        The minimal s is determined by the maximum over all directions k of:
          (max_p dot(p, n_k) - min_p dot(p, n_k)) / (2 * apothem/s)
        
        Actually, the width of the point set in direction n_k is w_k = max_p dot(p, n_k) - min_p dot(p, n_k).
        The hexagon with side s has width 2 * apothem = s * sqrt(3) in each of the 6 normal directions.
        So we need s * sqrt(3) >= w_k for all k, i.e., s >= w_k / sqrt(3) for all k.
        Thus s_min = max_k (w_k / sqrt(3)).
        """
        angle_rad = math.radians(angle_deg)
        normals = []
        for k in range(6):
            theta = angle_rad + math.radians(60 * k)
            normals.append(np.array([math.cos(theta), math.sin(theta)]))
        
        points = np.array(points)
        max_s = 0.0
        for n in normals:
            proj = np.dot(points, n)
            width = np.max(proj) - np.min(proj)
            s_needed = width / math.sqrt(3)
            max_s = max(max_s, s_needed)
        return max_s
    
    # Helper: get all vertices of all inner hexagons
    def get_all_vertices(inner):
        all_verts = []
        for x, y, ang in inner:
            verts = hex_vertices(x, y, ang)
            all_verts.extend(verts)
        return all_verts
    
    # Initial configuration: hexagonal packing
    # Unit hexagons in a hexagonal arrangement
    # For N=11, we can use a 3-4-3 arrangement or similar
    
    # Let's use a force-directed approach to find a good packing
    # State: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
    
    def initial_state():
        """Create initial hexagonal packing."""
        state = []
        # Hexagonal lattice spacing for unit hexagons
        # Distance between centers of adjacent hexagons = 2 * apothem = sqrt(3)
        dx = math.sqrt(3)
        dy = 1.5
        
        # Center at origin, arrange in hex pattern
        # For N=11: use rows
        # Row 0: 3 hexagons
        # Row 1: 4 hexagons  
        # Row 2: 3 hexagons
        # Or try a different arrangement
        
        # Let's use a simple grid-like arrangement first
        positions = []
        # 3 rows of 3-4-3 or 4-3-4
        # Try: place in a compact hexagonal cluster
        # For 11 hexagons: 1 + 6 + 4? No, that's 11 but not all touching.
        # Standard hexagonal numbers: 1, 7, 19, ...
        # 11 is between 7 and 19.
        
        # Let's just use a rectangular-ish grid
        # 3x4 grid = 12, so remove one
        for i in range(3):
            for j in range(4):
                if i == 2 and j == 3:
                    continue  # Skip to get 11
                x = j * dx
                y = i * dy
                # Offset alternating rows
                if i % 2 == 1:
                    x += dx / 2
                state.extend([x, y, 0.0])
        
        # Center the configuration
        xs = [state[3*i] for i in range(N)]
        ys = [state[3*i+1] for i in range(N)]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        for i in range(N):
            state[3*i] -= cx
            state[3*i+1] -= cy
        
        return np.array(state)
    
    def objective_and_grad(state):
        """
        Objective: minimize the side length of the enclosing hexagon,
        with a penalty for overlaps.
        
        We'll use a smooth approximation:
        - Overlap penalty: sum of squared penetration depths (approximated)
        - Enclosing hexagon side length
        """
        # Unpack state
        xs = state[0::3]
        ys = state[1::3]
        angs = state[2::3]
        
        # Compute vertices for each hexagon
        verts_list = [hex_vertices(xs[i], ys[i], math.degrees(angs[i])) for i in range(N)]
        
        # Overlap detection and penalty
        overlap_penalty = 0.0
        overlap_grad = np.zeros_like(state)
        
        for i in range(N):
            for j in range(i + 1, N):
                if polygons_overlap(verts_list[i], verts_list[j]):
                    # Compute approximate penetration depth
                    # Use distance between centers as a rough measure
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_dist = math.sqrt(3)  # Minimum center distance for non-overlap
                    if dist < min_dist:
                        penetration = min_dist - dist
                        # Gradient w.r.t. positions
                        if dist > 1e-10:
                            gx = -dx / dist * penetration
                            gy = -dy / dist * penetration
                            overlap_grad[3*i] += gx
                            overlap_grad[3*i+1] += gy
                            overlap_grad[3*j] -= gx
                            overlap_grad[3*j+1] -= gy
                        overlap_penalty += penetration * penetration
        
        # Enclosing hexagon side length
        all_verts = []
        for v in verts_list:
            all_verts.extend(v)
        all_verts = np.array(all_verts)
        
        side_len = min_enclosing_hex_side(all_verts, 0.0)
        
        # Gradient of side length w.r.t. positions
        # The side length is max over 6 directions of width/sqrt(3)
        # We'll compute a subgradient
        side_grad = np.zeros_like(state)
        
        angle_rad = 0.0
        normals = []
        for k in range(6):
            theta = angle_rad + math.radians(60 * k)
            normals.append(np.array([math.cos(theta), math.sin(theta)]))
        
        # Find which direction determines the max
        widths = []
        for n in normals:
            proj = np.dot(all_verts, n)
            width = np.max(proj) - np.min(proj)
            widths.append(width)
        
        max_width = max(widths)
        active_normals = [n for n, w in zip(normals, widths) if abs(w - max_width) < 1e-6]
        
        # Subgradient: for the active direction(s), the gradient is w.r.t. the extreme points
        for n in active_normals:
            proj = np.dot(all_verts, n)
            # Find which hexagon's vertex gives max and min
            # This is complex, so we'll use a finite difference approximation for the gradient
            pass
        
        # Use finite differences for the side length gradient (expensive but simple)
        eps = 1e-4
        for i in range(len(state)):
            state_plus = state.copy()
            state_plus[i] += eps
            # Compute side length at perturbed state
            xs_p = state_plus[0::3]
            ys_p = state_plus[1::3]
            angs_p = state_plus[2::3]
            verts_p = [hex_vertices(xs_p[k], ys_p[k], math.degrees(angs_p[k])) for k in range(N)]
            all_verts_p = []
            for v in verts_p:
                all_verts_p.extend(v)
            all_verts_p = np.array(all_verts_p)
            side_p = min_enclosing_hex_side(all_verts_p, 0.0)
            side_grad[i] = (side_p - side_len) / eps
        
        total_obj = side_len + 100.0 * overlap_penalty
        total_grad = side_grad + 100.0 * overlap_grad
        
        return total_obj, total_grad
    
    # Start with initial state
    best_state = initial_state()
    best_obj = None
    
    # Try several initial configurations and local optimizations
    num_restarts = 5
    max_iter_per_restart = 500
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Perturb initial state for diversity
        if restart == 0:
            state = initial_state()
        else:
            state = initial_state() + np.random.randn(len(initial_state())) * 0.5
        
        # Local optimization
        try:
            result = minimize(
                lambda s: objective_and_grad(s)[0],
                state,
                jac=lambda s: objective_and_grad(s)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_restart, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            
            if result.success or (best_obj is None or result.fun < best_obj):
                # Check for overlaps
                xs = result.x[0::3]
                ys = result.x[1::3]
                angs = result.x[2::3]
                verts_list = [hex_vertices(xs[i], ys[i], math.degrees(angs[i])) for i in range(N)]
                
                has_overlap = False
                for i in range(N):
                    for j in range(i + 1, N):
                        if polygons_overlap(verts_list[i], verts_list[j]):
                            has_overlap = True
                            break
                    if has_overlap:
                        break
                
                if not has_overlap:
                    all_verts = []
                    for v in verts_list:
                        all_verts.extend(v)
                    side_len = min_enclosing_hex_side(all_verts, 0.0)
                    
                    if best_obj is None or side_len < best_obj:
                        best_obj = side_len
                        best_state = result.x.copy()
                        
        except Exception as e:
            pass
    
    # If we found a valid solution, use it; otherwise fall back to a simple arrangement
    if best_state is not None and best_obj is not None:
        inner = []
        for i in range(N):
            inner.append([best_state[3*i], best_state[3*i+1], math.degrees(best_state[3*i+2])])
        
        # Compute the enclosing hexagon center
        xs = [inner[i][0] for i in range(N)]
        ys = [inner[i][1] for i in range(N)]
        center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        
        return inner, center, best_obj, 0.0
    
    # Fallback: simple grid arrangement
    dx = math.sqrt(3)
    dy = 1.5
    inner = []
    for i in range(3):
        for j in range(4):
            if i == 2 and j == 3:
                continue
            x = j * dx
            y = i * dy
            if i % 2 == 1:
                x += dx / 2
            inner.append([x, y, 0.0])
    
    # Center
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    
    # Compute side length
    all_verts = []
    for x, y, a in inner:
        verts = hex_vertices(x, y, a)
        all_verts.extend(verts)
    all_verts = np.array(all_verts)
    side_len = min_enclosing_hex_side(all_verts, 0.0)
    
    return inner, center, side_len, 0.0
# EVOLVE-BLOCK-END
