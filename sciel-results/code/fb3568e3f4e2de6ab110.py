import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    """Compute vertices of a regular hexagon."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(np.arange(0, 360, 60))
    return np.array([cx + s * np.cos(angles), cy + s * np.sin(angles)]).T

def _hex_normals(a_deg):
    """Compute edge normals of a regular hexagon."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(30) + np.deg2rad(np.arange(0, 360, 60))
    return np.array([np.cos(angles), np.sin(angles)]).T

def _pair_separated(v1, v2, tol=1e-7):
    """Check if two hexagons are separated (no intersection, not just touching).
    Uses the 12 edge normals approach as specified."""
    # Get normals for both hexagons
    n1 = _hex_normals(np.rad2deg(np.arctan2(v1[1,1]-v1[0,1], v1[1,0]-v1[0,0])))  # approximate
    # Better: compute normals from vertices
    def _normals_from_verts(v):
        normals = []
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1)%6]
            edge = p2 - p1
            # Normal perpendicular to edge, pointing outward
            n = np.array([-edge[1], edge[0]])
            # Ensure it points outward (away from center)
            center = np.mean(v, axis=0)
            mid = (p1 + p2) / 2
            if np.dot(n, mid - center) < 0:
                n = -n
            n = n / np.linalg.norm(n)
            normals.append(n)
        return np.array(normals)
    
    normals1 = _normals_from_verts(v1)
    normals2 = _normals_from_verts(v2)
    all_normals = np.vstack([normals1, normals2])
    
    # For each normal, project both polygons and check if intervals are apart
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # Intervals [min1, max1] and [min2, max2] are apart if max1 < min2 - tol or max2 < min1 - tol
        if max1 < min2 - tol or max2 < min1 - tol:
            return True  # Separated along this normal
    return False  # Not separated (intersecting or touching)

def _point_in_hexagon(p, hex_verts):
    """Check if point p is inside hexagon defined by hex_verts."""
    # Use cross product test for convex polygon
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i+1)%6]
        cross = np.cross(p2 - p1, p - p1)
        if cross < -1e-8:
            return False
    return True

def _inside_outer(inner_verts, outer_verts):
    """Check if all inner vertices are inside the outer hexagon."""
    for v in inner_verts:
        if not _point_in_hexagon(v, outer_verts):
            return False
    return True

def _min_enclosing_hex(centers, angles, center=[0,0], angle=0):
    """Find the minimum side length of a hexagon with given center and angle 
    that contains all inner hexagons."""
    all_verts = []
    for i in range(len(centers)):
        v = _hex_vertices(centers[i,0], centers[i,1], 1.0, angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    # Shift by outer center
    shifted = all_verts - np.array(center)
    
    # For a hexagon with center 0, side s, angle a:
    # A point (x,y) is inside if for all 6 edge normals n_k:
    # dot((x,y), n_k) <= s * sqrt(3) / 2
    a = np.deg2rad(angle)
    normals = _hex_normals(angle)
    
    # Compute max projection for each normal
    max_projs = []
    for n in normals:
        max_projs.append(np.max(shifted @ n))
    
    # The minimum s such that s * sqrt(3)/2 >= max_proj for all normals
    s_min = 0
    for mp in max_projs:
        s_candidate = mp * 2 / np.sqrt(3)
        if s_candidate > s_min:
            s_min = s_candidate
    
    return max(s_min, 1.0)

def _valid_solution(centers, angles, side, center=[0,0], outer_angle=0):
    """Validate the solution completely."""
    # Check all finite
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(angles)):
        return False
    if not np.isfinite(side) or side < np.sqrt(11) - 1e-8:
        return False
    
    # Get all inner vertices
    inner_verts = []
    for i in range(len(centers)):
        v = _hex_vertices(centers[i,0], centers[i,1], 1.0, angles[i])
        inner_verts.append(v)
    
    # Check pairwise separation
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            if not _pair_separated(inner_verts[i], inner_verts[j]):
                return False
    
    # Check containment in outer hexagon
    outer_verts = _hex_vertices(center[0], center[1], side, outer_angle)
    for iv in inner_verts:
        for p in iv:
            if not _point_in_hexagon(p, outer_verts):
                return False
    
    # Check edge lengths
    for iv in inner_verts:
        for i in range(6):
            edge_len = np.linalg.norm(iv[(i+1)%6] - iv[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    return True

def _soft_enclosing_side(centers, angles, center=[0,0], angle=0, softness=10):
    """Differentiable approximation of min enclosing hex side."""
    all_verts = []
    for i in range(len(centers)):
        v = _hex_vertices(centers[i,0], centers[i,1], 1.0, angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    shifted = all_verts - np.array(center)
    a = np.deg2rad(angle)
    normals = _hex_normals(angle)
    
    max_projs = []
    for n in normals:
        max_projs.append(np.max(shifted @ n))
    
    # Soft max of s_candidate = max_proj * 2/sqrt(3)
    s_candidates = np.array(max_projs) * 2 / np.sqrt(3)
    # Use log-sum-exp for soft max
    m = np.max(s_candidates)
    if m == 0:
        return 1.0
    soft_max = m + np.log(np.sum(np.exp(softness * (s_candidates - m)))) / softness
    return max(soft_max, 1.0)

def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0
    safety_margin = 10.0  # seconds before deadline to stop
    
    sqrt3 = np.sqrt(3)
    
    # Initial guess: hexagonal cluster of 7 + 4 more
    # Center + 6 ring + 4 in valleys
    centers = np.array([
        [0, 0],
        [sqrt3, 0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        [2*sqrt3, 0],
        [sqrt3, 3.0],
        [-2*sqrt3, 0],
        [-sqrt3, -3.0]
    ])
    angles = np.zeros(11)
    
    best_score = -1.0
    best_result = None
    
    def _try_configuration(c, a, max_time=None):
        """Try to optimize a configuration and return best valid solution found."""
        nonlocal best_score, best_result
        if max_time is None:
            max_time = deadline - safety_margin - (time.time() - start_time)
        if max_time <= 0:
            return
        
        # First compute initial enclosing side
        s_init = _min_enclosing_hex(c, a)
        
        def objective(params):
            if time.time() > deadline - safety_margin:
                return 1e10
            c_local = params[:22].reshape(11, 2)
            a_local = params[22:33]
            
            # Soft enclosing side
            s = _soft_enclosing_side(c_local, a_local, softness=20)
            
            # Pairwise overlap penalty
            penalty = 0.0
            inner_verts = []
            for i in range(11):
                v = _hex_vertices(c_local[i,0], c_local[i,1], 1.0, a_local[i])
                inner_verts.append(v)
            
            for i in range(11):
                for j in range(i+1, 11):
                    if not _pair_separated(inner_verts[i], inner_verts[j], tol=-1e-5):
                        # Compute overlap amount
                        # Simple: check if vertices are inside each other
                        for p in inner_verts[i]:
                            if _point_in_hexagon(p, inner_verts[j]):
                                penalty += 50.0
                        for p in inner_verts[j]:
                            if _point_in_hexagon(p, inner_verts[i]):
                                penalty += 50.0
            
            return s + penalty * 0.1
        
        p0 = np.concatenate([c.flatten(), a])
        
        # Try multiple optimization methods
        results = []
        
        # Nelder-Mead
        try:
            res = minimize(objective, p0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
            if time.time() < deadline - safety_margin:
                c_opt = res.x[:22].reshape(11, 2)
                a_opt = res.x[22:33]
                s_opt = _min_enclosing_hex(c_opt, a_opt)
                if _valid_solution(c_opt, a_opt, s_opt):
                    score = 1.0 / s_opt
                    results.append((score, c_opt.copy(), a_opt.copy(), s_opt))
        except Exception:
            pass
        
        # Powell method
        if time.time() < deadline - safety_margin - 30:
            try:
                res = minimize(objective, p0, method='Powell',
                               options={'maxiter': 1000, 'ftol': 1e-7})
                if time.time() < deadline - safety_margin:
                    c_opt = res.x[:22].reshape(11, 2)
                    a_opt = res.x[22:33]
                    s_opt = _min_enclosing_hex(c_opt, a_opt)
                    if _valid_solution(c_opt, a_opt, s_opt):
                        score = 1.0 / s_opt
                        results.append((score, c_opt.copy(), a_opt.copy(), s_opt))
            except Exception:
                pass
        
        if results:
            best = max(results, key=lambda x: x[0])
            if best[0] > best_score:
                best_score = best[0]
                inner_data = np.column_stack([best[1], best[2]])
                outer_data = np.array([0.0, 0.0, 0.0])
                best_result = (inner_data, outer_data, best[3])
    
    # Try initial configuration
    _try_configuration(centers, angles, max_time=120)
    
    # Try perturbed configurations
    if time.time() < deadline - safety_margin - 60:
        np.random.seed(42)
        for attempt in range(5):
            if time.time() > deadline - safety_margin - 30:
                break
            
            # Perturb the best configuration or initial
            c_pert = centers.copy() + np.random.normal(0, 0.5, (11, 2))
            a_pert = angles.copy() + np.random.normal(0, 0.3, 11)
            
            # Keep them roughly in place
            c_pert = np.clip(c_pert, -5, 5)
            a_pert = np.clip(a_pert, -90, 90)
            
            _try_configuration(c_pert, a_pert, max_time=45)
    
    # If no valid result, use fallback
    if best_result is None:
        # Use the initial configuration with computed enclosing hex
        s = _min_enclosing_hex(centers, angles)
        # Check if valid, if not, expand
        while not _valid_solution(centers, angles, s):
            s *= 1.05
        inner_data = np.column_stack([centers, angles])
        outer_data = np.array([0.0, 0.0, 0.0])
        best_result = (inner_data, outer_data, s)
    
    return best_result

if __name__ == "__main__":
    inner, outer, side = hexagon_packing_11()
    print(f"Outer side: {side}")
    print(f"Score: {1.0/side}")
