# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hexagon_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _hex_edge_normals(angle_deg):
    """Compute the 6 outward unit edge normals for a regular hexagon."""
    a = np.radians(angle_deg)
    normals_angles = a + np.radians(30) + np.radians(60) * np.arange(6)
    return np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])

def _check_separation(v1, v2, tol=1e-6):
    """Check if two hexagons are separated using SAT. Returns True if separated."""
    # Test normals from hex1
    n1 = _hex_edge_normals(0)  # We'll compute per-hexagon normals below
    # Actually, let's compute properly
    return _sat_separated(v1, v2, tol)

def _sat_separated(v1, v2, tol=1e-6):
    """SAT check. Returns True if separated."""
    for verts in [v1, v2]:
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            edge = p2 - p1
            normal = np.array([edge[1], -edge[0]])
            norm = np.linalg.norm(normal)
            if norm == 0:
                continue
            normal /= norm
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return True
    return False

def _point_in_hexagon(v, outer_verts):
    """Check if point v is inside or on boundary of convex polygon outer_verts (CCW)."""
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n]
        edge = p2 - p1
        cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
        if cross < -1e-6:
            return False
    return True

def _min_enclosing_hex_side(centers, angles, outer_angle=0.0):
    """
    Find minimum side length s of outer hexagon (centered at origin, given angle)
    that contains all inner hexagons.
    """
    outer_normals = _hex_edge_normals(outer_angle)
    sqrt3 = np.sqrt(3)
    max_required_s = 0.0
    
    for i in range(len(centers)):
        cx, cy = centers[i]
        a = angles[i]
        # Support function of unit hexagon at angle a in each outer normal direction
        inner_verts = _hexagon_vertices(0, 0, 1.0, a)
        for n in outer_normals:
            support = np.max(inner_verts @ n)
            # The farthest point in direction n from origin:
            #   dot((cx,cy), n) + support
            # This must be <= s * apothem_factor where apothem = s * sqrt(3)/2
            required_s = (cx * n[0] + cy * n[1] + support) / (sqrt3 / 2)
            if required_s > max_required_s:
                max_required_s = required_s
    
    return max_required_s

def _packing_cost(params):
    """
    Cost function. params: [s, ox, oy, oa, x1,y1,a1, ..., x12,y12,a12]
    Minimizes s + penalties.
    """
    s = params[0]
    ox, oy, oa = params[1], params[2], params[3]
    inner_data = params[4:].reshape(12, 3)
    
    if s < np.sqrt(12) - 0.1:
        return 1e6
    
    penalty = 0.0
    outer_verts = _hexagon_vertices(ox, oy, s, oa)
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i + 1, 6):
            d = np.linalg.norm(outer_verts[i] - outer_verts[j])
            if d < 1e-6:
                penalty += 1000.0
    
    inner_verts_list = []
    for i in range(12):
        x, y, a = inner_data[i]
        verts = _hexagon_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
    
    # Overlap check
    for i in range(12):
        for j in range(i + 1, 12):
            if not _sat_separated(inner_verts_list[i], inner_verts_list[j], 1e-8):
                penalty += 10.0
    
    # Containment check
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _point_in_hexagon(v, outer_verts):
                penalty += 10.0
    
    # Edge length check
    for i in range(12):
        verts = inner_verts_list[i]
        for k in range(6):
            d = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(d - 1.0) > 1e-8:
                penalty += 100.0
    
    return s + penalty

def _validate_result(inner_data, outer_data, s):
    """Validate the packing result. Returns True if valid."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hexagon_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-6:
                return False
    
    inner_verts_list = []
    for i in range(12):
        x, y, a = inner_data[i]
        verts = _hexagon_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
        
        for k in range(6):
            d = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(d - 1.0) > 1e-8:
                return False
        
        for v in verts:
            if not _point_in_hexagon(v, outer_verts):
                return False
    
    for i in range(12):
        for j in range(i + 1, 12):
            if not _sat_separated(inner_verts_list[i], inner_verts_list[j], 1e-6):
                return False
    
    return True

def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing outer side length.
    Returns (inner, outer, s).
    """
    start_time = time.time()
    deadline = start_time + 580
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    sqrt3 = np.sqrt(3)
    
    def _honeycomb_343():
        """3-4-3 honeycomb arrangement."""
        dy = 1.5 * sqrt3
        inner = []
        for i in range(3):
            x = (i - 1) * 2.0
            inner.append([x, 0, 0])
        for i in range(4):
            x = (i - 1.5) * 2.0
            inner.append([x, dy, 0])
        for i in range(3):
            x = (i - 1) * 2.0
            inner.append([x, 2 * dy, 0])
        return np.array(inner)
    
    def _honeycomb_2442():
        """2-4-4-2 arrangement."""
        dy = 1.5 * sqrt3
        inner = []
        for i in range(2):
            x = (i - 0.5) * 2.0
            inner.append([x, 0, 0])
        for i in range(4):
            x = (i - 1.5) * 2.0
            inner.append([x, dy, 0])
        for i in range(4):
            x = (i - 1.5) * 2.0
            inner.append([x, 2 * dy, 0])
        for i in range(2):
            x = (i - 0.5) * 2.0
            inner.append([x, 3 * dy, 0])
        return np.array(inner)
    
    def _circular_111():
        """1 center + 6 ring + 5 outer."""
        inner = [[0, 0, 0]]
        for i in range(6):
            a = np.radians(60 * i)
            x = 2.0 * np.cos(a)
            y = 2.0 * np.sin(a)
            inner.append([x, y, 0])
        for i in range(5):
            a = np.radians(72 * i + 30)
            x = 3.86 * np.cos(a)
            y = 3.86 * np.sin(a)
            inner.append([x, y, 0])
        return np.array(inner)
    
    def _grid_3x4():
        """3 columns, 4 rows."""
        inner = []
        for row in range(4):
            for col in range(3):
                x = (col - 1) * 2.0
                y = (row - 1.5) * 1.5 * sqrt3
                inner.append([x, y, 0])
        return np.array(inner)
    
    initial_configs = [_honeycomb_343(), _honeycomb_2442(), _circular_111(), _grid_3x4()]
    
    for config_idx, inner_init in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Center the arrangement
        xs = inner_init[:, 0]
        ys = inner_init[:, 1]
        cx = (xs.min() + xs.max()) / 2
        cy = (ys.min() + ys.max()) / 2
        inner_init = inner_init.copy()
        inner_init[:, 0] -= cx
        inner_init[:, 1] -= cy
        
        # Compute minimum enclosing hexagon side
        centers = inner_init[:, :2]
        angles = inner_init[:, 2]
        s_init = _min_enclosing_hex_side(centers, angles, 0.0) * 1.001
        
        params_init = np.concatenate([[s_init, 0.0, 0.0, 0.0], inner_init.flatten()])
        
        n_restarts = 4
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                params = params_init.copy()
            else:
                # Perturb positions and angles
                params = params_init.copy()
                params[4:26:3] += np.random.normal(0, 0.05, 12)  # x
                params[5:26:3] += np.random.normal(0, 0.05, 12)  # y
                params[6:26:3] += np.random.normal(0, 5, 12)     # angles (degrees)
                params[0] *= (1.0 + np.random.uniform(-0.02, 0.02))
            
            try:
                result = minimize(
                    _packing_cost,
                    params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-7,
                        'fatol': 1e-7,
                        'adaptive': True
                    }
                )
            except Exception:
                continue
            
            s, ox, oy, oa = result.x[0], result.x[1], result.x[2], result.x[3]
            inner_data = result.x[4:].reshape(12, 3)
            outer_data = np.array([ox, oy, oa])
            
            if _validate_result(inner_data, outer_data, s):
                if s < best_s:
                    best_s = s
                    best_inner = inner_data.copy()
                    best_outer = outer_data.copy()
    
    # Fallback
    if best_inner is None:
        inner = _honeycomb_343()
        inner[:, 0] -= (inner[:, 0].min() + inner[:, 0].max()) / 2
        inner[:, 1] -= (inner[:, 1].min() + inner[:, 1].max()) / 2
        s = _min_enclosing_hex_side(inner[:, :2], inner[:, 2], 0.0) * 1.01
        outer = np.array([0.0, 0.0, 0.0])
        return inner, outer, s
    
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
