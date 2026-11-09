# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _is_valid(inner, outer, s, tol=1e-6, edge_tol=1e-8):
    """Check validity of a packing configuration."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < math.sqrt(12) - 1e-9:
        return False
    
    ocx, ocy, oang = outer[0], outer[1], outer[2]
    outer_verts = _hex_vertices(ocx, ocy, s, oang)
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Check inner hexagons have unit edges
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > edge_tol:
                return False
    
    # Check separation between inner hexagons using SAT
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            
            normals = []
            for verts in [v1, v2]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1)%6]
                    edge = p2 - p1
                    n = np.array([-edge[1], edge[0]])
                    n_norm = np.linalg.norm(n)
                    if n_norm > 1e-12:
                        n = n / n_norm
                        normals.append(n)
            
            separated = False
            for n in normals:
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap > -tol:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check inner vertices inside outer hexagon
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for v in verts:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                edge = p2 - p1
                rel = v - p1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def _hex_lattice_positions(coords):
    """Convert hex grid coordinates to Cartesian positions.
    coords: list of (q, r) in axial coordinates
    Returns array of (x, y) positions for unit hexagons on hex lattice.
    Adjacent hexagons in hex lattice have center distance sqrt(3).
    """
    sqrt3 = math.sqrt(3)
    positions = np.zeros((len(coords), 2))
    for i, (q, r) in enumerate(coords):
        # Axial to Cartesian conversion for pointy-top hexagons
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        positions[i] = [x, y]
    return positions

def _min_s_for_config(inner, outer_angle):
    """Compute minimum outer hexagon side length s needed to contain all inner hexagons,
    given outer hexagon at origin with given angle."""
    outer_center = np.array([0.0, 0.0])
    
    # Get all inner vertices
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # For a regular hexagon centered at origin with side s and angle a,
    # a point p is inside if for each edge, the cross product condition holds.
    # The inradius of a regular hexagon with side s is s * sqrt(3)/2.
    # The apothem (distance from center to edge midpoint) is s * sqrt(3)/2.
    
    # For each edge of the outer hexagon, compute the required distance.
    # Edge k goes from vertex k to vertex k+1.
    # The inward normal direction for edge k can be computed.
    
    # Alternative: for a hexagon with center at origin, side s, angle a,
    # the supporting line for edge k has normal n_k pointing inward.
    # n_k = (cos(a + 60k + 90), sin(a + 60k + 90)) ? Let me think.
    
    # Vertices: v_k = s * (cos(a + 60k), sin(a + 60k))
    # Edge from v_k to v_{k+1}: direction d = v_{k+1} - v_k
    # Inward normal: rotate d by -90 degrees (clockwise) and normalize
    # Actually for CCW polygon, interior is to the left, so inward normal is left normal.
    # Left normal of (dx, dy) is (-dy, dx).
    
    # The distance from origin to edge k is the apothem = s * sqrt(3)/2.
    # The inward unit normal for edge k points from edge midpoint toward center.
    # Edge midpoint m_k = (v_k + v_{k+1})/2
    # Inward direction = -m_k / |m_k| (since center is at origin)
    
    # For point p to be inside: dot(p - m_k, inward_normal) >= 0 for all k
    # Equivalent: dot(p, inward_normal) >= dot(m_k, inward_normal) = |m_k| (since inward_normal = -m_k/|m_k|)
    # Wait: dot(m_k, -m_k/|m_k|) = -|m_k|. So dot(p, inward_normal) >= -|m_k|?
    # Let me redo. Inward normal n_k = -m_k / |m_k|.
    # For p inside: (p - m_k) dot n_k >= 0 => p dot n_k >= m_k dot n_k = m_k dot (-m_k/|m_k|) = -|m_k|
    # So: p dot n_k >= -|m_k| for all k.
    # |m_k| = apothem = s * sqrt(3)/2.
    # n_k = -m_k / |m_k| = -(v_k + v_{k+1}) / (2 * |m_k|)
    
    # For given angle a, compute n_k and |m_k| for each k.
    # Then for all points p, we need: max over k of (p dot n_k + |m_k|) <= 0? 
    # No: p dot n_k >= -|m_k| => p dot n_k + |m_k| >= 0.
    # So we need min over k of (p dot n_k + |m_k|) >= 0 for all p.
    # Equivalently, for all p: max over k of (-p dot n_k - |m_k|) <= 0.
    
    # To find minimum s: for fixed angle a, we need s such that for all p in all_verts:
    # For all k: p dot n_k(s,a) + |m_k(s,a)| >= 0
    # n_k depends on a but not s (it's a direction). |m_k| = s * sqrt(3)/2.
    # So: p dot n_k + s * sqrt(3)/2 >= 0 => s >= -2/sqrt(3) * p dot n_k
    # So s_min = max over p, k of (-2/sqrt(3) * p dot n_k)
    # But also s >= 0.
    
    a_rad = np.deg2rad(outer_angle)
    sqrt3 = math.sqrt(3)
    
    # Compute inward normals n_k for each edge
    n_k = np.zeros((6, 2))
    for k in range(6):
        # m_k direction: angle a + 60k + 30 degrees (midpoint between vertex k and k+1)
        # Actually, the midpoint of edge k is at angle a + 60k + 30 from center?
        # v_k at angle a + 60k, v_{k+1} at angle a + 60(k+1)
        # Midpoint angle = a + 60k + 30
        mid_angle = a_rad + np.deg2rad(60.0 * k + 30.0)
        # m_k = (s * sqrt(3)/2) * (cos(mid_angle), sin(mid_angle))
        # Inward normal = -m_k / |m_k| = -(cos(mid_angle), sin(mid_angle))
        n_k[k] = [-math.cos(mid_angle), -math.sin(mid_angle)]
    
    # For each point p, compute required s: s >= -2/sqrt(3) * p dot n_k for all k
    # s_min = max(0, max over p,k of (-2/sqrt(3) * p dot n_k))
    
    # Vectorized: for each p, compute p dot n_k for all k, then take max
    # p dot n_k for all p and all k: (N, 6) matrix
    dot_products = all_verts @ n_k.T  # (N, 6)
    # Required s for each (p, k): -2/sqrt(3) * dot_products[p, k]
    required_s = -2.0 / sqrt3 * dot_products  # (N, 6)
    s_min = max(0.0, np.max(required_s))
    
    return s_min

def _build_inner_from_lattice(lattice_coords, outer_angle):
    """Build inner hexagon array from lattice coordinates.
    All inner hexagons have angle 0 (aligned with each other).
    We can also optimize the lattice orientation relative to outer.
    """
    positions = _hex_lattice_positions(lattice_coords)
    inner = np.zeros((12, 3))
    inner[:, 0] = positions[:, 0]
    inner[:, 1] = positions[:, 1]
    inner[:, 2] = 0.0  # All aligned
    return inner

def _objective_lattice(params):
    """Objective: minimize s. Params: [outer_angle, lattice_rotation]
    We fix the lattice structure (which 12 hexagons) and optimize orientations."""
    outer_angle = params[0]
    lattice_rotation = params[1]
    
    # Standard 12-hexagon cluster: center + 6 neighbors + 5 more
    # Let's use a specific arrangement that's known to be compact.
    # Hex grid: center (0,0), ring 1: 6 hexes, then add 5 more from ring 2.
    # For 12 hexagons, a good arrangement is:
    # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) [7 hexes: center + ring 1]
    # Plus 5 more: (2,0), (1,1), (0,2), (-2,1), (-1,-1)? Let me think of a compact 12.
    # Actually, 12 = 1 + 6 + 5. The 5 from ring 2 should be chosen to minimize extent.
    
    # Let me use a different approach: try multiple lattice configurations and pick best.
    # For now, use a fixed configuration and optimize angles.
    
    # Configuration: center + 6 + 5 (selected from ring 2 to minimize bounding hexagon)
    coords = [
        (0, 0),    # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (1, 1), (0, 2), (-2, 1), (-1, -1)  # 5 from ring 2
    ]
    
    # Apply lattice rotation
    rot_rad = np.deg2rad(lattice_rotation)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    
    positions = _hex_lattice_positions(coords)
    positions = positions @ rot_matrix.T
    
    inner = np.zeros((12, 3))
    inner[:, 0] = positions[:, 0]
    inner[:, 1] = positions[:, 1]
    inner[:, 2] = 0.0
    
    # Compute minimum s
    s = _min_s_for_config(inner, outer_angle)
    
    # Check if valid
    outer = np.array([0.0, 0.0, outer_angle])
    if not _is_valid(inner, outer, s):
        return s + 50.0
    
    return s

def _objective_full(params):
    """Full optimization: 12*2 positions + outer_angle + 12*1 inner angles = 37 params.
    But we'll use a hybrid: optimize positions with constraints."""
    # This is the original approach but with better initialization
    cx_cy = params[:24].reshape(12, 2)
    inner_angles = params[24:36]
    outer_angle = params[36]
    
    inner = np.zeros((12, 3))
    inner[:, 0] = cx_cy[:, 0]
    inner[:, 1] = cx_cy[:, 1]
    inner[:, 2] = inner_angles
    
    outer = np.array([0.0, 0.0, outer_angle])
    
    # Compute minimum s for this configuration
    s = _min_s_for_config(inner, outer_angle)
    
    if not _is_valid(inner, outer, s):
        return s + 50.0
    
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s
    
    best_s = float('inf')
    best_inner = None
    best_outer = None
    
    def update_best(inner, outer, s):
        nonlocal best_s, best_inner, best_outer
        if s < best_s and _is_valid(inner, outer, s):
            best_s = s
            best_inner = inner.copy()
            best_outer = outer.copy()
    
    # Phase 1: Lattice-based optimization (fast, few params)
    # Try multiple lattice configurations
    lattice_configs = [
        # Config 1: center + ring1 + 5 from ring2
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
         (2,0), (1,1), (0,2), (-2,1), (-1,-1)],
        # Config 2: different 5 from ring2
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
         (2,0), (1,1), (0,2), (-1,2), (-2,1)],
        # Config 3: 
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
         (2,0), (1,1), (-1,2), (-2,1), (-2,0)],
    ]
    
    for config in lattice_configs:
        if time.time() > deadline - 60:
            break
        
        # Optimize outer_angle and lattice_rotation for this config
        def obj_lattice(params, cfg=config):
            outer_angle = params[0]
            lattice_rotation = params[1]
            
            rot_rad = np.deg2rad(lattice_rotation)
            cos_r = math.cos(rot_rad)
            sin_r = math.sin(rot_rad)
            rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            
            positions = _hex_lattice_positions(cfg)
            positions = positions @ rot_matrix.T
            
            inner = np.zeros((12, 3))
            inner[:, 0] = positions[:, 0]
            inner[:, 1] = positions[:, 1]
            inner[:, 2] = 0.0
            
            s = _min_s_for_config(inner, outer_angle)
            outer = np.array([0.0, 0.0, outer_angle])
            if not _is_valid(inner, outer, s):
                return s + 50.0
            return s
        
        try:
            res = differential_evolution(
                obj_lattice,
                bounds=[(-180, 180), (-180, 180)],
                maxiter=100,
                tol=1e-8,
                seed=42,
                polish=True,
                init='sobol'
            )
            if res.fun < best_s:
                outer_angle = res.x[0]
                lattice_rotation = res.x[1]
                
                rot_rad = np.deg2rad(lattice_rotation)
                cos_r = math.cos(rot_rad)
                sin_r = math.sin(rot_rad)
                rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
                
                positions = _hex_lattice_positions(config)
                positions = positions @ rot_matrix.T
                
                inner = np.zeros((12, 3))
                inner[:, 0] = positions[:, 0]
                inner[:, 1] = positions[:, 1]
                inner[:, 2] = 0.0
                
                s = _min_s_for_config(inner, outer_angle)
                outer = np.array([0.0, 0.0, outer_angle])
                update_best(inner, outer, s)
        except Exception:
            pass
    
    # Phase 2: Local refinement with full position optimization
    if best_inner is not None and time.time() < deadline - 30:
        # Use best as initial guess for local optimization
        def obj_refine(params):
            cx_cy = params[:24].reshape(12, 2)
            inner_angles = params[24:36]
            outer_angle = params[36]
            
            inner = np.zeros((12, 3))
            inner[:, 0] = cx_cy[:, 0]
            inner[:, 1] = cx_cy[:, 1]
            inner[:, 2] = inner_angles
            
            s = _min_s_for_config(inner, outer_angle)
            outer = np.array([0.0, 0.0, outer_angle])
            if not _is_valid(inner, outer, s):
                return s + 50.0
            return s
        
        x0 = np.concatenate([best_inner[:, :2].flatten(), best_inner[:, 2], [best_outer[2]]])
        
        try:
            res2 = differential_evolution(
                obj_refine,
                bounds=[(-5, 5)] * 24 + [(-180, 180)] * 12 + [(-180, 180)],
                maxiter=50,
                tol=1e-10,
                seed=123,
                polish=True,
                init='sobol'
            )
            if res2.fun < best_s:
                cx_cy = res2.x[:24].reshape(12, 2)
                inner_angles = res2.x[24:36]
                outer_angle = res2.x[36]
                
                inner = np.zeros((12, 3))
                inner[:, 0] = cx_cy[:, 0]
                inner[:, 1] = cx_cy[:, 1]
                inner[:, 2] = inner_angles
                
                s = _min_s_for_config(inner, outer_angle)
                outer = np.array([0.0, 0.0, outer_angle])
                update_best(inner, outer, s)
        except Exception:
            pass
    
    # Fallback if nothing found
    if best_inner is None:
        # Use seed-like solution
        inner = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.165, 0],
                [1.25, 2.165, 0],
                [-1.25, -2.165, 0],
                [1.25, -2.165, 0],
                [-3.75, 2.165, 0],
                [3.75, 2.165, 0],
                [-3.75, -2.165, 0],
                [3.75, -2.165, 0],
                [0, -4, 0],
            ]
        )
        outer = np.array([0, 0, 0])
        s = 8.0
        return inner, outer, s
    
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
