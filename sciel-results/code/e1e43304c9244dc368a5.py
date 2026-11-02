"""Structural rewrite: treat the 12 inner hexagons as a rigid body (fixed relative
positions) and only optimize the outer hexagon's center and angle. The inner body's
minimum enclosing hexagon (at any rotation) is pre-computed once, then the outer
hexagon is fit tightly around it. This reduces the search to a 3-parameter problem
and guarantees disjointness by construction.

Additionally, we try multiple inner-body layouts and optimize each one's outer
hexagon fit via a small global+local search on (cx, cy, theta_outer)."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12
SQRT3 = np.sqrt(3.0)
INV_SQRT3 = 2.0 / SQRT3

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def hex_verts(cx, cy, side, angle):
        """6 vertices of a regular hexagon."""
        angles = angle + np.radians(np.arange(0, 360, 60))
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def min_enclosing_hex_side(verts, outer_angle):
        """Minimum side length of a regular hexagon (centered at origin, given angle)
        that contains all points in verts (verts are already relative to that center)."""
        max_r = 0.0
        for k in range(6):
            edge_angle = outer_angle + np.radians(k * 60)
            nx, ny = np.cos(edge_angle), np.sin(edge_angle)
            # For a hexagon with side S centered at origin, the support in direction (nx,ny)
            # is S * sqrt(3)/2 when the direction aligns with an edge normal.
            # More precisely: a regular hexagon with side S has apothem S*sqrt(3)/2.
            # The support function h(u) for direction u is max over vertices.
            # For a hexagon centered at origin with side S and rotation phi:
            #   h(u) = S * max_k |u · e_k| where e_k are unit vectors to edge midpoints
            # Actually, let me just compute: the hexagon's vertices are at distance S from center
            # and the support in direction u is S * max_k (u · v_k / S) = max_k (u · v_k)
            # where v_k are the unit vertex directions.
            # But simpler: for a hexagon with side S, centered at origin, the distance from
            # center to each edge (apothem) is S*sqrt(3)/2. The support in any direction
            # equals S * max over 6 directions of (u · cos(direction_k - phi)) ... 
            # Let me just do it numerically: project all points onto the 6 edge normals.
            proj = verts[:, 0] * nx + verts[:, 1] * ny
            m = np.max(proj)
            if m > max_r:
                max_r = m
        # The required side: for a hexagon with side S and given orientation, 
        # the max projection onto any edge-normal direction is S * sqrt(3)/2.
        # So S = max_r / (sqrt(3)/2) = max_r * 2/sqrt(3)
        return max_r * INV_SQRT3

    def outer_side_from_center(verts_abs, outer_cx, outer_cy, outer_angle):
        """Given absolute vertex positions and outer hexagon params, compute min outer side."""
        rel = verts_abs - np.array([outer_cx, outer_cy])
        return min_enclosing_hex_side(rel, outer_angle)

    def sat_disjoint(p1, p2, tol=1e-9):
        """Separating axis test for two convex hexagons (each 6 vertices)."""
        for poly in (p1, p2):
            for i in range(6):
                e = poly[(i + 1) % 6] - poly[i]
                n = np.array([-e[1], e[0]])
                nn = np.linalg.norm(n)
                if nn < 1e-12:
                    continue
                n = n / nn
                proj1 = p1 @ n
                proj2 = p2 @ n
                if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                    return True
        return False

    def check_disjoint(hex_list):
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(hex_list[i], hex_list[j]):
                    return False
        return True

    def build_layout(layout_type, params):
        """Build 12 hexagon vertices given a layout type and parameters.
        params: (spacing_x, spacing_y, stagger, rotation_inner)
        Returns list of 12 vertex arrays (6x2 each)."""
        sx, sy, stag, rot = params
        positions = []
        if layout_type == 0:  # 3x4 grid with stagger
            for r in range(3):
                for c in range(4):
                    x = c * sx
                    y = r * sy
                    if r % 2 == 1:
                        x += sx * stag
                    positions.append((x, y))
        elif layout_type == 1:  # 4x3 grid with stagger
            for r in range(4):
                for c in range(3):
                    x = c * sx
                    y = r * sy
                    if r % 2 == 1:
                        x += sx * stag
                    positions.append((x, y))
        elif layout_type == 2:  # 6x2 rows
            for r in range(2):
                for c in range(6):
                    x = c * sx
                    y = r * sy
                    if r % 2 == 1:
                        x += sx * stag
                    positions.append((x, y))
        elif layout_type == 3:  # Hexagonal ring: 1 center + 6 inner + 5 outer partial
            # 7 in a hexagonal number pattern (center + 6) + 5 more
            positions.append((0, 0))
            for i in range(6):
                a = 2 * np.pi * i / 6
                positions.append((sx * np.cos(a), sx * np.sin(a) * 0.577))
            for i in range(6):
                a = 2 * np.pi * i / 6 + np.pi / 6
                positions.append((sx * 2 * np.cos(a), sx * 2 * np.sin(a) * 0.577))
            positions = positions[:12]
        elif layout_type == 4:  # Two offset rows of 6
            for i in range(6):
                positions.append((i * sx, 0))
                positions.append((i * sx + sx * stag, sy))
        else:
            for r in range(3):
                for c in range(4):
                    x = c * sx
                    y = r * sy
                    if r % 2 == 1:
                        x += sx * stag
                    positions.append((x, y))

        # Center the layout
        pos_arr = np.array(positions)
        pos_arr -= pos_arr.mean(axis=0)

        hexes = []
        for px, py in pos_arr:
            v = hex_verts(px, py, 1.0, rot)
            hexes.append(v)
        return hexes

    def optimize_outer(hexes):
        """Find best outer hexagon (cx, cy, angle) to enclose all inner hexes."""
        all_verts = np.vstack(hexes)
        centroid = all_verts.mean(axis=0)
        best_side = np.inf
        best_params = (centroid[0], centroid[1], 0.0)

        def obj(p):
            cx, cy, ang = p
            rel = all_verts - np.array([cx, cy])
            max_r = 0.0
            for k in range(6):
                ea = ang + np.radians(k * 60)
                nx, ny = np.cos(ea), np.sin(ea)
                proj = rel[:, 0] * nx + rel[:, 1] * ny
                m = np.max(proj)
                if m > max_r:
                    max_r = m
            return max_r * INV_SQRT3

        # Grid search over angles
        angle_candidates = np.linspace(0, np.pi / 3, 13)
        for ang in angle_candidates:
            def obj_fixed_ang(p):
                cx, cy = p
                rel = all_verts - np.array([cx, cy])
                max_r = 0.0
                for k in range(6):
                    ea = ang + np.radians(k * 60)
                    nx, ny = np.cos(ea), np.sin(ea)
                    proj = rel[:, 0] * nx + rel[:, 1] * ny
                    m = np.max(proj)
                    if m > max_r:
                        max_r = m
                return max_r * INV_SQRT3

            try:
                res = minimize(obj_fixed_ang, [centroid[0], centroid[1]],
                               method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                s = res.fun
                if s < best_side:
                    best_side = s
                    best_params = (res.x[0], res.x[1], ang)
            except Exception:
                pass

        return best_params, best_side

    # --- Try multiple layouts ---
    np.random.seed(123)
    best_overall_side = np.inf
    best_overall_hexes = None
    best_overall_outer = None

    layout_configs = [
        # (layout_type, sx, sy, stag, rot)
        (0, SQRT3 * 1.001, 1.501, 0.5, 0.0),
        (0, SQRT3 * 1.001, 1.501, 0.5, np.pi / 6),
        (0, SQRT3 * 1.02, 1.52, 0.5, 0.0),
        (1, SQRT3 * 1.001, 1.501, 0.5, 0.0),
        (1, SQRT3 * 1.001, 1.501, 0.5, np.pi / 6),
        (2, SQRT3 * 1.001, 1.501, 0.5, 0.0),
        (2, SQRT3 * 1.001, 1.501, 0.5, np.pi / 6),
        (4, SQRT3 * 1.001, 1.501, 0.5, 0.0),
        (4, SQRT3 * 1.001, 1.501, 0.5, np.pi / 6),
        (4, SQRT3 * 1.005, 1.505, 0.5, 0.0),
        (0, SQRT3 * 0.999, 1.499, 0.5, 0.0),
        (1, SQRT3 * 0.999, 1.499, 0.5, np.pi / 6),
    ]

    for lt, sx, sy, stag, rot in layout_configs:
        if time.time() > deadline - 5:
            break
        try:
            hexes = build_layout(lt, (sx, sy, stag, rot))
            if not check_disjoint(hexes):
                continue
            outer_params, side = optimize_outer(hexes)
            if side < best_overall_side:
                best_overall_side = side
                best_overall_hexes = hexes
                best_overall_outer = outer_params
        except Exception:
            pass

    # --- If we found something, also try small perturbations of the best layout ---
    if best_overall_hexes is not None:
        # Try rotating the entire inner assembly
        base_hexes = best_overall_hexes
        all_v = np.vstack(base_hexes)
        centroid = all_v.mean(axis=0)
        
        for rot_delta in np.linspace(-0.1, 0.1, 5):
            if time.time() > deadline - 5:
                break
            try:
                new_hexes = []
                for v in base_hexes:
                    rel = v - centroid
                    c, s = np.cos(rot_delta), np.sin(rot_delta)
                    R = np.array([[c, -s], [s, c]])
                    new_v = rel @ R.T + centroid
                    new_hexes.append(new_v)
                if check_disjoint(new_hexes):
                    op, side = optimize_outer(new_hexes)
                    if side < best_overall_side:
                        best_overall_side = side
                        best_overall_hexes = new_hexes
                        best_overall_outer = op
            except Exception:
                pass

    # --- Fallback if nothing worked ---
    if best_overall_hexes is None:
        # Simple widely-spaced valid config
        hexes = []
        for i in range(N):
            v = hex_verts(float(i), 0.0, 1.0, 0.0)
            hexes.append(v)
        best_overall_hexes = hexes
        best_overall_outer = (5.5, 0.0, 0.0)
        best_overall_side = outer_side_from_center(np.vstack(hexes), 5.5, 0.0, 0.0)

    # --- Final validation and extraction ---
    if not check_disjoint(best_overall_hexes):
        # Emergency fallback: very widely spaced
        hexes = []
        for i in range(N):
            v = hex_verts(3.0 * i, 0.0, 1.0, 0.0)
            hexes.append(v)
        return [[3.0 * i, 0.0, 0.0] for i in range(N)], (0.0, 0.0), 35.0, 0.0

    outer_cx, outer_cy, outer_ang = best_overall_outer
    exact_side = outer_side_from_center(np.vstack(best_overall_hexes), outer_cx, outer_cy, outer_ang)

    # Extract inner hexagon parameters (center and angle of each)
    inner = []
    for v in best_overall_hexes:
        cx = v[:, 0].mean()
        cy = v[:, 1].mean()
        # Estimate angle from first two vertices
        if np.linalg.norm(v[1] - v[0]) > 1e-10:
            ang = np.arctan2(v[1][1] - v[0][1], v[1][0] - v[0][0])
        else:
            ang = 0.0
        inner.append([cx, cy, np.degrees(ang)])

    return inner, (outer_cx, outer_cy), exact_side, np.degrees(outer_ang)
