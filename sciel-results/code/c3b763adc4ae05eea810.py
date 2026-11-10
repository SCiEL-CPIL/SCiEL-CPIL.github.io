# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 620.0
_EPS = 1e-9


def _hex_verts(cx, cy, d, a_deg):
    a = np.radians(a_deg)
    th = a + np.radians(60.0) * np.arange(6)
    return np.column_stack([cx + d * np.cos(th), cy + d * np.sin(th)])


def _min_containing_hex_side(pts):
    """Smallest regular hexagon side that contains all pts (centered at origin,
    with the hexagon orientation free to rotate)."""
    # For a regular hexagon of side s, the support function along direction u
    # is s*(|cos u| + |sin u|/sqrt(3)) for a hexagon with a vertex at angle 0.
    # We need the smallest s such that for all 6 edge normals, the max projection
    # of pts (relative to center) fits.
    # A regular hexagon of side s, centered at origin, with one vertex at angle a,
    # has edge normals at a + 30 + 60*i. The distance from center to each edge
    # (the inradius) is s * sqrt(3)/2.
    # The half-width of the hexagon along a normal direction n (at angle phi)
    # is: s * (cos(phi - a) * 0.5 + |sin(phi - a)| * sqrt(3)/2) ... 
    # Actually the support function of a regular hexagon (inradius r = s*sqrt(3)/2)
    # along direction at angle phi relative to a vertex direction a:
    # h(phi) = r * max over 6 vertices of cos(phi - v_i)  ... no.
    # Support function of regular hexagon with inradius r, vertex directions a+60i:
    # h(phi) = r * max(1, cos(phi - (a+30+60i))) for the 6 edge normals... 
    # Let's use: a regular hexagon = intersection of 6 half-planes. Edge i has
    # normal n_i at angle a + 30 + 60*i (pointing outward), and offset r = s*sqrt(3)/2.
    # So point p is inside iff dot(p, n_i) <= r for all i.
    # For all points to be inside: max_p dot(p, n_i) <= r for all i.
    # So we need r >= max_i (max_p dot(p, n_i)). And r = s*sqrt(3)/2, so
    # s >= (2/sqrt(3)) * max_i (max_p dot(p, n_i)).
    # We can rotate the hexagon (change a) to minimize this.
    # max_p dot(p, n_i) = max over points of dot(p, n_i). This is the support
    # function of the point set in direction n_i.
    # Let's compute for a given a: for each of 6 directions, the max projection.
    def s_for_angle(a):
        normals = a + np.radians(30.0) + np.radians(60.0) * np.arange(6)
        ns = np.column_stack([np.cos(normals), np.sin(normals)])
        # projections of all points onto each normal
        # pts is (N,2), ns is (6,2)
        proj = pts @ ns.T  # (N,6)
        maxproj = proj.max(axis=0)  # (6,)
        r_needed = maxproj.max()
        s = (2.0 / np.sqrt(3.0)) * r_needed
        return s
    # Optimize over angle a in [0, 60)
    best = 1e18
    best_a = 0.0
    # Coarse scan then refine
    for a0 in np.linspace(0, np.pi/3, 361, endpoint=False):
        s = s_for_angle(a0)
        if s < best:
            best = s
            best_a = a0
    # Refine around best_a
    def obj(aa):
        return s_for_angle(aa)
    res = minimize(obj, [best_a], method='Nelder-Mead',
                   options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000})
    if res.fun < best:
        best = res.fun
        best_a = res.x[0]
    return best, best_a


def _objective(params, base_pts, base_angles_deg):
    """params: [dx, dy, rot_deg, scale] applied to base points.
    Returns the containing hexagon side length (to minimize)."""
    dx, dy, rot_deg, scale = params
    rot = np.radians(rot_deg)
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]])
    pts = base_pts * scale
    pts = pts @ R.T
    pts = pts + np.array([dx, dy])
    # center at centroid for containing hex (optimal centering is centroid of
    # the point set's convex hull; using centroid of all points is a good start,
    # but the true optimal center minimizes max projection. Use centroid.)
    center = pts.mean(axis=0)
    pts_c = pts - center
    side, _ = _min_containing_hex_side(pts_c)
    return side


def hexagon_packing_12():
    t0 = time.time()

    # 12 unit hexagons on a hexagonal lattice. Lattice spacing for non-overlapping
    # unit hexagons: adjacent hexagons (sharing an edge) have centers at distance
    # sqrt(3) apart (since inradius = sqrt(3)/2, two inradii = sqrt(3)).
    # Use axial coordinates (q, r) in the hex grid; positions:
    # x = sqrt(3) * (q + r/2), y = 1.5 * r  (for pointy... let's use flat-top)
    # For flat-top hexagons with side 1:
    #   width = 2, height = sqrt(3)
    #   horizontal spacing between columns = 1.5 (x), vertical = sqrt(3)
    #   adjacent columns offset by sqrt(3)/2
    # Let me just place 12 hexagons in a compact hexagonal cluster shape.
    # A hexagonal cluster of 12: center + ring of 6 + partial second ring.
    # Actually 1 = center, 7 = center+ring1, 19 = center+ring1+ring2.
    # 12 is not a perfect hexagonal number. Let's use a shape:
    # rows of hexagons. Let me define positions directly in a tight arrangement.

    # Use a "flower" of 7 (center + 6 around) plus 5 more to make 12.
    # The 7 hexagons form a perfect hexagon of side 2 (in hex units).
    # Add 5 more attached to the outside.

    # Let me define in axial coords and convert. Flat-top, side=1:
    # pos(q, r) = ( 1.5*q, sqrt(3)*(r + q/2) )
    sqrt3 = np.sqrt(3.0)

    # 12 axial coordinates forming a compact shape
    # Center at (0,0). Ring 1: 6 neighbors. That's 7.
    # Add 5 more: extend one side.
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1 (7 total)
        (2, 0), (2, -1), (1, 1), (0, 2), (-1, 2),  # 5 more (12 total)
    ]
    base_pts = np.array([
        [1.5 * q, sqrt3 * (r + q / 2.0)] for q, r in axial
    ])
    base_angles_deg = np.zeros(12)

    # Optimize the affine transform: translate, rotate, scale
    # Start with identity
    x0 = np.array([0.0, 0.0, 0.0, 1.0])

    best_side = 1e18
    best_params = x0.copy()

    # Nelder-Mead on the 4 transform parameters
    res = minimize(_objective, x0, args=(base_pts, base_angles_deg),
                   method='Nelder-Mead',
                   options={'xatol': 1e-9, 'fatol': 1e-11, 'maxiter': 50000,
                            'adaptive': True})
    if res.fun < best_side:
        best_side = res.fun
        best_params = res.x

    # Try a few random restarts to escape local minima
    rng = np.random.default_rng(42)
    for restart in range(20):
        if time.time() > _DEADLINE - 30.0:
            break
        x0r = np.array([
            rng.uniform(-3, 3),
            rng.uniform(-3, 3),
            rng.uniform(-30, 30),
            rng.uniform(0.7, 1.3),
        ])
        res = minimize(_objective, x0r, args=(base_pts, base_angles_deg),
                       method='Nelder-Mead',
                       options={'xatol': 1e-9, 'fatol': 1e-11, 'maxiter': 20000,
                                'adaptive': True})
        if res.fun < best_side:
            best_side = res.fun
            best_params = res.x

    dx, dy, rot_deg, scale = best_params
    rot = np.radians(rot_deg)
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]])

    inner_pts = base_pts * scale
    inner_pts = inner_pts @ R.T
    inner_pts = inner_pts + np.array([dx, dy])
    center = inner_pts.mean(axis=0)
    inner_pts = inner_pts - center  # shift so centroid at origin

    # Final containing hexagon
    side, hex_a = _min_containing_hex_side(inner_pts)
    # hex_a is the vertex angle of the containing hexagon

    inner_hex_data = np.column_stack([inner_pts, base_angles_deg * 0 + rot_deg])
    # Actually the inner hexagons should have the same rotation as the transform
    inner_hex_data = np.column_stack([inner_pts[:, 0], inner_pts[:, 1],
                                      np.full(12, rot_deg)])
    outer_hex_data = np.array([0.0, 0.0, np.degrees(hex_a)])
    outer_hex_side_length = float(side)

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
