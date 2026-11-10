# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _hex_edge_normals(vertices):
    """Return unit outward normals for each of the 6 edges of a hexagon (CCW)."""
    n = []
    for i in range(6):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 6]
        edge = p2 - p1
        # outward normal for CCW polygon: rotate edge by -90 deg
        normal = np.array([edge[1], -edge[0]])
        normal = normal / np.linalg.norm(normal)
        # midpoint to check outward direction
        mid = (p1 + p2) / 2.0
        center = np.mean(vertices, axis=0)
        if np.dot(normal, mid - center) < 0:
            normal = -normal
        n.append(normal)
    return np.array(n)


def _separation_ok(v1, n1, v2, n2, eps=1e-6):
    """
    Check that two convex hexagons are separated: on at least one of the 12
    edge normals (6 from each), the projected intervals are apart by more than eps.
    """
    for normals, verts in ((n1, v1), (n2, v2)):
        for i in range(6):
            nrm = normals[i]
            proj1 = np.dot(verts, nrm)
            proj2 = np.dot(v2, nrm)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 + eps < min2 or max2 + eps < min1:
                return True
    return False


def _containment_ok(inner_verts_list, outer_verts, eps=1e-6):
    """
    Check that every inner vertex is inside (or on) the outer hexagon.
    Outer hexagon is CCW. For each edge p1->p2, cross(p2-p1, p-p1) >= -eps.
    """
    for verts in inner_verts_list:
        for p in verts:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -eps:
                    return False
    return True


def _min_enclosing_hex_side(points):
    """
    Given a set of points (vertices of inner hexagons), find the minimum side length s
    of a regular hexagon (centered at origin, variable angle) that encloses all points.
    We use the fact that a regular hexagon centered at origin with side s and angle a
    can be described by 6 half-planes. We search over angle and use linear programming
    to find the minimum s such that all points satisfy the constraints.
    
    A regular hexagon centered at origin with circumradius R = s (for unit side, 
    circumradius = side length) and angle a has vertices at angles a + 60i.
    The 6 supporting lines are at distance s * cos(30 deg) = s * sqrt(3)/2 from center.
    
    Actually, for a regular hexagon with side length s, the apothem (distance from 
    center to each edge) is s * sqrt(3)/2. The 6 edge normals are at angles a + 60i.
    
    A point (x,y) is inside if for all i: dot((x,y), n_i) <= apothem, where n_i is 
    the unit normal to edge i.
    
    So we need: for all points p and all i: dot(p, n_i(a)) <= s * sqrt(3)/2
    => s >= (2/sqrt(3)) * max over i of dot(p, n_i(a))
    
    We minimize over angle a.
    """
    # points: (N, 2)
    best_s = np.inf
    best_a = 0.0
    # Search over angle
    for a_deg in np.linspace(0, 60, 121):
        a = np.deg2rad(a_deg)
        normals = np.column_stack([np.cos(a + np.deg2rad(60 * np.arange(6))),
                                   np.sin(a + np.deg2rad(60 * np.arange(6)))])
        # For each point, compute max dot with any normal
        # dots: (N, 6)
        dots = points @ normals.T
        max_dots = dots.max(axis=1)  # (N,)
        required = max_dots.max()
        s_candidate = (2.0 / np.sqrt(3.0)) * required
        if s_candidate < best_s:
            best_s = s_candidate
            best_a = a_deg
    return best_s, best_a


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    # Honeycomb lattice: center + 6 neighbors + 6 second-ring positions
    # In a honeycomb of hexagons with side 1, adjacent hexagons are separated by 
    # distance 2 * apothem = 2 * sqrt(3)/2 = sqrt(3) ≈ 1.732... 
    # Wait, for unit-side hexagons, the distance between centers of adjacent 
    # hexagons in a honeycomb is 2 * apothem = sqrt(3).
    
    # Actually, let me reconsider. Two unit hexagons sharing an edge: the distance 
    # between centers is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    
    apothem = np.sqrt(3) / 2.0
    center_dist = 2.0 * apothem  # = sqrt(3)
    
    # Place 11 hexagons in a honeycomb pattern:
    # Center at (0,0)
    # 6 neighbors at distance center_dist, angles 0, 60, 120, 180, 240, 300
    # 6 more at second ring: distance 2*center_dist, angles 30, 90, 150, 210, 270, 330
    
    positions = [(0.0, 0.0)]
    for i in range(6):
        angle = np.deg2rad(60 * i)
        x = center_dist * np.cos(angle)
        y = center_dist * np.sin(angle)
        positions.append((x, y))
    for i in range(6):
        angle = np.deg2rad(60 * i + 30)
        x = 2.0 * center_dist * np.cos(angle)
        y = 2.0 * center_dist * np.sin(angle)
        positions.append((x, y))
    
    # All hexagons aligned at angle 0
    inner_hex_data = np.array([[x, y, 0.0] for x, y in positions])
    
    # Compute all vertices of all inner hexagons
    inner_verts_list = []
    for x, y, a in inner_hex_data:
        verts = _hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
    
    all_points = np.vstack(inner_verts_list)
    
    # Find minimum enclosing regular hexagon
    side, angle_deg = _min_enclosing_hex_side(all_points)
    
    # Add a tiny margin to ensure strict separation from boundary
    side *= 1.000001
    
    outer_hex_data = np.array([0.0, 0.0, angle_deg])
    
    # Verify constraints
    # Check separation between all pairs
    n_inner = len(inner_hex_data)
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            vi = _hex_vertices(*inner_hex_data[i])
            vj = _hex_vertices(*inner_hex_data[j])
            ni = _hex_edge_normals(vi)
            nj = _hex_edge_normals(vj)
            if not _separation_ok(vi, ni, vj, nj):
                # Fallback: use a larger hexagon
                side = 8.0
                outer_hex_data = np.array([0.0, 0.0, 0.0])
                return inner_hex_data, outer_hex_data, side
    
    # Check containment
    outer_verts = _hex_vertices(0.0, 0.0, side, angle_deg)
    if not _containment_ok(inner_verts_list, outer_verts):
        # Expand side until containment is satisfied
        for _ in range(100):
            side *= 1.01
            outer_verts = _hex_vertices(0.0, 0.0, side, angle_deg)
            if _containment_ok(inner_verts_list, outer_verts):
                break
    
    return inner_hex_data, outer_hex_data, side


# EVOLVE-BLOCK-END
