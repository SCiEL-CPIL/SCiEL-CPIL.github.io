import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Unit regular hexagon has inradius sqrt(3)/2, circumradius 1.
    # To pack disjoint unit hexagons in a lattice, the minimum center-to-center
    # distance in the "hexagonal lattice" arrangement is 2 * sqrt(3) / 2 = sqrt(3)
    # in one direction, and the lattice vectors are:
    #   a1 = (sqrt(3), 0)
    #   a2 = (sqrt(3)/2, 3/2)
    # The inradius (apothem) of a unit hexagon is sqrt(3)/2, so two adjacent
    # hexagons in this lattice are tangent (distance between centers = sqrt(3)
    # and the sum of apothems = 2 * sqrt(3)/2 = sqrt(3)).

    sqrt3 = np.sqrt(3)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3 / 2.0, 1.5])

    # We need N=11 hexagons. Place them in a compact arrangement.
    # Try to find a set of integer coefficients (i, j) such that the set of
    # 11 points is compact. A common compact set for 11 points in a hex lattice:
    # We can use a 4x3 grid minus corners, or a cross-like shape.
    # Let's generate all points in a bounding box and pick the 11 closest to origin.

    # Generate candidate lattice points
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * a1 + j * a2
            candidates.append((np.linalg.norm(p), i, j, p))

    # Sort by distance to origin
    candidates.sort(key=lambda x: x[0])

    # Pick the first N points
    selected = []
    for dist, i, j, p in candidates:
        if len(selected) >= N:
            break
        selected.append(p)

    # Compute the bounding hexagon.
    # The outer hexagon must contain all inner hexagons.
    # Each inner hexagon centered at p has extent: for any direction theta,
    # the farthest point is at distance 1 (circumradius) from p.
    # The outer hexagon of side length S, centered at c, rotated by angle phi,
    # contains a point q if for all 6 face normals n_k, (q - c) . n_k <= S * apothem_outer
    # where apothem_outer = S * sqrt(3)/2.
    # Actually, for a regular hexagon with side S, the apothem is S * sqrt(3)/2.
    # The containment condition: for each of the 6 directions (face normals),
    # max over all inner hexagon vertices of (v - c) . n_k <= S * sqrt(3)/2.
    # The vertices of an inner hexagon centered at p with rotation alpha are:
    #   p + R * (cos(alpha + k*60deg), sin(alpha + k*60deg)) for k=0..5

    # Since we can rotate the outer hexagon, let's try a few angles and find the
    # minimum S that works.

    def min_side_length_for_angle(points, outer_angle_deg):
        phi = np.radians(outer_angle_deg)
        # 6 face normals of the outer hexagon
        normals = []
        for k in range(6):
            angle = phi + np.radians(60 * k) + np.radians(30)
            # Wait, the face normal of a regular hexagon...
            # For a regular hexagon with vertices at angles 0, 60, 120, ...
            # the face between vertices at angle 30 and 90 has normal at angle 60.
            # Actually, the outward normal of the face opposite to vertex at angle theta
            # is at angle theta. Let me reconsider.
            # A regular hexagon centered at origin with a vertex at angle 0:
            # vertices at 0, 60, 120, 180, 240, 300 degrees.
            # The face between vertices at 30 and 90... no, vertices are at 0,60,...
            # The edge between vertex at 0 and vertex at 60 has midpoint direction at 30.
            # The outward normal of that edge is at angle 30.
            # So the 6 face normals are at 30, 90, 150, 210, 270, 330 degrees.
            normals.append(np.array([np.cos(angle + np.radians(30)), np.sin(angle + np.radians(30))]))
        
        # For each inner hexagon, compute the max projection on each normal
        max_proj = np.zeros(6)
        for p in points:
            # Inner hexagon vertices (rotation 0 for simplicity)
            for k in range(6):
                v_angle = np.radians(60 * k)
                v = p + np.array([np.cos(v_angle), np.sin(v_angle)])
                for m in range(6):
                    proj = np.dot(v, normals[m])
                    if proj > max_proj[m]:
                        max_proj[m] = proj
        
        # S * sqrt(3)/2 >= max_proj[m] for all m
        # So S >= max_proj[m] * 2/sqrt(3)
        S = np.max(max_proj) * 2.0 / sqrt3
        return S

    # Try different outer angles
    best_S = float('inf')
    best_angle = 0.0
    for angle_deg in range(0, 60, 5):
        S = min_side_length_for_angle(selected, angle_deg)
        if S < best_S:
            best_S = S
            best_angle = angle_deg

    # Refine around best angle
    for angle_deg in np.arange(best_angle - 5, best_angle + 5.1, 1.0):
        S = min_side_length_for_angle(selected, angle_deg)
        if S < best_S:
            best_S = S
            best_angle = angle_deg

    inner = []
    for p in selected:
        inner.append([float(p[0]), float(p[1]), 0.0])

    return inner, (0.0, 0.0), float(best_S), float(best_angle)
# EVOLVE-BLOCK-END
