"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _icosahedral_symmetries():
    """Return the 60 rotation matrices of the icosahedral group."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 icosahedron vertices
    v = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    # Generate all 60 rotations by permuting axes and flipping signs,
    # then keep only those that map the vertex set to itself.
    mats = []
    for perm in [(0,1,2),(1,2,0),(2,0,1),(0,2,1),(2,1,0),(1,0,2)]:
        for s in [(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),
                  (-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1)]:
            M = np.zeros((3,3))
            for r in range(3):
                M[r, perm[r]] = s[r]
            det = np.linalg.det(M)
            if abs(det - 1.0) > 1e-9:
                continue
            # Check it maps vertex set to itself
            Mv = M @ v.T
            Mv = Mv / np.linalg.norm(Mv, axis=0, keepdims=True)
            ok = True
            for j in range(12):
                d = np.min(np.sum((Mv[:, j] - v.T) ** 2, axis=1))
                if d > 1e-8:
                    ok = False
                    break
            if ok:
                # deduplicate
                dup = False
                for existing in mats:
                    if np.allclose(M, existing, atol=1e-9):
                        dup = True
                        break
                if not dup:
                    mats.append(M)
    return np.array(mats)


def _covering_radius(pts):
    """Compute the covering radius in radians: max over sphere of min angular distance to pts.
    This equals the max circumradius of the convex hull facets."""
    from scipy.spatial import ConvexHull
    pts = np.asarray(pts, dtype=float)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    # For each facet (simplex), compute the circumradius of the spherical cap
    # The facet points lie on the unit sphere; the cap center is the normalized
    # centroid direction, and the radius is the max angle to the facet vertices.
    max_r = 0.0
    for simplex in hull.simplices:
        fpts = pts[simplex]
        # centroid direction
        c = fpts.mean(axis=0)
        nc = np.linalg.norm(c)
        if nc < 1e-12:
            # degenerate, skip
            continue
        c = c / nc
        # angle from c to each facet vertex
        dots = np.clip(fpts @ c, -1.0, 1.0)
        angles = np.arccos(dots)
        r = np.max(angles)
        if r > max_r:
            max_r = r
    return max_r


def _covering_radius_grad(pts, grad):
    """Compute covering radius and its gradient w.r.t. pts (on the sphere).
    Uses the max facet circumradius. For the active facet, the gradient is
    the direction that would most reduce the cap radius."""
    from scipy.spatial import ConvexHull
    pts = np.asarray(pts, dtype=float)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, np.zeros_like(pts)
    max_r = 0.0
    active_simplex = None
    active_c = None
    for simplex in hull.simplices:
        fpts = pts[simplex]
        c = fpts.mean(axis=0)
        nc = np.linalg.norm(c)
        if nc < 1e-12:
            continue
        c = c / nc
        dots = np.clip(fpts @ c, -1.0, 1.0)
        angles = np.arccos(dots)
        r = np.max(angles)
        if r > max_r:
            max_r = r
            active_simplex = simplex
            active_c = c
    # Gradient: move facet vertices toward the cap center c to reduce radius
    g = np.zeros_like(pts)
    if active_simplex is not None:
        for idx in active_simplex:
            p = pts[idx]
            # gradient of arccos(p.c) w.r.t. p is -c / sqrt(1-(p.c)^2)
            dot = np.clip(p @ active_c, -1.0, 1.0)
            denom = np.sqrt(max(1e-12, 1.0 - dot * dot))
            g[idx] = -active_c / denom
    return max_r, g


def _make_icosahedral_config():
    """Construct the 50-point icosahedral-symmetric configuration."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    pts = []
    # 2 poles
    pts.append([0.0, 0.0, 1.0])
    pts.append([0.0, 0.0, -1.0])
    # 12 icosahedron vertices
    ico = [
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ]
    for v in ico:
        v = np.array(v, dtype=float)
        pts.append(v / np.linalg.norm(v))
    # 30 edge midpoints: the icosahedron has 30 edges
    # Edges connect vertices that are closest. Find all pairs at min distance.
    ico_arr = np.array(ico, dtype=float)
    ico_arr = ico_arr / np.linalg.norm(ico_arr, axis=1, keepdims=True)
    # edge length in angle: arccos of min dot
    dots = ico_arr @ ico_arr.T
    np.fill_diagonal(dots, -2)
    min_dot = np.max(dots)  # closest neighbors have max dot
    edge_midpoints = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dots[i, j] - min_dot) < 1e-9:
                m = (ico_arr[i] + ico_arr[j]) / 2.0
                m = m / np.linalg.norm(m)
                edge_midpoints.append(m)
    assert len(edge_midpoints) == 30, f"Expected 30 edges, got {len(edge_midpoints)}"
    for m in edge_midpoints:
        pts.append(m)
    # 6 face centers: the icosahedron has 20 faces, but we need 6 points with
    # icosahedral symmetry. The 6 face-centers of the DUAL octahedron? No.
    # Actually 6 = vertices of an octahedron. In the HSS 50-point construction,
    # the 6 points are the face centers of the icosahedron's 20 faces grouped
    # by symmetry, OR they are the 6 vertices of the dual octahedron.
    # The dual of the icosahedron is the dodecahedron (20 vertices), not 6.
    # 6 points with icosahedral symmetry: the vertices of an octahedron.
    # But an octahedron is NOT a subset of the icosahedral orbit structure alone.
    # Let me reconsider: 2 + 12 + 30 + 6 = 50. The 6 could be the centers of
    # the 6 pairs of opposite faces? No, icosahedron has 20 faces = 10 pairs.
    # Alternative: the 6 points are the face-centers of the 20 faces, but only
    # 6 distinct directions under some subgroup? 
    # Actually, in the HSS construction, the 50 points are:
    # 2 poles + 12 icosahedron vertices + 30 edge midpoints + 6 face centers
    # where the 6 face centers are the centers of the 6 "great" faces or
    # the 6 vertices of the dual octahedron inscribed.
    # Let me use the 6 vertices of a regular octahedron: (±1,0,0),(0,±1,0),(0,0,±1)
    # But these may not align with the icosahedral symmetry of the rest.
    # Actually the icosahedral group DOES contain the octahedral symmetry as a
    # subgroup? No, the icosahedral group is larger. The 6 octahedron vertices
    # are NOT invariant under the full icosahedral group.
    # Let me reconsider the HSS construction. Perhaps the 6 points are the
    # centers of the 20 triangular faces, but since opposite faces have the
    # same center direction (antipodal), that gives 10 points, not 6.
    # Hmm. Let me try: the 6 points might be at the face centers of the
    # icosahedron, but the icosahedron has 20 faces. Under the icosahedral
    # group, the 20 face centers form a single orbit (the dodecahedron vertices).
    # So 6 is not a face-center orbit.
    # Let me just use a different decomposition. Maybe:
    # 2 + 12 + 30 + 6 where the 6 are something else.
    # For now, let me place 6 points at the octahedron vertices and let
    # the optimizer fix them.
    octa = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
    for v in octa:
        v = np.array(v, dtype=float)
        pts.append(v / np.linalg.norm(v))
    pts = np.array(pts, dtype=float)
    assert len(pts) == 50
    return pts


def _gradient_descent(pts, max_time=100.0):
    """Optimize the covering radius using gradient descent on the sphere."""
    pts = np.array(pts, dtype=float)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    best_pts = pts.copy()
    best_r = _covering_radius(pts)
    start = time.time()
    lr = 0.01
    for it in range(5000):
        if time.time() - start > max_time:
            break
        r, g = _covering_radius_grad(pts, None)
        if r < best_r:
            best_r = r
            best_pts = pts.copy()
        # Normalize gradient
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
        g = g / gn
        # Move in negative gradient direction, project back to sphere
        pts_new = pts - lr * g
        pts_new = pts_new / np.linalg.norm(pts_new, axis=1, keepdims=True)
        # Check if it improved
        r_new = _covering_radius(pts_new)
        if r_new < r:
            pts = pts_new
            lr = min(lr * 1.05, 0.1)
        else:
            lr = max(lr * 0.5, 1e-4)
    return best_pts, best_r


def _lloyd_like(pts, max_time=30.0):
    """A Lloyd-like iteration: for each point, move it to the center of its
    Voronoi cell (the point on the sphere that minimizes the max distance
    to the cell boundary). This is a heuristic."""
    pts = np.array(pts, dtype=float)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    start = time.time()
    for it in range(100):
        if time.time() - start > max_time:
            break
        # For each point, find the farthest boundary point of its Voronoi cell
        # and move toward it. Simplified: move each point to reduce the max
        # distance to its neighbors.
        new_pts = pts.copy()
        for i in range(len(pts)):
            # Find the 3-4 nearest neighbors
            dots = pts @ pts[i]
            dots[i] = -2
            # The Voronoi boundary is where two points are equidistant
            # Approximate: find the direction of the farthest point in the cell
            # which is roughly the direction opposite to the centroid of neighbors
            # that are close
            close = np.where(dots > 0.0)[0]
            if len(close) > 0:
                # Move away from the centroid of close neighbors
                centroid = pts[close].mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                # Move in the direction away from centroid
                new_pts[i] = pts[i] + 0.01 * (-centroid)
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        r_new = _covering_radius(new_pts)
        r_old = _covering_radius(pts)
        if r_new < r_old:
            pts = new_pts
        else:
            break
    return pts


def construct():
    start = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Start from the icosahedral-symmetric configuration
    pts = _make_icosahedral_config()
    best_pts = pts.copy()
    best_r = _covering_radius(pts)

    # Gradient descent polish
    if time.time() - start < deadline - 20:
        pts2, r2 = _gradient_descent(pts, max_time=deadline - 25 - (time.time() - start))
        if r2 < best_r:
            best_r = r2
            best_pts = pts2

    # Lloyd-like refinement
    if time.time() - start < deadline - 10:
        pts3 = _lloyd_like(best_pts, max_time=deadline - 15 - (time.time() - start))
        r3 = _covering_radius(pts3)
        if r3 < best_r:
            best_r = r3
            best_pts = pts3

    # Final gradient descent on the best
    if time.time() - start < deadline - 5:
        pts4, r4 = _gradient_descent(best_pts, max_time=deadline - 8 - (time.time() - start))
        if r4 < best_r:
            best_r = r4
            best_pts = pts4

    # Ensure valid output
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
# EVOLVE-BLOCK-END
