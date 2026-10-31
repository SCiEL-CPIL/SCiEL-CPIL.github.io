"""Circle packing in unit square using Voronoi-based radii optimization."""
import numpy as np
from scipy.spatial import Voronoi
import time

N = 26
DEADLINE = 140  # seconds, safety margin before 150s kill

def _compute_voronoi_radii(points):
    """Compute the inradius (max circle radius) for each Voronoi cell,
    clipped to the unit square. Returns array of radii of shape (n,)."""
    n = len(points)
    if n < 3:
        # Degenerate case: just compute min distance to boundary and other points
        radii = np.zeros(n)
        for i in range(n):
            # Distance to square boundary
            d_boundary = min(points[i][0], 1 - points[i][0], points[i][1], 1 - points[i][1])
            # Distance to other points
            d_other = np.inf
            for j in range(n):
                if i != j:
                    d_other = min(d_other, np.linalg.norm(points[i] - points[j]))
            radii[i] = min(d_boundary, d_other / 2)
        return radii

    # Add boundary points to ensure Voronoi diagram covers the square
    # We need to handle the bounded domain [0,1]^2
    # Strategy: compute Voronoi of the points, then clip each cell to the square
    # and compute the inradius of the clipped polygon.

    try:
        vor = Voronoi(points)
    except Exception:
        # Fallback: compute min distance to boundary and half min distance to neighbors
        radii = np.zeros(n)
        for i in range(n):
            d_boundary = min(points[i][0], 1 - points[i][0], points[i][1], 1 - points[i][1])
            d_other = np.inf
            for j in range(n):
                if i != j:
                    d_other = min(d_other, np.linalg.norm(points[i] - points[j]))
            radii[i] = min(d_boundary, d_other / 2)
        return radii

    radii = np.zeros(n)
    for i in range(n):
        # Get the Voronoi region for point i
        # vor.regions[vor.point_region[i]] gives vertex indices
        region_idx = vor.point_region[i]
        if region_idx == -1:
            # Point is outside the convex hull, use fallback
            d_boundary = min(points[i][0], 1 - points[i][0], points[i][1], 1 - points[i][1])
            d_other = np.inf
            for j in range(n):
                if i != j:
                    d_other = min(d_other, np.linalg.norm(points[i] - points[j]))
            radii[i] = min(d_boundary, d_other / 2)
            continue

        region = vor.regions[region_idx]
        if -1 in region:
            # Unbounded region: need to clip to square
            # Get the finite vertices
            finite_verts = [vor.vertices[v] for v in region if v >= 0]
            if not finite_verts:
                # All vertices are at infinity, use fallback
                d_boundary = min(points[i][0], 1 - points[i][0], points[i][1], 1 - points[i][1])
                d_other = np.inf
                for j in range(n):
                    if i != j:
                        d_other = min(d_other, np.linalg.norm(points[i] - points[j]))
                radii[i] = min(d_boundary, d_other / 2)
                continue

            # For unbounded regions, we need to find the intersection of the Voronoi edges
            # with the square boundary, then compute the inradius of the resulting polygon.
            # This is complex. Alternative: use a large bounding box and compute Voronoi,
            # then clip to unit square.
            # Simpler approach: compute the minimum distance from point i to:
            #   - each Voronoi edge (line segment between consecutive finite vertices,
            #     extended to the square boundary)
            #   - the square boundary
            # For now, use a conservative estimate: min distance to square boundary
            # and half the distance to nearest neighbor, then refine.

            # Better approach: compute the Voronoi diagram in a larger domain
            # and clip. Let's use the edge-based approach.

            # Get the edges of this Voronoi region
            # vor.ridge_points gives pairs of point indices for each ridge
            # vor.ridge_vertices gives vertex indices for each ridge
            # Find ridges that belong to region i

            # For each ridge involving point i, get the line segment
            # and compute distance from points[i] to that line, clipped to the segment
            # Also check distance to square boundary

            min_dist = min(points[i][0], 1 - points[i][0], points[i][1], 1 - points[i][1])

            # Find all ridges connected to point i
            for ridge_idx in range(len(vor.ridge_points)):
                rp = vor.ridge_points[ridge_idx]
                if i not in rp:
                    continue
                rv = vor.ridge_vertices[ridge_idx]
                if -1 in rv:
                    # One or both vertices at infinity
                    # The ridge is a ray. Get the finite vertex and the direction
                    # The direction is perpendicular to the line between the two points
                    other_pt = rp[0] if rp[1] == i else rp[1]
                    mid = (points[i] + points[other_pt]) / 2
                    # Perpendicular direction
                    diff = points[other_pt] - points[i]
                    perp = np.array([-diff[1], diff[0]])
                    perp = perp / (np.linalg.norm(perp) + 1e-15)

                    # The ridge is the line through mid in direction perp
                    # Compute distance from points[i] to this line
                    # Distance from point P to line through A in direction D:
                    # |cross(P-A, D)| / |D|
                    # Since D is unit, it's |cross(P-A, D)|
                    v = points[i] - mid
                    dist_to_line = abs(v[0] * perp[1] - v[1] * perp[0])
                    min_dist = min(min_dist, dist_to_line)

                    # Also, the ridge extends from the finite vertex to infinity
                    # Check if the closest point on the line is within the square
                    # For now, we also need to check the segment from finite vertex
                    # to the square boundary. This is an approximation.

                    # If there's a finite vertex, also check distance to the segment
                    # from that vertex along the ray
                    for vidx in rv:
                        if vidx >= 0:
                            vertex = vor.vertices[vidx]
                            # Segment from vertex to infinity in direction perp
                            # The closest point on the ray to points[i]
                            # Parametrize: vertex + t * perp, t >= 0
                            w = points[i] - vertex
                            t = np.dot(w, perp)
                            if t >= 0:
                                closest = vertex + t * perp
                                d = np.linalg.norm(points[i] - closest)
                                min_dist = min(min_dist, d)
                            else:
                                # Closest point is the vertex itself
                                d = np.linalg.norm(points[i] - vertex)
                                min_dist = min(min_dist, d)
                else:
                    # Both vertices finite: it's a line segment
                    v1 = vor.vertices[rv[0]]
                    v2 = vor.vertices[rv[1]]
                    # Distance from points[i] to segment v1-v2
                    d_seg = v2 - v1
                    len_sq = np.dot(d_seg, d_seg)
                    if len_sq < 1e-15:
                        d = np.linalg.norm(points[i] - v1)
                    else:
                        t = np.dot(points[i] - v1, d_seg) / len_sq
                        t = np.clip(t, 0, 1)
                        closest = v1 + t * d_seg
                        d = np.linalg.norm(points[i] - closest)
                    min_dist = min(min_dist, d)

            radii[i] = min_dist
        else:
            # Bounded region: all vertices finite
            verts = np.array([vor.vertices[v] for v in region])
            # Clip to unit square
            # Simple approach: check if all vertices are in [0,1]^2
            # If not, we need to clip the polygon. For now, assume they are
            # (Voronoi vertices of interior points should be inside the convex hull,
            #  but may extend outside [0,1]^2)

            # Check if any vertex is outside [0,1]^2
            outside = (verts[:, 0] < -1e-9) | (verts[:, 0] > 1 + 1e-9) | \
                      (verts[:, 1] < -1e-9) | (verts[:, 1] > 1 + 1e-9)

            if np.any(outside):
                # Need to clip the polygon to the unit square
                # Use Sutherland-Hodgman clipping
                def clip_polygon_to_rect(poly, xmin, xmax, ymin, ymax):
                    def clip_edge(poly, edge):
                        # edge: (normal, offset) where normal·p <= offset
                        if len(poly) == 0:
                            return []
                        result = []
                        n = len(poly)
                        for i in range(n):
                            curr = poly[i]
                            prev = poly[(i - 1) % n]
                            curr_inside = np.dot(edge[0], curr) <= edge[1] + 1e-12
                            prev_inside = np.dot(edge[0], prev) <= edge[1] + 1e-12
                            if curr_inside:
                                if not prev_inside:
                                    # Intersection
                                    d = np.dot(edge[0], curr - prev)
                                    if abs(d) > 1e-15:
                                        t = (edge[1] - np.dot(edge[0], prev)) / d
                                        inter = prev + t * (curr - prev)
                                        result.append(inter)
                                result.append(curr)
                            else:
                                if prev_inside:
                                    d = np.dot(edge[0], curr - prev)
                                    if abs(d) > 1e-15:
                                        t = (edge[1] - np.dot(edge[0], prev)) / d
                                        inter = prev + t * (curr - prev)
                                        result.append(inter)
                        return result

                    poly = clip_edge(poly, (np.array([1, 0]), xmax))
                    poly = clip_edge(poly, (np.array([-1, 0]), -xmin))
                    poly = clip_edge(poly, (np.array([0, 1]), ymax))
                    poly = clip_edge(poly, (np.array([0, -1]), -ymin))
                    return poly

                verts = clip_polygon_to_rect(verts, 0, 1, 0, 1)
                if len(verts) < 3:
                    # Degenerate, use fallback
                    d_boundary = min(points[i][0], 1 - points[i][0], points[i][1], 1 - points[i][1])
                    d_other = np.inf
                    for j in range(n):
                        if i != j:
                            d_other = min(d_other, np.linalg.norm(points[i] - points[j]))
                    radii[i] = min(d_boundary, d_other / 2)
                    continue

            # Compute inradius of the polygon: min distance from center to each edge
            # For a convex polygon, the inradius is the min over edges of
            # the perpendicular distance from the center to the edge line
            m = len(verts)
            min_dist = np.inf
            for k in range(m):
                v1 = verts[k]
                v2 = verts[(k + 1) % m]
                # Edge from v1 to v2
                edge_vec = v2 - v1
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < 1e-15:
                    continue
                # Normal to the edge (inward)
                # The inward normal depends on polygon orientation
                # For a CCW polygon, the inward normal is (edge_vec[1], -edge_vec[0]) normalized
                # But we need to check orientation
                normal = np.array([edge_vec[1], -edge_vec[0]])
                normal = normal / edge_len
                # Check if this normal points inward
                # The centroid of the polygon should be on the inside
                centroid = np.mean(verts, axis=0)
                if np.dot(normal, centroid - v1) < 0:
                    normal = -normal
                # Distance from center to the line
                dist = np.dot(points[i] - v1, normal)
                # Also check that the closest point is on the segment
                t = np.dot(points[i] - v1, edge_vec) / edge_len**2
                if t < 0 or t > 1:
                    # Closest point is an endpoint
                    d1 = np.linalg.norm(points[i] - v1)
                    d2 = np.linalg.norm(points[i] - v2)
                    dist = min(d1, d2)
                min_dist = min(min_dist, dist)

            if np.isinf(min_dist):
                min_dist = min(points[i][0], 1 - points[i][0], points[i][1], 1 - points[i][1])
            radii[i] = min_dist

    return radii


def _objective_and_grad(points):
    """Compute the sum of Voronoi radii and its gradient (numerical)."""
    radii = _compute_voronoi_radii(points)
    obj = np.sum(radii)

    # Numerical gradient
    eps = 1e-6
    grad = np.zeros_like(points)
    for i in range(N):
        for d in range(2):
            orig = points[i, d]
            points[i, d] = orig + eps
            r_plus = np.sum(_compute_voronoi_radii(points))
            points[i, d] = orig - eps
            r_minus = np.sum(_compute_voronoi_radii(points))
            points[i, d] = orig
            grad[i, d] = (r_plus - r_minus) / (2 * eps)

    return obj, grad, radii


def _is_valid(points, radii, tol=1e-6):
    """Check that circles are inside the unit square and non-overlapping."""
    for i in range(N):
        x, y, r = points[i, 0], points[i, 1], radii[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def _post_process(points, radii):
    """Adjust radii down slightly to ensure strict non-overlap and boundary containment."""
    # Reduce each radius by a tiny amount
    eps = 1e-7
    radii = np.maximum(radii - eps, 0)
    # Also clamp positions to ensure circles stay in [0,1]^2
    for i in range(N):
        r = radii[i]
        points[i, 0] = np.clip(points[i, 0], r, 1 - r)
        points[i, 1] = np.clip(points[i, 1], r, 1 - r)
    return points, radii


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None

    # Grid initial
    side = int(np.ceil(np.sqrt(N)))
    grid_pts = np.array([[(i + 0.5) / side, (j + 0.5) / side] for i in range(side) for j in range(side)])[:N]

    # Generate diverse random inits
    rng = np.random.default_rng(42)
    inits = [grid_pts.copy()]
    for k in range(7):
        pts = rng.uniform(0.05, 0.95, size=(N, 2))
        inits.append(pts)

    for init_idx, init_pts in enumerate(inits):
        if time.time() - start_time > DEADLINE:
            break

        # Multi-stage penalty: start with a smoothed objective, then refine
        # Stage 1: Coarse optimization with L-BFGS-B
        points = init_pts.copy()

        # Use scipy.optimize.minimize with L-BFGS-B
        from scipy.optimize import minimize

        def neg_obj(x):
            pts = x.reshape(N, 2)
            # Keep points in [0.01, 0.99] to avoid boundary issues
            pts = np.clip(pts, 0.01, 0.99)
            r = _compute_voronoi_radii(pts)
            return -np.sum(r)

        def neg_grad(x):
            pts = x.reshape(N, 2)
            pts = np.clip(pts, 0.01, 0.99)
            r = _compute_voronoi_radii(pts)
            # Numerical gradient of the negative objective
            eps = 1e-5
            g = np.zeros_like(x)
            for i in range(N):
                for d in range(2):
                    orig = x[i * 2 + d]
                    x[i * 2 + d] = orig + eps
                    f_plus = neg_obj(x)
                    x[i * 2 + d] = orig - eps
                    f_minus = neg_obj(x)
                    x[i * 2 + d] = orig
                    g[i * 2 + d] = (f_plus - f_minus) / (2 * eps)
            return g

        x0 = init_pts.flatten()
        bounds = [(0.01, 0.99)] * (2 * N)

        # Stage 1: L-BFGS-B with moderate tolerance
        res1 = minimize(neg_obj, x0, method='L-BFGS-B', jac=neg_grad,
                        bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-4})

        if time.time() - start_time > DEADLINE:
            break

        x_opt = res1.x.reshape(N, 2)
        x_opt = np.clip(x_opt, 0.01, 0.99)
        radii = _compute_voronoi_radii(x_opt)
        score = np.sum(radii)

        if score > best_score:
            best_score = score
            best_result = (x_opt.copy(), radii.copy())

        # Stage 2: Refine with tighter tolerance
        if time.time() - start_time > DEADLINE - 20:
            break

        def neg_obj_tight(x):
            pts = x.reshape(N, 2)
            pts = np.clip(pts, 0.005, 0.995)
            r = _compute_voronoi_radii(pts)
            return -np.sum(r)

        def neg_grad_tight(x):
            pts = x.reshape(N, 2)
            pts = np.clip(pts, 0.005, 0.995)
            eps = 1e-6
            g = np.zeros_like(x)
            for i in range(N):
                for d in range(2):
                    orig = x[i * 2 + d]
                    x[i * 2 + d] = orig + eps
                    f_plus = neg_obj_tight(x)
                    x[i * 2 + d] = orig - eps
                    f_minus = neg_obj_tight(x)
                    x[i * 2 + d] = orig
                    g[i * 2 + d] = (f_plus - f_minus) / (2 * eps)
            return g

        bounds_tight = [(0.005, 0.995)] * (2 * N)
        res2 = minimize(neg_obj_tight, x_opt.flatten(), method='L-BFGS-B', jac=neg_grad_tight,
                        bounds=bounds_tight, options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-5})

        if time.time() - start_time > DEADLINE:
            break

        x_opt2 = res2.x.reshape(N, 2)
        x_opt2 = np.clip(x_opt2, 0.005, 0.995)
        radii2 = _compute_voronoi_radii(x_opt2)
        score2 = np.sum(radii2)

        if score2 > best_score:
            best_score = score2
            best_result = (x_opt2.copy(), radii2.copy())

    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    points, radii = best_result
    points, radii = _post_process(points, radii)

    result = np.zeros((N, 3))
    result[:, :2] = points
    result[:, 2] = radii
    return result
