"""Improved covering of the sphere with 50 points using geodesic start + Laplacian smoothing."""
import numpy as np
import time

N = 50
DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _covering_radius(pts):
    """Compute the covering radius in degrees for a set of unit vectors."""
    # Compute all pairwise angular distances
    # pts shape (N, 3)
    # For each point on the sphere, find nearest neighbor
    # Use the fact that the covering radius is the max over the sphere of min distance to any point
    # This equals the largest circumradius of the Voronoi cells on the sphere
    # A practical approximation: for each point, compute its Voronoi cell vertices (intersection of bisectors)
    # But a simpler and robust method: the covering radius is the max over all pairs of adjacent points
    # of the distance from their midpoint (on the great circle) to the nearest point... no that's not right either.
    
    # Actually, the covering radius equals the maximum, over all points y on the sphere, of min_i angle(y, p_i).
    # The maximum occurs at a vertex of the spherical Voronoi diagram, i.e., at a point equidistant to 
    # (at least) 3 points, or at a point on the sphere that is the circumcenter of a triangle of points.
    
    # Practical approach: the covering radius is the largest circumradius of the triangles formed by
    # triples of points that form a facet of the convex hull. But computing the convex hull on the sphere
    # is complex.
    
    # Alternative: use the fact that for a good configuration, the covering radius is approximately
    # the maximum over all pairs (i,j) that are nearest neighbors, of the angular distance from the
    # midpoint of the arc between them to the nearest point. But this is also an approximation.
    
    # Most robust practical method: sample many points on the sphere and compute min distance to each.
    # But this is slow and imprecise.
    
    # Better: compute the spherical Delaunay triangulation implicitly. The covering radius is the max
    # circumradius of the Delaunay triangles. We can approximate this by considering all triples and
    # checking if the circumcenter is in the interior of the triangle and not covered by a closer point.
    
    # For 50 points, we can do a reasonable approximation:
    # 1. Compute all pairwise distances
    # 2. For each pair (i,j), find the midpoint on the great circle
    # 3. The covering radius is at least half the max nearest-neighbor distance
    # 4. But the true covering radius is the max circumradius of Voronoi cells
    
    # Let's use a more direct approach: the covering radius equals the maximum over all y of min_i d(y, p_i).
    # This is a max-min problem. We can solve it by noting that the maximum occurs at a point where
    # at least 3 points are equidistant (a Voronoi vertex) or 2 points (on an edge) or 1 point (unlikely).
    
    # For 50 points, let's compute all triples and find their circumcenters, then check if the circumcenter
    # is valid (not closer to any other point), and take the max circumradius.
    
    # This is O(N^3) which for N=50 is 125,000 triples, manageable.
    
    # Actually, a simpler and correct approach for the covering radius:
    # The covering radius is the largest r such that there exists a point y with d(y, p_i) >= r for all i.
    # This is equivalent to: the largest r such that the intersection of the complements of the caps
    # of radius r around each p_i is non-empty.
    
    # For practical computation with 50 points, let's use the following:
    # The covering radius is the maximum, over all Voronoi vertices, of the distance to the nearest point.
    # A Voronoi vertex is equidistant to 3 (or more) points.
    
    # Let's compute this by checking all triples.
    
    pts = np.asarray(pts)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Compute all pairwise dot products
    dots = pts @ pts.T
    np.fill_diagonal(dots, 2.0)
    dists = np.arccos(np.clip(dots, -1.0, 1.0))
    
    # For each triple (i, j, k), find the circumcenter on the sphere
    # The circumcenter is equidistant to i, j, k
    # This is the point y such that y.p_i = y.p_j = y.p_k (in terms of dot product, since on unit sphere)
    # Actually, the circumcenter satisfies: the angular distances to i, j, k are equal.
    # In terms of dot products: y.p_i = y.p_j = y.p_k = c (same c)
    # And ||y|| = 1.
    
    # We can find y by solving: y is in the plane equidistant from i and j, and equidistant from i and k.
    # The set of points equidistant from i and j is the great circle through the midpoint of the arc ij
    # perpendicular to the arc. In terms of dot products: y.p_i = y.p_j, i.e., y.(p_i - p_j) = 0.
    # So y is in the plane spanned by (p_i - p_j) perpendicular... wait, y.(p_i - p_j) = 0 means y is
    # in the plane through the origin with normal (p_i - p_j).
    
    # So y satisfies:
    #   y . (p_i - p_j) = 0
    #   y . (p_i - p_k) = 0
    #   ||y|| = 1
    
    # This gives a linear system for y (up to sign), then normalize.
    
    max_r = 0.0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                # Solve for circumcenter
                a = pts[i] - pts[j]
                b = pts[i] - pts[k]
                # y is perpendicular to both a and b
                y = np.cross(a, b)
                ny = np.linalg.norm(y)
                if ny < 1e-10:
                    continue
                y /= ny
                # Check both signs
                for sign in [1.0, -1.0]:
                    yc = sign * y
                    # Compute distance to nearest point
                    d_to_nearest = np.min(np.arccos(np.clip(pts @ yc, -1.0, 1.0)))
                    # Check that this is actually a Voronoi vertex: no point is closer than d_to_nearest
                    # to y, and at least 3 points are at distance d_to_nearest
                    # Actually, if y is the circumcenter of (i,j,k), then d(y, p_i) = d(y, p_j) = d(y, p_k)
                    # We need to verify that no other point is closer.
                    all_dists = np.arccos(np.clip(pts @ yc, -1.0, 1.0))
                    min_d = np.min(all_dists)
                    # Check that i, j, k are among the closest
                    if (abs(all_dists[i] - min_d) < 1e-8 and 
                        abs(all_dists[j] - min_d) < 1e-8 and 
                        abs(all_dists[k] - min_d) < 1e-8):
                        max_r = max(max_r, min_d)
    
    return np.degrees(max_r)


def _laplacian_smooth(pts, weights=None):
    """Spherical Laplacian smoothing: move each point toward the weighted average of its neighbors."""
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    
    # Find nearest neighbors (k=6 for icosahedral topology)
    dots = pts @ pts.T
    np.fill_diagonal(dots, -2.0)
    
    # For each point, find its 6 nearest neighbors
    k = 6
    neighbor_idx = np.argsort(-dots, axis=1)[:, :k]
    
    new_pts = np.zeros_like(pts)
    for i in range(n):
        nbrs = neighbor_idx[i]
        if weights is not None:
            w = weights[i, nbrs]
            w = w / np.sum(w)
            new_pts[i] = np.sum(pts[nbrs] * w[:, None], axis=0)
        else:
            new_pts[i] = np.mean(pts[nbrs], axis=0)
        # Normalize
        norm = np.linalg.norm(new_pts[i])
        if norm > 1e-10:
            new_pts[i] /= norm
        else:
            new_pts[i] = pts[i]
    
    return new_pts


def _build_geodesic_start():
    """Build an initial configuration from an icosahedron with 1st subdivision (12 + 30 = 42 points)
    plus 8 additional points, or use a 2nd subdivision and merge/remove to get 50."""
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # First subdivision: each face becomes 4 triangles, adding midpoint of each edge
    # Icosahedron has 20 faces, 30 edges
    # After 1st subdivision: 12 + 30 = 42 vertices
    # After 2nd subdivision: 42 + 60 = 102 vertices (too many)
    
    # Let's use the 1st subdivision (42 points) and add 8 more points to reach 50.
    # Or better: use a different approach.
    
    # Actually, let's just use the Fibonacci spiral as a good starting point,
    # but initialize it on a geodesic-like structure.
    
    # For now, use the icosahedron + edge midpoints (42 points) + 8 carefully placed points.
    
    # Compute all edges of the icosahedron
    from scipy.spatial import ConvexHull
    hull = ConvexHull(ico)
    faces = hull.simplices
    
    # Find all edges
    edges = set()
    for face in faces:
        for i in range(3):
            for j in range(i+1, 3):
                a, b = face[i], face[j]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
    
    edge_midpoints = []
    for a, b in edges:
        mid = ico[a] + ico[b]
        mid /= np.linalg.norm(mid)
        edge_midpoints.append(mid)
    
    pts = np.vstack([ico, np.array(edge_midpoints)])
    # Now we have 42 points
    # Add 8 more points: place them at the centers of the 20 faces of the icosahedron? No, that's only 20.
    # Let's add points at the centers of the 20 triangular faces of the icosahedron, but we only need 8.
    # Actually, let's just add 8 points using a golden angle distribution in the gaps.
    
    # Better: use the face centers of the icosahedron (20 points) and pick 8 of them that are most separated.
    face_centers = []
    for face in faces:
        fc = np.mean(ico[face], axis=0)
        fc /= np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    # Greedily pick 8 face centers that maximize minimum distance to existing points
    remaining = list(range(20))
    additional = []
    for _ in range(8):
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            fc = face_centers[idx]
            min_d = np.min(np.arccos(np.clip(pts @ fc, -1.0, 1.0)))
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        if best_idx >= 0:
            additional.append(face_centers[best_idx])
            remaining.remove(best_idx)
    
    if additional:
        pts = np.vstack([pts, np.array(additional)])
    
    # Normalize all
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def construct():
    t0 = time.time()
    
    # Start with geodesic-based initial configuration
    pts = _build_geodesic_start()
    
    # Ensure we have exactly N points
    if len(pts) < N:
        # Add more points via golden angle
        i = np.arange(N - len(pts)) + 0.5
        z = 1.0 - 2.0 * i / (N - len(pts))
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        extra = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        pts = np.vstack([pts, extra])
    elif len(pts) > N:
        pts = pts[:N]
    
    # Iterative Laplacian smoothing with adaptive weights
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    for iteration in range(500):
        if time.time() - t0 > DEADLINE:
            break
        
        # Compute weights: inverse of local spacing
        dots = pts @ pts.T
        np.fill_diagonal(dots, -2.0)
        dists = np.arccos(np.clip(dots, -1.0, 1.0))
        
        # For each point, weight its neighbors by 1/distance^2
        k = 6
        neighbor_idx = np.argsort(dists, axis=1)[:, :k]
        
        new_pts = np.zeros_like(pts)
        for i in range(N):
            nbrs = neighbor_idx[i]
            d = dists[i, nbrs]
            w = 1.0 / (d * d + 1e-10)
            w /= np.sum(w)
            new_pts[i] = np.sum(pts[nbrs] * w[:, None], axis=0)
            norm = np.linalg.norm(new_pts[i])
            if norm > 1e-10:
                new_pts[i] /= norm
            else:
                new_pts[i] = pts[i]
        
        pts = new_pts
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Compute covering radius
        cr = _covering_radius(pts)
        if cr < best_radius:
            best_radius = cr
            best_pts = pts.copy()
    
    # Final polish: try to improve the worst Voronoi cell
    # Move the point that is farthest from the worst circumcenter
    for polish_iter in range(100):
        if time.time() - t0 > DEADLINE:
            break
        
        # Find the worst point (the one contributing to the max covering radius)
        # Approximate: find the point with the largest nearest-neighbor distance
        dots = best_pts @ best_pts.T
        np.fill_diagonal(dots, -2.0)
        dists = np.arccos(np.clip(dots, -1.0, 1.0))
        nn_dists = np.min(dists, axis=1)
        
        # Move each point slightly toward the centroid of its k nearest neighbors
        k = 6
        neighbor_idx = np.argsort(dists, axis=1)[:, :k]
        
        improved = False
        for i in range(N):
            nbrs = neighbor_idx[i]
            centroid = np.mean(best_pts[nbrs], axis=0)
            centroid /= np.linalg.norm(centroid)
            
            # Try moving point i toward centroid
            for step in [0.01, 0.005, 0.001]:
                new_pt = best_pts[i] + step * (centroid - best_pts[i])
                new_pt /= np.linalg.norm(new_pt)
                
                trial_pts = best_pts.copy()
                trial_pts[i] = new_pt
                
                cr = _covering_radius(trial_pts)
                if cr < best_radius:
                    best_pts = trial_pts
                    best_radius = cr
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_pts
