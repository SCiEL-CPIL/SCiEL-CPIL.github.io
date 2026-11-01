import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def fibonacci_initial(phase=0.0):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + phase)
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_hull_data(p):
        """Compute convex hull, facet circumcenters and covering radii."""
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
        except Exception:
            return None, None, None

        n_facets = len(facets)
        if n_facets == 0:
            return None, None, None

        a = p[facets[:, 0]]
        b = p[facets[:, 1]]
        c = p[facets[:, 2]]

        v1 = a - b
        v2 = b - c
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1)

        valid = norm_n > 1e-12
        circumcenters = np.zeros((n_facets, 3))
        radii = np.zeros(n_facets)

        if np.any(valid):
            cc = n[valid] / norm_n[valid, np.newaxis]
            # Ensure circumcenter points outward (positive dot with face normal direction)
            dots = np.sum(cc * a[valid], axis=1)
            cc[dots < 0] *= -1
            circumcenters[valid] = cc
            cos_r = np.clip(np.sum(cc * a[valid], axis=1), -1.0, 1.0)
            radii[valid] = np.arccos(cos_r)

        return facets, circumcenters, radii

    def get_covering_radius(pts):
        facets, circumcenters, radii = compute_hull_data(pts)
        if radii is None:
            return float('inf')
        return float(np.max(radii))

    def lloyd_step(pts, n_iters=1):
        """
        Lloyd's algorithm for sphere covering:
        For each point, compute the centroid of its Voronoi cell on the sphere.
        The Voronoi cell is the set of points on the sphere closer to this point
        than to any other. We approximate by finding the Delaunay dual.
        
        On the sphere, the Voronoi cell of point i is bounded by the great circles
        that are the perpendicular bisectors. The centroid can be computed as
        the normalized sum of:
          - The point itself (weighted by solid angle)
          - The circumcenters of adjacent Delaunay triangles
        
        Simpler approach: use the convex hull facets. Each point's Voronoi cell
        corresponds to the dual. We compute the centroid as the normalized sum
        of the facet circumcenters adjacent to each vertex.
        """
        pts = pts.copy()
        for _ in range(n_iters):
            if time.time() > deadline - 5:
                break
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break

            # For each vertex, accumulate the circumcenters of facets containing it
            # weighted by the area (solid angle) of the Voronoi cell
            n_facets = len(facets)
            # Build vertex-to-facets mapping
            vertex_facets = [[] for _ in range(N)]
            for k in range(n_facets):
                for v in facets[k]:
                    vertex_facets[v].append(k)

            new_pts = np.zeros_like(pts)
            for v in range(N):
                if len(vertex_facets[v]) == 0:
                    new_pts[v] = pts[v]
                    continue
                # Centroid = normalized sum of circumcenters of adjacent facets
                # weighted by the angle subtended (approximate by equal weight)
                sum_cc = np.zeros(3)
                for k in vertex_facets[v]:
                    sum_cc += circumcenters[k]
                norm = np.linalg.norm(sum_cc)
                if norm > 1e-10:
                    new_pts[v] = sum_cc / norm
                else:
                    new_pts[v] = pts[v]

            # Check if all points are still distinct and on sphere
            if not np.all(np.isfinite(new_pts)):
                break
            pts = normalize(new_pts)

            # Verify no coincident points
            for i in range(N):
                for j in range(i+1, N):
                    if np.linalg.norm(pts[i] - pts[j]) < 1e-8:
                        # Nudge apart
                        pts[j] += 0.01 * np.random.randn(3)
                        pts[j] = pts[j] / np.linalg.norm(pts[j])

        return pts

    def lloyd_with_repulsion(pts, n_iters=1, repulse_strength=0.05):
        """Lloyd's step with additional repulsion to avoid coincident points."""
        pts = pts.copy()
        for _ in range(n_iters):
            if time.time() > deadline - 5:
                break
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break

            n_facets = len(facets)
            vertex_facets = [[] for _ in range(N)]
            for k in range(n_facets):
                for v in facets[k]:
                    vertex_facets[v].append(k)

            new_pts = np.zeros_like(pts)
            for v in range(N):
                if len(vertex_facets[v]) == 0:
                    new_pts[v] = pts[v]
                    continue
                sum_cc = np.zeros(3)
                for k in vertex_facets[v]:
                    sum_cc += circumcenters[k]
                norm = np.linalg.norm(sum_cc)
                if norm > 1e-10:
                    new_pts[v] = sum_cc / norm
                else:
                    new_pts[v] = pts[v]

            # Add repulsion from nearby points
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    d = np.linalg.norm(new_pts[i] - new_pts[j])
                    if d < 0.1:
                        repulse = (new_pts[i] - new_pts[j]) / max(d, 1e-10)
                        new_pts[i] += repulse_strength * repulse * (0.1 - d)

            pts = normalize(new_pts)

            for i in range(N):
                for j in range(i+1, N):
                    if np.linalg.norm(pts[i] - pts[j]) < 1e-8:
                        pts[j] += 0.01 * np.random.randn(3)
                        pts[j] = pts[j] / np.linalg.norm(pts[j])

        return pts

    def simulated_annealing(pts, t0=0.3, t_end=0.001, cooling=0.9995, max_steps=5000, rng=None):
        """
        Simulated annealing on the sphere.
        Move: pick a random point, perturb it by a random tangent vector.
        Energy: covering radius (max facet circumradius).
        """
        if rng is None:
            rng = np.random.RandomState(42)

        pts = pts.copy()
        best_pts = pts.copy()
        best_radius = get_covering_radius(pts)
        current_radius = best_radius
        t = t0

        for step in range(max_steps):
            if time.time() > deadline - 10:
                break

            # Pick a random point to move
            idx = rng.randint(N)
            
            # Generate random tangent direction
            v = pts[idx]
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, v) * v
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-10:
                continue
            tangent /= tangent_norm

            # Move magnitude proportional to temperature
            magnitude = t * 0.5
            new_pts = pts.copy()
            new_pts[idx] = v + magnitude * tangent
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])

            # Evaluate new energy
            new_radius = get_covering_radius(new_pts)
            
            if not np.isfinite(new_radius):
                t *= cooling
                continue

            # Accept or reject
            delta_e = new_radius - current_radius
            if delta_e < 0 or rng.random() < np.exp(-delta_e / max(t, 1e-10)):
                pts = new_pts
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()
            # else: reject, keep old

            t *= cooling
            if t < t_end:
                break

        return best_pts, best_radius

    def annealing_with_lloyd(pts, t0=0.2, t_end=0.005, cooling=0.999, max_steps=3000, rng=None):
        """
        Hybrid: simulated annealing with periodic Lloyd refinement.
        """
        if rng is None:
            rng = np.random.RandomState(123)

        pts = pts.copy()
        best_pts = pts.copy()
        best_radius = get_covering_radius(pts)
        current_radius = best_radius
        t = t0

        lloyd_interval = 200  # Do Lloyd every 200 SA steps

        for step in range(max_steps):
            if time.time() > deadline - 10:
                break

            # Periodic Lloyd refinement
            if step > 0 and step % lloyd_interval == 0:
                pts_lloyd = lloyd_step(pts, n_iters=3)
                rad_lloyd = get_covering_radius(pts_lloyd)
                if rad_lloyd < current_radius:
                    pts = pts_lloyd
                    current_radius = rad_lloyd
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = pts.copy()

            # SA move
            idx = rng.randint(N)
            v = pts[idx]
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, v) * v
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-10:
                continue
            tangent /= tangent_norm

            magnitude = t * 0.3
            new_pts = pts.copy()
            new_pts[idx] = v + magnitude * tangent
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])

            new_radius = get_covering_radius(new_pts)
            
            if not np.isfinite(new_radius):
                t *= cooling
                continue

            delta_e = new_radius - current_radius
            if delta_e < 0 or rng.random() < np.exp(-delta_e / max(t, 1e-10)):
                pts = new_pts
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()

            t *= cooling
            if t < t_end:
                break

        # Final Lloyd refinement
        if time.time() < deadline - 5:
            pts_final = lloyd_step(pts, n_iters=5)
            rad_final = get_covering_radius(pts_final)
            if rad_final < best_radius:
                best_radius = rad_final
                best_pts = pts_final.copy()

        return best_pts, best_radius

    def facet_biased_annealing(pts, t0=0.15, t_end=0.002, cooling=0.999, max_steps=4000, rng=None):
        """
        Annealing that biases moves toward vertices of the worst facet.
        """
        if rng is None:
            rng = np.random.RandomState(777)

        pts = pts.copy()
        best_pts = pts.copy()
        best_radius = get_covering_radius(pts)
        current_radius = best_radius
        t = t0

        for step in range(max_steps):
            if time.time() > deadline - 10:
                break

            # Occasionally do Lloyd
            if step > 0 and step % 300 == 0:
                pts_lloyd = lloyd_step(pts, n_iters=2)
                rad_lloyd = get_covering_radius(pts_lloyd)
                if rad_lloyd < current_radius:
                    pts = pts_lloyd
                    current_radius = rad_lloyd
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_pts = pts.copy()

            # Compute hull to find worst facet
            facets, circumcenters, radii = compute_hull_data(pts)
            if facets is None:
                break

            # With probability 0.5, pick a vertex from the worst facet
            if rng.random() < 0.5:
                worst_idx = np.argmax(radii)
                facet_verts = facets[worst_idx]
                idx = facet_verts[rng.randint(len(facet_verts))]
            else:
                idx = rng.randint(N)

            v = pts[idx]
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, v) * v
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-10:
                continue
            tangent /= tangent_norm

            magnitude = t * 0.25
            new_pts = pts.copy()
            new_pts[idx] = v + magnitude * tangent
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])

            new_radius = get_covering_radius(new_pts)
            
            if not np.isfinite(new_radius):
                t *= cooling
                continue

            delta_e = new_radius - current_radius
            if delta_e < 0 or rng.random() < np.exp(-delta_e / max(t, 1e-10)):
                pts = new_pts
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = pts.copy()

            t *= cooling
            if t < t_end:
                break

        # Final Lloyd
        if time.time() < deadline - 5:
            pts_final = lloyd_step(pts, n_iters=5)
            rad_final = get_covering_radius(pts_final)
            if rad_final < best_radius:
                best_radius = rad_final
                best_pts = pts_final.copy()

        return best_pts, best_radius

    def fine_tune(pts, n_steps=200, rng=None):
        """
        Fine-tuning: very small random moves + Lloyd, accepting only improvements.
        """
        if rng is None:
            rng = np.random.RandomState(999)

        pts = pts.copy()
        best_pts = pts.copy()
        best_radius = get_covering_radius(pts)

        for step in range(n_steps):
            if time.time() > deadline - 3:
                break

            idx = rng.randint(N)
            v = pts[idx]
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, v) * v
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm < 1e-10:
                continue
            tangent /= tangent_norm

            magnitude = 0.005
            new_pts = pts.copy()
            new_pts[idx] = v + magnitude * tangent
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])

            new_radius = get_covering_radius(new_pts)
            if np.isfinite(new_radius) and new_radius < best_radius:
                pts = new_pts
                best_radius = new_radius
                best_pts = pts.copy()

            # Periodic Lloyd
            if step % 50 == 49:
                pts_lloyd = lloyd_step(pts, n_iters=3)
                rad_lloyd = get_covering_radius(pts_lloyd)
                if rad_lloyd < best_radius:
                    best_radius = rad_lloyd
                    best_pts = pts_lloyd.copy()
                    pts = pts_lloyd

        return best_pts, best_radius

    # --- Main optimization pipeline ---

    # Generate diverse initial configurations
    inits = [
        fibonacci_initial(0.0),
        fibonacci_initial(0.37),
        fibonacci_initial(0.618),
        fibonacci_initial(0.25),
    ]

    # Add random initializations
    rng_init = np.random.RandomState(42)
    for _ in range(3):
        pts_rand = rng_init.randn(N, 3)
        pts_rand = normalize(pts_rand)
        inits.append(pts_rand)

    # Evaluate all inits
    best_pts = None
    best_radius = float('inf')
    for init in inits:
        rad = get_covering_radius(init)
        if rad < best_radius:
            best_radius = rad
            best_pts = init.copy()

    # Phase 1: Simulated Annealing from multiple starts
    if time.time() < deadline - 60:
        for init_idx, init in enumerate(inits[:4]):
            if time.time() > deadline - 40:
                break
            rng_sa = np.random.RandomState(100 + init_idx)
            pts_sa, rad_sa = simulated_annealing(
                init, t0=0.3, t_end=0.005, cooling=0.9995,
                max_steps=2000, rng=rng_sa
            )
            if rad_sa < best_radius:
                best_radius = rad_sa
                best_pts = pts_sa.copy()

    # Phase 2: Hybrid annealing + Lloyd from best
    if time.time() < deadline - 40:
        rng_h = np.random.RandomState(200)
        pts_h, rad_h = annealing_with_lloyd(
            best_pts, t0=0.15, t_end=0.003, cooling=0.999,
            max_steps=3000, rng=rng_h
        )
        if rad_h < best_radius:
            best_radius = rad_h
            best_pts = pts_h.copy()

    # Phase 3: Facet-biased annealing
    if time.time() < deadline - 25:
        rng_fb = np.random.RandomState(300)
        pts_fb, rad_fb = facet_biased_annealing(
            best_pts, t0=0.1, t_end=0.002, cooling=0.999,
            max_steps=3000, rng=rng_fb
        )
        if rad_fb < best_radius:
            best_radius = rad_fb
            best_pts = pts_fb.copy()

    # Phase 4: Lloyd refinement
    if time.time() < deadline - 15:
        pts_lloyd = lloyd_with_repulsion(best_pts, n_iters=10, repulse_strength=0.03)
        rad_lloyd = get_covering_radius(pts_lloyd)
        if rad_lloyd < best_radius:
            best_radius = rad_lloyd
            best_pts = pts_lloyd.copy()

    # Phase 5: Fine tuning
    if time.time() < deadline - 8:
        rng_ft = np.random.RandomState(400)
        pts_ft, rad_ft = fine_tune(best_pts, n_steps=300, rng=rng_ft)
        if rad_ft < best_radius:
            best_radius = rad_ft
            best_pts = pts_ft.copy()

    # Phase 6: Another round of annealing + Lloyd from current best
    if time.time() < deadline - 10:
        rng_a2 = np.random.RandomState(500)
        pts_a2, rad_a2 = annealing_with_lloyd(
            best_pts, t0=0.08, t_end=0.001, cooling=0.9995,
            max_steps=2000, rng=rng_a2
        )
        if rad_a2 < best_radius:
            best_radius = rad_a2
            best_pts = pts_a2.copy()

    # Phase 7: Final Lloyd + fine tune
    if time.time() < deadline - 5:
        pts_final_lloyd = lloyd_step(best_pts, n_iters=5)
        rad_fl = get_covering_radius(pts_final_lloyd)
        if rad_fl < best_radius:
            best_radius = rad_fl
            best_pts = pts_final_lloyd.copy()

    if time.time() < deadline - 3:
        rng_ft2 = np.random.RandomState(600)
        pts_ft2, rad_ft2 = fine_tune(best_pts, n_steps=100, rng=rng_ft2)
        if rad_ft2 < best_radius:
            best_radius = rad_ft2
            best_pts = pts_ft2.copy()

    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = fibonacci_initial()

    # Check no coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                best_pts[j] += 0.01 * np.random.RandomState(i*100+j).randn(3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])

    best_pts = normalize(best_pts)

    # Final check: ensure not in a hemisphere (convex hull should have N facets roughly)
    try:
        hull = ConvexHull(best_pts)
        if len(hull.simplices) < 20:  # Too few facets, likely degenerate
            best_pts = fibonacci_initial()
    except Exception:
        best_pts = fibonacci_initial()

    return best_pts
