import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 600  # safety margin before 630s kill

    def true_score(points):
        dists_sq = pdist(points, metric='sqeuclidean')
        if len(dists_sq) == 0:
            return 0.0
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-12:
            return 0.0
        return d_min_sq / d_max_sq

    def normalize_rms(points):
        rms = np.sqrt(np.mean(np.sum(points**2, axis=1)))
        if rms < 1e-6:
            return points
        return points / rms

    def smooth_obj(x_flat):
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-8)
        return np.sum(1.0 / dists_sq)

    def barrier_obj(x_flat, mu=0.1):
        """Barrier-augmented objective: minimize -min_dist_sq + barrier on max_dist_sq > 1.
        This directly targets the (d_min/d_max)^2 ratio by maximizing min distance
        while penalizing max distance exceeding 1."""
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-12:
            return 1e10
        # Barrier term: -mu * log(1 - d_max_sq) if d_max_sq < 1, else large penalty
        if d_max_sq < 1.0 - 1e-6:
            barrier = -mu * np.log(1.0 - d_max_sq)
        else:
            barrier = mu * (d_max_sq - 1.0) ** 2
        # Primary objective: maximize d_min_sq (minimize negative)
        return -d_min_sq + barrier

    def barrier_obj_grad(x_flat, mu=0.1):
        """Gradient of barrier_obj via finite differences."""
        eps = 1e-6
        grad = np.zeros_like(x_flat)
        f0 = barrier_obj(x_flat, mu)
        for i in range(len(x_flat)):
            x_plus = x_flat.copy()
            x_plus[i] += eps
            f_plus = barrier_obj(x_plus, mu)
            x_minus = x_flat.copy()
            x_minus[i] -= eps
            f_minus = barrier_obj(x_minus, mu)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    # DIRECT construction: symmetry-based candidate configurations
    candidates = []

    # Candidate 1: Icosahedron vertices + 2 poles
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    candidates.append(np.vstack([ico, poles]))

    # Candidate 2: Regular dodecagon in xy-plane + 2 poles
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    dodec = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(12)])
    candidates.append(np.vstack([dodec, poles]))

    # Candidate 3: Dodecagon tilted + 2 poles (optimize tilt)
    for tilt in [0.1, 0.2, 0.3, 0.4, 0.5]:
        dodec_t = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(12)])
        # Rotate around x-axis by tilt
        c, s = np.cos(tilt), np.sin(tilt)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        dodec_t = dodec_t @ R.T
        # Scale poles to same RMS
        all_pts = np.vstack([dodec_t, poles])
        all_pts = normalize_rms(all_pts)
        candidates.append(all_pts)

    # Candidate 4: Two concentric hexagons (offset) + 2 poles
    for offset in [0.1, 0.2, 0.3, 0.5]:
        ang1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ang2 = ang1 + np.pi / 6  # offset by 30 degrees
        hex1 = np.column_stack([np.cos(ang1), np.sin(ang1), np.ones(6) * offset])
        hex2 = np.column_stack([np.cos(ang2), np.sin(ang2), np.ones(6) * (-offset)])
        all_pts = np.vstack([hex1, hex2, poles])
        all_pts = normalize_rms(all_pts)
        candidates.append(all_pts)

    # Candidate 5: Icosahedron with slightly moved poles
    for pole_shift in [0.1, 0.2, 0.3, 0.5, 0.8]:
        p1 = np.array([0.0, 0.0, 1.0]) / np.sqrt(1 + pole_shift**2)
        p2 = np.array([0.0, 0.0, -1.0]) / np.sqrt(1 + pole_shift**2)
        # Actually, let's shift the poles radially inward
        scale = 1.0 - pole_shift
        p1 = np.array([0.0, 0.0, scale])
        p2 = np.array([0.0, 0.0, -scale])
        all_pts = np.vstack([ico, p1.reshape(1, 3), p2.reshape(1, 3)])
        all_pts = normalize_rms(all_pts)
        candidates.append(all_pts)

    # Candidate 6: Cuboctahedron (14 vertices? No, cuboctahedron has 12 vertices)
    # Use truncated octahedron? No, let's use a different approach.
    # Place 14 points as close to a known optimal as possible.
    # Try: 8 points on a cube + 6 on octahedron faces
    cube = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ], dtype=float)
    cube = cube / np.sqrt(3.0)
    octa = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    all_pts = np.vstack([cube, octa])
    all_pts = normalize_rms(all_pts)
    candidates.append(all_pts)

    # Candidate 7: 7 points on upper hemisphere + 7 on lower (antipodal pairs)
    # Use 7 random-ish points on sphere, then antipodes
    rng = np.random.RandomState(42)
    for seed in range(5):
        rng = np.random.RandomState(seed)
        pts_upper = rng.randn(7, 3)
        pts_upper[:, 2] = np.abs(pts_upper[:, 2])  # ensure upper hemisphere
        pts_upper = pts_upper / np.linalg.norm(pts_upper, axis=1, keepdims=True)
        pts_lower = -pts_upper
        all_pts = np.vstack([pts_upper, pts_lower])
        all_pts = normalize_rms(all_pts)
        candidates.append(all_pts)

    # Candidate 8: icosahedron with poles at different z
    for z_pole in [0.5, 0.7, 0.9, 1.0, 1.2]:
        p1 = np.array([0.0, 0.0, z_pole])
        p2 = np.array([0.0, 0.0, -z_pole])
        all_pts = np.vstack([ico, p1.reshape(1, 3), p2.reshape(1, 3)])
        all_pts = normalize_rms(all_pts)
        candidates.append(all_pts)

    best_score = 0.0
    best_points = None

    # Score all candidates first
    for cand in candidates:
        cand = normalize_rms(cand)
        s = true_score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()

    # Now polish each candidate with optimization
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break

        cand = normalize_rms(cand)

        try:
            # Phase 1: L-BFGS-B on smooth surrogate (inverse distance sum)
            res1 = minimize(
                smooth_obj,
                cand.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pts1 = normalize_rms(res1.x.reshape(n, d))

            # Phase 2: Barrier-based optimization to directly target min/max ratio
            # Scale points so max distance is close to 1
            dists_sq_1 = pdist(pts1, metric='sqeuclidean')
            max_d_sq_1 = np.max(dists_sq_1)
            if max_d_sq_1 > 1e-12:
                pts1_scaled = pts1 / np.sqrt(max_d_sq_1)
            else:
                pts1_scaled = pts1

            # Try multiple barrier strengths
            best_barrier_pts = pts1
            best_barrier_score = true_score(pts1)
            for mu in [0.01, 0.05, 0.1, 0.5]:
                if time.time() > deadline - 25:
                    break
                try:
                    res_b = minimize(
                        lambda x: barrier_obj(x, mu),
                        pts1_scaled.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    pts_b = normalize_rms(res_b.x.reshape(n, d))
                    s_b = true_score(pts_b)
                    if s_b > best_barrier_score:
                        best_barrier_score = s_b
                        best_barrier_pts = pts_b
                except Exception:
                    continue

            # Phase 3: Nelder-Mead on true objective from barrier result
            def nm_obj(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return -true_score(pts)

            res2 = minimize(
                nm_obj,
                best_barrier_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14,
                         'adaptive': True}
            )
            pts2 = normalize_rms(res2.x.reshape(n, d))

            s = true_score(pts2)
            if s > best_score:
                best_score = s
                best_points = pts2.copy()

        except Exception:
            continue

    # If time permits, do more local searches from best
    if best_points is not None and time.time() < deadline - 30:
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            try:
                perturbed = best_points + np.random.randn(n, d) * 0.05
                perturbed = normalize_rms(perturbed)

                res1 = minimize(
                    smooth_obj,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                pts1 = normalize_rms(res1.x.reshape(n, d))

                # Barrier refinement
                dists_sq_1 = pdist(pts1, metric='sqeuclidean')
                max_d_sq_1 = np.max(dists_sq_1)
                if max_d_sq_1 > 1e-12:
                    pts1_scaled = pts1 / np.sqrt(max_d_sq_1)
                else:
                    pts1_scaled = pts1

                best_barrier_pts = pts1
                best_barrier_score = true_score(pts1)
                for mu in [0.01, 0.1, 0.5]:
                    if time.time() > deadline - 20:
                        break
                    try:
                        res_b = minimize(
                            lambda x: barrier_obj(x, mu),
                            pts1_scaled.flatten(),
                            method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                        )
                        pts_b = normalize_rms(res_b.x.reshape(n, d))
                        s_b = true_score(pts_b)
                        if s_b > best_barrier_score:
                            best_barrier_score = s_b
                            best_barrier_pts = pts_b
                    except Exception:
                        continue

                def nm_obj(x_flat):
                    pts = x_flat.reshape(n, d)
                    pts = normalize_rms(pts)
                    return -true_score(pts)

                res2 = minimize(
                    nm_obj,
                    best_barrier_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                             'adaptive': True}
                )
                pts2 = normalize_rms(res2.x.reshape(n, d))
                s = true_score(pts2)
                if s > best_score:
                    best_score = s
                    best_points = pts2.copy()
            except Exception:
                continue

    # Final high-precision polish pass
    if best_points is not None and time.time() < deadline - 10:
        try:
            def nm_obj_tight(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return -true_score(pts)

            res_final = minimize(
                nm_obj_tight,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16,
                         'adaptive': True}
            )
            pts_final = normalize_rms(res_final.x.reshape(n, d))
            s_final = true_score(pts_final)
            if s_final > best_score:
                best_score = s_final
                best_points = pts_final.copy()
        except Exception:
            pass

    # Fallback
    if best_points is None:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
            [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        best_points = np.vstack([ico, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]])])

    if not np.all(np.isfinite(best_points)):
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
            [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        best_points = np.vstack([ico, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]])])

    return best_points
