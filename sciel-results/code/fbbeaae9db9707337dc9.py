import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # safety margin before 630s limit
    n = 14

    def to_cartesian(sph):
        """Convert spherical coordinates (theta, phi) to Cartesian on unit sphere."""
        theta = sph[:, 0]  # polar angle [0, pi]
        phi = sph[:, 1]    # azimuthal angle [0, 2*pi]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])

    def ratio_from_cart(cart):
        d = pdist(cart)
        d_min = float(d.min())
        d_max = float(d.max())
        if d_max <= 1e-15:
            return 0.0
        return d_min / d_max

    def neg_ratio_sph(sph):
        cart = to_cartesian(sph)
        return -ratio_from_cart(cart)

    # Initial configurations in spherical coordinates
    def fib_sphere(n):
        k = np.arange(n)
        phi = np.arccos(1 - 2 * (k + 0.5) / n)
        golden = np.pi * (1 + np.sqrt(5))
        theta = phi
        phi_az = golden * k
        return np.column_stack([theta, phi_az])

    def icosahedron_sphere():
        t = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [-1, 1, t], [1, 1, t], [-1, -1, t], [1, -1, t],
            [1, t, -1], [1, t, 1], [-1, t, -1], [-1, t, 1],
            [t, -1, -1], [t, 1, -1], [t, -1, 1], [t, 1, 1]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        # Add 2 poles
        poles = np.array([[0, 0, 1.0], [0, 0, -1.0]])
        all_pts = np.vstack([ico, poles])
        # Convert to spherical
        r = np.linalg.norm(all_pts, axis=1)
        theta = np.arccos(all_pts[:, 2] / r)
        phi = np.arctan2(all_pts[:, 1], all_pts[:, 0])
        phi[phi < 0] += 2 * np.pi
        return np.column_stack([theta, phi])

    # Generate diverse initial candidates
    np.random.seed(12345)
    candidates = [fib_sphere(n), icosahedron_sphere()]
    
    # Add perturbed versions
    for base in [fib_sphere(n), icosahedron_sphere()]:
        for noise_level in [0.05, 0.1, 0.2, 0.3]:
            perturbed = base.copy()
            perturbed[:, 0] += noise_level * np.random.randn(n)
            perturbed[:, 1] += noise_level * np.random.randn(n)
            # Wrap angles
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0, np.pi)
            perturbed[:, 1] = perturbed[:, 1] % (2 * np.pi)
            candidates.append(perturbed)
    
    # Add random sphere points
    for _ in range(10):
        theta = np.random.uniform(0, np.pi, n)
        phi = np.random.uniform(0, 2 * np.pi, n)
        candidates.append(np.column_stack([theta, phi]))

    best_sph = None
    best_r = -1.0

    # Phase 1: Multi-start Nelder-Mead on spherical coordinates
    for i, cand in enumerate(candidates[:12]):
        if time.time() > deadline - 150:
            break
        try:
            res = minimize(neg_ratio_sph, cand, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11,
                                    'adaptive': True})
            r = -float(res.fun)
            if np.isfinite(r) and r > best_r:
                best_r = r
                best_sph = res.x.copy()
        except Exception:
            pass

    # Phase 2: Powell optimization from best and a few other starts
    if best_sph is not None:
        starts = [best_sph]
    else:
        starts = [fib_sphere(n)]
    starts.extend(candidates[12:16])
    
    for i, cand in enumerate(starts):
        if time.time() > deadline - 80:
            break
        try:
            res = minimize(neg_ratio_sph, cand, method='Powell',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            r = -float(res.fun)
            if np.isfinite(r) and r > best_r:
                best_r = r
                best_sph = res.x.copy()
        except Exception:
            pass

    # Phase 3: Final high-precision Nelder-Mead polish
    if best_sph is not None and time.time() < deadline - 30:
        try:
            res2 = minimize(neg_ratio_sph, best_sph, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-13,
                                     'adaptive': True})
            r2 = -float(res2.fun)
            if np.isfinite(r2) and r2 > best_r:
                best_r = r2
                best_sph = res2.x.copy()
        except Exception:
            pass

    # Phase 4: One more round of simulated annealing in spherical space
    if best_sph is not None and time.time() < deadline - 10:
        rng = np.random.RandomState(999)
        pts = best_sph.copy()
        current_score = -neg_ratio_sph(pts)
        best_pts = pts.copy()
        best_score = current_score
        T_start = 0.05
        T_end = 0.0005
        steps = 1500
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            T = T_start * (T_end / T_start) ** (step / max(steps - 1, 1))
            
            n_perturb = rng.choice([1, 2], p=[0.7, 0.3])
            indices = rng.choice(n, size=n_perturb, replace=False)
            
            new_pts = pts.copy()
            for idx in indices:
                new_pts[idx, 0] += rng.randn() * 0.05 * T / T_start + 0.005
                new_pts[idx, 1] += rng.randn() * 0.05 * T / T_start + 0.005
                new_pts[idx, 0] = np.clip(new_pts[idx, 0], 0, np.pi)
                new_pts[idx, 1] = new_pts[idx, 1] % (2 * np.pi)
            
            new_score = -neg_ratio_sph(new_pts)
            
            if new_score > current_score or rng.rand() < np.exp((new_score - current_score) / max(T, 1e-10)):
                pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = pts.copy()
        
        if best_score > best_r:
            best_r = best_score
            best_sph = best_pts

    if best_sph is None:
        best_sph = fib_sphere(n)

    # Convert to Cartesian and return
    result = to_cartesian(best_sph)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure all points are on unit sphere
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    mask = norms < 1e-12
    if np.any(mask):
        norms[mask] = 1.0
        result[mask, 0] = 1.0
        result[mask, 1:] = 0.0
    result = result / norms
    
    return result
