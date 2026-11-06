import time
import math
import numpy as np
from scipy.optimize import minimize

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def compute_score(pts):
        """Compute the score for a set of points."""
        pts = np.asarray(pts)
        if not np.all(np.isfinite(pts)):
            return 0.0
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        if D < 1e-12:
            return 0.0
        # Scale to diameter 2
        scaled_d = d * (2.0 / D)
        prod_unordered = np.prod(scaled_d)
        Delta = prod_unordered ** 2  # Product over ordered pairs
        return Delta / (n ** n)

    def objective(coeffs):
        """
        Objective function for polynomial coefficients.
        coeffs: [c7, c6, c5, c4, c3, c2, c1, c0]
        Maximizes sum(log(d_ij)) - 28*log(D)
        """
        # Construct polynomial
        # P(z) = z^8 + c7*z^7 + ... + c0
        poly_coeffs = np.concatenate(([1.0], coeffs))
        
        try:
            roots = np.roots(poly_coeffs)
        except Exception:
            return 1e10
            
        if not np.all(np.isfinite(roots)):
            return 1e10
            
        # Check for distinct roots
        # Compute pairwise distances
        diffs = roots[:, np.newaxis] - roots[np.newaxis, :]
        dists = np.abs(diffs)
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        
        if np.min(d) < 1e-10:
            return 1e10
            
        D = np.max(d)
        if D < 1e-10:
            return 1e10
            
        # Objective: - (sum(log(d)) - 28*log(D))
        # We want to minimize this.
        log_sum = np.sum(np.log(d))
        val = -(log_sum - 28.0 * np.log(D))
        
        return val

    best_score = 0.0
    best_pts = None

    def try_candidate(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts)
        if not np.all(np.isfinite(pts)):
            return 0.0
        # Check distinctness
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-8:
            return 0.0
        s = compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # Initial guesses for coefficients
    # 1. Regular Octagon: z^8 - 1 = 0 -> coeffs [0,0,0,0,0,0,0,-1]
    inits = []
    inits.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    
    # 2. Perturbed Regular Octagon
    np.random.seed(42)
    for _ in range(5):
        c = np.random.randn(8) * 0.1
        c[7] = -1.0 + np.random.randn() * 0.1 # Keep constant term near -1
        inits.append(c)
        
    # 3. Two squares structure
    # Roots of (z^4 - r1^4)(z^4 - r2^4 * e^{i theta * 4})? No, simpler to just randomize.
    # Let's try some structured ones.
    # (z^2 - a)(z^2 - b)(z^2 - c)(z^2 - d)
    for _ in range(5):
        a = np.random.rand()
        b = np.random.rand()
        c = np.random.rand()
        d = np.random.rand()
        # Expand (z^2-a)(z^2-b) = z^4 - (a+b)z^2 + ab
        # This is getting complicated to expand manually. 
        # Just use random coeffs for now, the optimizer will find structure.
        c_rand = np.random.randn(8) * 0.5
        inits.append(c_rand)

    # Optimization
    for init in inits:
        if time.time() > deadline - 15.0:
            break
        try:
            res = minimize(
                objective,
                init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < 1e9:
                # Extract roots
                poly_coeffs = np.concatenate(([1.0], res.x))
                roots = np.roots(poly_coeffs)
                if np.all(np.isfinite(roots)):
                    pts = np.column_stack([roots.real, roots.imag])
                    s = try_candidate(pts)
                    
                    # Polish with Nelder-Mead on the points directly if good
                    if s > 1.01 and time.time() < deadline - 5.0:
                        # We can also try to optimize the points directly from this start
                        # But the polynomial representation is the point of this change.
                        # Let's do a second pass with tighter tolerance on coeffs.
                        res2 = minimize(
                            objective,
                            res.x,
                            method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-14}
                        )
                        if res2.fun < 1e9:
                            poly_coeffs2 = np.concatenate(([1.0], res2.x))
                            roots2 = np.roots(poly_coeffs2)
                            if np.all(np.isfinite(roots2)):
                                pts2 = np.column_stack([roots2.real, roots2.imag])
                                try_candidate(pts2)
        except Exception:
            pass

    # Fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])

    # Final validation
    best_pts = np.asarray(best_pts)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])

    # Scale to ensure coords are small
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()
