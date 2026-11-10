# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Finds 14 points in 3D maximizing (d_min/d_max)^2.
    Uses gradient descent on a smooth proxy (minimizing sum of inverse distances)
    on the unit sphere, followed by Nelder-Mead refinement on the true objective.
    """
    n = 14
    d = 3
    t0 = time.time()
    deadline = t0 + 580.0  # Safety margin before 630s limit

    def remaining():
        return deadline - time.time()

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        return pts / norms

    def score(pts):
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        dmin = np.min(dists)
        dmax = np.max(dists)
        if dmax <= 1e-15 or dmin <= 1e-15:
            return 0.0
        return (dmin / dmax) ** 2

    def fib_sphere(n_pts):
        phi = np.pi * (3.0 - np.sqrt(5.0))
        pts = []
        for i in range(n_pts):
            y = 1.0 - 2.0 * i / (n_pts - 1)
            r = np.sqrt(max(0.0, 1.0 - y * y))
            th = phi * i
            pts.append([np.cos(th) * r, y, np.sin(th) * r])
        return np.array(pts)

    def grad_desc_sphere(pts, iters=2000, lr=0.008, mom=0.9, k=4):
        """
        Gradient descent on the sphere to minimize sum_{i!=j} 1/d_ij^k.
        This is a smooth proxy for maximizing d_min (since d_min is the bottleneck).
        """
        p = pts.copy()
        v = np.zeros_like(p)
        
        for it in range(iters):
            if remaining() < 5.0:
                break
            
            # Compute pairwise squared distances
            diff = p[:, None, :] - p[None, :, :]
            d2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(d2, 1e-12)
            
            # Compute gradient of sum 1/d^k = sum d^(-k)
            # d/dp_i (d_ij^(-k)) = -k * d_ij^(-k-2) * (p_i - p_j)
            # So grad = -k * sum_{j!=i} (p_i - p_j) / d_ij^(k+2)
            
            # For efficiency, we compute weights w_ij = 1/d_ij^(k+2)
            # d2^(k/2 + 1) = d^(k+2)
            exp_term = (k // 2) + 1
            if k % 2 == 0:
                w = 1.0 / (d2 ** exp_term)
            else:
                w = 1.0 / (d2 ** (exp_term - 0.5)) * (1.0 / np.sqrt(d2))
            
            np.fill_diagonal(w, 0.0)
            
            # grad_i = -k * sum_j w_ij (p_i - p_j)
            # sum_j w_ij (p_i - p_j) = p_i * sum_j w_ij - sum_j w_ij p_j
            sum_w = w.sum(axis=1, keepdims=True)
            sum_wp = (w[:, :, None] * p).sum(axis=1)
            
            grad = -k * (p * sum_w - sum_wp)
            
            # Momentum update
            v = mom * v - lr * grad
            p = p + v
            p = normalize(p)
            
        return p

    def nm_refine(pts, maxfev=2000):
        """
        Nelder-Mead refinement on the true objective, constrained to the sphere.
        """
        def obj(x):
            if remaining() < 2.0:
                return 1e6
            pp = x.reshape(n, d)
            pp = normalize(pp)
            s = score(pp)
            return -s if s > 0 else 1e6

        x0 = pts.flatten()
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxfev': maxfev, 'xatol': 1e-8, 'fatol': 1e-8})
            pp = normalize(res.x.reshape(n, d))
            return pp
        except Exception:
            return pts

    def sa_refine(pts, n_iter=5000, t0=0.05, t1=0.001):
        """
        Simulated Annealing refinement to escape local minima.
        Uses the true score as the objective.
        """
        p = pts.copy()
        current_score = score(p)
        best_p = p.copy()
        best_score = current_score
        rng = np.random.RandomState(42)
        
        for it in range(n_iter):
            if remaining() < 10.0:
                break
            
            # Temperature schedule (geometric cooling)
            t = t0 * (t1 / t0) ** (it / n_iter)
            
            # Propose a move: perturb one point slightly
            idx = rng.randint(n)
            step = 0.02 * t / t0
            new_p = p.copy()
            new_p[idx] += rng.randn(d) * step
            new_p = normalize(new_p)
            
            new_score = score(new_p)
            delta = new_score - current_score
            
            if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-10)):
                p = new_p
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_p = p.copy()
        
        return best_p

    # --- Generate diverse initializations ---
    inits = []
    inits.append(fib_sphere(n))

    np.random.seed(12345)
    for trial in range(10):
        pts = np.random.randn(n, d)
        inits.append(normalize(pts))

    # Perturbed Fibonacci
    for eps in [0.05, 0.1, 0.2]:
        pts = fib_sphere(n) + np.random.randn(n, d) * eps
        inits.append(normalize(pts))

    # Rotated Fibonacci
    fib = fib_sphere(n)
    for _ in range(4):
        R = Rotation.from_quat(np.random.randn(4)).as_matrix()
        inits.append(normalize(fib @ R.T))

    best_score = -1.0
    best_points = inits[0]

    # --- Phase 1: Gradient descent on smooth proxy ---
    # We use different k values (2 and 4) and learning rates to explore the landscape.
    # k=2 is smoother, k=4 is more aggressive in pushing points apart.
    gd_results = []
    for init in inits:
        if remaining() < 60.0:
            break
        
        for k in [2, 4]:
            for lr in [0.005, 0.01, 0.02]:
                if remaining() < 30.0:
                    break
                pts = grad_desc_sphere(init, iters=1500, lr=lr, mom=0.9, k=k)
                s = score(pts)
                gd_results.append((s, pts))
                if s > best_score:
                    best_score = s
                    best_points = pts

    # Sort by score and keep top distinct candidates
    gd_results.sort(key=lambda x: x[0], reverse=True)
    top = []
    for s, pts in gd_results:
        dup = False
        for _, p2 in top:
            # Check if similar
            if np.max(np.abs(pts - p2)) < 1e-3:
                dup = True
                break
        if not dup:
            top.append((s, pts))
        if len(top) >= 8:
            break

    # --- Phase 2: Nelder-Mead + Simulated Annealing refinement on top candidates ---
    for s, pts in top:
        if remaining() < 20.0:
            break
        
        # NM refinement
        refined = nm_refine(pts, maxfev=1500)
        rs = score(refined)
        if rs > best_score:
            best_score = rs
            best_points = refined

        # SA refinement from the NM result
        if remaining() > 40.0:
            sa_result = sa_refine(refined, n_iter=3000, t0=0.03, t1=0.0005)
            sa_score = score(sa_result)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_result

        # Try a second NM pass from a slightly perturbed version
        if remaining() > 30.0:
            perturbed = normalize(pts + np.random.randn(n, d) * 0.02)
            refined2 = nm_refine(perturbed, maxfev=1000)
            rs2 = score(refined2)
            if rs2 > best_score:
                best_score = rs2
                best_points = refined2

    # --- Phase 3: Final polish with SA + NM if time remains ---
    if remaining() > 60.0:
        for _ in range(3):
            if remaining() < 30.0:
                break
            # SA polish
            sa_result = sa_refine(best_points, n_iter=2000, t0=0.02, t1=0.0003)
            sa_score = score(sa_result)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_result
            
            # NM polish
            nm_result = nm_refine(best_points, maxfev=2000)
            nm_score = score(nm_result)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_result

    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        best_points = fib_sphere(n)
        best_points = normalize(best_points)

    return best_points

# EVOLVE-BLOCK-END
