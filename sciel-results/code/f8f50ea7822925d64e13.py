"""Program for covering the sphere with 130 equal caps using differentiable min-max optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def generate_grid(n_lat, n_lon):
        lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Use a moderately dense grid for the soft objective to balance accuracy and speed
    grid_main = generate_grid(80, 160)

    def lloyd_covering_step(pts, grid, n_iters=50, alpha=0.5):
        current = pts.copy()
        for it in range(n_iters):
            if time.time() > deadline - 15.0:
                break
            dots = np.dot(grid, current.T)
            nearest = np.argmax(dots, axis=1)
            new_pts = np.zeros_like(current)
            for i in range(N):
                mask = (nearest == i)
                if not np.any(mask):
                    new_pts[i] = current[i]
                    continue
                cell_pts = grid[mask]
                d = current[i] @ cell_pts.T
                d = np.clip(d, -1.0, 1.0)
                ang = np.arccos(d)
                temp = 3.0
                w = np.exp(ang * temp)
                w /= np.sum(w)
                new_c = np.sum(cell_pts * w[:, None], axis=0)
                nrm = np.linalg.norm(new_c)
                if nrm < 1e-10:
                    new_pts[i] = current[i]
                else:
                    new_pts[i] = new_c / nrm
            current = (1 - alpha) * current + alpha * new_pts
            current /= np.linalg.norm(current, axis=1, keepdims=True)
        return current

    def covering_radius(pts, grid=None):
        if grid is None:
            grid = grid_main
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        return float(np.max(np.min(ang, axis=1)))

    def soft_covering_and_grad(pts, beta=50.0, grid=None):
        """
        Computes the soft covering radius and its gradient using a differentiable min-max formulation.
        """
        if grid is None:
            grid = grid_main
        
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        
        min_d = np.min(ang, axis=1)
        m = np.max(min_d)
        
        # Soft max
        exp_terms = np.exp(beta * (min_d - m))
        sum_exp = np.sum(exp_terms)
        val = m + np.log(sum_exp) / beta
        
        # Gradient computation
        # d(val)/d(min_d_i) = (1/beta) * exp(beta*(min_d_i - m)) / sum_exp
        # d(min_d_i)/d(pts_j) is non-zero only for j = argmin(ang_i)
        # d(ang)/d(p) = -grid / sqrt(1 - dot^2) + (grid . p) / sqrt(1 - dot^2) * p
        #               = (p - dot * grid) / sqrt(1 - dot^2)  ... wait, derivative of arccos(x) is -1/sqrt(1-x^2)
        # d(ang)/d(p) = -1/sqrt(1-dot^2) * d(dot)/d(p) = -1/sqrt(1-dot^2) * grid
        
        # We need to project the gradient to the tangent space of the sphere for each point.
        # grad_j = sum_{i: argmin(ang_i)=j} coeff_i * (-grid_i / sqrt(1-dot_ij^2))
        
        argmins = np.argmin(ang, axis=1)
        coeffs = exp_terms / sum_exp  # (M,)
        
        # We need to accumulate contributions to each point j
        grad = np.zeros_like(pts)
        
        # For efficiency, we can use bincount or simple loops if N is small, but vectorized is better
        # Let's use an approach where we sum over grid points
        # For each grid point i, it contributes to point j = argmins[i]
        # Contribution: coeffs[i] * (-grid[i] / denom[i])
        
        # Compute denominators
        dots_safe = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
        denom = np.sqrt(1.0 - dots_safe * dots_safe)
        
        # Raw gradient vectors for each grid point
        # Shape (M, 3)
        raw_grads = -grid / denom[:, None]
        
        # Weighted raw gradients
        weighted_grads = raw_grads * coeffs[:, None]
        
        # Accumulate into points
        # grad[j] += weighted_grads[i] for all i where argmins[i] == j
        for j in range(N):
            mask = (argmins == j)
            if np.any(mask):
                grad[j] = np.sum(weighted_grads[mask], axis=0)
        
        # Project gradient to tangent space
        # grad_tangent = grad - (grad . p) * p
        dots_grad = np.sum(grad * pts, axis=1, keepdims=True)
        grad = grad - dots_grad * pts
        
        return val, grad

    def optimize_smooth(pts, beta=80.0, max_iter=30):
        if time.time() > deadline - 15.0:
            return pts
        x0 = pts.flatten()
        def obj(x):
            if time.time() > deadline - 8.0:
                return 1e10
            v, _ = soft_covering_and_grad(x.reshape(N, 3), beta)
            return v
        def jac(x):
            if time.time() > deadline - 8.0:
                return np.zeros_like(x)
            _, g = soft_covering_and_grad(x.reshape(N, 3), beta)
            return g.flatten()
        try:
            res = minimize(obj, x0, jac=jac, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-5})
            if res.fun < obj(x0):
                p = res.x.reshape(N, 3)
                p /= np.linalg.norm(p, axis=1, keepdims=True)
                return p
        except Exception:
            pass
        return pts

    best_pts = None
    best_rad = np.inf

    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
        if r == 0:
            pts = fibonacci_spiral(N)
        elif r == 1:
            rng = np.random.RandomState(42)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 2:
            rng = np.random.RandomState(123)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 3:
            rng = np.random.RandomState(789)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r >= 4 and best_pts is not None:
            rng = np.random.RandomState(1000 + r)
            scale = max(0.01, 0.04 - 0.005 * r)
            pts = best_pts + scale * rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            pts = fibonacci_spiral(N)
        
        cand = lloyd_covering_step(pts, grid_main, n_iters=40, alpha=0.5)
        cand = optimize_smooth(cand, beta=50, max_iter=20)
        cand = optimize_smooth(cand, beta=100, max_iter=15)
        cand = optimize_smooth(cand, beta=200, max_iter=10)
        
        rad = covering_radius(cand)
        if rad < best_rad:
            best_rad = rad
            best_pts = cand.copy()
            print(f"Restart {r}: {np.degrees(best_rad):.6f} deg")

    if best_pts is not None and time.time() < deadline - 10.0:
        fp = lloyd_covering_step(best_pts, grid_main, n_iters=20, alpha=0.3)
        fp = optimize_smooth(fp, beta=300, max_iter=10)
        fr = covering_radius(fp)
        if fr < best_rad:
            best_rad = fr
            best_pts = fp

    if best_pts is None:
        best_pts = fibonacci_spiral(N)

    return best_pts
