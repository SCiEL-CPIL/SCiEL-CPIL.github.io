"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class Optimizer:
    """GP-EI global exploration with proper hyperparameter fitting, followed by Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, n, dim, skip=1):
        """Generate Halton sequence for space-filling initial design."""
        points = []
        for i in range(skip, skip + n):
            point = []
            for d in range(1, dim + 1):
                # Prime for dimension d
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                p = primes[d - 1]
                f = 1.0
                i_val = i
                while i_val > 0:
                    f /= p
                    i_val = i_val // p
                point.append(f)
            points.append(point)
        return np.array(points)

    def _rbf_kernel(self, A, B, length_scale):
        """Compute RBF kernel between sets A and B."""
        sq_dists = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-sq_dists / (2 * length_scale**2))

    def _gp_predict(self, X_train, Y_train, X_pred, length_scale, sigma_f, noise=1e-8):
        """Compute GP mean and std at X_pred given training data."""
        n = len(X_train)
        K = self._rbf_kernel(X_train, X_train, length_scale) + noise * np.eye(n)
        Kf = self._rbf_kernel(X_pred, X_train, length_scale)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            mu = Kf @ alpha
            v = np.linalg.solve(L, Kf.T)
            var = sigma_f**2 - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-12)
            std = np.sqrt(var)
        except np.linalg.LinAlgError:
            mu = np.full(len(X_pred), np.mean(Y_train))
            std = np.full(len(X_pred), np.std(Y_train) + 1e-6)
        
        return mu, std

    def _expected_improvement(self, f_best, mu, std):
        """Compute Expected Improvement for minimization."""
        z = (f_best - mu) / (std + 1e-10)
        ei = (f_best - mu) * norm.cdf(z) + std * norm.pdf(z)
        ei = np.where(std < 1e-8, 0.0, ei)
        return ei

    def _fit_gp_hyperparams(self, X, Y, n_trials=20, rng=None):
        """Fit GP hyperparameters via marginal likelihood maximization."""
        if rng is None:
            rng = np.random.default_rng(42)
        
        n = len(X)
        if n < 5:
            return 1.0, 1.0
        
        best_lml = -np.inf
        best_ls = 1.0
        best_sf = 1.0
        
        # Grid search over log-scale
        ls_candidates = np.logspace(-1, 1, 10)  # 0.1 to 10
        sf_candidates = np.logspace(-1, 1, 10)  # 0.1 to 10
        
        noise = 1e-8
        for ls in ls_candidates:
            for sf in sf_candidates:
                K = self._rbf_kernel(X, X, ls) + noise * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                    lml = -0.5 * Y @ np.linalg.solve(L.T, np.linalg.solve(L, Y)) \
                          - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
                    if lml > best_lml:
                        best_lml = lml
                        best_ls = ls
                        best_sf = sf
                except np.linalg.LinAlgError:
                    continue
        
        return best_ls, best_sf

    def _nelder_mead(self, func, x0, budget, rng, dim, bounds=(-5, 5)):
        """Nelder-Mead simplex method for local refinement."""
        if budget < dim + 2:
            return x0, func(x0)
        
        # Initialize simplex
        x0 = np.array(x0, dtype=float)
        f0 = func(x0)
        
        # Create initial simplex
        simplex = [x0.copy()]
        simplex_f = [f0]
        
        for i in range(dim):
            xi = x0.copy()
            step = 0.5 if xi[i] < bounds[0] + 0.5 else -0.5
            xi[i] += step
            xi = np.clip(xi, bounds[0], bounds[1])
            fi = func(xi)
            simplex.append(xi)
            simplex_f.append(fi)
        
        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        max_iter = budget - (dim + 1)
        
        best_x = x0.copy()
        best_f = f0
        
        for _ in range(max_iter):
            # Sort simplex by function value
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()
            
            # Check convergence
            if len(simplex_f) > 1 and abs(simplex_f[0] - simplex_f[-1]) < 1e-8:
                break
            
            # Centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, bounds[0], bounds[1])
            fr = func(xr)
            
            if fr < simplex_f[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, bounds[0], bounds[1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                # Simple reflection
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Contraction
                if fr < simplex_f[-1]:
                    # Outside contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (simplex[-1] - centroid)
                xc = np.clip(xc, bounds[0], bounds[1])
                fc = func(xc)
                
                if fc < simplex_f[-1]:
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                        simplex_f[i] = func(simplex[i])
        
        return best_x, best_f

    def __call__(self, func):
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Space-filling initial design
        n_init = min(20, budget // 5)
        halton_pts = self._halton(n_init, dim, skip=1)
        # Map Halton [0,1]^d to [-5,5]^d
        halton_pts = halton_pts * 10.0 - 5.0
        halton_pts = np.clip(halton_pts, -5, 5)
        
        X = np.zeros((0, dim))
        Y = np.zeros(0)
        
        def add_point(x, f):
            nonlocal X, Y
            X = np.vstack([X, x.reshape(1, -1)])
            Y = np.append(Y, f)
        
        best_x = None
        best_f = np.inf
        
        for x0 in halton_pts:
            f0 = func(x0)
            add_point(x0, f0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
        
        # Phase 2: GP-EI exploration
        gp_budget = min(80, budget - n_init - 30)  # Leave budget for local refinement
        
        # Fit initial hyperparameters
        length_scale, sigma_f = self._fit_gp_hyperparams(X, Y, rng=rng)
        
        for i in range(gp_budget):
            # Re-fit hyperparameters periodically
            if i % 10 == 0 and len(X) >= 10:
                length_scale, sigma_f = self._fit_gp_hyperparams(X, Y, rng=rng)
            
            # Generate diverse candidates
            n_candidates = 100
            candidates = np.zeros((n_candidates, dim))
            
            # 40% random in the box
            n_rand = int(n_candidates * 0.4)
            candidates[:n_rand] = rng.uniform(-5, 5, (n_rand, dim))
            
            # 30% local perturbations around best point
            n_local = int(n_candidates * 0.3)
            if best_x is not None:
                for j in range(n_local):
                    candidates[n_rand + j] = best_x + rng.standard_normal(dim) * 0.5
            
            # 30% near top-k good points
            n_recent = n_candidates - n_rand - n_local
            if len(X) > 0:
                sorted_idx = np.argsort(Y)
                top_k = sorted_idx[:min(5, len(X))]
                for j in range(n_recent):
                    base = X[top_k[j % len(top_k)]]
                    candidates[n_rand + n_local + j] = base + rng.standard_normal(dim) * 0.3
            
            candidates = np.clip(candidates, -5, 5)
            
            # Compute GP predictions
            mu, std = self._gp_predict(X, Y, candidates, length_scale, sigma_f)
            
            # Compute Expected Improvement
            f_best_gp = np.min(Y)
            ei = self._expected_improvement(f_best_gp, mu, std)
            
            # Select next point using EI with some exploration
            ei_max = np.max(ei)
            ei_min = np.min(ei)
            if ei_max - ei_min < 1e-10:
                x_next = candidates[rng.integers(len(candidates))]
            else:
                # Temperature-based selection with decreasing temperature
                temperature = max(0.05, (ei_max - ei_min) * 0.05 * (1 - i / gp_budget))
                ei_shifted = ei - ei_min
                probs = np.exp(ei_shifted / temperature)
                probs = probs / np.sum(probs)
                x_next = candidates[rng.choice(len(candidates), p=probs)]
            
            x_next = np.clip(x_next, -5, 5)
            f_next = func(x_next)
            add_point(x_next, f_next)
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
        
        # Phase 3: Local refinement with Nelder-Mead
        remaining_budget = budget - len(X)
        
        if remaining_budget > 10 and best_x is not None:
            best_x, best_f = self._nelder_mead(func, best_x, remaining_budget, rng, dim)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
