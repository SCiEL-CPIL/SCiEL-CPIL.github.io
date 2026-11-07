"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class Optimizer:
    """GP-EI global exploration with adaptive hyperparameters, followed by CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _rbf_kernel(self, A, B, length_scale):
        """Compute RBF kernel between sets A and B."""
        sq_dists = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-sq_dists / (2 * length_scale**2))

    def _gp_predict(self, X_train, Y_train, X_pred, length_scale, sigma_f):
        """Compute GP mean and std at X_pred given training data."""
        n = len(X_train)
        K = self._rbf_kernel(X_train, X_train, length_scale) + 1e-8 * np.eye(n)
        Kf = self._rbf_kernel(X_pred, X_train, length_scale)
        
        try:
            # Use Cholesky for numerical stability
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            mu = Kf @ alpha
            v = np.linalg.solve(L, Kf.T)
            var = sigma_f**2 - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
        except np.linalg.LinAlgError:
            # Fallback: uniform predictions
            mu = np.full(len(X_pred), np.mean(Y_train))
            std = np.full(len(X_pred), np.std(Y_train) + 1e-6)
        
        return mu, std

    def _expected_improvement(self, f_best, mu, std):
        """Compute Expected Improvement for minimization."""
        z = (f_best - mu) / (std + 1e-10)
        ei = (f_best - mu) * norm.cdf(z) + std * norm.pdf(z)
        ei = np.where(std < 1e-8, 0.0, ei)
        return ei

    def _cma_es_refine(self, func, x0, f0, budget, rng, dim):
        """Simple CMA-ES for local refinement."""
        if budget < 10:
            return x0, f0
        
        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))
        pop_size = min(pop_size, budget // 3)
        if pop_size < 2:
            pop_size = 2
        
        x = x0.copy()
        sigma = 0.5
        C = np.eye(dim)
        cs = 1 / (pop_size + dim + 2)
        cmu = min(0.4, 2 / ((dim + 2)**2 - 1))
        cc = (4 + dim / 4) / (dim + 4 + 2 / cs)
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        ps = np.zeros(dim)
        remaining = budget
        best_x = x0.copy()
        best_f = f0
        no_improve = 0
        
        while remaining > 0 and sigma > 1e-8:
            # Generate population
            offspring = []
            for j in range(pop_size):
                if remaining <= 0:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (C @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                remaining -= 1
                offspring.append((fy, y))
            
            if not offspring:
                break
            
            offspring.sort(key=lambda t: t[0])
            
            # Update best
            for fy, y in offspring:
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            # Check for improvement
            if offspring[0][0] < f0:
                no_improve = 0
            else:
                no_improve += 1
            
            # Update mean
            weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
            weights = weights / np.sum(weights)
            mu_eff = 1 / np.sum(weights[:pop_size // 2]**2)
            mu = np.zeros(dim)
            for i in range(min(pop_size // 2, len(offspring))):
                mu += weights[i] * offspring[i][1]
            mu = mu / np.sum(weights[:min(pop_size // 2, len(offspring))])
            
            # Update covariance
            C = (1 - cmu) * C
            for i in range(min(pop_size // 2, len(offspring))):
                d = offspring[i][1] - mu
                C += weights[i] * np.outer(d, d)
            
            # Update sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (C @ (mu - x))
            sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-8, 10)
            
            # Update mean
            x = mu
            f0 = offspring[0][0]
            
            if no_improve > pop_size * 3:
                break
        
        return best_x, best_f

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: GP-based exploration with adaptive hyperparameters
        gp_budget = min(60, budget // 4)
        
        X = np.zeros((0, dim))
        Y = np.zeros(0)
        
        def add_point(x, f):
            nonlocal X, Y
            X = np.vstack([X, x.reshape(1, -1)])
            Y = np.append(Y, f)

        # Initial sampling: stratified points to cover the space
        init_points = [np.zeros(dim)]
        # Add corners and midpoints for better initial coverage
        for i in range(dim):
            p1 = np.zeros(dim)
            p2 = np.zeros(dim)
            p1[i] = -4.0
            p2[i] = 4.0
            init_points.append(p1)
            init_points.append(p2)
        
        # Deduplicate and limit initial points
        init_points = list(set(tuple(p) for p in init_points))
        init_points = [np.array(p) for p in init_points[:8]]
        
        best_x = None
        best_f = np.inf
        
        for x0 in init_points:
            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            add_point(x0, f0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
        
        # Adaptive GP exploration
        length_scale = 1.5  # Initial length scale
        sigma_f = 1.0  # Initial signal variance
        
        for i in range(gp_budget - len(init_points)):
            if len(X) == 0:
                break
            
            # Adapt hyperparameters based on data spread
            if len(X) >= 5:
                # Estimate length scale from data
                dists = np.sqrt(np.sum((X[:, None, :] - X[None, :, :])**2, axis=-1))
                dists = dists[np.triu_indices(len(X), k=1)]
                if len(dists) > 0 and np.mean(dists) > 1e-6:
                    # Scale length_scale based on typical distance
                    length_scale = np.clip(np.mean(dists) * 0.5, 0.3, 3.0)
                # Estimate sigma_f
                if len(Y) > 1:
                    sigma_f = np.clip(np.std(Y) + 0.1, 0.1, 10.0)
            
            # Generate diverse candidates: mix of random and local perturbations
            n_candidates = 80
            candidates = np.zeros((n_candidates, dim))
            
            # 50% random in the box
            n_rand = n_candidates // 2
            candidates[:n_rand] = rng.uniform(-5, 5, (n_rand, dim))
            
            # 30% local perturbations around best points
            n_local = n_candidates // 3
            if best_x is not None:
                for j in range(n_local):
                    candidates[n_rand + j] = best_x + rng.standard_normal(dim) * 0.8
            
            # 20% near recent good points
            n_recent = n_candidates - n_rand - n_local
            if len(X) > 0:
                # Pick a few recent points with low f
                sorted_idx = np.argsort(Y)
                recent_idx = sorted_idx[:min(3, len(X))]
                for j in range(n_recent):
                    base = X[recent_idx[j % len(recent_idx)]]
                    candidates[n_rand + n_local + j] = base + rng.standard_normal(dim) * 0.5
            
            candidates = np.clip(candidates, -5, 5)
            
            # Compute GP predictions
            mu, std = self._gp_predict(X, Y, candidates, length_scale, sigma_f)
            
            # Compute Expected Improvement
            f_best_gp = np.min(Y)
            ei = self._expected_improvement(f_best_gp, mu, std)
            
            # Select next point with some exploration
            # Use softmax sampling to avoid getting stuck
            ei_max = np.max(ei)
            ei_min = np.min(ei)
            if ei_max - ei_min < 1e-10:
                x_next = candidates[rng.integers(len(candidates))]
            else:
                # Temperature-based selection
                temperature = max(0.1, (ei_max - ei_min) * 0.1)
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
        
        # Phase 2: CMA-ES local refinement
        remaining_budget = budget - len(X)
        
        if remaining_budget > 10 and best_x is not None:
            best_x, best_f = self._cma_es_refine(func, best_x, best_f, remaining_budget, rng, dim)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
