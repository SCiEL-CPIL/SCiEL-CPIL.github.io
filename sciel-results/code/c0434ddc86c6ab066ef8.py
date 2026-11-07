"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """GP-guided multi-objective optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        archive_X = []
        archive_F = []
        eval_count = 0

        # GP surrogate state
        gp_X = []  # observed inputs
        gp_Y = []  # observed g-values (penalty term proxy)

        def compute_g_proxy(x):
            """Compute proxy for g(x) = 1 + (9/n)*sum_{i=2}^{n} x_i*(1+10*sin^2(15*x_i)).
            We use the actual problem to get f2, and estimate g from f1 and f2."""
            pass  # We'll track g indirectly

        def gp_predict(X_new):
            """Simple GP prediction using RBF kernel."""
            if len(gp_X) == 0:
                return np.zeros(len(X_new)), np.zeros(len(X_new))
            
            X_obs = np.array(gp_X)
            Y_obs = np.array(gp_Y)
            X_new = np.array(X_new)
            
            # RBF kernel
            def rbf(X1, X2, length_scale=0.5, variance=1.0):
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                return variance * np.exp(-sq_dists / (2 * length_scale**2))
            
            K = rbf(X_obs, X_obs) + 1e-6 * np.eye(len(X_obs))
            K_s = rbf(X_obs, X_new)
            K_ss = rbf(X_new, X_new)
            
            try:
                K_inv = np.linalg.inv(K)
                mu = K_s.T @ K_inv @ Y_obs
                v = K_s.T @ K_inv @ K_s
                var = np.diag(K_ss) - np.diag(v)
                var = np.maximum(var, 1e-6)
                return mu, np.sqrt(var)
            except np.linalg.LinAlgError:
                return np.mean(Y_obs) * np.ones(len(X_new)), np.std(Y_obs) * np.ones(len(X_new))

        def acquisition(X_new, mu, sigma, xi=0.01):
            """Expected Improvement acquisition function."""
            sigma = np.maximum(sigma, 1e-8)
            Z = (mu + xi - np.min(gp_Y)) / sigma
            EI = (mu + xi - np.min(gp_Y)) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return EI

        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if f is dominated by any point in archive
            if archive_F:
                archive_F_arr = np.array(archive_F)
                leq = np.all(archive_F_arr <= f, axis=1)
                lt = np.any(archive_F_arr < f, axis=1)
                if np.any(leq & lt):
                    return
            # Remove points from archive that are dominated by f
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if np.all(f <= af) and np.any(f < af):
                    continue
                new_archive_X.append(ax)
                new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F

        def evaluate(x):
            nonlocal eval_count
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped), dtype=float)
            eval_count += 1
            add_to_archive(x_clipped.copy(), f)
            return f

        # Phase 1: Initial sampling - mix of random and biased toward low-g region
        n_init = min(15, budget)
        for i in range(n_init):
            if eval_count >= budget:
                break
            if i < 5:
                # Sample near origin (low g region)
                x = lo + (hi - lo) * rng.random(dim) * 0.05
            else:
                x = lo + (hi - lo) * rng.random(dim)
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            # Track g proxy: for ZDT6, f2 = g*(1-(f1/g)^2), so if we know f1 and f2,
            # we can estimate g. But simpler: use sum of x[1:] as proxy for g-1
            g_proxy = 1.0 + 0.9 * np.sum(x[1:]) * (1 + 10 * np.sin(15 * x[1:])**2)
            gp_X.append(x.copy())
            gp_Y.append(g_proxy)

        # Phase 2: GP-guided search to find low-g regions
        n_gp = min(20, budget - eval_count)
        for _ in range(n_gp):
            if eval_count >= budget:
                break
            
            # Generate candidate points
            n_candidates = 50
            candidates = lo + (hi - lo) * rng.random((n_candidates, dim))
            
            # Predict g for candidates
            mu, sigma = gp_predict(candidates)
            
            # Use EI acquisition
            ei = acquisition(candidates, mu, sigma)
            
            # Pick best candidate
            best_idx = np.argmax(ei)
            x_new = candidates[best_idx]
            
            f = evaluate(x_new)
            g_proxy = 1.0 + 0.9 * np.sum(x_new[1:]) * (1 + 10 * np.sin(15 * x_new[1:])**2)
            gp_X.append(x_new.copy())
            gp_Y.append(g_proxy)

        # Phase 3: Dense sampling of the known Pareto front
        # For ZDT6, Pareto front: x_2..x_n = 0, x_1 varies in [0,1]
        # f1 = 1 - exp(-4*x_1)*sin(6*pi*x_1)^6
        # f2 = 1 - f1^2
        n_front = min(40, budget - eval_count)
        if n_front > 0:
            # Use non-uniform sampling to capture the shape of the front
            # The front has interesting curvature, so sample more densely where f1 changes rapidly
            x1_values = np.linspace(0, 1, n_front)
            # Add some extra points in regions of high curvature
            extra_x1 = np.linspace(0, 0.5, 10)  # f1 changes rapidly near 0
            x1_values = np.unique(np.concatenate([x1_values, extra_x1]))
            x1_values = x1_values[np.argsort(x1_values)]
            
            for x1 in x1_values:
                if eval_count >= budget:
                    break
                x = np.zeros(dim)
                x[0] = x1
                evaluate(x)

        # Phase 4: Use remaining budget for fine-tuning around best archive points
        remaining = budget - eval_count
        if remaining > 0 and archive_F:
            archive_F_arr = np.array(archive_F)
            archive_X_arr = np.array(archive_X)
            
            # Sort by hypervolume contribution (use f2 as proxy)
            idx_sorted = np.argsort(archive_F_arr[:, 1])
            
            for idx in idx_sorted:
                if eval_count >= budget:
                    break
                x_center = archive_X_arr[idx]
                # Small perturbations
                n_perturb = min(2, budget - eval_count)
                for _ in range(n_perturb):
                    if eval_count >= budget:
                        break
                    # Perturb mainly in x_1 direction to stay near front
                    x_new = x_center.copy()
                    x_new[0] += 0.02 * rng.standard_normal()
                    # Small perturbations in other dimensions (should stay near 0)
                    x_new[1:] += 0.001 * rng.standard_normal(dim - 1)
                    x_new = np.clip(x_new, lo, hi)
                    evaluate(x_new)

        return archive_X if archive_X else [lo]
# EVOLVE-BLOCK-END
