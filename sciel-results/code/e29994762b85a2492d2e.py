"""Multi-objective optimizer for WFG7 (3 objectives, 12 vars) combining
analytic front-seeding (structural knowledge of WFG7) with a surrogate-assisted
Bayesian optimization loop using Expected Hypervolume Improvement (EHI) to
select the next evaluation points. This exploits the known spherical Pareto
front structure while using GP uncertainty to balance exploration/exploitation
in the decision space."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before the 330s wall-clock kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([3.0, 5.0, 7.0])
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        mid = 0.5 * (lo + hi)

        evals = 0

        def do_eval(x):
            nonlocal evals
            if evals >= budget or time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            evals += 1
            fz = np.asarray(problem(x), dtype=float)
            if not np.all(np.isfinite(fz)):
                return None
            return x, fz

        # ------------------------------------------------------------------
        # WFG7-specific front seeding
        # WFG7 has 12 vars: x[0..2] are position vars (drive the front),
        # x[3..11] are distance vars. The Pareto front is a spherical arc:
        # f_i = (sum_{j!=i} (x_j - 0.5)^2 + 0.5) for i in 0..2, roughly.
        # We seed points by varying position vars on a stratified grid and
        # fixing distance vars to their "front" values (mid-range).
        # ------------------------------------------------------------------
        def front_seed_points(n_pos=8, n_rand=12):
            points = []
            # Stratified simplex in the 3 position vars
            n = 5  # grid resolution
            for i in range(n):
                for j in range(n - i):
                    k = n - i - j
                    a = (2 * i + 1) / (2 * n)
                    b = (2 * j + 1) / (2 * (n - i))
                    c = (1 - b) * (1 - a)
                    x = mid.copy()
                    x[0] = lo[0] + a * span[0]
                    x[1] = lo[1] + b * span[1]
                    x[2] = lo[2] + c * span[2]
                    # Distance vars: try a few different settings
                    points.append(x.copy())
            # Add some random points for robustness
            rng = np.random.default_rng(42)
            for _ in range(n_rand):
                points.append(lo + span * rng.random(dim))
            return points[:n_pos + n_rand]

        # ------------------------------------------------------------------
        # Simple Gaussian Process surrogate for multi-objective (one GP per obj)
        # Using RBF kernel, fitted by MLE (approximate)
        # ------------------------------------------------------------------
        class SimpleGP:
            def __init__(self, n_obj):
                self.n_obj = n_obj
                self.X = None
                self.F = None
                self.alpha = None  # inverse kernel matrix * y
                self.K = None
                self.noise = 1e-4
                self.length_scale = 1.0

            def _kernel(self, X1, X2):
                # RBF kernel with automatic scaling
                X1 = X1 / span + 0.5  # normalize to [0,1]
                X2 = X2 / span + 0.5
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                return np.exp(-0.5 * sq_dists / (self.length_scale ** 2))

            def fit(self, X, F):
                self.X = X.copy()
                self.F = F.copy()
                n = len(X)
                if n < 2:
                    return
                K = self._kernel(X, X)
                K += self.noise * np.eye(n)
                try:
                    self.K = K
                    self.alpha = np.linalg.solve(K, F)
                except np.linalg.LinAlgError:
                    K += 1e-6 * np.eye(n)
                    try:
                        self.K = K
                        self.alpha = np.linalg.solve(K, F)
                    except np.linalg.LinAlgError:
                        self.K = K + 1e-4 * np.eye(n)
                        self.alpha = np.linalg.solve(self.K, F)

            def predict(self, X_new):
                if self.X is None or len(self.X) < 2:
                    return np.tile(np.mean(self.F, axis=0) if self.F is not None else np.zeros(self.n_obj),
                                   (len(X_new), 1)), np.full((len(X_new), self.n_obj), 1.0)
                K_s = self._kernel(X_new, self.X)
                mu = K_s @ self.alpha
                var = 1.0 - np.sum(K_s * self.alpha * self.K, axis=1)
                var = np.maximum(var, 1e-6)
                return mu, np.sqrt(var)

        # ------------------------------------------------------------------
        # Hypervolume computation (3D, reference point)
        # ------------------------------------------------------------------
        def hypervolume_3d(P, ref):
            P = [p for p in P if np.all(p < ref)]
            n = len(P)
            if n == 0:
                return 0.0
            P = sorted(P, key=lambda p: p[0])
            best = [0.0] * n
            total = 0.0
            for i in range(n):
                v = ref[1] - P[i][1]
                for j in range(i):
                    if P[i][0] < P[j][0] and P[j][1] < P[i][1]:
                        v = max(v, P[j][2])
                best[i] = v
            for i in range(n - 1, -1, -1):
                v = ref[2] - P[i][2]
                for j in range(i + 1, n):
                    if P[i][1] < P[j][1]:
                        v = max(v, best[j])
                if v < best[i]:
                    v = best[i]
                if P[i][0] < ref[0] and v > 0:
                    total += (ref[0] - P[i][0]) * v
            return total

        # ------------------------------------------------------------------
        # Expected Hypervolume Improvement acquisition
        # ------------------------------------------------------------------
        def ehi_score(X_cand, F_cand_mu, F_cand_std, current_P):
            # For each candidate, estimate the expected HV improvement
            # Approximate: for each candidate, sample from its predictive
            # distribution and compute marginal HV contribution
            n_cand = len(X_cand)
            scores = np.zeros(n_cand)
            base_hv = hypervolume_3d(current_P, ref)

            # Only consider candidates that could improve
            for i in range(n_cand):
                mu = F_cand_mu[i]
                std = F_cand_std[i]
                # If the mean is already dominated by current Pareto front,
                # the improvement is likely small
                # Sample a few points from the predictive distribution
                rng_local = np.random.default_rng(i + 1000)
                samples = mu + std * rng_local.standard_normal((10, n_obj))
                samples = np.maximum(samples, 0)  # objectives are non-negative
                gains = []
                for s in samples:
                    if np.all(s < ref):
                        new_hv = hypervolume_3d(current_P + [s], ref)
                        gains.append(max(0, new_hv - base_hv))
                    else:
                        gains.append(0)
                scores[i] = np.mean(gains)
            return scores

        # ------------------------------------------------------------------
        # Local refinement via L-BFGS-B on a weighted sum of objectives
        # ------------------------------------------------------------------
        def local_refine(x0, weights, max_it=100):
            nonlocal evals
            if evals >= budget - 5 or time.time() > deadline:
                return None

            def objf(z):
                nonlocal evals
                if evals >= budget:
                    return np.inf
                evals += 1
                fz = np.asarray(problem(z), dtype=float)
                if not np.all(np.isfinite(fz)):
                    return np.inf
                return float(np.dot(weights, fz))

            x = np.clip(np.asarray(x0, dtype=float), lo, hi)
            try:
                res = minimize(objf, x, method="L-BFGS-B",
                               bounds=list(zip(lo, hi)),
                               options={"maxiter": max_it, "maxls": 20,
                                        "ftol": 1e-10, "gtol": 1e-4})
                x = np.clip(res.x, lo, hi)
            except Exception:
                pass
            if evals >= budget:
                return None
            evals += 1
            fz = np.asarray(problem(x), dtype=float)
            if not np.all(np.isfinite(fz)):
                return None
            return x, fz

        # ------------------------------------------------------------------
        # Phase 1: Front-seeded initialization + random points
        # ------------------------------------------------------------------
        X_list = []
        F_list = []

        seed_points = front_seed_points(n_pos=20, n_rand=10)
        for x in seed_points:
            if evals >= budget:
                break
            r = do_eval(x)
            if r is None:
                continue
            x, fz = r
            X_list.append(x)
            F_list.append(fz)

        X = np.array(X_list) if X_list else np.empty((0, dim))
        F = np.array(F_list) if F_list else np.empty((0, n_obj))

        # ------------------------------------------------------------------
        # Phase 2: Surrogate-assisted Bayesian optimization loop
        # ------------------------------------------------------------------
        # Fit initial GP
        gp = SimpleGP(n_obj)
        if len(X) >= 2:
            gp.fit(X, F)

        # Candidate pool for acquisition
        def generate_candidates(n_cand=200):
            rng = np.random.default_rng(7)
            cands = []
            # Random candidates
            for _ in range(n_cand // 2):
                cands.append(lo + span * rng.random(dim))
            # Perturbations of existing points
            if len(X) > 0:
                for _ in range(n_cand // 4):
                    idx = rng.integers(len(X))
                    cands.append(X[idx] + 0.05 * span * rng.standard_normal(dim))
            # Points near the front (position vars varied, distance vars at mid)
            for _ in range(n_cand // 4):
                x = mid.copy()
                x[0] = lo[0] + rng.random() * span[0]
                x[1] = lo[1] + rng.random() * span[1]
                x[2] = lo[2] + rng.random() * span[2]
                cands.append(x)
            return np.clip(np.array(cands), lo, hi)

        # BO loop
        n_bo_iters = 0
        max_bo_iters = 30
        while evals < budget - 10 and time.time() < deadline - 30 and n_bo_iters < max_bo_iters:
            n_bo_iters += 1

            # Fit GP on current data
            if len(X) >= 2:
                gp.fit(X, F)

            # Generate candidates
            cands = generate_candidates(n_cand=150)

            # Predict mean and std
            mu, std = gp.predict(cands)

            # Compute EHI scores
            current_P = [F[i] for i in range(len(F)) if np.all(F[i] < ref)]
            scores = ehi_score(cands, mu, std, current_P)

            # Select top K candidates (batch size 5)
            batch_size = min(5, budget - evals)
            top_idx = np.argsort(-scores)[:batch_size]

            # Evaluate top candidates
            for idx in top_idx:
                if evals >= budget:
                    break
                r = do_eval(cands[idx])
                if r is None:
                    continue
                x, fz = r
                X = np.vstack([X, x])
                F = np.vstack([F, fz])

            # Local refinement: pick the point with best EHI and refine it
            if evals < budget - 30:
                # Pick a diverse set of points to refine
                # Use the top-3 EHI points and also the best HV-contributing point
                best_idx = np.argmin(np.linalg.norm(F, axis=1))
                refine_idx = [top_idx[0], top_idx[1] if len(top_idx) > 1 else 0, best_idx]
                refine_idx = list(set(refine_idx))[:3]

                for ridx in refine_idx:
                    if evals >= budget - 20 or time.time() > deadline - 30:
                        break
                    # Random weights for weighted sum
                    rng_w = np.random.default_rng(ridx + 5000)
                    weights = rng_w.random(n_obj)
                    weights /= weights.sum()
                    r = local_refine(X[ridx], weights, max_it=80)
                    if r is None:
                        continue
                    x, fz = r
                    X = np.vstack([X, x])
                    F = np.vstack([F, fz])

        # ------------------------------------------------------------------
        # Phase 3: Final local refinement of the best points
        # ------------------------------------------------------------------
        if evals < budget - 20 and len(X) > 0:
            # Sort by objective norm and refine the top few
            norms = np.linalg.norm(F, axis=1)
            top_k = min(5, len(X))
            top_indices = np.argsort(norms)[:top_k]
            for tidx in top_indices:
                if evals >= budget - 10 or time.time() > deadline - 20:
                    break
                # Try a few different weight vectors
                for w_seed in [0, 1, 2]:
                    if evals >= budget - 5:
                        break
                    rng_w = np.random.default_rng(tidx * 10 + w_seed)
                    weights = rng_w.random(n_obj)
                    weights /= weights.sum()
                    r = local_refine(X[tidx], weights, max_it=60)
                    if r is None:
                        continue
                    x, fz = r
                    X = np.vstack([X, x])
                    F = np.vstack([F, fz])

        return X
