"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Quadratic model estimation + Nelder-Mead polish for Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        self.calls = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)
        self.dim = self.dim
        N = self.dim
        bounds_low = -5.0
        bounds_high = 5.0

        def wrapped_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")

            x = np.clip(x, bounds_low, bounds_high)

            val = func(x)
            self.calls += 1

            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()

            return val

        # PHASE 1: QUADRATIC MODEL FITTING
        # We assume f(x) = x^T A x + b^T x + c
        # We need to estimate A (symmetric NxN), b (Nx1), c (scalar)
        # Total params: N(N+1)/2 + N + 1
        # For N=5, params = 15 + 5 + 1 = 21
        # We will use 25 points for least squares
        
        D = 4.0
        points = []
        
        # 1. Origin
        points.append(np.zeros(N))
        
        # 2. Axis points +/- D
        for i in range(N):
            p_pos = np.zeros(N)
            p_pos[i] = D
            points.append(p_pos)
            
            p_neg = np.zeros(N)
            p_neg[i] = -D
            points.append(p_neg)
            
        # 3. Pairwise points to capture off-diagonal terms
        # Use D * (e_i + e_j) for i < j
        for i in range(N):
            for j in range(i + 1, N):
                p_pair = np.zeros(N)
                p_pair[i] = D
                p_pair[j] = D
                points.append(p_pair)
                
                p_pair_neg = np.zeros(N)
                p_pair_neg[i] = -D
                p_pair_neg[j] = -D
                points.append(p_pair_neg)

        # If we have more points than needed, we can add random ones or just use these.
        # For N=5: 1 + 10 + 10 = 21 points.
        # Let's add a few random points for robustness if budget allows, or just stick to 21-25.
        # Let's add 4 random points to make it 25.
        rng = np.random.default_rng(42)
        for _ in range(4):
            p_rand = rng.uniform(-D, D, N)
            points.append(p_rand)

        values = []
        for p in points:
            try:
                v = wrapped_func(p)
                values.append(v)
            except (ValueError, TimeoutError, Exception):
                break

        if len(values) < N + 1:
            # Fallback: Random search
            for _ in range(20):
                if self.calls >= self.budget: break
                x = rng.uniform(bounds_low, bounds_high, N)
                try:
                    wrapped_func(x)
                except Exception:
                    break
            return (self.best_x, self.best_f)

        # Construct design matrix for least squares
        # f(x) = sum_i sum_j A_ij x_i x_j + sum_i b_i x_i + c
        # We only need unique A_ij due to symmetry.
        # Let's vectorize the quadratic terms.
        # x^T A x = sum_{i<=j} A_{ij} x_i x_j (with factor 2 for i!=j if we use full matrix, 
        # but easier to just create features for all unique pairs)
        
        M = len(values)
        # Number of features: N(N+1)/2 (quadratic) + N (linear) + 1 (const)
        n_quad = N * (N + 1) // 2
        n_feat = n_quad + N + 1
        
        X = np.zeros((M, n_feat))
        y = np.array(values)
        
        for k, p in enumerate(points[:M]):
            idx = 0
            # Quadratic terms
            for i in range(N):
                for j in range(i, N):
                    X[k, idx] = p[i] * p[j]
                    idx += 1
            # Linear terms
            for i in range(N):
                X[k, idx] = p[i]
                idx += 1
            # Constant
            X[k, idx] = 1.0

        try:
            # Solve least squares
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            
            # Extract A, b, c
            A_flat = coeffs[:n_quad]
            b = coeffs[n_quad:n_quad+N]
            c = coeffs[n_quad+N]
            
            # Reconstruct A (symmetric)
            A = np.zeros((N, N))
            idx = 0
            for i in range(N):
                for j in range(i, N):
                    val = A_flat[idx]
                    A[i, j] = val
                    if i != j:
                        A[j, i] = val
                    idx += 1
            
            # Find minimum: grad f = 2 A x + b = 0 => x* = -0.5 A^{-1} b
            # Check if A is invertible
            try:
                x_est = -0.5 * np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse or just use best found so far
                x_est = self.best_x
                
        except Exception:
            x_est = self.best_x

        x_start = np.clip(x_est, bounds_low, bounds_high)

        # Evaluate start point if not already done (it might be one of the points, but usually not)
        try:
            if not np.allclose(x_start, self.best_x, atol=1e-8):
                wrapped_func(x_start)
        except Exception:
            pass

        # PHASE 2: NELDER-MEAD POLISH
        if self.calls < self.budget and time.time() < deadline:
            try:
                self._run_nelder_mead(wrapped_func, x_start, N)
            except Exception:
                pass

        return (self.best_x, self.best_f)

    def _run_nelder_mead(self, func, x0, dim):
        """Nelder-Mead implementation."""
        x0 = np.clip(np.asarray(x0, dtype=float), -5.0, 5.0)
        
        # Initialize simplex
        pts = [x0.copy()]
        vals = [func(x0)]
        
        # Add dim more points
        for i in range(dim):
            p = x0.copy()
            p[i] = x0[i] + 0.25
            p = np.clip(p, -5.0, 5.0)
            pts.append(p)
            vals.append(func(p))
            
            if self.calls >= self.budget:
                return

        # Sort
        idx = np.argsort(vals)
        pts = [pts[i] for i in idx]
        vals = [vals[i] for i in idx]
        
        # Update best
        if vals[0] < self.best_f:
            self.best_f = vals[0]
            self.best_x = pts[0].copy()

        lam, gam, rho = 1.0, 0.5, -1.0
        
        # Main loop
        while self.calls < self.budget and time.time() < deadline - 5:
            centroid = np.mean(pts[:-1], axis=0)
            
            # Reflection
            xr = centroid + lam * (centroid - pts[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = func(xr)
            
            if fr < vals[0]:
                # Expansion
                xe = centroid + gam * (xr - centroid)
                xe = np.clip(xe, -5.0, 5.0)
                fe = func(xe)
                
                if fe < fr:
                    pts[-1] = xe
                    vals[-1] = fe
                else:
                    pts[-1] = xr
                    vals[-1] = fr
            elif fr < vals[-2]:
                # Accept reflection
                pts[-1] = xr
                vals[-1] = fr
            else:
                # Contraction
                xc = centroid + rho * (centroid - pts[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = func(xc)
                
                if fc < vals[-1]:
                    pts[-1] = xc
                    vals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(pts)):
                        pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                        pts[i] = np.clip(pts[i], -5.0, 5.0)
                        vals[i] = func(pts[i])
                        
                        if self.calls >= self.budget:
                            return

            # Re-sort
            idx = np.argsort(vals)
            pts = [pts[i] for i in idx]
            vals = [vals[i] for i in idx]
            
            if vals[0] < self.best_f:
                self.best_f = vals[0]
                self.best_x = pts[0].copy()

        return (self.best_x, self.best_f)

# EVOLVE-BLOCK-END
