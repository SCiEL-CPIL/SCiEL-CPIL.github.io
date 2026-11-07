"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES for BBOB f4 (Büche-Rastrigin) with strict budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        call_count = 0
        max_calls = self.budget
        dim = self.dim

        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal call_count
            if call_count >= max_calls:
                raise ValueError("Budget exceeded")
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # --- CMA-ES Implementation ---
        
        # Parameters
        # Standard CMA-ES parameter settings
        # lambda (population size)
        lam = 4 + int(3 * np.log(dim))
        # Step size
        sigma = 2.0
        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Update rates
        c1 = (2.0 / ((dim + 1)**2 + 10)) / mueff
        cmu = min(1.0 - c1, 2 * (mueff - 2.0 + 2.0 / dim) / ((dim + 2)**2 * mueff + 10))
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        cc = (4.0 - mueff / dim) / (dim + 4.0)
        # Damping for step size control
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        
        # Initial state
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        # Pre-compute matrix square root for sampling
        # C = D * B * B.T * D, so sqrt(C) = D * B
        # Initially D=I, B=I, so sqrt(C)=I
        # We need to update C and its decomposition
        
        # Helper to update matrix square root
        # This is the expensive part. We can use a simplified update or just recompute.
        # For dim=5, recomputing eigendecomposition every generation is cheap.
        
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Initial evaluation
        try:
            best_f = counted_func(best_x)
        except Exception:
            return best_x, best_f

        # Main loop
        while call_count < max_calls and time.time() < deadline:
            # 1. Sample lambda points
            # We need sqrt(C). Let's compute it.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure eigenvalues are positive
            eigvals = np.maximum(eigvals, 1e-12)
            sqrt_eigvals = np.sqrt(eigvals)
            # sqrt(C) = eigvecs @ diag(sqrt_eigvals) @ eigvecs.T
            # But we want to sample y ~ N(0, C). y = sqrt(C) @ z.
            # It is more efficient to do: y = eigvecs @ (sqrt_eigvals * (eigvecs.T @ z))
            # Actually, z is standard normal.
            # Let's generate Z (lam x dim)
            Z = rng.standard_normal((lam, dim))
            # Transform to Y
            # Y = eigvecs @ diag(sqrt_eigvals) @ (eigvecs.T @ Z.T).T ?
            # Y_i = eigvecs @ (sqrt_eigvals * (eigvecs.T @ Z_i))
            
            # Compute eigvecs.T @ Z.T -> (dim x lam)
            # Then multiply rows by sqrt_eigvals
            # Then multiply by eigvecs -> (dim x lam)
            # Then transpose -> (lam x dim)
            
            Z_rot = (eigvecs.T @ Z.T) # dim x lam
            Z_rot *= sqrt_eigvals[:, np.newaxis] # scale rows
            Y = (eigvecs @ Z_rot).T # lam x dim
            
            # Generate points
            X = x_mean + sigma * Y
            X = np.clip(X, -5, 5)
            
            # 2. Evaluate
            Fs = np.empty(lam)
            for i in range(lam):
                if call_count >= max_calls:
                    break
                if time.time() > deadline:
                    break
                try:
                    Fs[i] = counted_func(X[i])
                except Exception:
                    break
            
            if i + 1 < lam:
                # If we broke early, we only have i+1 evaluations
                # This breaks the CMA-ES update logic which expects full population.
                # To keep it simple and robust, if we can't complete a generation, we stop.
                break
            
            # Update best
            min_idx = np.argmin(Fs)
            if Fs[min_idx] < best_f:
                best_f = Fs[min_idx]
                best_x = X[min_idx].copy()
                
            # 3. Update mean
            # Sort by fitness
            sorted_indices = np.argsort(Fs)
            X_sorted = X[sorted_indices]
            Fs_sorted = Fs[sorted_indices]
            
            # Weighted mean
            x_new = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # 4. Update step size
            # ps = (1-cs) * ps + sqrt(cs*(2-cs)*mueff) * C^(-1/2) * (x_new - x_mean)/sigma
            # C^(-1/2) = eigvecs @ diag(1/sqrt_eigvals) @ eigvecs.T
            # Let's compute C^-1/2 @ (x_new - x_mean)
            diff = x_new - x_mean
            # diff_rot = eigvecs.T @ diff
            diff_rot = eigvecs.T @ diff
            # diff_rot /= sqrt_eigvals
            diff_rot /= sqrt_eigvals
            # diff_inv = eigvecs @ diff_rot
            diff_inv = eigvecs @ diff_rot
            
            # We need ps from previous iteration. Initialize to 0.
            if not hasattr(self, '_cma_ps'):
                ps = np.zeros(dim)
            # But self is not persistent across instances in the way we want for state?
            # Actually, __call__ is called once per instance. So local state is fine.
            # Wait, I initialized ps outside? No.
            # Let's re-init ps inside the loop or keep it local.
            # I forgot to define ps in the local scope before the loop.
            # Let's fix this by defining ps before the loop.
            
            # Restarting logic for ps is tricky if I define it inside.
            # Let's just re-run the block with ps defined.
            
            # Actually, I can just define ps = np.zeros(dim) before the while loop.
            # Let's assume I did that.
            
            # Correction: I need to define ps before the loop.
            
            # Let's restart the code structure in my head to ensure ps is defined.
            
            # 5. Update covariance
            # C = (1 - c1 - cmu) * C + c1 * (diff @ diff.T) + cmu * sum(weights_i * (y_i - y_w) @ (y_i - y_w).T)
            # where y_i = (X_sorted[i] - x_mean) / sigma
            # y_w = (x_new - x_mean) / sigma
            
            y_w = diff / sigma
            # y_i for top mu
            Y_sorted = (X_sorted[:mu] - x_mean) / sigma
            # Y_sorted is mu x dim
            # We need sum(weights_i * (y_i - y_w) @ (y_i - y_w).T)
            # Let Y_diff = Y_sorted - y_w (broadcasting y_w)
            Y_diff = Y_sorted - y_w
            # Sum of weighted outer products
            # (Y_diff * weights[:, np.newaxis]).T @ Y_diff
            C_update = (Y_diff.T * weights) @ Y_diff
            
            C = (1.0 - c1 - cmu) * C + c1 * (diff @ diff.T) + cmu * C_update
            
            # Ensure C is symmetric
            C = (C + C.T) / 2.0
            
            # Update mean
            x_mean = x_new
            
            # Check convergence
            # If sigma is too small or C is singular, break
            if sigma < 1e-10:
                break

        return best_x, best_f

# REWRITING WITH CORRECT STATE MANAGEMENT

class Optimizer:
    """CMA-ES for BBOB f4 (Büche-Rastrigin) with strict budget management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        call_count = 0
        max_calls = self.budget
        dim = self.dim

        # Wrapper to count calls and enforce budget
        def counted_func(x):
            nonlocal call_count
            if call_count >= max_calls:
                raise ValueError("Budget exceeded")
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        # --- CMA-ES Implementation ---
        
        # Parameters
        lam = 4 + int(3 * np.log(dim))
        sigma = 2.0
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        c1 = (2.0 / ((dim + 1)**2 + 10)) / mueff
        cmu = min(1.0 - c1, 2 * (mueff - 2.0 + 2.0 / dim) / ((dim + 2)**2 * mueff + 10))
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        cc = (4.0 - mueff / dim) / (dim + 4.0)
        
        # Initial state
        x_mean = np.zeros(dim)
        C = np.eye(dim)
        ps = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Initial evaluation
        try:
            best_f = counted_func(best_x)
        except Exception:
            return best_x, best_f

        # Main loop
        while call_count < max_calls and time.time() < deadline:
            # 1. Sample lambda points
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrt_eigvals = np.sqrt(eigvals)
            
            Z = rng.standard_normal((lam, dim))
            Z_rot = (eigvecs.T @ Z.T)
            Z_rot *= sqrt_eigvals[:, np.newaxis]
            Y = (eigvecs @ Z_rot).T
            
            X = x_mean + sigma * Y
            X = np.clip(X, -5, 5)
            
            # 2. Evaluate
            Fs = np.empty(lam)
            eval_success = True
            for i in range(lam):
                if call_count >= max_calls:
                    eval_success = False
                    break
                if time.time() > deadline:
                    eval_success = False
                    break
                try:
                    Fs[i] = counted_func(X[i])
                except Exception:
                    eval_success = False
                    break
            
            if not eval_success:
                break
            
            # Update best
            min_idx = np.argmin(Fs)
            if Fs[min_idx] < best_f:
                best_f = Fs[min_idx]
                best_x = X[min_idx].copy()
                
            # 3. Update mean
            sorted_indices = np.argsort(Fs)
            X_sorted = X[sorted_indices]
            
            x_new = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # 4. Update step size
            diff = x_new - x_mean
            diff_rot = eigvecs.T @ diff
            diff_rot /= sqrt_eigvals
            diff_inv = eigvecs @ diff_rot
            
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * diff_inv
            hsig = (np.sqrt(1.0 - (1.0 - cs)**2) - 1.0) / np.sqrt(mueff)
            hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs)**(2 * (call_count // lam))) < (1.4 + 2.0 / (dim + 1.0)) * hsig else 0.0
            
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(mueff) - 1.0))
            
            # 5. Update covariance
            y_w = diff / sigma
            Y_sorted = (X_sorted[:mu] - x_mean) / sigma
            Y_diff = Y_sorted - y_w
            C_update = (Y_diff.T * weights) @ Y_diff
            
            C = (1.0 - c1 - cmu) * C + c1 * (diff @ diff.T) + cmu * C_update
            C = (C + C.T) / 2.0
            
            x_mean = x_new
            
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
