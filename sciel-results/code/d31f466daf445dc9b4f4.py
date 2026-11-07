"""CMA-ES implementation for BBOB-style linear slope function f5."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES Optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES Parameters
        # Population size lambda
        lambda_ = int(4 + 3 * np.log(dim))
        # Weighting scheme
        mu = lambda_ // 2
        # Weights for the weighted recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # Variance-effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        # Step-size control (CSA)
        # c_sigma: learning rate for sigma
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        # d_sigma: damping for sigma
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        # Path lengths
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # State variables
        # Initial center: Random point in [-5, 5]^dim
        # Since f5 is linear, the optimum is at the boundary.
        # However, we don't know the direction.
        # Starting near 0 is a safe bet, but random is better to avoid symmetry.
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-5, 5, dim)
        
        # Initial sigma
        # The box is [-5, 5]. A reasonable initial step is to cover a significant part of the box.
        # Let's start with sigma = 1.0 or 2.0.
        sigma = 1.0
        
        # Covariance matrix C (identity)
        C = np.eye(dim)
        # Eigen decomposition state for efficient updates (optional optimization, but let's stick to direct update for simplicity first, dim=5 is small)
        # Actually, for dim=5, direct matrix inversion/update is fine.
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Call management
        call_count = 0
        best_x = x_mean.copy()
        best_f = np.inf
        
        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget or time.time() > deadline:
                return None
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            try:
                val = func(x)
                call_count += 1
                if np.isfinite(val):
                    if val < best_f:
                        best_f = val
                        best_x = x.copy()
                    return val
                return np.inf
            except Exception:
                call_count += 1
                return np.inf

        # Initial evaluation
        f_init = safe_call(x_mean)
        if f_init is None:
            return best_x, best_f
        if not np.isfinite(f_init):
            f_init = 1e10

        # Main Loop
        while call_count < budget and time.time() < deadline:
            # 1. Sample lambda points
            # Generate standard normal samples
            z_samples = rng.standard_normal((lambda_, dim))
            
            # Transform to x_samples
            # x = x_mean + sigma * (B * (D * z))
            # Since we update C directly, we need the square root of C.
            # For dim=5, np.linalg.cholesky is fast enough.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, reset it
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                sigma *= 0.5
                L = np.linalg.cholesky(C)
            
            x_samples = x_mean + sigma * (z_samples @ L.T)
            
            # Clip to box
            x_samples = np.clip(x_samples, -5.0, 5.0)
            
            # Evaluate
            f_samples = []
            valid_indices = []
            for i in range(lambda_):
                if call_count >= budget or time.time() > deadline:
                    break
                f_val = safe_call(x_samples[i])
                if f_val is not None:
                    f_samples.append(f_val)
                    valid_indices.append(i)
                else:
                    f_samples.append(np.inf)
                    valid_indices.append(i) # Keep index for sorting, but inf will be last
            
            if len(f_samples) < lambda_:
                break
                
            # Sort by fitness
            order = np.argsort(f_samples)
            x_sorted = x_samples[order]
            f_sorted = f_samples[order]
            
            # Check if we have enough valid points
            # If many are inf, CMA-ES might struggle, but let's proceed with what we have.
            # Ideally, we want mu valid points.
            valid_mask = np.isfinite(f_sorted)
            if np.sum(valid_mask) < mu:
                # Too many failures, restart or reduce sigma
                sigma *= 0.5
                continue

            # 2. Update mean
            # Use the best mu points
            # If some of the best mu are inf, we might have issues.
            # Let's assume the first mu are the best.
            x_w = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # Save old mean for path updates
            x_mean_old = x_mean.copy()
            x_mean = x_w
            
            # 3. Update path evolution strategies
            # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mueff) * (1/sigma) * C^(-1/2) * (x_w - x_mean_old)
            # C^(-1/2) is inv(L)
            # Note: x_w - x_mean_old is in x-space. We need to transform to z-space.
            # z_w = C^(-1/2) * (x_w - x_mean_old) / sigma
            
            try:
                # Solve L * v = (x_w - x_mean_old) / sigma
                diff = (x_w - x_mean_old) / sigma
                v = np.linalg.solve(L, diff)
            except np.linalg.LinAlgError:
                v = np.zeros(dim)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * v
            
            # p_c = (1 - c_c) * p_c + sqrt(c_c * (2 - c_c) * mueff) * h_sigma * (1/sigma) * C^(-1/2) * (x_w - x_mean_old)
            # h_sigma = ||p_sigma|| / chi_n < 1.4 + c_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if (norm_p_sigma / np.sqrt(1 - (1 - c_sigma)**(2 * (call_count // lambda_))) / chi_n) < 1.4 + c_sigma else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * v
            
            # 4. Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c^T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(w_i * (y_i * y_i^T))
            # where y_i = (x_i - x_mean_old) / sigma
            
            # Compute Hessian-like term from paths
            H_path = np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C
            
            # Compute rank-mu update
            # y_samples = (x_sorted[:mu] - x_mean_old) / sigma
            y_samples = (x_sorted[:mu] - x_mean_old) / sigma
            # Rank-1 updates weighted
            # sum(w_i * y_i * y_i^T)
            # Efficient way: Y = y_samples.T @ np.diag(weights)
            # H_rank = Y @ Y.T
            Y = y_samples.T @ weights
            H_rank = Y @ y_samples.T
            
            C = (1 - c_1 - c_mu) * C + c_1 * H_path + c_mu * H_rank
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # 5. Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / chi_n - 1))
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Clamp sigma to prevent explosion or vanishing
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # If sigma becomes very small, we might be stuck.
            # For linear functions, we usually want to push to the boundary.
            # If we are at the boundary and sigma is small, we are likely close to optimum.

        return best_x, best_f

# EVOLVE-BLOCK-END
