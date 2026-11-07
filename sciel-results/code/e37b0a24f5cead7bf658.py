import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # For dim=5 and budget=500, a small population is appropriate.
        # lambda = 4 + floor(3 * log(dim)) is a common heuristic, but for small dim/budget, 
        # lambda=4 or 5 is efficient. Let's use lambda=4 to maximize generations (500/4=125 gens).
        # Actually, larger lambda gives better direction info. Let's try lambda=5.
        lam = 5
        mu = lam // 2  # Number of best points used for update
        
        # Weights for the selected points
        # Standard weights: log(mu + 0.5) - log(1..mu)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_w = np.sum(weights ** 2)
        
        # CMA-ES parameters
        c_sigma = (mu_w + 2) / (dim + mu_w)
        c_c = (4 + mu_w / dim) / (dim + 4 + 2 * mu_w / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_w)
        c_mu = min(1 - c_1, 2 * (mu_w - 2 + 2 / mu_w) / ((dim + 2) ** 2 + mu_w))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_w - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initialization
        # Start at center of the box
        mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, covers a good portion of [-5, 5]
        
        # State variables
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute square root of C for sampling
        # We will update B and D (Cholesky/Eigen) every generation
        
        def update_sqrt():
            nonlocal C, B, D
            # Eigen decomposition for CMA-ES
            # In standard CMA-ES, we maintain B and D such that C = B diag(D^2) B^T
            # For simplicity and stability in this constrained budget, we can use
            # the standard update rules.
            # However, calculating eigenvalues every step is O(n^3). For n=5, it's fast.
            # Let's stick to the standard CMA-ES update which maintains B and D.
            # But implementing the full rank-mu update with B,D updates is complex.
            # Alternative: Use the simpler "Covariance Matrix Adaptation" where we 
            # just update C directly and use SVD/Cholesky for sampling.
            # Given the small dim, direct C update and SVD is robust.
            pass # We will use a simplified approach: update C, then SVD for sampling.
            
        # Simplified CMA-ES:
        # 1. Sample X ~ N(mean, sigma^2 * C)
        # 2. Evaluate
        # 3. Select best mu
        # 4. Update mean
        # 5. Update sigma (path-based)
        # 6. Update C (rank-1 + rank-mu)
        # 7. Ensure C is positive definite
        
        rng = np.random.default_rng(42)
        
        calls_made = 0
        best_f = np.inf
        best_x = mean.copy()
        
        # Helper to sample from N(0, C)
        def sample_from_cov(C, rng):
            # Use SVD for stability
            U, s, Vt = np.linalg.svd(C)
            # C = U diag(s) U^T
            # sqrt(C) = U diag(sqrt(s)) U^T
            # z ~ N(0, I)
            z = rng.standard_normal(dim)
            # y = U diag(sqrt(s)) z
            y = (U * np.sqrt(s)) @ z
            return y

        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # 1. Sample population
            # We need to sample lam points
            # To save time, we can generate all z's at once
            Z = rng.standard_normal((lam, dim))
            
            # Get current sqrt(C)
            # For efficiency, we can cache the SVD if C doesn't change drastically, 
            # but C changes every gen. For dim=5, SVD is ~microseconds.
            U, s, Vt = np.linalg.svd(C)
            sqrt_C = (U * np.sqrt(s)) @ U.T
            
            # X_i = mean + sigma * (sqrt_C @ z_i)
            # Z is (lam, dim). sqrt_C is (dim, dim).
            # We want X (lam, dim)
            # X_i = mean + sigma * sum_k z_{ik} v_k d_k
            # Using matrix mult: X = mean + sigma * (Z @ sqrt_C.T) ? 
            # Let's check: y_i = sqrt_C @ z_i. 
            # If Z is (lam, dim), then Z @ sqrt_C.T gives (lam, dim) where row i is z_i^T @ sqrt_C^T = (sqrt_C @ z_i)^T.
            # Yes.
            X = mean[None, :] + sigma * (Z @ sqrt_C.T)
            
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # 2. Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lam
            
            # Track best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()
                
            # 3. Select best mu
            # Sort indices by fitness
            idx_sorted = np.argsort(F)
            idx_selected = idx_sorted[:mu]
            X_selected = X[idx_selected]
            F_selected = F[idx_selected]
            
            # 4. Update mean
            # m_new = sum(weights * X_selected)
            m_new = np.sum(weights[:, None] * X_selected, axis=0)
            
            # 5. Update sigma
            # Calculate y_old = (m_new - m) / sigma
            y_old = (m_new - mean) / sigma
            
            # Update ps
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_w) * (sqrt(C) @ (y_old / ||y_old||))
            norm_y_old = np.linalg.norm(y_old)
            if norm_y_old > 1e-12:
                # We need sqrt(C) @ (y_old / norm)
                # Note: y_old is in "standard normal" space relative to C?
                # No, y_old is in the transformed space.
                # The standard formula is:
                # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_w) * (B * (D * (B^T * y_old / sigma))) / ||...||
                # Wait, the path ps is in the "principal axes" space?
                # Let's use the standard CMA-ES update from Hansen's tutorial.
                # y = (m_new - m) / sigma
                # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_w) * (B * (D * (B.T * y))) / ||y||
                # This requires B and D. Since we are using SVD, B=U, D=sqrt(s).
                # Let's compute the term.
                # y is (dim,).
                # B.T @ y = U.T @ y
                # D * (U.T @ y) = sqrt(s) * (U.T @ y)
                # B * (...) = U * (sqrt(s) * (U.T @ y))
                # This vector is the "untransformed" direction?
                # Actually, the norm check is usually on the transformed vector.
                # Let's simplify: if we just use the success rate rule for sigma, it's less robust but simpler.
                # However, path-based is better.
                
                # Let's implement the path update carefully.
                # y_sigma = (m_new - mean) / sigma
                # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_w) * (B * (D * (B.T * y_sigma))) / ||y_sigma||
                # Note: The denominator is ||y_sigma||.
                
                vec = U * (np.sqrt(s) * (U.T @ y_old))
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) * (vec / norm_y_old)
            else:
                ps = (1 - c_sigma) * ps
                
            # Update sigma
            # h_sigma = (||ps|| / sqrt(1 - (1 - c_sigma)^(2 * age)) < (1 + 2 / (dim + 1)) * chi_N) / chi_N
            # age is number of generations since last sigma update? No, it's the generation counter for the path.
            # We need to track age.
            # For simplicity and robustness in a short budget, we can use a simpler adaptation:
            # sigma *= exp(1 / d_sigma * (||ps|| / chi_N - 1))
            # chi_N = sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim^2))
            chi_N = np.sqrt(dim) * (1 - 1.0/(4.0*dim) + 1.0/(21.0*dim**2))
            sigma *= np.exp((1.0 / d_sigma) * (np.linalg.norm(ps) / chi_N - 1.0))
            
            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # 6. Update C
            # h_c = (||pc|| / sqrt(1 - (1 - c_c)^(2 * age)) < (1 + 2 / (dim + 1)) * chi_N) / chi_N
            # pc = (1 - c_c) * pc + h_c * sqrt(c_c * (2 - c_c) * mu_w) * y_old
            # We need age for pc too.
            # Let's just use the simpler condition: if norm(y_old) < chi_N * (1 + 2/(dim+1)), update pc.
            # Actually, the h_c is a Heaviside function to avoid updating C if the step was too small (noise).
            
            # For a robust implementation without tracking complex ages:
            # We update C using rank-1 and rank-mu.
            
            # Rank-1 update
            # h_c approx: if ||pc|| is small, h_c=1.
            # Let's just set h_c=1 for simplicity in this constrained context, 
            # or use a simple check.
            # pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mu_w) * y_old
            # The h_c term is to prevent noise.
            # Let's include a simple h_c:
            # If ||pc|| < (1 + 2 / (dim + 1)) * chi_N, then h_c = 1 else 0.
            # But pc needs to be updated first? No, h_c is calculated with the old pc.
            
            # Let's refine the pc update:
            # y_old is the normalized step.
            # pc_new = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mu_w) * y_old
            # But we need to check if the step was significant.
            # Standard CMA-ES uses a Heaviside function based on the norm of the *path* pc.
            # Let's assume h_c = 1 for now, as dim=5 is small and noise is manageable.
            
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_w) * y_old
            
            # Rank-1 part: c_1 * (pc * pc.T - C)
            # Rank-mu part: c_mu * sum(weights_i * (y_i * y_i.T))
            # where y_i = (x_i - mean_old) / sigma
            
            # Calculate y_i for selected points
            Y_selected = (X_selected - mean[None, :]) / sigma
            
            # Rank-mu sum
            # C_mu = sum(weights_i * np.outer(y_i, y_i))
            C_mu = np.zeros((dim, dim))
            for i in range(mu):
                C_mu += weights[i] * np.outer(Y_selected[i], Y_selected[i])
                
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * C_mu
            
            # Ensure C is positive definite and symmetric
            C = (C + C.T) / 2.0
            # Add small diagonal to ensure PD
            C += 1e-12 * np.eye(dim)
            
            # Update mean
            mean = m_new
            
            # Check for stagnation or collapse
            # If sigma is too small or C has collapsed eigenvalues, restart?
            # With 500 budget, a restart might be too expensive if it happens late.
            # Let's just continue.
            
            # If we have used up budget, break
            if calls_made >= budget:
                break
                
        return best_x, best_f
