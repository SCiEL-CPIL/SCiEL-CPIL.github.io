"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population-based global search and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # For dim=5, lambda=10 is a good balance between exploration and exploitation
        # Standard CMA-ES uses lambda = 4 + floor(3 * log(dim))
        # For dim=5, 4 + 3*1.6 = 8.8 -> 8. But for Rastrigin (multimodal), larger pop helps.
        # Let's use lambda=10.
        lam = 10
        
        # Initial center: random in [-4, 4] to avoid boundaries initially
        # But CMA-ES works best if start is reasonable. Let's use a random start.
        rng = np.random.default_rng(42)
        
        # We will run CMA-ES for the entire budget.
        # No restarts, relying on the population diversity to handle multimodality.
        # If it gets stuck, the covariance matrix will shrink, but sigma adaptation
        # might keep it moving. However, CMA-ES is not great at jumping out of deep local minima
        # once the population collapses. 
        # To mitigate this, we can add a "restart" logic if progress stalls, 
        # but the directive says "without relying on discrete restarts".
        # Actually, the prompt says "Replace ... with ... that uses a large initial population ... to maintain global diversity ... without relying on discrete restarts".
        # So I will implement a single CMA-ES run.
        
        # However, a single CMA-ES run on Rastrigin might get stuck in a local minimum 
        # if the initial population is not diverse enough or if it converges too early.
        # A pure CMA-ES is a local optimizer. 
        # To make it a global optimizer without "discrete restarts" (in the sense of throwing away the population and starting over),
        # we can use a "population of CMA-ES" or a "multi-start" that is managed by a higher-level scheduler.
        # But the directive specifically asks for "a population-based CMA-ES ... that uses a large initial population ... to maintain global diversity".
        # This implies the population itself is the mechanism for global search.
        # If the population collapses, we are stuck.
        
        # Let's try a standard CMA-ES implementation.
        
        # Initial state
        # Start at origin? Or random? 
        # The minimizer is in [-4, 4]^5. 
        # Let's start at 0.
        m = np.zeros(dim)
        sigma = 2.0  # Initial step size, covering a good portion of the box
        
        # Population
        # Generate initial population
        # We need to track evaluations.
        # CMA-ES step:
        # 1. Sample lam points from N(m, sigma^2 * C)
        # 2. Evaluate
        # 3. Sort by fitness
        # 4. Update m, C, sigma
        
        # We'll implement a simplified CMA-ES that is robust to the box constraints.
        # Clipping is handled by the evaluator, but for the sampler, we should probably
        # sample from the distribution and let the evaluator clip.
        # This is inefficient if we clip a lot, but for dim=5 and box [-5,5], 
        # with sigma=2 and m=0, most samples will be inside.
        
        # To be safe and efficient, we can sample and then project/clip.
        # But clipping distorts the distribution.
        # Given the small dimension, simple clipping is likely fine.
        
        # CMA-ES parameters
        # mu = lam // 2
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Learning rates
        # Standard formulas
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff)))
        
        # Initial covariance
        C = np.eye(dim)
        # Inverse square root of C (initially identity)
        C_sqrt_inv = np.eye(dim)
        # Eigen decomposition cache
        D = np.ones(dim)
        B = np.eye(dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Helper to update C inverse square root
        def update_c_sqrt_inv():
            nonlocal C, C_sqrt_inv, D, B
            # Update C
            # C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c.T + (1-delta_cs)*C) + cmu * sum(wi * (yi @ yi.T))
            # This is expensive to compute directly.
            # Standard CMA-ES updates C via eigen decomposition or rank-one/rank-mu updates.
            # For dim=5, we can do a full eigen decomposition every step.
            pass # We'll do the update inline in the main loop for clarity and speed in small dims
            
        # Actually, let's use a simpler, robust CMA-ES variant or just use SciPy's implementation if available?
        # "Python standard library, NumPy and SciPy are available."
        # SciPy has `scipy.optimize` but not a direct CMA-ES black-box optimizer in the standard top-level API that is easy to call with a budget in a loop.
        # `scipy.optimize.minimize` with method='COBYLA' or 'Nelder-Mead' are local.
        # There is `cma` package, but it's not standard.
        # So I must implement CMA-ES manually.
        
        # Let's implement a simplified CMA-ES that works for small dimensions.
        # Key insight: For dim=5, we can afford to do eigen decomposition every generation.
        
        # State variables
        m = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Pre-compute weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff)))
        hsig = (1 + 2 / (mueff + 1)) * (1 - cc)
        
        # Initial population
        # We need to generate lam points.
        # To handle the box [-5, 5], we will sample from N(m, sigma^2 C) and clip.
        
        # Main loop
        calls_used = 0
        best_x = m.copy()
        best_f = np.inf
        
        # First, evaluate the initial center? No, CMA-ES uses the population.
        # But we can evaluate m if we want, though standard CMA-ES doesn't require it.
        # Let's just run the generations.
        
        # To ensure we don't exceed budget, we check inside the loop.
        
        # We need to handle the case where the population collapses (sigma becomes too small).
        # If sigma < 1e-10, we can stop or restart.
        # The directive says "without relying on discrete restarts", so I will just stop if it converges.
        # But if it converges to a local minimum, we are stuck.
        # To mitigate this, I will initialize the population with a wider spread.
        # sigma=2.0 is good.
        
        # Let's run until budget is exhausted or time limit.
        
        # We will store the current population points and their fitnesses.
        # In each generation, we generate new points.
        
        # Optimization: We can reuse the best point from the previous generation?
        # Standard CMA-ES does not reuse, it generates lam new points.
        
        # Let's start.
        
        # To speed up, we can use vectorized operations for the population sampling.
        
        # Eigen decomposition of C
        # Initially C=I, so D=1, B=I
        D = np.ones(dim)
        B = np.eye(dim)
        
        # Main loop
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a full generation
            if budget - calls_used < lam:
                # Do a partial generation? 
                # Or just stop? 
                # Stopping might waste budget. 
                # Let's do a partial generation by sampling one by one.
                # But CMA-ES update requires the full population.
                # So if we can't complete a generation, we should probably do a simple (1+1)-ES step or just stop.
                # Given lam=10 and budget=500, we will have 50 full generations.
                # So this case is unlikely unless we are at the very end.
                # Let's just break if we can't do a full generation to keep it simple.
                # Or, we can reduce lam for the last few steps? No, that complicates the update.
                # Let's just break.
                break
            
            # 1. Sample population
            # Y = B @ (D * (m + sigma * Z))
            # Z is standard normal
            Z = rng.standard_normal((lam, dim))
            Y = m[None, :] + sigma * (Z @ B.T * D)
            
            # Clip to box
            Y = np.clip(Y, -5, 5)
            
            # 2. Evaluate
            F = np.array([func(y) for y in Y])
            calls_used += lam
            
            # 3. Sort
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()
            
            # 4. Update mean
            # m_new = m + sigma * sum(wi * Yi_sorted)
            # Actually, the weighted sum is in the standard normal space?
            # No, m is updated in the original space.
            # The formula is: m_new = m + sigma * B @ (D * (sum(wi * Zi_sorted)))
            # Wait, Y_i = m + sigma * B @ (D * Z_i)
            # So Y_i - m = sigma * B @ (D * Z_i)
            # sum(wi * (Y_i - m)) = sigma * B @ (D * (sum(wi * Z_i)))
            # So m_new = m + sum(wi * (Y_i - m)) = m + sigma * B @ (D * (sum(wi * Z_i)))
            
            # We need the sorted Z values.
            # We have Y_sorted, but we generated Y from Z.
            # We need to keep track of Z.
            # Let's re-sample and keep Z.
            
            # Re-do sampling with Z tracking
            Z = rng.standard_normal((lam, dim))
            # We already evaluated Y from Z. 
            # But we lost the mapping if we just used Y.
            # We need to map Y_sorted back to Z_sorted.
            # Since Y = m + sigma * (Z @ B.T * D), and the transformation is linear and deterministic,
            # the order of Y is the same as the order of the transformed Z.
            # But we can just compute the weighted sum of Y_sorted directly?
            # m_new = m + sum(wi * (Y_i_sorted - m))
            # This is correct because the weights are applied to the selected points.
            
            m_new = m + np.sum(w[:, None] * (Y_sorted - m[None, :]), axis=0)
            
            # 5. Update step size
            # hsig = ||p_sigma_new|| < (1 + 2/(mueff+1)) * E||N(0, I)||
            # p_sigma_new = (1-cs)*p_sigma + sqrt(cs*(2-cs))* (1/sigma) * (m_new - m)
            # Note: (m_new - m) / sigma is the step in the standard normal space?
            # m_new - m = sigma * B @ (D * (sum(wi * Zi)))
            # So (m_new - m) / sigma = B @ (D * (sum(wi * Zi)))
            
            # We need the sum of weighted Zs.
            # Let's compute it from the sorted Ys?
            # It's easier to keep Z.
            # Let's modify the sampling block to keep Z.
            
            # Re-evaluate? No, we already have F.
            # We just need Z_sorted.
            # Since Y = m + sigma * T(Z), and T is a linear bijection (if C is positive definite),
            # the rank of Y is the rank of T(Z).
            # We can recover Z_sorted by inverting the transformation.
            # Z = D_inv @ B_inv @ (Y - m) / sigma
            # This is expensive.
            
            # Alternative: Just keep Z in the loop.
            
            # Let's restart the loop logic with Z kept.
            pass # This block is getting complex. Let's write the full clean implementation.
        
        # Let's write the full clean implementation in the return.
        
        # --- Clean Implementation ---
        
        dim = self.dim
        budget = self.budget
        lam = 10
        mu = lam // 2
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff)))
        hsig = (1 + 2 / (mueff + 1)) * (1 - cc)
        
        m = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        D = np.ones(dim)
        B = np.eye(dim)
        
        calls_used = 0
        best_x = m.copy()
        best_f = np.inf
        
        # Precompute expected norm of standard normal
        # E||N(0, I)|| = sqrt(2) * Gamma((dim+1)/2) / Gamma(dim/2)
        # For dim=5, this is approx 2.4
        from scipy.special import gamma
        c_n = np.sqrt(2) * gamma((dim + 1) / 2.0) / gamma(dim / 2.0)
        
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            if budget - calls_used < lam:
                break
            
            # 1. Sample
            Z = rng.standard_normal((lam, dim))
            # Y = m + sigma * B @ (D * Z)
            # Note: B @ (D * Z) is equivalent to (Z * D) @ B.T
            Y = m[None, :] + sigma * ((Z * D) @ B.T)
            Y = np.clip(Y, -5, 5)
            
            # 2. Evaluate
            F = np.array([func(y) for y in Y])
            calls_used += lam
            
            # 3. Sort
            order = np.argsort(F)
            Y_sorted = Y[order]
            Z_sorted = Z[order]
            F_sorted = F[order]
            
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()
            
            # 4. Update mean
            # m_new = m + sigma * B @ (D * (sum(wi * Zi_sorted)))
            # Let v = sum(wi * Zi_sorted)
            v = np.sum(w[:, None] * Z_sorted, axis=0)
            m_new = m + sigma * (B @ (D * v))
            
            # 5. Update step size
            # p_sigma_new = (1-cs)*p_sigma + sqrt(cs*(2-cs)) * v
            # Note: v is in the standard normal space
            p_sigma_new = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs)) * v
            
            # hsig condition
            if np.linalg.norm(p_sigma_new) < hsig * c_n:
                hsig_val = 1.0
            else:
                hsig_val = 0.0
            
            # Update p_c
            # p_c_new = (1-cc)*p_c + hsig * sqrt(cc*(2-cc)) * (m_new - m) / sigma
            # (m_new - m) / sigma = B @ (D * v)
            step = B @ (D * v)
            p_c_new = (1 - cc) * p_c + hsig_val * np.sqrt(cc * (2 - cc)) * step
            
            # Update sigma
            # sigma_new = sigma * exp(cs / lambda_sigma * (||p_sigma_new|| / c_n - 1))
            # lambda_sigma is typically 1
            lambda_sigma = 1.0
            sigma_new = sigma * np.exp(cs / lambda_sigma * (np.linalg.norm(p_sigma_new) / c_n - 1))
            
            # Update C
            # C_new = (1 - c1 - cmu) * C + c1 * (p_c_new @ p_c_new.T + (1 - delta_cs) * C) + cmu * sum(wi * (yi @ yi.T))
            # delta_cs = 1 if hsig_val == 1 else 0
            delta_cs = 1.0 if hsig_val == 1.0 else 0.0
            
            # The term sum(wi * (yi @ yi.T)) is in the original space?
            # No, the standard CMA-ES update is in the standard normal space?
            # Actually, the covariance matrix C is in the standard normal space?
            # No, C is the covariance of the distribution N(0, C) in the standard normal space?
            # The distribution is N(m, sigma^2 * B C B.T)?
            # No, the distribution is N(m, sigma^2 * C) where C is the covariance matrix in the original space?
            # In standard CMA-ES, C is the covariance matrix of the distribution in the original space?
            # No, C is the covariance matrix of the "standard normal" distribution that is transformed by B and D?
            # Actually, the transformation is Y = m + sigma * B @ (D * Z).
            # Cov(Y) = sigma^2 * B @ (D @ D) @ B.T = sigma^2 * B @ (C) @ B.T if C = D @ D?
            # No, C is the covariance matrix such that B @ C @ B.T is the covariance in the original space?
            # In standard CMA-ES, C is the covariance matrix in the "evolution path" space?
            # Let's stick to the standard implementation where C is updated in the space of the weights.
            
            # The standard update for C is:
            # C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c.T + (1 - delta_cs) * C) + cmu * sum(wi * (yi @ yi.T))
            # where yi are the steps in the "C-space"?
            # Actually, the steps are Y_i - m.
            # But the weights are applied to the sorted Zs?
            # No, the weights are applied to the sorted Ys?
            # The formula uses the "rank-mu update" with the selected points.
            # The points are Y_i.
            # So we need Y_i - m.
            
            # Let's compute the rank-mu term
            # Y_i - m = sigma * B @ (D * Z_i)
            # So (Y_i - m) @ (Y_i - m).T = sigma^2 * B @ (D * Z_i) @ (D * Z_i).T @ B.T
            # This is expensive.
            
            # Alternative: Update C in the Z-space?
            # If we define C_z such that Cov(Z) = C_z, then Cov(Y) = sigma^2 * B @ (D @ C_z @ D) @ B.T?
            # No, Z is standard normal, so Cov(Z) = I.
            # The transformation is Y = m + sigma * B @ (D * Z).
            # So Cov(Y) = sigma^2 * B @ (D @ I @ D) @ B.T = sigma^2 * B @ (D^2) @ B.T.
            # So the effective covariance is C_eff = B @ (D^2) @ B.T.
            # In standard CMA-ES, C is the covariance matrix such that Cov(Y) = sigma^2 * C.
            # So C = B @ (D^2) @ B.T.
            
            # The update for C is done on the "C-matrix" in the original space?
            # Or in the "eigen-space"?
            # Most implementations update C in the original space.
            
            # Let's compute the rank-mu term in the original space.
            # Steps = Y_sorted[:mu] - m
            # Rank-mu term = sum(wi * (Steps_i @ Steps_i.T))
            
            Steps = Y_sorted[:mu] - m[None, :]
            rank_mu_term = np.sum(w[:mu, None, None] * (Steps[:, :, None] * Steps[:, None, :]), axis=0)
            
            # Rank-one term
            rank_one_term = p_c_new @ p_c_new.T + (1 - delta_cs) * C
            
            C_new = (1 - c1 - cmu) * C + c1 * rank_one_term + cmu * rank_mu_term
            
            # Update eigen decomposition
            # C_new should be symmetric positive definite
            C_new = (C_new + C_new.T) / 2
            
            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C_new)
            
            # Clip small eigenvalues
            eigvals = np.clip(eigvals, 1e-10, None)
            
            # Update B and D
            # B = eigvecs
            # D = sqrt(eigvals)
            B = eigvecs
            D = np.sqrt(eigvals)
            
            # Update state
            m = m_new
            sigma = sigma_new
            C = C_new
            p_sigma = p_sigma_new
            p_c = p_c_new
            
            # Convergence check
            if sigma < 1e-10:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
