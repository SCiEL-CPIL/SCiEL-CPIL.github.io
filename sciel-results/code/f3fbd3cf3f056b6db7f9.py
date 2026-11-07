"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with global best archive for black-box continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        max_time = 295.0  # Safety margin before 300s kill

        global_best_x = None
        global_best_f = np.inf
        
        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        # For dim=5, lambda (population size) is typically 4 + floor(3*log(n)) = 4 + 3*1.6 = 8
        # But for multimodal functions, a slightly larger population helps exploration.
        # We will adapt population size based on remaining budget.
        
        # Initial mean: center of the box or a random point?
        # Random start is better to avoid bias towards center if optimum is shifted.
        # However, CMA-ES converges faster if started reasonably close.
        # Let's start with a random point in [-4, 4]^5.
        
        mean = rng.uniform(-4, 4, size=dim)
        
        # Initial standard deviation
        # The box is [-5, 5]. The optimum is in [-4, 4].
        # A sigma of 2.0 covers the space well initially.
        sigma = 2.0
        
        # CMA-ES weights (default: linear weights)
        # mu = lambda // 2
        # weights = log(mu + 0.5) - log(arange(1, mu+1))
        # weights /= sum(weights)
        
        # We will implement a simplified but robust CMA-ES loop.
        # Key features:
        # 1. Sample lambda points.
        # 2. Evaluate.
        # 3. Update mean, sigma, and covariance matrix C.
        # 4. Track global best.
        
        # To handle the "500 calls" constraint strictly:
        # Each iteration consumes lambda calls.
        # We need to ensure we don't exceed total_budget.
        
        calls_made = 0
        
        # Adaptive Lambda
        # Start with a larger population for exploration, reduce as we converge?
        # Or keep it constant?
        # For dim=5, lambda=8 is standard. 
        # 500 / 8 = 62.5 iterations.
        # This is a decent number of CMA-ES iterations for dim=5.
        
        # Let's try a slightly larger population initially to find the basin.
        # lambda = 12? 500/12 = 41 iterations.
        # Let's stick to standard lambda=8 for stability, but maybe start with 10?
        # Let's use lambda = 8.
        
        lambda_pop = 8
        
        # Precompute weights
        mu = lambda_pop // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        w_eff = 1 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (4 + w_eff / dim) / (dim + 4 + 2 * w_eff / dim)
        c_c = (4 + w_eff / dim) / (dim + 12 + 2 * w_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + w_eff)
        c_mu = min(1 - c_1, 2 * ((w_eff + 2) / (dim + 2) - 1))
        
        # State variables
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(20*dim**2))
        
        # Main Loop
        while calls_made < total_budget and time.time() - start_time < max_time:
            # Check if we have enough budget for one full iteration
            if calls_made + lambda_pop > total_budget:
                # If not enough for full population, do a final single-step polish?
                # Or just break. Breaking is safer to avoid partial updates.
                # But we can do a few single evaluations if we want to squeeze out value.
                # Let's just break to keep CMA-ES logic clean.
                break
            
            # 1. Sample population
            B = rng.multivariate_normal(np.zeros(dim), C, size=lambda_pop)
            # Whitening transformation: B = B * sigma? No, B is already from C.
            # Standard CMA: x = mean + sigma * B
            # Where B is sampled from N(0, C)
            # Actually, scipy's multivariate_normal handles the covariance.
            # But for efficiency, we can use Cholesky or eigendecomposition.
            # Given dim=5, multivariate_normal is fast enough.
            
            # However, we need to update C.
            # Standard CMA-ES uses the transformation matrix B to update C.
            # Let's use a simpler implementation that updates C via eigendecomposition each step?
            # Or use the standard rank-one and rank-mu updates.
            
            # Let's stick to the standard update rules.
            # We need the eigendecomposition of C to sample efficiently?
            # For dim=5, np.linalg.eigh is fast.
            
            # Actually, sampling from N(0, C) using scipy is fine.
            # But we need the "whitened" points to update C correctly.
            # Let's compute the eigendecomposition of C at the start of the loop.
            
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                eigvals = np.maximum(eigvals, 1e-10)
                
                # Sample from N(0, I) and transform
                Z = rng.standard_normal((lambda_pop, dim))
                # B = Z @ diag(sqrt(eigvals)) @ eigvecs.T
                # This is equivalent to sampling from N(0, C)
                B = Z @ (np.sqrt(eigvals)[:, np.newaxis] * eigvecs.T)
                
                # Generate candidates
                X = mean + sigma * B
                X = np.clip(X, -5, 5) # Clip to box
                
                # 2. Evaluate
                F = np.array([func(x) for x in X])
                calls_made += lambda_pop
                
                # Update global best
                best_idx = np.argmin(F)
                if F[best_idx] < global_best_f:
                    global_best_f = F[best_idx]
                    global_best_x = X[best_idx].copy()
                
                # 3. Sort by fitness
                order = np.argsort(F)
                X_sorted = X[order]
                F_sorted = F[order]
                
                # 4. Update mean
                # Only use the best mu points
                mean_new = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
                
                # 5. Update ps
                # pc update needs the old mean and sigma
                # pc = (1 - c_sigma) * pc + c_sigma * sqrt(w_eff) / sigma * (mean_new - mean)
                # Note: The standard formula uses the unsorted mean? 
                # No, the mean is updated using the weighted sum of the best points.
                # The "evolution path" ps is updated with the normalized step.
                
                # We need the "step" vector.
                # step = (mean_new - mean) / sigma
                # But mean_new is computed from sorted points.
                # The standard CMA-ES algorithm updates the mean first, then ps.
                
                # Let's follow the pseudo-code from Hansen's website.
                # 1. Sample X
                # 2. Evaluate F
                # 3. Sort X by F
                # 4. Update mean: m' = sum w_i x_i (best mu)
                # 5. Update ps: ps = (1-c_sigma)ps + c_sigma * sqrt(w_eff) / sigma * (m' - m)
                # 6. Update pc: pc = (1-c_c)pc + c_c * sqrt(w_eff) * (m' - m) / sigma  (if ||ps|| < (1.4+c_sigma)chi_n)
                # 7. Update C: C = (1 - c_1 - c_mu)C + c_1 (pc pc^T + (1 - (1-c_sigma)^2) C) + c_mu sum w_i (y_i y_i^T)
                #    where y_i = (x_i - m) / sigma (using the OLD mean m)
                
                # Let's implement this carefully.
                
                # Old mean is 'mean'
                # New mean is 'mean_new'
                
                # Update ps
                ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(w_eff) / sigma * (mean_new - mean)
                
                # Update pc
                if np.linalg.norm(ps) / chi_n < (1.4 + c_sigma):
                    pc = (1 - c_c) * pc + c_c * np.sqrt(w_eff) * (mean_new - mean) / sigma
                    # Rank-one update of C
                    # C = (1 - c_1) * C + c_1 * (pc * pc.T + (1 - (1-c_sigma)**2) * C)
                    # Note: The term (1 - (1-c_sigma)**2) is approximately 2*c_sigma for small c_sigma
                    C = (1 - c_1) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_sigma)**2) * C)
                else:
                    # If condition not met, just decay pc? Or skip pc update?
                    # Standard algorithm: if condition fails, pc is not updated? 
                    # Actually, the condition is for the rank-one update part.
                    # If it fails, we might just skip the pc contribution to C.
                    C = (1 - c_1) * C # And add the rank-mu part?
                    # Let's simplify: if condition fails, we still do the rank-mu part, 
                    # but skip the pc part in the rank-one update.
                    # So C = (1 - c_1 - c_mu) * C + c_mu * ...
                    # We will handle the rank-mu part below.
                    # For now, just decay C slightly? No, let's stick to the formula.
                    # If condition fails, the rank-one update term involving pc is omitted.
                    # So C = (1 - c_1) * C is wrong because we also have c_mu.
                    # Let's restructure the C update.
                
                # Rank-mu update
                # Y = (X_sorted[:mu] - mean) / sigma  (using OLD mean)
                Y = (X_sorted[:mu] - mean) / sigma
                # C_mu = sum w_i * outer(Y_i, Y_i)
                C_mu = np.sum(w[:, np.newaxis, np.newaxis] * Y[:, np.newaxis, :] * Y[:, :, np.newaxis], axis=0)
                
                # Combine updates
                # If pc condition was met:
                # C_new = (1 - c_1 - c_mu) * C + c_1 * (pc pc^T + (1 - (1-c_sigma)^2) C) + c_mu * C_mu
                # If pc condition was NOT met:
                # C_new = (1 - c_1 - c_mu) * C + c_mu * C_mu  (pc part omitted)
                
                # Let's assume the condition is usually met. If not, we just skip the pc term.
                if np.linalg.norm(ps) / chi_n < (1.4 + c_sigma):
                    C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - (1 - c_sigma)**2) * C) + c_mu * C_mu
                else:
                    C = (1 - c_1 - c_mu) * C + c_mu * C_mu
                
                # Symmetrize C to avoid numerical drift
                C = (C + C.T) / 2
                
                # Update mean
                mean = mean_new
                
                # Update sigma
                sigma = sigma * np.exp(c_sigma / (5 * dim) * (np.linalg.norm(ps) / chi_n - 1))
                
                # Clip sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-6, 10.0)
                
                # Optional: If we are very close to convergence, we could reduce lambda?
                # Or just let it run.
                
            except Exception:
                # If numerical issues occur, break
                break

        return global_best_x, global_best_f

# EVOLVE-BLOCK-END
