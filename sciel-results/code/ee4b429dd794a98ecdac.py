import numpy as np
import time

class Optimizer:
    """CMA-ES based continuous optimizer for BBOB-style f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320.0

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # For dim=5, population size lambda is typically 4 + floor(3*log(dim + 1))
        # For dim=5: 4 + floor(3*log(6)) = 4 + floor(3*1.79) = 4 + 5 = 9
        # Let's use a slightly larger population for better exploration initially, or stick to standard.
        # Standard: lambda = 4 + int(3 * log(dim + 1))
        lam = 4 + int(3 * np.log(dim + 1))
        # For dim=5, lam=9.
        
        # Initial mean
        # Uniform initialization in [-5, 5]
        # But CMA-ES works best if the initial sigma is appropriate.
        # Box is [-5, 5], so width is 10.
        # Initial sigma ~ 1.0 or 2.0?
        # Let's start with sigma = 1.0.
        
        rng = np.random.default_rng(42)
        mean = rng.uniform(-5, 5, dim)
        sigma = 1.0
        
        # CMA-ES weights
        # Default weights: mu = lam // 2
        # w_i = log(mu + 0.5) - log(i)
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        cc = (4. + mueff / dim) / (dim + 4. + 2. * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Hessian matrix
        C = np.eye(dim)
        # Damping
        damp = 1 + 2 * max(0, np.sqrt((mueff - 2 + 1 / mueff) / (dim + 2) - 1)) / cs + c1
        
        # Eigenvalue decomposition for C
        # We will update C and its decomposition periodically or every step.
        # For dim=5, every step is cheap.
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Evolution path for covariance matrix adaptation
        pc = np.zeros(dim)
        # Evolution path for sigma adaptation
        ps = np.zeros(dim)
        
        # Track best solution
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Pre-allocate population
        pop = np.zeros((lam, dim))
        fvals = np.zeros(lam)
        
        # Main loop
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Sample population
            for i in range(lam):
                # Bivariate normal sampling using current C and sigma
                # z ~ N(0, I)
                z = rng.standard_normal(dim)
                # y = B * z, where B = eigvecs * diag(sqrt(eigvals))
                # Efficiently: y = (eigvecs * sqrt(eigvals)) @ z
                # Or: y = eigvecs @ (sqrt(eigvals) * z)
                y = eigvecs @ (np.sqrt(eigvals) * z)
                pop[i] = mean + sigma * y
                fvals[i] = func(pop[i])
                calls += 1
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
                
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = pop[min_idx].copy()
                
            # Sort population by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            
            # Compute weighted mean of best mu points
            new_mean = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
            
            # Update evolution paths
            # pc = (1 - cc) * pc + sqrt(cc * (2 - cc) * mueff) * C^-1/2 * (new_mean - mean) / sigma
            # C^-1/2 = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
            # But we have B = eigvecs @ diag(sqrt(eigvals)), so C^1/2 = B
            # C^-1/2 = B^-1
            # Actually, the term is (new_mean - mean) / sigma projected into the C^-1/2 space.
            # Let's compute C^-1/2 explicitly or use the eigen decomposition.
            # C^-1/2 * v = eigvecs @ (v @ eigvecs.T / sqrt(eigvals))
            
            diff = (new_mean - mean) / sigma
            # Project diff into C^-1/2
            # v_proj = eigvecs @ ( (diff @ eigvecs.T) / sqrt(eigvals) )
            v_proj = eigvecs @ ((diff @ eigvecs.T) / np.sqrt(eigvals))
            
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * v_proj
            
            # Update ps
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) * C^-1/2 * (new_mean - mean) / sigma
            # Wait, the formula is:
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * C^{-1/2} * (new_mean - mean)/sigma
            # This is the same v_proj as above?
            # No, the standard CMA-ES update for ps uses the same term.
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * v_proj
            
            # Update sigma
            # chiN = sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim^2))
            chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / chiN - 1))
            
            # Limit sigma
            if sigma < 1e-12:
                sigma = 1e-12
            if sigma > 1e6:
                sigma = 1e6
                
            # Update covariance matrix
            # DeltaH = (1/mueff) * sum(w_i * y_i * y_i^T) - I
            # Where y_i = (pop[i] - mean) / sigma
            # But we need to use the sorted population.
            # y = (pop[:mu] - mean) / sigma
            y = (pop[:mu] - mean) / sigma
            # DeltaH = (1/mueff) * (Y^T W Y) - I
            # W = diag(weights)
            # Efficiently:
            # YW = y * np.sqrt(weights)[:, np.newaxis]
            # DeltaH = (YW.T @ YW) / mueff - I
            YW = y * np.sqrt(weights)[:, np.newaxis]
            DeltaH = (YW.T @ YW) / mueff - np.eye(dim)
            
            # C = (1 - c1 - cmu + c1 * cc * (1 - cc)) * C + c1 * (pc * pc.T + (1 - cc) * I) + cmu * DeltaH
            # The term (1 - cc) * I is part of the pc update correction?
            # Standard formula:
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - cc) * I) + cmu * DeltaH
            # Wait, the pc * pc.T term already includes the history.
            # Let's use the standard update:
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - cc) * I) + cmu * DeltaH
            # Actually, the term is:
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - cc) * I) + cmu * DeltaH
            # But pc is updated in the C^-1/2 space? No, pc is in the original space?
            # In the standard CMA-ES algorithm, pc is maintained in the original coordinate system?
            # No, pc is usually an evolution path in the original space, but the update involves C^-1/2.
            # Let's stick to the standard implementation details.
            
            # Correction: pc is updated using C^-1/2.
            # The update for C is:
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - cc) * I) + cmu * DeltaH
            # This assumes pc is in the original space?
            # Actually, the term pc * pc.T should be in the original space.
            # My v_proj was in the C^-1/2 space.
            # So pc is in the C^-1/2 space?
            # Let's re-read the CMA-ES paper.
            # pc = (1-cc)*pc + sqrt(cc*(2-cc)*mueff) * C^{-1/2} * (m_new - m_old)/sigma
            # So pc is in the "normalized" space?
            # When updating C, we need pc in the original space?
            # No, the formula C = ... + c1 * (pc * pc.T + ...) implies pc is in the original space?
            # Actually, the standard code often keeps pc in the original space but the update step projects it.
            # Let's assume pc is in the original space for the C update.
            # But my v_proj was C^-1/2 * diff.
            # So pc is in the C^-1/2 space.
            # To get pc in the original space, we need C^{1/2} * pc.
            # Let's define pc_orig = C^{1/2} * pc.
            # Then the term in C update is pc_orig * pc_orig.T.
            
            # Let's compute pc_orig
            # C^{1/2} = eigvecs @ diag(sqrt(eigvals))
            pc_orig = eigvecs @ (np.sqrt(eigvals) * pc)
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc_orig, pc_orig) + (1 - cc) * np.eye(dim)) + cmu * DeltaH
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            
            # Update eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            # Ensure eigvals are positive
            eigvals = np.maximum(eigvals, 1e-12)
            
            # Update mean
            mean = new_mean
            
            # Optional: If sigma is very small, we might be converged.
            # But we have a budget, so we just continue.
            
        return None
