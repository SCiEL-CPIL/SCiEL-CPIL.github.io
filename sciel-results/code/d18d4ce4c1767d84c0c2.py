"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

class Optimizer:
    """CMA-ES based optimizer for BBOB f12."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280
        dim = self.dim
        
        # Initial setup
        # Start near center but slightly randomized to avoid exact symmetry if any
        x0 = np.zeros(dim)
        
        # Initial sigma
        # For bent cigar, a larger sigma helps escape the narrow valley initially
        sigma0 = 2.0
        
        # Population size
        # Standard CMA-ES lambda = 4 + floor(3 log n)
        lam = max(4, int(4 + 3 * np.log(dim)))
        
        # Budget allocation
        # We have ~500 calls. 
        # Each iteration costs lam calls.
        # Max iterations = budget // lam
        max_iter = self.budget // lam
        
        best_x = x0
        best_f = np.inf
        
        if HAS_CMA:
            # Use pycma if available for robustness
            # Set seed to 0 for reproducibility as per original
            opts = {'seed': 0, 'verbose': -9, 'maxiter': max_iter, 'bounds': [-5, 5]}
            # Start from x0
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            
            while not es.stop() and es.countiter < max_iter:
                if time.time() > deadline:
                    break
                
                # Ask for next population
                xs = es.ask()
                
                # Evaluate
                fs = []
                for x in xs:
                    # Clip to bounds just in case, though CMA handles it via reflection usually
                    # The task says func clips, so we can pass raw x
                    fs.append(func(x))
                
                # Tell
                es.tell(xs, fs)
                
                # Update best
                current_best = es.result.xbest
                current_best_f = es.result.fbest
                if current_best_f < best_f:
                    best_f = current_best_f
                    best_x = current_best.copy()
        else:
            # Fallback: Simple CMA-ES implementation in NumPy
            # This is a simplified version, less robust but works without dependencies
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            step_size = 1.0
            
            # Precompute weights
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            c1 = (2 / ((dim + 1)**2 + 10)) / mueff
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            for i in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Generate population
                zs = np.random.randn(lam, dim)
                # Cholesky of C
                L = np.linalg.cholesky(C)
                xs = np.array([mean + sigma * (L @ z.T).T for z in zs])
                
                # Evaluate
                fs = np.array([func(x) for x in xs])
                
                # Sort
                idx = np.argsort(fs)
                xs = xs[idx]
                fs = fs[idx]
                
                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = xs[0].copy()
                    
                # Weighted mean
                new_mean = np.sum(weights[:, None] * xs[:mu], axis=0)
                
                # Update covariance
                diff = xs[:mu] - mean
                C = (1 - c1 - cmu) * C + c1 * (np.outer(mean - new_mean, mean - new_mean) + (mean - new_mean) @ (mean - new_mean).T) \
                    + cmu * np.sum(weights[:, None, None] * diff[:, None, :] * diff[:, None, :], axis=0)
                
                # Update step size
                ps = (1 - cs) * ps + cs * np.sqrt(mueff) / sigma * (L @ (new_mean - mean).T).T
                hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / np.sqrt(1 - (1 - cs)**(2 * (i + 1))) < (1.4 + 2 / (dim + 1)) * np.trace(C) / dim else 0
                pc = (1 - c1) * pc + c1 * hsig * (new_mean - mean) / sigma
                
                # Check eigenvalues
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C = np.eye(dim)
                    sigma = sigma0
                    mean = x0.copy()
                    break
                
                mean = new_mean
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (i + 1))) - 1))
                
                # Bounds handling: simple reflection or clipping
                # CMA-ES usually handles bounds via transformation, but here we just clip
                # If we clip, the distribution is distorted. Better to keep within bounds.
                # For simplicity in fallback, we just clip the mean and hope for the best.
                mean = np.clip(mean, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
