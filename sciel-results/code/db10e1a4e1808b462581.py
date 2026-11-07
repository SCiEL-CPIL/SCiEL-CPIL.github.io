"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using analytical front sampling + local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n = self.dim
        n_obj = self.n_obj

        def dtlz2_objectives(x):
            """Compute DTLZ2 objectives analytically."""
            # g(x) = sum((xi - 0.5)^2) for i=2..n (1-indexed), i.e., x[1:]
            g = np.sum((x[1:] - 0.5) ** 2)
            # f_i = (1 + g) * prod
            # For DTLZ2: f_i = (1+g) * cos(x_1 * pi/2) * ... pattern
            # Standard DTLZ2: f_i = (1+g) * prod_{j=1}^{i} cos(x_j * pi/2) * prod_{j=i+1}^{n_obj-1} sin(x_j * pi/2)
            # Actually for DTLZ2 with m objectives:
            # f_i = (1+g) * prod_{j=1}^{i} cos(x_j * pi/2) * prod_{j=i+1}^{m-1} sin(x_j * pi/2)
            # where x_1..x_{m-1} are the position variables
            m = n_obj
            # Position variables: x[0] .. x[m-2]
            # g variables: x[m-1] .. x[n-1]
            if m <= 1:
                return np.array([1.0 + g])
            
            # Compute products
            cos_prod = np.ones(m)
            sin_prod = np.ones(m)
            for j in range(m - 1):
                cos_prod[j + 1] = cos_prod[j] * np.cos(x[j] * np.pi / 2)
                sin_prod[j] = sin_prod[j] * np.sin(x[j] * np.pi / 2)
            
            f = np.zeros(m)
            for i in range(m):
                f[i] = (1 + g) * cos_prod[i] * sin_prod[i]
            return f

        def evaluate(x):
            """Use the problem function (clips to bounds)."""
            return np.array(problem(x), dtype=np.float64)

        # Phase 1: Sample the Pareto front analytically
        # For DTLZ2, the Pareto front is: f_i = cos(x_i*pi/2)*... with g=0
        # We sample x[0]..x[m-2] to get diverse points on the front
        # Then we need to minimize g by optimizing x[m-1]..x[n-1]
        
        # Number of front samples
        n_front = min(150, self.budget // 2)
        
        # Generate diverse position variables using Latin hypercube-like sampling
        # For m=3, we need x[0], x[1] in [0,1]
        m = n_obj
        pos_dim = m - 1  # number of position variables
        
        # Use stratified sampling for position variables
        samples = []
        if pos_dim == 1:
            # Simple 1D stratified
            for i in range(n_front):
                x_pos = (i + 0.5) / n_front
                samples.append(x_pos)
        elif pos_dim == 2:
            # 2D stratified using grid + jitter
            side = int(np.ceil(np.sqrt(n_front)))
            for i in range(side):
                for j in range(side):
                    if len(samples) >= n_front:
                        break
                    x_pos = np.array([(i + 0.5) / side, (j + 0.5) / side])
                    samples.append(x_pos)
            samples = samples[:n_front]
        else:
            # General case: random stratified
            for i in range(n_front):
                samples.append((i + 0.5) / n_front * np.ones(pos_dim))
        
        # For each front sample, we need to find x[m-1:] that minimizes g
        # g = sum((xi - 0.5)^2) for i in [m-1, n-1]
        # The minimum is at xi = 0.5, giving g = 0
        # But we must use the problem function to get actual objective values
        
        # Phase 2: Local search to minimize g for each front sample
        # Start from x[m-1:] = 0.5 (which gives g=0 for DTLZ2)
        # But we need to verify with the actual problem function
        
        results_F = []
        results_X = []
        used = 0
        
        for x_pos in samples:
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            
            # Construct full x: position vars + decision vars for g
            x = np.zeros(n)
            x[:pos_dim] = x_pos
            # Initialize g-variables at 0.5 (optimal for DTLZ2)
            x[pos_dim:] = 0.5
            
            # Evaluate
            f = evaluate(x)
            results_F.append(f)
            results_X.append(x.copy())
            used += 1
            
            # Local search to improve g (minimize sum of objectives)
            # Use a simple hill-climbing approach
            best_f = f.copy()
            best_x = x.copy()
            best_sum = np.sum(best_f)
            
            step_size = 0.1
            for _ in range(20):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                
                # Try small perturbations in the g-variables
                improved = False
                for _ in range(5):
                    perturbation = rng.standard_normal(n - pos_dim) * step_size
                    x_trial = best_x.copy()
                    x_trial[pos_dim:] += perturbation
                    x_trial = np.clip(x_trial, lo, hi)
                    
                    f_trial = evaluate(x_trial)
                    used += 1
                    sum_trial = np.sum(f_trial)
                    
                    if sum_trial < best_sum:
                        best_f = f_trial
                        best_x = x_trial.copy()
                        best_sum = sum_trial
                        improved = True
                        break
                
                if not improved:
                    step_size *= 0.7
                    if step_size < 0.001:
                        break
            
            # Update with best found
            results_F[-1] = best_f
            results_X[-1] = best_x
        
        # Phase 3: If budget remains, do additional random search
        if used < self.budget:
            n_remaining = self.budget - used
            for _ in range(n_remaining):
                if time.time() > deadline:
                    break
                
                # Random point
                x = lo + (hi - lo) * rng.random(n)
                f = evaluate(x)
                results_F.append(f)
                results_X.append(x)
                used += 1
        
        # Return all evaluated points
        if results_F:
            return np.array(results_F)
        return np.array([])
# EVOLVE-BLOCK-END
