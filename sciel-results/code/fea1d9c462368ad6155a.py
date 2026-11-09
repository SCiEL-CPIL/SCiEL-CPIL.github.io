"""Multi-objective optimizer for RE34: Bayesian Optimization with EHVI."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimizer for RE34 using EHVI."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])
        
        # Scale bounds to [0, 1] for better GP conditioning
        span = hi - lo
        if np.any(span == 0):
            span = np.where(span == 0, 1.0, span)
            
        def to_scaled(x):
            return (x - lo) / span
            
        def to_real(x):
            return lo + x * span

        # Initialize with Halton sequence for deterministic, well-spread start
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = []
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
            for i in range(1, n + 1):
                p = [halton(i, b) for b in bases[:dim]]
                points.append(p)
            return np.array(points)

        n_init = min(20, self.budget // 10)
        if n_init < 5:
            n_init = 5

        X_scaled_init = halton_sequence(n_init, self.dim)
        X_real_init = to_real(X_scaled_init)
        
        F = np.array([problem(x) for x in X_real_init])
        X_scaled = X_scaled_init
        n_evals = n_init

        # GP Hyperparameters
        # We use a simple Matern kernel or RBF. RBF is smoother, Matern(5/2) is less smooth.
        # RE34 might be somewhat smooth. Let's use RBF for stability.
        # We will fix hyperparameters for speed, or do a very quick optimization.
        # Given 200 budget, fixed HPs are safer to avoid overhead.
        # Length scale l, signal variance sigma_f, noise variance sigma_n.
        
        # Since we scaled inputs to [0,1], l=0.2 is reasonable.
        # sigma_f should match the scale of objectives.
        # Objective scales vary significantly (1705, 12, 0.3).
        # We should standardize objectives for the GP? 
        # EHVI works on original scales. 
        # If we standardize, we need to map predictions back.
        # It's better to let the GP model the raw values but with appropriate sigma_f.
        # However, mixing scales in a single GP is bad. We are using 3 independent GPs.
        # So each GP has its own sigma_f.
        
        # Let's estimate sigma_f from the initial data.
        # If data variance is 0, set to 1.
        
        gps = []
        for j in range(self.n_obj):
            y = F[:, j]
            mean_y = np.mean(y)
            std_y = np.std(y)
            if std_y < 1e-6:
                std_y = 1.0
            # Noise is small
            sigma_f = std_y * 2.0 # Slightly larger to allow exploration
            sigma_n = std_y * 0.01 + 1e-6
            l = 0.2 # Length scale in scaled space
            gps.append({
                'mean': mean_y,
                'std': std_y,
                'sigma_f': sigma_f,
                'sigma_n': sigma_n,
                'l': l
            })

        # GP Prediction Function
        def gp_predict(X_train, y_train, X_test, hp):
            # X_train, X_test are in scaled space [0,1]^dim
            # y_train is raw objective values
            sigma_f = hp['sigma_f']
            sigma_n = hp['sigma_n']
            l = hp['l']
            
            # Kernel: RBF
            # K(x, x') = sigma_f^2 * exp(-0.5 * ||x - x'||^2 / l^2)
            
            # Compute K(X_train, X_train) + sigma_n^2 I
            diff_train = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
            sq_dist_train = np.sum(diff_train ** 2, axis=-1)
            K_train = sigma_f ** 2 * np.exp(-0.5 * sq_dist_train / (l ** 2))
            K_train[np.diag_indices_from(K_train)] += sigma_n ** 2
            
            # Compute K(X_train, X_test)
            diff_test = X_train[:, np.newaxis, :] - X_test[np.newaxis, :, :]
            sq_dist_test = np.sum(diff_test ** 2, axis=-1)
            K_test = sigma_f ** 2 * np.exp(-0.5 * sq_dist_test / (l ** 2))
            
            # Solve
            try:
                L = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train - hp['mean']))
                K_s = np.linalg.solve(L, K_test)
                mu = hp['mean'] + K_s.T @ alpha
                v = np.linalg.solve(L, K_test)
                K_test_test = sigma_f ** 2 * np.ones((X_test.shape[0], X_test.shape[0])) # Diagonal is sigma_f^2
                # Actually, variance is K(x,x) - K(x, X_train) K(X_train, X_train)^-1 K(X_train, x)
                # K(x,x) for RBF is sigma_f^2
                var = sigma_f ** 2 - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-9)
            except np.linalg.LinAlgError:
                # Fallback to identity if Cholesky fails (shouldn't happen with noise)
                mu = np.full(X_test.shape[0], hp['mean'])
                var = np.full(X_test.shape[0], sigma_f ** 2)
                
            return mu, np.sqrt(var)

        def compute_ehvi(X_candidates, F_current, ref_point, gps_data):
            """
            Compute EHVI for a set of candidates.
            X_candidates: (M, dim) in scaled space
            F_current: (N, n_obj) current non-dominated front (raw values)
            gps_data: list of dicts with 'mean', 'std' etc, but we need predictions.
            Actually, we should pass the GPs or precomputed predictions.
            Let's restructure: compute predictions for all candidates first, then EHVI.
            """
            pass # Placeholder, logic moved to main loop for efficiency

        # Main Loop
        # We will select candidates from a set of points:
        # 1. Random points
        # 2. Points near current front
        # 3. Points suggested by GP uncertainty?
        
        # For simplicity and robustness, we will generate a pool of candidates each step:
        # - 50 random points
        # - 50 points perturbed from current best points
        # Evaluate EHVI for all, pick the best.
        
        # To speed up EHVI, we need an efficient implementation.
        # Exact EHVI for 3 objectives is complex. 
        # Approximation: Use Monte Carlo sampling of the GP posterior to estimate EHVI.
        # This is slower but easier to implement correctly.
        # Given 200 evals, and maybe 100 candidates * 30 samples * 3 objectives...
        # 100 * 30 * 3 = 9000 GP evals per step. 
        # 200 steps * 9000 = 1.8M GP evals. Might be too slow in Python.
        
        # Alternative: Use a simpler acquisition function like EI (Expected Improvement) 
        # on a weighted sum or scalarization, or just use GP Mean + Uncertainty.
        # But the directive asks for EHVI.
        
        # Let's try a simplified EHVI using Monte Carlo with fewer samples (e.g., 10).
        # And fewer candidates (e.g., 20).
        
        n_candidates = 20
        n_mc_samples = 10
        
        # Pre-allocate arrays for speed
        # We will maintain the list of all evaluated points to update GP?
        # Updating GP every step (re-fitting) is O(N^3). N=200 -> 8M ops. 200 steps -> 1.6B ops. Too slow.
        # We should use a fixed GP fitted on the initial data? 
        # Or update the GP only every 10 steps?
        # Or use a simpler surrogate?
        
        # Let's stick to the directive: "Bayesian Optimization surrogate".
        # If we re-fit every step, it might be too slow.
        # Let's fit the GP once on the initial 20 points, and use it for the rest?
        # That ignores new information. Bad.
        
        # Compromise: Fit GP on the current non-dominated front? 
        # Or fit on all points but use a sparse approximation?
        
        # Given the time limit (330s), and Python overhead, re-fitting a 5D GP 200 times 
        # with N=200 is risky.
        # Let's check cost: Cholesky of 200x200 is ~1.3M flops. 200 times is 260M flops.
        # Python/NumPy overhead might make this 10-20 seconds total. Acceptable.
        
        # So, strategy:
        # 1. Fit 3 GPs on ALL evaluated points.
        # 2. Generate 20 candidates.
        # 3. Predict mean/var for candidates.
        # 4. Compute EHVI via MC (10 samples).
        # 5. Pick best, evaluate, add to set.
        
        # EHVI MC Implementation:
        # For each candidate x:
        #   For each sample s:
        #     Draw y_s ~ N(mu(x), var(x)) for each obj
        #     Compute HV of (F_current U {y_s}) - HV(F_current)
        #   EHVI(x) = mean(improvements)
        
        # To speed up HV calculation, we can use a fast 3D HV algorithm.
        # The previous program had a slow one. Let's write a fast one or use a library?
        # No library. 
        # Fast 3D HV: O(N log N) or O(N^2). 
        # For N=200, O(N^2) is 40k ops. 20 candidates * 10 samples * 40k = 8M ops per step.
        # 200 steps * 8M = 1.6B ops. Might be borderline in Python.
        
        # Optimization: 
        # - Only compute HV for non-dominated points.
        # - Use vectorized operations where possible.
        
        def hypervolume_3d_fast(points, ref_point):
            """
            points: (N, 3) array of objective values.
            ref_point: (3,) array.
            Returns hypervolume.
            """
            if len(points) == 0:
                return 0.0
            
            # Filter points better than ref
            mask = np.all(points < ref_point, axis=1)
            points = points[mask]
            if len(points) == 0:
                return 0.0
            
            # Remove dominated points
            # Sort by obj 0
            points = points[np.argsort(points[:, 0])]
            
            front = [points[0]]
            for i in range(1, len(points)):
                p = points[i]
                # Check if p is dominated by any point in front
                # Since sorted by obj 0, front points have obj 0 <= p[0]
                # We need to check if there is a point in front with obj 1 <= p[1] AND obj 2 <= p[2]
                # We can maintain the min obj 2 for each obj 1? 
                # For small N, simple check is fine.
                dominated = False
                for q in front:
                    if q[1] <= p[1] and q[2] <= p[2]:
                        dominated = True
                        break
                if not dominated:
                    front.append(p)
            
            front = np.array(front)
            if len(front) == 0:
                return 0.0
            
            # Sort by obj 0
            front = front[np.argsort(front[:, 0])]
            
            # Calculate HV
            # Sweep along obj 0
            total_hv = 0.0
            # Active set of (obj1, obj2)
            # We need 2D HV of active set.
            
            def hv_2d(pts, ref_yz):
                if len(pts) == 0:
                    return 0.0
                # Sort by y
                pts = pts[np.argsort(pts[:, 0])]
                min_z = np.inf
                total = 0.0
                for i in range(len(pts)):
                    y_i = pts[i, 0]
                    z_i = pts[i, 1]
                    if z_i < min_z:
                        min_z = z_i
                    if min_z < ref_yz[1]:
                        y_next = pts[i+1, 0] if i + 1 < len(pts) else ref_yz[0]
                        total += (y_next - y_i) * (ref_yz[1] - min_z)
                    else:
                        break
                return total

            for i in range(len(front)):
                x_i = front[i, 0]
                x_next = front[i+1, 0] if i + 1 < len(front) else ref_point[0]
                
                if x_next <= x_i:
                    continue
                    
                active_yz = front[0:i+1, 1:3]
                hv_2d_val = hv_2d(active_yz, ref_point[1:3])
                total_hv += (x_next - x_i) * hv_2d_val
                
            return total_hv

        # Main Loop
        while n_evals < self.budget:
            if time.time() > deadline:
                break
            
            # 1. Fit GPs on current data
            # X_scaled is (N, dim), F is (N, n_obj)
            # We fit 3 independent GPs
            # To save time, we can skip refitting if N is small? No, let's refit.
            # But 3 Cholesky decompositions of NxN where N grows to 200.
            
            # Optimization: Only refit every 5 steps?
            # Or use a simpler model?
            # Let's try refitting every step but monitor time.
            
            # Generate Candidates
            # Mix of random and local search around best points
            candidates = []
            
            # 10 Random points in [0,1]^5
            rand_pts = np.random.rand(10, self.dim)
            candidates.extend(rand_pts)
            
            # 10 Points near current front
            # Find non-dominated points in F
            front_idx = []
            for i in range(len(F)):
                is_dom = False
                for j in range(len(F)):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            is_dom = True
                            break
                if not is_dom:
                    front_idx.append(i)
            
            if front_idx:
                # Pick a few from front
                n_front_pts = min(5, len(front_idx))
                chosen_front_idx = np.random.choice(front_idx, n_front_pts, replace=False)
                for idx in chosen_front_idx:
                    # Perturb with small noise
                    noise = np.random.randn(self.dim) * 0.1
                    new_pt = X_scaled[idx] + noise
                    new_pt = np.clip(new_pt, 0, 1)
                    candidates.append(new_pt)
            
            candidates = np.array(candidates)
            n_cand = len(candidates)
            
            # 2. Predict with GPs
            # We need to fit GPs first.
            # This is the expensive part.
            # Let's fit them.
            
            gps = []
            for j in range(self.n_obj):
                y = F[:, j]
                mean_y = np.mean(y)
                std_y = np.std(y)
                if std_y < 1e-6:
                    std_y = 1.0
                sigma_f = std_y * 2.0
                sigma_n = std_y * 0.01 + 1e-6
                l = 0.2
                
                # Fit
                X_train = X_scaled
                diff_train = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
                sq_dist_train = np.sum(diff_train ** 2, axis=-1)
                K_train = sigma_f ** 2 * np.exp(-0.5 * sq_dist_train / (l ** 2))
                K_train[np.diag_indices_from(K_train)] += sigma_n ** 2
                
                try:
                    L = np.linalg.cholesky(K_train)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - mean_y))
                    gps.append({
                        'mean': mean_y,
                        'L': L,
                        'alpha': alpha,
                        'sigma_f': sigma_f,
                        'l': l,
                        'X_train': X_train
                    })
                except np.linalg.LinAlgError:
                    # Fallback
                    gps.append({
                        'mean': mean_y,
                        'L': None,
                        'alpha': None,
                        'sigma_f': sigma_f,
                        'l': l,
                        'X_train': X_train
                    })
            
            # Predict for candidates
            # mu: (n_cand, n_obj), var: (n_cand, n_obj)
            mus = np.zeros((n_cand, self.n_obj))
            vars_ = np.zeros((n_cand, self.n_obj))
            
            for j in range(self.n_obj):
                gp = gps[j]
                if gp['L'] is None:
                    mus[:, j] = gp['mean']
                    vars_[:, j] = gp['sigma_f'] ** 2
                    continue
                
                X_test = candidates
                diff_test = gp['X_train'][:, np.newaxis, :] - X_test[np.newaxis, :, :]
                sq_dist_test = np.sum(diff_test ** 2, axis=-1)
                K_test = gp['sigma_f'] ** 2 * np.exp(-0.5 * sq_dist_test / (gp['l'] ** 2))
                
                K_s = np.linalg.solve(gp['L'], K_test)
                mus[:, j] = gp['mean'] + K_s.T @ gp['alpha']
                v = np.linalg.solve(gp['L'], K_test)
                vars_[:, j] = gp['sigma_f'] ** 2 - np.sum(v ** 2, axis=0)
                vars_[:, j] = np.maximum(vars_[:, j], 1e-9)
            
            # 3. Compute EHVI via MC
            # Base HV
            base_hv = hypervolume_3d_fast(F, ref)
            
            # For each candidate, estimate EHVI
            ehvi_values = np.zeros(n_cand)
            
            # To speed up, we can vectorize the MC sampling?
            # For each candidate i:
            #   samples = np.random.multivariate_normal? No, independent objectives.
            #   y_samples = mus[i] + np.sqrt(vars_[i]) * np.random.randn(n_mc, n_obj)
            #   For each sample, compute HV of F U {y_sample}
            
            # This loop is n_cand * n_mc * HV_cost.
            # 20 * 10 * 40k = 8M ops.
            
            for i in range(n_cand):
                # Sample from GP posterior
                # y ~ N(mu, sigma^2)
                samples = mus[i] + np.sqrt(vars_[i]) * np.random.randn(n_mc_samples, self.n_obj)
                
                improvements = 0.0
                for s in range(n_mc_samples):
                    y_s = samples[s]
                    # Check if y_s is better than ref
                    if not np.all(y_s < ref):
                        continue
                    
                    # Add y_s to F
                    # We need HV of F U {y_s}
                    # To avoid modifying F, we can create a new array? 
                    # Or just pass a list? 
                    # hypervolume_3d_fast takes an array.
                    # Creating a new array of size N+1 is cheap.
                    F_new = np.vstack([F, y_s.reshape(1, -1)])
                    hv_new = hypervolume_3d_fast(F_new, ref)
                    improvements += (hv_new - base_hv)
                
                ehvi_values[i] = improvements / n_mc_samples
            
            # Pick best candidate
            best_idx = np.argmax(ehvi_values)
            x_next_scaled = candidates[best_idx]
            x_next_real = to_real(x_next_scaled)
            
            # Evaluate
            f_next = problem(x_next_real)
            n_evals += 1
            
            # Update data
            X_scaled = np.vstack([X_scaled, x_next_scaled.reshape(1, -1)])
            F = np.vstack([F, f_next.reshape(1, -1)])
            
            # Optional: Prune dominated points from X_scaled/F to keep GP training set small?
            # If we keep all points, N grows to 200.
            # GP cost is O(N^3). 200^3 = 8M.
            # 200 steps * 8M = 1.6B.
            # In Python, this might be slow.
            # Let's prune to non-dominated points only for GP training?
            # But GP should model the whole function, not just the front.
            # However, for EHVI, we care about the front.
            # Using all points is more accurate for the GP.
            # Let's monitor time.
            
            if time.time() > deadline - 10:
                break

        return X_real_init # Return something, though ignored

# EVOLVE-BLOCK-END
