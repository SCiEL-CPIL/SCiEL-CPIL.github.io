import numpy as np
import time
from scipy.stats import norm
from scipy.spatial import ConvexHull
import warnings

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        max_time = 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        used = 0
        archive_x = []
        archive_f = []
        
        def is_dominated(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        def update_archive(x, f):
            nonlocal archive_x, archive_f
            to_remove = []
            for i in range(len(archive_x)):
                if is_dominated(f, archive_f[i]):
                    to_remove.append(i)
            
            if to_remove:
                for i in reversed(to_remove):
                    archive_x.pop(i)
                    archive_f.pop(i)
                
                for i in range(len(archive_x)):
                    if is_dominated(archive_f[i], f):
                        return False
            
            archive_x.append(x)
            archive_f.append(f)
            return True

        # Hypervolume calculation for 3D using recursive decomposition
        # Reference point is fixed at [1.1, 1.3, 1.1]
        ref_point = np.array([1.1, 1.3, 1.1])
        
        def calc_hypervolume(points, ref):
            if not points:
                return 0.0
            # Filter points strictly better than ref
            valid_points = [p for p in points if np.all(p < ref)]
            if not valid_points:
                return 0.0
            
            # For 3D, we can use a simple recursive slicing or convex hull volume approximation
            # Since the number of points is small (< 50 usually), we can use a simple O(N^2) or O(N^3) method
            # Let's use the standard recursive algorithm for 3D HV
            # Sort by first objective
            pts = np.array(valid_points)
            pts = pts[np.argsort(pts[:, 0])]
            
            # Recursive helper
            def hv_recursive(pts_2d, ref_2d):
                if len(pts_2d) == 0:
                    return 0.0
                if len(pts_2d) == 1:
                    return np.prod(ref_2d - pts_2d[0])
                
                # Sort by first coord
                pts_2d = pts_2d[np.argsort(pts_2d[:, 0])]
                
                # Remove dominated points in 2D
                # In 2D, if sorted by x, we only keep points with decreasing y
                pareto_2d = [pts_2d[0]]
                min_y = pts_2d[0, 1]
                for p in pts_2d[1:]:
                    if p[1] < min_y:
                        pareto_2d.append(p)
                        min_y = p[1]
                
                hv = 0.0
                # Integrate over y
                # The volume is sum of boxes
                # For 2D: sum (x_{i+1} - x_i) * (ref_y - y_i) + (ref_x - x_last) * (ref_y - y_last)
                # Wait, the standard formula for 2D HV with ref:
                # Sort by x ascending.
                # HV = sum_{i=0}^{N-1} (x_{i+1} - x_i) * (ref_y - y_i) where x_N = ref_x
                # Actually, it's easier to just compute the union of rectangles.
                # Given small N, let's just do the standard recursive decomposition for 3D.
                
                # Let's implement a generic 3D HV using the "slice" method
                # This is getting complex. For < 50 points, we can use a simpler approach.
                # Since we only need EHVI for selection, and the archive is small,
                # we can approximate EHVI by sampling.
                # But let's try to be exact if possible.
                
                # Actually, for the acquisition function, we don't need the *exact* HV of the current archive
                # if we are just comparing candidates. But we do need it to calculate the improvement.
                
                # Let's use a standard 3D HV algorithm.
                # Reference: "A simple and efficient algorithm for computing the hypervolume indicator in 3D"
                # Or just use the fact that N is small.
                
                # Let's implement a simple O(N^3) HV for 3D.
                # It works by projecting onto 2D planes.
                
                # Actually, for the sake of robustness and speed, let's use a Monte Carlo estimate for EHVI
                # if exact calculation is too slow/complex to code correctly in short time.
                # But 3D HV is standard.
                
                # Let's stick to a robust recursive HV for 3D.
                # From literature:
                # HV(S, r) = sum_{i in S} (r_i - f_i) * HV({j in S | f_j <= f_i and f_j != f_i}, f_i) ... this is not right.
                
                # Let's use the "box counting" method for small N.
                # Generate a grid? No, that's inaccurate.
                
                # I will use a known O(N^2 log N) or O(N^3) algorithm for 3D.
                # Here is a simple recursive implementation for 3D:
                
                def hv_3d(pts, ref):
                    if len(pts) == 0:
                        return 0.0
                    # Sort by first objective
                    pts = pts[np.argsort(pts[:, 0])]
                    # Remove points dominated by the first point in the sorted list? No.
                    # Standard algorithm:
                    # 1. Sort by f1.
                    # 2. For each point i, calculate the volume of the box defined by (f_i, ref)
                    #    that is not covered by points j < i (which have smaller f1).
                    #    This reduces to a 2D HV problem for the remaining dimensions.
                    
                    total_hv = 0.0
                    # We process points from best f1 to worst f1.
                    # Actually, it's easier to process from worst f1 to best f1?
                    # Let's use the recursive definition:
                    # HV(S, r) = (r_1 - f_{1,min}) * HV_2D(proj(S), r_2d) ... no.
                    
                    # Let's use this specific 3D algorithm:
                    # 1. Sort points by f1 ascending.
                    # 2. Keep a list of "active" points that are not dominated in the remaining 2D space.
                    #    As we move f1 from low to high, we add points.
                    #    The contribution of a point is the volume of the region it dominates
                    #    that hasn't been dominated by points with smaller f1.
                    
                    # This is effectively integrating the 2D HV of the "front" at each f1 slice.
                    
                    # Given the complexity, and N < 50, let's use a simple grid-based approximation
                    # for the HV calculation inside the EHVI loop? No, that's slow.
                    
                    # Let's use the `pymoo` logic if available? No, standard library only.
                    
                    # I will implement a simple O(N^3) HV for 3D.
                    # It iterates over all pairs and calculates the volume of the box
                    # that is uniquely dominated by a point.
                    
                    # Actually, for 3D, we can project to 2D.
                    # HV_3D(S, r) = sum_{i=1}^N (r_1 - f_{1,i}) * HV_2D(S_i, r_{2:}) where S_i are points dominated by i in 2D?
                    # No.
                    
                    # Let's use this reliable method for small N:
                    # Generate all corner points of the "staircase".
                    # In 3D, the Pareto front defines a staircase.
                    # We can compute the volume by summing the volumes of the rectangular boxes
                    # that form the staircase.
                    
                    # For N=1: V = prod(r - f)
                    # For N>1:
                    # Sort by f1.
                    # The volume is the sum over i of:
                    # (f_{1,i+1} - f_{1,i}) * (r_2 - f_{2,i}) * (r_3 - f_{3,i}) ... this is only valid for 2D.
                    
                    # Okay, I will use a Monte Carlo estimate for EHVI to ensure correctness and speed.
                    # With 1000 samples, it should be stable enough for selection.
                    pass

                # Let's switch strategy: Use a simple random sampling for EHVI.
                # It is robust, fast, and avoids complex HV bugs.
                # EHVI(x) = E[ HV(A ∪ {f(x)}) - HV(A) ]
                # We estimate this by sampling from the GP posterior.
                
                return 0.0 # Placeholder

        # 1. Initialization
        # Use Halton sequence for better coverage as per lesson
        def halton(dim, n):
            def halton_index(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f = f / base
                    r = r + f * (index % base)
                    index = index // base
                return r

            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            points = []
            for i in range(1, n + 1):
                p = []
                for d in range(dim):
                    p.append(halton_index(i, primes[d]))
                points.append(p)
            return np.array(points)

        init_size = min(20, budget // 5)
        X_init_norm = halton(dim, init_size)
        X_init = lo + X_init_norm * (hi - lo)
        
        for x in X_init:
            f = problem(x)
            used += 1
            update_archive(x, f)
            if used >= budget:
                return archive_x
            if time.time() - start_time > max_time:
                return archive_x

        # 2. MBEA Loop
        # GP Model
        # We need to fit a GP. Since we can't use sklearn, we implement a simple GP.
        # K(x, x') = sigma_f^2 * exp(-0.5 * ||x - x'||^2 / l^2) + sigma_n^2 * I
        
        class SimpleGP:
            def __init__(self, dim, n_obj, alpha=1e-6):
                self.dim = dim
                self.n_obj = n_obj
                self.alpha = alpha
                self.l = 0.5
                self.sigma_f = 1.0
                self.sigma_n = 1e-4
                
            def fit(self, X, Y):
                self.X = X
                self.Y = Y
                n = len(X)
                # Standardize Y? GP is scale invariant if we handle kernel correctly.
                # Let's standardize Y for numerical stability
                self.mean_y = np.mean(Y, axis=0)
                self.std_y = np.std(Y, axis=0) + 1e-8
                Y_std = (Y - self.mean_y) / self.std_y
                self.Y_std = Y_std
                
                # Kernel matrix
                self.K = self._kernel(X, X) + self.alpha * np.eye(n)
                self.L = np.linalg.cholesky(self.K)
                self.alpha_vec = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y_std))
                
            def _kernel(self, X1, X2):
                # Squared exponential
                # X1: (n1, dim), X2: (n2, dim)
                # Compute squared distances
                X1_sq = np.sum(X1**2, axis=1)[:, None]
                X2_sq = np.sum(X2**2, axis=1)[None, :]
                dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
                dist_sq = np.maximum(dist_sq, 0)
                return self.sigma_f**2 * np.exp(-0.5 * dist_sq / self.l**2)
            
            def predict(self, X_new):
                K_s = self._kernel(self.X, X_new)
                mu_std = K_s.T @ self.alpha_vec
                mu = mu_std * self.std_y + self.mean_y
                
                # Variance
                # var(x) = k(x,x) - k(x, X)^T K^{-1} k(x, X)
                # We need K^{-1} k(x, X)
                # K^{-1} = L^{-1} L^{-T}
                # Let v = L^{-1} k(x, X)
                # Then K^{-1} k(x, X) = L^{-T} v
                
                n_new = len(X_new)
                # K_s is (n, n_new)
                # We want var for each new point
                # v_i = L^{-1} K_s[:, i]
                # This is solving L v = K_s[:, i] for each i
                # We can do it all at once: L V = K_s.T => V = solve(L, K_s.T)
                # V shape: (n, n_new)
                V = np.linalg.solve(self.L, K_s.T)
                # var_std_i = self.sigma_f**2 - sum(V[:, i]**2)
                var_std = self.sigma_f**2 - np.sum(V**2, axis=0)
                var = var_std * self.std_y**2 # Approximate variance scaling
                # Note: Scaling variance by std_y^2 is an approximation for the covariance in original space.
                # For EHVI, we usually work in standardized space or transform the EHVI.
                # Let's work in standardized space for EHVI calculation to keep GP assumptions valid.
                # So we need to transform the archive and reference point to standardized space.
                
                return mu, var

        # EHVI in standardized space
        # We need to calculate HV in standardized space.
        # The reference point in standardized space is (ref - mean) / std.
        # The objectives in standardized space are (f - mean) / std.
        # Minimizing f is equivalent to minimizing f_std.
        
        # We will maintain the archive in original space, but for EHVI,
        # we will standardize the archive and the candidate predictions.
        
        # Since the GP model is refit every iteration, the standardization changes.
        # This is computationally expensive but N is small.
        
        # To save time, we will only refit the GP every few iterations or use a fixed standardization
        # based on the initial points? No, that's bad.
        # Let's refit every time. N=20..200. Cholesky is O(N^3). 200^3 = 8M. Fast enough.
        
        # For the HV calculation in standardized space, we need a robust HV calculator.
        # Given the complexity, I will use a simple grid-based HV for the *small* number of archive points
        # in the standardized space, or just use the Monte Carlo EHVI directly in original space?
        # No, GP predictions are in original space (mu, var).
        # If we use original space, the GP assumptions (Gaussian) might be violated if objectives are skewed.
        # But for 4D RE37, it might be okay.
        
        # Let's try EHVI in original space with a simple HV approximation.
        # Actually, let's use the "Expected Improvement" (EI) for each objective and sum them?
        # That's not EHVI.
        
        # Let's go with a simplified acquisition function: 
        # Maximize the minimum of the EIs for each objective?
        # Or use a hypervolume indicator approximation.
        
        # Given the time constraints and complexity of exact 3D HV,
        # I will use a **random sampling based EHVI** in the **original space**.
        # We assume the GP predictions (mu, var) are correct in original space.
        
        def ehvi_acquisition(X_candidates, archive_f, gp_model, ref_point, n_samples=100):
            # X_candidates: (M, dim)
            # archive_f: (N, n_obj)
            # ref_point: (n_obj,)
            
            M = len(X_candidates)
            # Get predictions
            mu, var = gp_model.predict(X_candidates)
            # mu: (M, n_obj), var: (M, n_obj)
            
            # Sample from posterior
            # f_s ~ N(mu, diag(var))
            # We need to calculate HV(archive ∪ {f_s}) - HV(archive) for each sample.
            
            # To speed up, we can pre-calculate HV(archive).
            # But we need HV(archive ∪ {f_s}) for each s.
            
            # Since N (archive size) is small, we can do this.
            # However, we need a HV function.
            
            # Let's implement a simple HV for 3D using the "box" method.
            # It works by iterating through points and calculating the volume of the box
            # that is uniquely dominated by that point.
            
            def hv_3d_simple(points, ref):
                if not points:
                    return 0.0
                pts = np.array(points)
                # Filter
                pts = pts[np.all(pts < ref, axis=1)]
                if len(pts) == 0:
                    return 0.0
                
                # Sort by f0
                idx = np.argsort(pts[:, 0])
                pts = pts[idx]
                
                # For 3D, we can use the following:
                # The volume is the sum of the volumes of the boxes defined by
                # (f_i, ref) minus the volume of the boxes defined by points that dominate i.
                # This is complex.
                
                # Alternative: Use the fact that N is small.
                # We can compute the volume of the union of boxes.
                # In 3D, we can project to 2D.
                # HV = integral over f0 of Area_2D(f0) df0.
                # Area_2D(f0) is the HV of the projection of points with f1 < f0 onto (f1, f2).
                
                # This is O(N^2).
                # Let's implement this.
                
                # Sort by f0
                # For each unique f0 value (or point), calculate the 2D HV of the points with smaller f0?
                # No, the "front" changes.
                
                # Let's use a standard O(N^2) 2D HV and integrate.
                # Actually, for N < 50, we can just use a grid.
                # But grid is inaccurate.
                
                # I will use a recursive HV for 3D from a known source.
                # https://www.jmlr.org/papers/volume12/10-094/10-094.pdf
                # Algorithm 1.
                
                def hv_recursive(pts, ref):
                    n = len(pts)
                    if n == 0:
                        return 0.0
                    if n == 1:
                        return np.prod(ref - pts[0])
                    
                    # Sort by first dimension
                    pts = pts[np.argsort(pts[:, 0])]
                    
                    # Remove dominated points
                    # In 3D, if sorted by f0, a point is dominated if there is a point j < i
                    # such that f_j <= f_i in all dimensions.
                    # We can do a simple filter.
                    pareto = [pts[0]]
                    for i in range(1, n):
                        p = pts[i]
                        dominated = False
                        for q in pareto:
                            if np.all(q <= p):
                                dominated = True
                                break
                        if not dominated:
                            pareto.append(p)
                            # Remove points dominated by p
                            new_pareto = []
                            for q in pareto:
                                if q is not p and not np.all(p <= q):
                                    new_pareto.append(q)
                            pareto = new_pareto + [p]
                            
                    # Now we have the Pareto front sorted by f0.
                    # The volume is the sum of the volumes of the boxes.
                    # For each point i in pareto, the box is defined by (f_i, ref).
                    # But we must subtract the volume of boxes that overlap.
                    # The recursive formula is:
                    # HV(pareto, ref) = (ref_0 - f_{0,0}) * HV_2D(proj(pareto_1..N), ref_1..N)
                    #                 + ... this is not quite right.
                    
                    # Let's use the "slice" method.
                    # The volume is the sum over i of:
                    # (f_{0, i+1} - f_{0, i}) * HV_2D( {j | f_{0,j} <= f_{0,i}} , ref_1..N )
                    # Where f_{0, N+1} = ref_0.
                    
                    # We need a 2D HV function.
                    def hv_2d(pts_2d, ref_2d):
                        if len(pts_2d) == 0:
                            return 0.0
                        pts_2d = np.array(pts_2d)
                        pts_2d = pts_2d[np.all(pts_2d < ref_2d, axis=1)]
                        if len(pts_2d) == 0:
                            return 0.0
                        
                        # Sort by x
                        pts_2d = pts_2d[np.argsort(pts_2d[:, 0])]
                        
                        # Remove dominated
                        pareto_2d = [pts_2d[0]]
                        for i in range(1, len(pts_2d)):
                            p = pts_2d[i]
                            dominated = False
                            for q in pareto_2d:
                                if np.all(q <= p):
                                    dominated = True
                                    break
                            if not dominated:
                                pareto_2d.append(p)
                                new_pareto = []
                                for q in pareto_2d:
                                    if q is not p and not np.all(p <= q):
                                        new_pareto.append(q)
                                pareto_2d = new_pareto + [p]
                                
                        # Calculate HV
                        # Sort by x
                        pareto_2d = np.array(pareto_2d)
                        pareto_2d = pareto_2d[np.argsort(pareto_2d[:, 0])]
                        
                        hv = 0.0
                        prev_x = pareto_2d[0, 0]
                        prev_y = pareto_2d[0, 1]
                        # The area is the sum of rectangles
                        # For each point, the width is (next_x - curr_x)
                        # The height is (ref_y - curr_y)
                        # Wait, the "front" is decreasing y.
                        # So the area is sum_{i} (x_{i+1} - x_i) * (ref_y - y_i)
                        # Where x_N = ref_x.
                        
                        x_coords = pareto_2d[:, 0]
                        y_coords = pareto_2d[:, 1]
                        
                        # Add ref_x
                        x_coords = np.append(x_coords, ref_2d[0])
                        
                        widths = np.diff(x_coords)
                        heights = ref_2d[1] - y_coords
                        
                        return np.sum(widths * heights)
                    
                    # Now apply slice method for 3D
                    # pareto is sorted by f0
                    # We need to calculate HV_2D for prefixes of pareto.
                    # To be efficient, we can just do it for each i.
                    
                    total_hv = 0.0
                    f0_coords = pareto[:, 0]
                    f0_coords = np.append(f0_coords, ref[0])
                    widths = np.diff(f0_coords)
                    
                    for i in range(len(pareto)):
                        # Points with f0 <= pareto[i, 0]
                        # This is just pareto[0:i+1]
                        pts_slice = pareto[0:i+1, 1:]
                        hv_2d_val = hv_2d(pts_slice, ref[1:])
                        total_hv += widths[i] * hv_2d_val
                        
                    return total_hv
                
                return hv_recursive(pts, ref)

            # Pre-calculate HV of archive
            hv_archive = hv_3d_simple(archive_f, ref_point)
            
            # Sample from GP
            # f_s = mu + sqrt(var) * Z
            Z = np.random.normal(0, 1, size=(n_samples, M, n_obj))
            f_samples = mu[None, :, :] + np.sqrt(var)[None, :, :] * Z
            
            # For each sample, calculate HV(archive ∪ {f_s})
            # To speed up, we can vectorize the HV calculation? No, it's complex.
            # But n_samples=100, M=50 (grid size), N=20.
            # 100 * 50 = 5000 HV calculations.
            # Each HV calculation is O(N^2) ~ 400 ops.
            # 5000 * 400 = 2M ops. Fast.
            
            ehvi_vals = np.zeros(M)
            for m in range(M):
                for s in range(n_samples):
                    f_new = f_samples[s, m]
                    if np.any(f_new >= ref_point):
                        # If any objective is worse than ref, HV contribution is 0?
                        # Actually, if f_new is not better than ref in all objectives,
                        # it might still improve HV if it dominates some archive points?
                        # No, HV only counts points strictly better than ref.
                        # If f_new is not strictly better than ref, it doesn't add to the "valid" set.
                        # But it might dominate archive points that are valid?
                        # If f_new is invalid, it cannot dominate a valid point (which is < ref).
                        # So we can skip.
                        continue
                        
                    # Add f_new to archive
                    # We need to check if f_new is dominated by archive.
                    # If yes, HV doesn't change.
                    dominated = False
                    for f_a in archive_f:
                        if np.all(f_a <= f_new):
                            dominated = True
                            break
                    if dominated:
                        continue
                    
                    # Calculate new HV
                    # We can just calculate HV of the union.
                    # To be fast, we can update the archive and calculate HV.
                    # But we are in a loop, so we should avoid modifying the actual archive.
                    # Let's just calculate HV of (archive_f + [f_new]).
                    
                    new_archive = np.vstack([archive_f, [f_new]])
                    hv_new = hv_3d_simple(new_archive, ref_point)
                    ehvi_vals[m] += (hv_new - hv_archive)
                
                ehvi_vals[m] /= n_samples
            
            return ehvi_vals

        # Grid for acquisition
        # We will use a random grid of candidates.
        # Size: 100 candidates.
        
        while used < budget:
            if time.time() - start_time > max_time:
                break
            
            if len(archive_x) < 2:
                # Random
                x = lo + np.random.random(dim) * (hi - lo)
            else:
                # Fit GP
                X_archive = np.array(archive_x)
                F_archive = np.array(archive_f)
                
                gp = SimpleGP(dim, n_obj)
                gp.fit(X_archive, F_archive)
                
                # Generate candidates
                n_cands = 100
                X_cands = lo + np.random.random((n_cands, dim)) * (hi - lo)
                
                # Calculate EHVI
                ehvi_vals = ehvi_acquisition(X_cands, F_archive, gp, ref_point, n_samples=50)
                
                # Select best
                best_idx = np.argmax(ehvi_vals)
                x = X_cands[best_idx]
                
            f = problem(x)
            used += 1
            update_archive(x, f)
            
        return archive_x
