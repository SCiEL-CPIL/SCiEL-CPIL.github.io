import numpy as np
import time
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        max_time = 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Reference point for scoring
        ref_point = np.array([1.1, 1.3, 1.1])
        
        used = 0
        X_archive = []
        F_archive = []
        
        def is_dominated(f_new, f_old):
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        def update_archive(x, f):
            nonlocal X_archive, F_archive
            # Check if new point dominates existing ones
            to_remove = []
            for i in range(len(F_archive)):
                if is_dominated(f, F_archive[i]):
                    to_remove.append(i)
            
            if to_remove:
                for i in reversed(to_remove):
                    X_archive.pop(i)
                    F_archive.pop(i)
                
                # Check if new point is dominated by any remaining
                for i in range(len(F_archive)):
                    if is_dominated(F_archive[i], f):
                        return False
            
            X_archive.append(x)
            F_archive.append(f)
            return True

        def calc_hypervolume(F, ref):
            if len(F) == 0:
                return 0.0
            F = np.array(F)
            # Filter points strictly better than ref
            mask = np.all(F < ref, axis=1)
            F = F[mask]
            if len(F) == 0:
                return 0.0
            
            # Sort by first objective
            F = F[np.argsort(F[:, 0])]
            
            # Recursive HV calculation for 3D
            def hv_recursive(points, ref):
                if len(points) == 0:
                    return 0.0
                if len(points) == 1:
                    return np.prod(ref - points[0])
                
                # Sort by first objective
                points = points[np.argsort(points[:, 0])]
                
                # Hypervolume is sum of slices
                hv = 0.0
                prev_point = points[0]
                for i in range(1, len(points)):
                    curr_point = points[i]
                    # Volume of the box from prev_point to curr_point in first dim, and ref to curr_point in other dims
                    # Wait, standard decomposition:
                    # Slice between prev_point and curr_point (in first obj)
                    # The "height" in other dims is determined by the current point?
                    # Actually, simpler: 
                    # HV = sum over i of (F[i,0] - F[i-1,0]) * HV_2D(F[i:][1:], ref[1:])
                    # But F is sorted ascending.
                    # Let's use the standard recursive definition:
                    # HV(S, ref) = (ref[0] - S[0][0]) * HV(S[1:], ref[1:]) + (S[0][0] - S[1][0]) * HV(S[1:], ref[1:])? No.
                    
                    # Correct recursive step for 3D (sorted by x):
                    # HV = (ref[0] - x_0) * HV_2D({(y_i, z_i)}, (ref_y, ref_z)) 
                    #      + sum_{i=1}^{N-1} (x_i - x_{i-1}) * HV_2D({(y_j, z_j) for j >= i}, (ref_y, ref_z))
                    
                    # This is O(N^2) if we recalculate HV_2D each time.
                    # For N < 100, it's fine.
                    
                    pass
                
                # Let's implement a simpler O(N^3) or O(N^2 log N) for 3D
                # Since N is small (< 100), we can use the inclusion-exclusion or slicing method.
                
                # Slicing method:
                # Sort by x.
                # For each point i, the volume it contributes is:
                # (x_{i+1} - x_i) * (y_{min(i)} - ref_y) * (z_{min(i)} - ref_z) ? No.
                
                # Let's use the standard algorithm for 3D:
                # 1. Sort by x.
                # 2. For each point i, find the point j > i that minimizes y.
                #    Actually, we can project to 2D.
                
                # Easier: Use the "union of boxes" logic via inclusion-exclusion is hard.
                # Let's use the recursive slice:
                # HV(S, ref) where S is sorted by x.
                # HV = (ref[0] - S[0][0]) * HV_2D(S, ref[1:]) 
                #      + sum_{i=1}^{N-1} (S[i][0] - S[i-1][0]) * HV_2D(S[i:], ref[1:])
                
                def hv_2d(points_2d, ref_2d):
                    if len(points_2d) == 0:
                        return 0.0
                    points_2d = points_2d[np.argsort(points_2d[:, 0])]
                    hv = 0.0
                    prev_y = ref_2d[1]
                    for p in points_2d:
                        if p[1] < prev_y:
                            hv += (ref_2d[0] - p[0]) * (prev_y - p[1])
                            prev_y = p[1]
                    return hv

                hv = 0.0
                # First slice: from S[0][0] to ref[0]
                # The 2D points involved are all points in S (since x is the smallest)
                hv += (ref[0] - points[0][0]) * hv_2d(points[:, 1:], ref[1:])
                
                # Subsequent slices
                for i in range(1, len(points)):
                    # Slice from points[i-1][0] to points[i][0]
                    # The 2D points involved are points[i:] (those with x >= points[i][0])
                    # Wait, the slice is between x_{i-1} and x_i.
                    # The height in y,z is determined by the points with index >= i.
                    # Actually, the standard formula is:
                    # HV = sum_{i=0}^{N-1} (x_{i+1} - x_i) * HV_2D(S[i:], ref[1:])
                    # where x_N = ref[0]
                    # Let's re-verify.
                    # If we have points sorted by x.
                    # The volume to the right of x_i is determined by the "lower envelope" of points i..N-1.
                    # Yes.
                    
                    # So:
                    # Term i: (x_{i+1} - x_i) * HV_2D(points[i:], ref[1:])
                    # For i=0: x_1 - x_0
                    # ...
                    # For i=N-1: ref[0] - x_{N-1}
                    
                    pass
                
                # Let's just loop i from 0 to N-1
                # x_next = points[i+1][0] if i < N-1 else ref[0]
                for i in range(len(points)):
                    x_next = points[i+1][0] if i < len(points) - 1 else ref[0]
                    width = x_next - points[i][0]
                    if width <= 0:
                        continue
                    hv += width * hv_2d(points[i:, 1:], ref[1:])
                
                return hv

            return hv_recursive(F, ref)

        # 1. Initialization
        init_size = min(30, budget // 5)
        X_init = lo + np.random.random((init_size, dim)) * (hi - lo)
        
        for x in X_init:
            f = problem(x)
            used += 1
            update_archive(x, f)
            
            if time.time() - start_time > max_time:
                break
            if used >= budget:
                break

        if used >= budget or time.time() - start_time > max_time:
            return X_archive

        # 2. Surrogate Model Based Search
        # We will fit GPs to the archive and use EHVI to select next point.
        # To save time, we will only do this for the remaining budget.
        # Grid for acquisition function: 20x20x20x20 = 160k points.
        # GP prediction on 160k points might be slow if done naively.
        # We can subsample the grid or use a smaller grid.
        # Let's try a 10x10x10x10 = 10k grid.
        
        grid_res = 10
        grid_points = []
        for d in range(dim):
            grid_points.append(np.linspace(lo[d], hi[d], grid_res))
        
        # Create meshgrid
        G = np.meshgrid(*grid_points, indexing='ij')
        grid = np.vstack([g.flatten() for g in G])
        
        # EHVI Acquisition
        # We need to compute HV of (Archive U {predicted_f})
        # Approximation: Use mean prediction for HV calculation.
        # This is "Expected HV" if we ignore uncertainty, or we can use a simple uncertainty bonus.
        # Given time constraints, let's use Mean Prediction + Uncertainty Bonus (like EI but for HV).
        
        while used < budget:
            if time.time() - start_time > max_time:
                break
            
            if len(F_archive) < 2:
                # Need more data
                x = lo + np.random.random(dim) * (hi - lo)
                f = problem(x)
                used += 1
                update_archive(x, f)
                continue
            
            X_arch = np.array(X_archive)
            F_arch = np.array(F_archive)
            
            # Fit GPs
            # Simple GP: y = K(x, x') alpha
            # To avoid heavy computation, we use a simplified kernel or just RBF with fixed lengthscale.
            # Since dim=4, and N<50, it's manageable.
            
            # Precompute kernel matrix for training data
            # K[i,j] = exp(-0.5 * ||x_i - x_j||^2 / l^2)
            # Let's use l=0.3 (normalized space)
            l = 0.3
            diff = X_arch[:, np.newaxis, :] - X_arch[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            K = np.exp(-sq_dist / (2 * l**2))
            K += 1e-6 * np.eye(len(X_arch))
            
            # Solve for alpha: K alpha = y
            try:
                alpha = np.linalg.solve(K, F_arch)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, F_arch, rcond=None)[0]
            
            # Predict on grid
            # K_grid_train: (M, N)
            diff_grid = grid[:, np.newaxis, :] - X_arch[np.newaxis, :, :]
            sq_dist_grid = np.sum(diff_grid**2, axis=2)
            K_grid = np.exp(-sq_dist_grid / (2 * l**2))
            
            # Mean prediction
            F_pred = K_grid @ alpha
            
            # Variance prediction
            # v(x) = 1 - k(x)^T K^{-1} k(x)
            # We don't have K^{-1}, but we have alpha = K^{-1} y.
            # This doesn't give us K^{-1} directly.
            # We can compute v(x) = 1 - k(x)^T (K^{-1} k(x))
            # Let's solve K v_vec = k(x) for each x? That's expensive.
            # Alternatively, use Cholesky decomposition of K.
            
            # Let's re-do with Cholesky for efficiency and stability
            try:
                L = np.linalg.cholesky(K)
                # Solve L w = y
                w = np.linalg.solve(L, F_arch)
                # Solve L^T alpha = w
                alpha = np.linalg.solve(L.T, w)
                
                # Predictions
                # L_grid = cholesky(K_grid)? No, K_grid is MxN.
                # We need K_grid @ alpha
                F_pred = K_grid @ alpha
                
                # Variance
                # v = 1 - sum_j (L^{-1} k_j)^2 ?
                # Let V = L^{-1} K_grid.T  (N x M)
                # Then v = 1 - sum(V^2, axis=0)
                
                # Solve L V^T = K_grid -> V^T = L^{-1} K_grid
                # V^T is N x M
                V_T = np.linalg.solve(L, K_grid.T)
                v = 1.0 - np.sum(V_T**2, axis=0)
                v = np.clip(v, 0, 1)
                
            except np.linalg.LinAlgError:
                # Fallback to random if GP fails
                x = lo + np.random.random(dim) * (hi - lo)
                f = problem(x)
                used += 1
                update_archive(x, f)
                continue
            
            # Calculate HV for each candidate
            # HV(Archive U {f_i})
            # Optimization: Only points that might be non-dominated are interesting.
            # But let's just compute HV for all to be safe, as M=10k is small.
            
            # We can speed this up by only considering candidates that are not dominated by archive.
            # But for simplicity and robustness, we compute HV for a subset or all.
            # 10k candidates * HV calc (N~30) is ~300k ops. Fast.
            
            HV_scores = np.zeros(len(grid))
            for i in range(len(grid)):
                # Create new front
                # Filter dominated points from archive?
                # Just union and calculate.
                # To speed up HV, we can use the fact that if f_i is dominated by archive, HV doesn't change.
                # Check dominance
                # If any archive point dominates f_i, HV is same as archive.
                # If f_i dominates some archive points, HV increases.
                
                # Simple check:
                # If f_i is dominated by any point in F_arch, skip (score = HV_arch)
                # Actually, we want to maximize improvement.
                
                # Let's just compute HV for the union.
                # To avoid overhead of list creation, we can use numpy.
                
                # Optimization: 
                # 1. Check if f_i is dominated by archive. If so, HV = HV_arch.
                # 2. Else, compute HV of (F_arch U {f_i}).
                
                # For now, just compute.
                # We can cache HV_arch
                if not hasattr(self, '_hv_arch_cache'):
                    self._hv_arch_cache = -1
                if self._hv_arch_cache == -1:
                    self._hv_arch_cache = calc_hypervolume(F_arch, ref_point)
                
                # Check if dominated
                dominated = False
                for f_a in F_arch:
                    if np.all(f_a <= F_pred[i]) and np.any(f_a < F_pred[i]):
                        dominated = True
                        break
                
                if dominated:
                    HV_scores[i] = self._hv_arch_cache
                else:
                    # Compute new HV
                    # Combine arrays
                    new_F = np.vstack([F_arch, F_pred[i].reshape(1, -1)])
                    HV_scores[i] = calc_hypervolume(new_F, ref_point)
            
            # Add exploration bonus
            # bonus = sigma * sqrt(v)
            # Let's use a simple bonus: 0.05 * sqrt(v)
            bonus = 0.05 * np.sqrt(v)
            scores = HV_scores + bonus
            
            # Select best
            best_idx = np.argmax(scores)
            x_next = grid[best_idx]
            
            f_next = problem(x_next)
            used += 1
            update_archive(x_next, f_next)
            
            # Invalidate cache
            self._hv_arch_cache = -1

        return X_archive
