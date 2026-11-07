"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_3d(self, points, ref):
        """
        Compute hypervolume for a set of points in 3D using inclusion-exclusion.
        points: (N, 3) array, ref: (3,) array
        Returns scalar hypervolume.
        """
        # Filter points that are strictly better than reference
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        if len(pts) == 0:
            return 0.0
        
        # Sort by first objective
        pts = pts[np.argsort(pts[:, 0])]
        
        # Use a simple O(n^2) approach for 3D HV
        # For each point, compute its contribution
        hv = 0.0
        n = len(pts)
        for i in range(n):
            p = pts[i]
            # Volume dominated by p but not by any other point
            # This is complex; use a simpler approximation
            # Contribution: box from p to ref, minus overlaps
            # For simplicity, use the standard 3D HV algorithm
            pass
        
        # Better: use the standard O(n^2 log n) 3D HV algorithm
        # Sort by x, then for each point compute the 2D HV of its "shadow"
        sorted_idx = np.argsort(pts[:, 0])
        pts_sorted = pts[sorted_idx]
        
        hv = 0.0
        for i in range(n):
            xi, yi, zi = pts_sorted[i]
            # Find all points j with x_j > x_i
            # Their y,z define a 2D region that subtracts from i's contribution
            # Contribution of i: (ref[0]-xi) * (ref[1]-yi) * (ref[2]-zi)
            # minus overlaps with points that dominate in at least one dimension
            
            # Simpler: use the fact that we can compute incrementally
            # For each point, its "private" volume is the box [xi, ref[0]] x [yi, ref[1]] x [zi, ref[2]]
            # minus boxes from points that are "closer" in some dimensions
            
            # This is getting complex. Let's use a different approach.
            break
        
        # Use a practical approach: compute HV by sorting and using sweep line
        # For 3D, we can sort by first coord and maintain a 2D structure
        # But for simplicity and correctness, let's use a known algorithm
        
        # Actually, for the selection in MO-CMA-ES, we don't need exact HV
        # We can use a simpler metric: sum of inverse ranks or similar
        # But let's implement a correct but simple HV
        
        # Filter valid points
        valid = np.all(pts < ref, axis=1)
        pts = pts[valid]
        if len(pts) == 0:
            return 0.0
        
        n = len(pts)
        # Sort by first objective
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        hv = 0.0
        # For each point, compute the volume it uniquely contributes
        # A point contributes if no other point dominates it in all 3 objectives
        # This is non-dominated front contribution
        
        # Simpler correct approach: 
        # HV = sum over all subsets... no, that's exponential
        # Use the standard algorithm: sort by x, then for each point, 
        # compute the 2D HV of the "skyline" in yz from that point onwards
        
        # Let's just use a simple but correct O(n^2) method
        # For each point p_i, its contribution is the volume of the box
        # [p_i, ref] that is not covered by any other point
        # This is hard to compute directly
        
        # Alternative: use the fact that HV can be computed by:
        # Sort points, then use inclusion-exclusion on the sorted order
        # For 3D: HV = sum_i (ref_0 - x_i) * HV_2D({(y_j, z_j) : j >= i, y_j < y_i or z_j < z_i...})
        # This is still complex
        
        # For MO-CMA-ES selection, we can use a simpler fitness:
        # For each point, compute a "quality" score based on how well it covers the front
        # Use: for each point, the minimum distance to reference in each objective
        # Or use: sum of (ref_k - f_k) for k=1..3, weighted
        
        # Actually, let's just implement a correct HV using a known method
        # Use the "incremental" approach: sort by x, maintain 2D skyline
        
        # For now, use a simple but potentially suboptimal metric for selection:
        # Sum of (ref - f) across objectives, but this doesn't capture coverage well
        
        # Let's use a proper HV computation
        # Reference: https://en.wikipedia.org/wiki/Hypervolume_indicator
        
        # For 3D, O(n^2 log n) algorithm:
        # 1. Sort by x
        # 2. For i from 0 to n-1:
        #    a. Compute 2D HV of points {j >= i} in yz-plane, with reference (ref_y, ref_z)
        #       but only considering points with y < ref_y and z < ref_z
        #    b. Contribution of point i: (ref_x - x_i) * (HV_2D - HV_2D_without_i)
        # This is still complex to implement correctly
        
        # For practical purposes in MO-CMA-ES, many implementations use:
        # - Non-dominated sorting + crowding (like NSGA-II)
        # - Or a simple HV approximation
        
        # Let's use a simple but effective selection: 
        # Keep the non-dominated front, and use crowding distance for truncation
        # This is what NSGA-II does, and it works well
        
        return 0.0  # Placeholder; we'll use non-dominated sorting for selection

    def _non_dominated_sort(self, F):
        """
        Perform non-dominated sorting on F (n x m array).
        Returns a list of lists of indices, where each inner list is a front.
        """
        n = len(F)
        S = [[] for _ in range(n)]
        n_dom = np.zeros(n, dtype=int)
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dom[i] += 1
        
        current_front = [i for i in range(n) if n_dom[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        return fronts

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for a set of indices in F.
        Returns a dictionary {index: distance}.
        """
        if len(indices) <= 2:
            return {i: float('inf') for i in indices}
        
        distances = {i: 0.0 for i in indices}
        m = self.n_obj
        for k in range(m):
            sorted_indices = sorted(indices, key=lambda i: F[i, k])
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            f_min = F[sorted_indices[0], k]
            f_max = F[sorted_indices[-1], k]
            if f_max - f_min == 0:
                continue
                
            for j in range(1, len(sorted_indices) - 1):
                i_prev = sorted_indices[j - 1]
                i_next = sorted_indices[j + 1]
                distances[sorted_indices[j]] += (F[i_next, k] - F[i_prev, k]) / (f_max - f_min)
        
        return distances

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3.0, 5.0, 7.0])
        
        # MO-CMA-ES parameters
        lambda_ = 24  # Population size
        mu = 12       # Number of parents
        dim = self.dim
        
        # Initialize CMA-ES
        # Start from center of bounds
        mean = (lo + hi) / 2.0
        sigma = 0.25 * np.mean(hi - lo)  # Initial step size
        C = np.eye(dim)  # Covariance matrix
        C_inv = np.eye(dim)
        
        # Precompute eigendecomposition for sampling
        eigvals, eigvecs = np.linalg.eigh(C)
        
        # Weights for recombination
        weights = np.zeros(mu)
        weights[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff + c_sigma) / (dim + 4 - mu_eff)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Archive for non-dominated points
        archive_X = []
        archive_F = []
        evaluations = 0
        
        # Initial population
        X = np.zeros((lambda_, dim))
        for i in range(lambda_):
            z = np.random.standard_normal(dim)
            X[i] = mean + sigma * (eigvecs @ (eigvals * z))
            X[i] = np.clip(X[i], lo, hi)
        
        F = np.array([problem(x) for x in X])
        evaluations += lambda_
        
        # Add initial points to archive
        for i in range(lambda_):
            archive_X.append(X[i].copy())
            archive_F.append(F[i].copy())
        
        # Main loop
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            # Sort by objective sum (simple proxy for ranking)
            # In proper MO-CMA-ES, we'd use HV-based selection
            # For simplicity, use non-dominated sorting + crowding
            fronts = self._non_dominated_sort(F)
            
            # Select mu parents from the fronts
            parent_indices = []
            for front in fronts:
                if len(parent_indices) + len(front) <= mu:
                    parent_indices.extend(front)
                else:
                    need = mu - len(parent_indices)
                    cd = self._crowding_distance(F, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    parent_indices.extend(sorted_front[:need])
                    break
            
            if len(parent_indices) < mu:
                # Fill with random if not enough
                while len(parent_indices) < mu:
                    parent_indices.append(np.random.randint(lambda_))
            
            parent_indices = np.array(parent_indices[:mu])
            
            # Compute new mean
            new_mean = np.sum(weights[:, None] * X[parent_indices], axis=0)
            
            # Update evolution paths
            C_sqrt_inv = np.linalg.cholesky(C_inv)
            z_bar = (new_mean - mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv.T @ z_bar
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) / chi_n) < (1.4 + 2/(dim+1))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * z_bar
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                z_i = (X[parent_indices[i]] - mean) / sigma
                rank_mu += weights[i] * np.outer(z_i, z_i)
            
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure C is positive definite
            C = (C + C.T) / 2
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            if np.any(eigvals_C < 1e-12):
                eigvals_C = np.maximum(eigvals_C, 1e-12)
                C = eigvecs_C @ np.diag(eigvals_C) @ eigvecs_C.T
            
            C_inv = np.linalg.inv(C)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10 * np.mean(hi - lo))
            
            mean = new_mean
            
            # Generate new population
            X_new = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = np.random.standard_normal(dim)
                X_new[i] = mean + sigma * (eigvecs @ (eigvals * z))
                X_new[i] = np.clip(X_new[i], lo, hi)
            
            F_new = np.array([problem(x) for x in X_new])
            evaluations += lambda_
            
            # Update archive with new non-dominated points
            for i in range(lambda_):
                # Check if new point is non-dominated with respect to archive
                is_dominated = False
                for j in range(len(archive_F)):
                    if np.all(archive_F[j] <= F_new[i]) and np.any(archive_F[j] < F_new[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    # Check if it dominates any archive points
                    for j in range(len(archive_F) - 1, -1, -1):
                        if np.all(F_new[i] <= archive_F[j]) and np.any(F_new[i] < archive_F[j]):
                            archive_X.pop(j)
                            archive_F.pop(j)
                    archive_X.append(X_new[i].copy())
                    archive_F.append(F_new[i].copy())
            
            # Limit archive size
            if len(archive_X) > 200:
                # Keep the best 200 based on a simple criterion
                archive_F_arr = np.array(archive_F)
                # Use sum of objectives as a simple criterion
                sums = np.sum(archive_F_arr, axis=1)
                keep = np.argsort(sums)[:200]
                archive_X = [archive_X[i] for i in keep]
                archive_F = [archive_F[i] for i in keep]
            
            X = X_new
            F = F_new
        
        # Return the archive (or a subset)
        if len(archive_X) > 0:
            return np.array(archive_X)
        else:
            return X
# EVOLVE-BLOCK-END
