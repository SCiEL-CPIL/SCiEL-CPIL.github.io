"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Construct 300 equal circle centers inside the unit disk to maximize the minimum
    common radius. Uses a vectorized soft-min gradient ascent with local refinement.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    target_r = 0.0531542991469

    def _hex_init(n):
        """Generate a hexagonal lattice packing of n points in the unit disk."""
        s = np.sqrt(2 * np.pi / (n * np.sqrt(3))) * 0.95
        points = []
        max_coord = int(1.0 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if x * x + y * y <= (1.0 - s / 2) ** 2:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < n:
            s *= 0.9
            points = []
            max_coord = int(1.0 / s) + 2
            for j in range(-max_coord, max_coord + 1):
                for i in range(-max_coord, max_coord + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3) / 2
                    if x * x + y * y <= (1.0 - s / 2) ** 2:
                        points.append([x, y])
            points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            return points[idx]
        else:
            angles = np.random.uniform(0, 2 * np.pi, n)
            radii = np.sqrt(np.random.uniform(0, 1, n))
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    def _compute_radius(pts):
        """Compute the exact minimum common radius."""
        n = len(pts)
        dist_boundary = 1.0 - np.linalg.norm(pts, axis=1)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        return min(min_pair_dist / 2.0, np.min(dist_boundary))

    def _soft_min_logsumexp(values, beta):
        """
        Soft-min of values using log-sum-exp.
        For large beta, this approximates min(values).
        We use: -log(sum(exp(-beta*v))) / beta ≈ min(v)
        """
        v = -beta * values
        v_max = np.max(v)
        # Clip to avoid overflow
        v_shifted = v - v_max
        log_sum = np.log(np.sum(np.exp(v_shifted))) + v_max
        return -log_sum / beta

    def _objective_and_grad(pts, beta):
        """
        Compute the soft-min objective and its gradient.
        Objective: soft-min of {d_ij/2 for all i<j, 1-|c_i| for all i}
        We maximize this, so we do gradient ascent.
        """
        n = len(pts)
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2) + 1e-30)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        norms = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - norms
        
        # All "constraints" values
        # Pairwise: d_ij / 2
        # Boundary: 1 - |c_i|
        pair_vals = dists / 2.0
        # Create full list
        all_vals = np.concatenate([np.minimum(pair_vals, pair_vals.T).ravel(), boundary_dists.ravel()])
        
        # Soft-min
        obj = _soft_min_logsumexp(all_vals, beta)
        
        # Gradient computation
        # d/dx_j of soft-min ≈ weighted average of gradients
        # For pairwise dist d_ij: grad wrt c_i = (c_i - c_j)/d_ij, wrt c_j = (c_j - c_i)/d_ij
        # For boundary: grad wrt c_i = -c_i/|c_i|
        
        # Compute weights: w_k = exp(-beta * v_k) / sum(exp(-beta * v_k))
        neg_beta_vals = -beta * all_vals
        neg_beta_vals_max = np.max(neg_beta_vals)
        exp_vals = np.exp(neg_beta_vals - neg_beta_vals_max)
        weights = exp_vals / np.sum(exp_vals)
        
        grad = np.zeros_like(pts)
        
        # Pairwise part: indices 0 to n*(n-1)
        n_pairs = n * (n - 1)
        # The pair_vals matrix is n x n with inf on diagonal
        # We need to map flat index to (i, j)
        # pair_vals.ravel() gives row-major: (0,0), (0,1), ..., (0,n-1), (1,0), ...
        # But we only want i < j pairs. Let's use the upper triangle.
        
        # Actually, let me restructure: use only i<j pairs
        # Number of i<j pairs: n*(n-1)/2
        n_pairs_upper = n * (n - 1) // 2
        pair_flat = np.zeros(n_pairs_upper)
        for k in range(n_pairs_upper):
            i = k // (n - 1)
            j = k % (n - 1)
            if j >= i:
                j += 1
            pair_flat[k] = dists[i, j] / 2.0
        
        # Boundary part
        # All vals: [pair_vals (n_pairs_upper), boundary_dists (n)]
        total_vals = np.concatenate([pair_flat, boundary_dists])
        neg_beta_total = -beta * total_vals
        neg_beta_total_max = np.max(neg_beta_total)
        exp_total = np.exp(neg_beta_total - neg_beta_total_max)
        total_weights = exp_total / np.sum(exp_total)
        
        obj = _soft_min_logsumexp(total_vals, beta)
        
        pair_weights = total_weights[:n_pairs_upper]
        boundary_weights = total_weights[n_pairs_upper:]
        
        # Pairwise gradients
        for k in range(n_pairs_upper):
            i = k // (n - 1)
            j = k % (n - 1)
            if j >= i:
                j += 1
            w = pair_weights[k]
            d = dists[i, j]
            if d > 1e-10:
                direction = (pts[i] - pts[j]) / d
                grad[i] += w * 0.5 * direction  # d/2 gradient: 0.5 * (c_i-c_j)/d
                grad[j] -= w * 0.5 * direction
        
        # Boundary gradients
        for i in range(n):
            w = boundary_weights[i]
            norm_i = norms[i]
            if norm_i > 1e-10:
                grad[i] -= w * pts[i] / norm_i
        
        return obj, grad

    def _local_refine(pts, time_budget, beta_start=50.0, beta_end=500.0):
        """
        Gradient ascent on the soft-min objective with adaptive beta.
        """
        n = len(pts)
        best_pts = pts.copy()
        best_score = _compute_radius(pts)
        current_pts = pts.copy()
        
        # Phase 1: moderate beta gradient ascent
        beta = beta_start
        step_size = 0.005
        no_improve_count = 0
        
        while time.time() < time_budget:
            obj, grad = _objective_and_grad(current_pts, beta)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm
            
            # Take step
            new_pts = current_pts + step_size * grad
            
            # Project back to unit disk
            new_norms = np.linalg.norm(new_pts, axis=1)
            mask = new_norms > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / new_norms[mask] * 0.999
            
            new_score = _compute_radius(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                no_improve_count = 0
                # Increase beta slowly
                beta = min(beta * 1.05, beta_end)
                step_size = min(step_size * 1.02, 0.01)
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    # Reduce step and reset
                    step_size *= 0.7
                    beta = max(beta * 0.9, beta_start)
                    no_improve_count = 0
                    if step_size < 0.0005:
                        break
                current_pts = new_pts
        
        # Phase 2: fine-tune with higher beta
        if time.time() < time_budget - 5:
            beta = beta_end
            step_size = 0.001
            no_improve_count = 0
            current_pts = best_pts.copy()
            
            while time.time() < time_budget - 2:
                obj, grad = _objective_and_grad(current_pts, beta)
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-10:
                    break
                if grad_norm > 1e-12:
                    grad = grad / grad_norm
                
                new_pts = current_pts + step_size * grad
                new_norms = np.linalg.norm(new_pts, axis=1)
                mask = new_norms > 1.0
                if np.any(mask):
                    new_pts[mask] = new_pts[mask] / new_norms[mask] * 0.999
                
                new_score = _compute_radius(new_pts)
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > 100:
                        step_size *= 0.8
                        no_improve_count = 0
                        if step_size < 0.0001:
                            break
                    current_pts = new_pts
        
        return best_pts, best_score

    def _tighten_min_constraint(pts, time_budget):
        """
        Final polish: iteratively push the most constrained points apart.
        """
        n = len(pts)
        best_pts = pts.copy()
        best_score = _compute_radius(pts)
        
        while time.time() < time_budget:
            score = _compute_radius(best_pts)
            
            # Find the binding constraint
            dist_boundary = 1.0 - np.linalg.norm(best_pts, axis=1)
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2) + 1e-30)
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            min_boundary = np.min(dist_boundary)
            
            if min_pair_dist / 2.0 < min_boundary:
                # Pairwise constraint is binding
                i, j = np.unravel_index(np.argmin(dists), dists.shape)
                d = dists[i, j]
                if d < 1e-10:
                    continue
                direction = (best_pts[i] - best_pts[j]) / d
                push = 0.001
                best_pts[i] += direction * push
                best_pts[j] -= direction * push
            else:
                # Boundary constraint is binding
                i = np.argmin(dist_boundary)
                norm_i = np.linalg.norm(best_pts[i])
                if norm_i < 1e-10:
                    continue
                direction = -best_pts[i] / norm_i
                best_pts[i] += direction * 0.001
            
            # Project to disk
            norms = np.linalg.norm(best_pts, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                best_pts[mask] = best_pts[mask] / norms[mask] * 0.9995
            
            new_score = _compute_radius(best_pts)
            if new_score > best_score:
                best_score = new_score
            else:
                # Revert if it got worse
                pass  # We already modified in place, so this is tricky
                # Let's just continue
        
        return best_pts, best_score

    # Main: multi-start with gradient refinement
    best_pts = None
    best_score = 0.0
    
    # Start with hex init
    pts = _hex_init(N)
    norms = np.linalg.norm(pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / norms[mask] * 0.999
    
    # First refinement pass
    time_budget1 = deadline - 40
    pts, score = _local_refine(pts, time_budget1, beta_start=30.0, beta_end=300.0)
    best_pts = pts
    best_score = score
    
    # Second pass with perturbation
    if time.time() < deadline - 30 and best_score < target_r:
        pts2 = best_pts + np.random.normal(0, 0.008, size=(N, 2))
        norms2 = np.linalg.norm(pts2, axis=1)
        mask2 = norms2 > 1.0
        if np.any(mask2):
            pts2[mask2] = pts2[mask2] / norms2[mask2] * 0.999
        
        time_budget2 = deadline - 20
        pts2, score2 = _local_refine(pts2, time_budget2, beta_start=50.0, beta_end=500.0)
        if score2 > best_score:
            best_pts = pts2
            best_score = score2
    
    # Third pass: fine polish
    if time.time() < deadline - 10 and best_score < target_r:
        time_budget3 = deadline - 5
        best_pts, best_score = _tighten_min_constraint(best_pts, time_budget3)
    
    # Final validation
    if best_pts is None:
        best_pts = _hex_init(N)
    
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / norms[mask] * 0.99
    
    return best_pts
# EVOLVE-BLOCK-END
