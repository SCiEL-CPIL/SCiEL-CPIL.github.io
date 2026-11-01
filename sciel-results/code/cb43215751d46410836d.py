"""Rewritten packing solver using binary search + projected gradient descent."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 50

def construct():
    start_time = time.time()
    # Safety margin: return by 140s (10s buffer before 150s kill)
    deadline = start_time + 140.0
    
    def _score(pts):
        """Return the achievable common radius for centre array (N,2)."""
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return -np.inf
        # Boundary
        d_bound = np.min(0.5 - np.abs(pts), axis=1)
        r = float(np.min(d_bound))
        # Pairwise
        if N > 1:
            tree = cKDTree(pts)
            d, _ = tree.query(pts, k=2)
            min_pair = float(np.min(d[:, 1])) / 2.0
            r = min(r, min_pair)
        return r

    def _soft_min_score(pts, alpha=50.0):
        """
        Differentiable proxy: penalize all violations softly.
        Lower is better (we minimize negative of this, or maximize this).
        We return the negative so we can minimize.
        """
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return 1e10
        # Boundary: want pts >= -0.5 and pts <= 0.5
        # Violation: max(0, -0.5 - pts) + max(0, pts - 0.5)
        lower_viol = np.maximum(0.0, -0.5 - pts)
        upper_viol = np.maximum(0.0, pts - 0.5)
        bound_viol = np.sum(lower_viol**2 + upper_viol**2)
        
        # Pairwise: want ||p_i - p_j|| >= 2r
        # We approximate using a soft min over pairwise distances
        # Use a smaller alpha for pairwise to keep gradients reasonable
        if N > 1:
            tree = cKDTree(pts)
            # Only compute soft min for closest few neighbors to save time?
            # For N=50, full pairwise is 1225 pairs, which is fine.
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            # Soft min: -1/alpha * log(sum(exp(-alpha * dist)))
            # We want to maximize the minimum distance.
            # So we minimize -log-sum-exp(-alpha * dist)
            # But log-sum-exp is expensive. Let's use a simpler soft min:
            # soft_min = (1/alpha) * (-log(sum(exp(-alpha * dist))))
            # Actually, for optimization, we can just use the standard soft min formula.
            # However, computing exp for all pairs might be slow or overflow.
            # Let's use a trick: subtract max to avoid overflow.
            min_d = np.min(dists)
            exp_vals = np.exp(-alpha * (dists - min_d))
            exp_vals[np.diag_indices_from(exp_vals)] = 0.0
            soft_min = min_d - np.log(np.sum(exp_vals)) / alpha
            pair_viol = -soft_min  # We want to maximize soft_min, so minimize -soft_min
        else:
            pair_viol = 0.0
            
        return bound_viol + pair_viol

    def _soft_min_grad(pts, alpha=50.0):
        """Gradient of _soft_min_score."""
        if pts.shape != (N, 2):
            return np.zeros_like(pts)
        
        grad = np.zeros_like(pts)
        
        # Boundary gradient
        # d/dp max(0, -0.5 - p)^2 + max(0, p - 0.5)^2
        # If p < -0.5: d/dp (-0.5 - p)^2 = -2(-0.5 - p) = 2(0.5 + p)
        # If p > 0.5: d/dp (p - 0.5)^2 = 2(p - 0.5)
        lower_viol = np.maximum(0.0, -0.5 - pts)
        upper_viol = np.maximum(0.0, pts - 0.5)
        grad -= 2 * lower_viol
        grad += 2 * upper_viol
        
        # Pairwise gradient
        if N > 1:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            min_d = np.min(dists)
            exp_vals = np.exp(-alpha * (dists - min_d))
            exp_vals[np.diag_indices_from(exp_vals)] = 0.0
            sum_exp = np.sum(exp_vals)
            
            # Gradient of -log(sum(exp(-alpha*(d - min_d)))) w.r.t p_i
            # Let S = sum(exp(-alpha*(d_j - min_d)))
            # d/dp_i (-log S) = - (1/S) * sum_j (d/dp_i exp(-alpha*(d_j - min_d)))
            # d_j = ||p_i - p_j||
            # d/dp_i d_j = (p_i - p_j) / d_j
            # d/dp_i exp(-alpha*(d_j - min_d)) = exp(-alpha*(d_j - min_d)) * (-alpha) * (p_i - p_j) / d_j
            # So term is: - (1/S) * sum_j exp(...) * (-alpha) * (p_i - p_j) / d_j
            #           = (alpha/S) * sum_j exp(...) * (p_i - p_j) / d_j
            
            # We need to compute this sum for all i.
            # Note: min_d is constant w.r.t p_i in the gradient calculation?
            # Actually, min_d changes with p_i. This makes the gradient of the soft min complex.
            # A simpler approximation: ignore the derivative of min_d.
            # Or use a different soft min that is easier to differentiate.
            
            # Alternative: use L2 regularization on distances below a threshold.
            # Let's stick to a simpler heuristic for now: just push apart close points.
            # If dist < threshold, add force.
            
            # Let's re-implement a simpler differentiable objective:
            # Objective = sum( max(0, threshold - dist)^2 )
            # This is simpler and differentiable (piecewise quadratic).
            # But it requires knowing a threshold.
            
            # Let's go back to the log-sum-exp but handle the gradient carefully.
            # Actually, for N=50, we can just compute the gradient numerically or use a simpler method.
            # Given time constraints, let's use a simpler gradient:
            # Force between each pair proportional to overlap.
            
            # Let's try a different approach for the gradient:
            # For each pair (i,j) with dist < 1.0 (arbitrary threshold),
            # add a repulsive force.
            force = np.zeros_like(pts)
            # We only need to consider pairs that are close.
            # Using the tree, find neighbors within 1.0
            neigh = tree.query_ball_point(pts, r=1.0)
            for i in range(N):
                for j in neigh[i]:
                    if j > i:
                        d = dists[i, j]
                        if d > 1e-8 and d < 1.0:
                            # Repulsive force
                            f = alpha * (1.0 - d) / d
                            force[i] += f * (pts[i] - pts[j])
                            force[j] -= f * (pts[i] - pts[j])
            grad += force
        return grad

    def _hex_init():
        """Generate hexagonal lattice initialization."""
        # Approximate spacing
        d = np.sqrt(1.0 / (N * 0.866))
        pts = []
        y = -0.5
        row = 0
        while y <= 0.5:
            x_offset = (d / 2) if row % 2 else 0
            x = -0.5 + x_offset
            while x <= 0.5:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts.append([x, y])
                x += d
            y += d * np.sqrt(3) / 2
            row += 1
            
        pts = np.array(pts)
        if len(pts) < N:
            # Fallback to grid
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            pos = grid[:N].copy()
        else:
            np.random.shuffle(pts)
            pos = pts[:N].copy()
            
        pos = np.clip(pos, -0.49, 0.49)
        return pos

    def _grid_init():
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N].copy()

    def _project(pts):
        """Project points into the square."""
        return np.clip(pts, -0.5, 0.5)

    best_pos = None
    best_r = 0.0

    # Binary search on radius
    r_lo = 0.05
    r_hi = 0.08  # Best known is ~0.0714
    
    # Initialize
    pos = _hex_init()
    best_pos = pos.copy()
    best_r = _score(pos)
    
    # Optimize for a range of radii
    # We try to find the largest r such that we can place circles of radius r.
    # We use projected gradient ascent on the objective:
    # f(pts) = sum( max(0, 2r - dist_ij)^2 ) + boundary violations
    # We minimize this. If it reaches 0, the configuration is feasible for r.
    
    def _feasible(pts, r, alpha=10.0):
        """Check if pts is feasible for radius r."""
        if r <= 0:
            return True
        # Boundary
        if np.any(pts < -0.5 + r - 1e-9) or np.any(pts > 0.5 - r + 1e-9):
            return False
        # Pairwise
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=2)
        if np.any(d[:, 1] < 2 * r - 1e-9):
            return False
        return True

    def _objective(pts, r):
        """Objective to minimize: violation of constraints for radius r."""
        # Boundary
        d_bound = np.min(0.5 - np.abs(pts), axis=1)
        bound_viol = np.sum(np.maximum(0, r - d_bound)**2)
        
        # Pairwise
        if N > 1:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_d = np.min(dists)
            # Only penalize if min_d < 2r
            # Use a smooth penalty
            pair_viol = np.sum(np.maximum(0, 2*r - dists)**2)
        else:
            pair_viol = 0.0
        return bound_viol + pair_viol

    def _grad_objective(pts, r):
        """Gradient of _objective."""
        grad = np.zeros_like(pts)
        
        # Boundary
        d_bound_x = 0.5 - np.abs(pts[:, 0])
        d_bound_y = 0.5 - np.abs(pts[:, 1])
        # d_bound = min(dx, dy)
        # This is tricky. Let's approximate by using L2 distance to boundary?
        # No, the problem is L-inf distance (square).
        # Let's just use the gradient of the squared violation for each side independently.
        # Violation_x_lower = max(0, -0.5 + r - pts[:,0])
        # Violation_x_upper = max(0, pts[:,0] - (0.5 - r))
        # Similarly for y.
        
        v_x_low = np.maximum(0, -0.5 + r - pts[:, 0])
        v_x_up = np.maximum(0, pts[:, 0] - (0.5 - r))
        v_y_low = np.maximum(0, -0.5 + r - pts[:, 1])
        v_y_up = np.maximum(0, pts[:, 1] - (0.5 - r))
        
        grad[:, 0] += -2 * v_x_low + 2 * v_x_up
        grad[:, 1] += -2 * v_y_low + 2 * v_y_up
        
        # Pairwise
        if N > 1:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Penalty: sum max(0, 2r - dist)^2
            # Gradient w.r.t p_i:
            # sum_j max(0, 2r - d_ij)^2 * d/dp_i (2r - d_ij)
            # = sum_j max(0, 2r - d_ij)^2 * (-1) * d/dp_i d_ij
            # d/dp_i d_ij = (p_i - p_j) / d_ij
            # So term = - max(0, 2r - d_ij)^2 * (p_i - p_j) / d_ij
            
            force = np.zeros_like(pts)
            for i in range(N):
                for j in range(i+1, N):
                    d = dists[i, j]
                    if d < 2*r and d > 1e-8:
                        w = (2*r - d)**2 / d
                        force[i] += -w * (pts[i] - pts[j])
                        force[j] -= -w * (pts[i] - pts[j])
            grad += force
        return grad

    # Optimize using projected gradient descent
    # We try to find a feasible configuration for increasing r.
    
    # Initial estimate
    r_current = best_r
    
    # Step size
    step_size = 0.01
    
    # Main loop: try to increase r
    while time.time() < deadline - 5.0:
        # Try to find a better configuration for current r_current
        # Or try to increase r_current slightly
        r_test = r_current + 0.001
        if r_test > r_hi:
            r_test = r_hi
            
        # Optimize for r_test
        obj = _objective(pos, r_test)
        
        converged = False
        for it in range(200):
            if time.time() > deadline - 5.0:
                break
            grad = _grad_objective(pos, r_test)
            pos_new = pos - step_size * grad
            pos_new = _project(pos_new)
            
            obj_new = _objective(pos_new, r_test)
            if obj_new < obj:
                pos = pos_new
                obj = obj_new
            else:
                # Reduce step size
                step_size *= 0.5
                if step_size < 1e-6:
                    break
            if obj < 1e-6:
                converged = True
                break
        
        if converged:
            # We found a feasible configuration for r_test
            best_r = r_test
            best_pos = pos.copy()
            r_current = r_test
        else:
            # Failed to improve for r_test
            # Try a smaller increment
            r_test = r_current + 0.0005
            if r_test > r_hi:
                break
            # If we fail multiple times, try to polish current best
            if time.time() < deadline - 10.0:
                # Try to polish best_pos with current r_current
                pos = best_pos.copy()
                for it in range(100):
                    if time.time() > deadline - 5.0:
                        break
                    grad = _grad_objective(pos, r_current)
                    pos_new = pos - step_size * grad
                    pos_new = _project(pos_new)
                    obj_new = _objective(pos_new, r_current)
                    if obj_new < _objective(pos, r_current):
                        pos = pos_new
                    else:
                        break
                new_r = _score(pos)
                if new_r > best_r:
                    best_r = new_r
                    best_pos = pos.copy()
            else:
                break

    # Final polish
    if time.time() < deadline - 5.0:
        pos = best_pos.copy()
        # Use the actual radius as the target for the objective
        r_target = best_r
        for it in range(200):
            if time.time() > deadline - 2.0:
                break
            grad = _grad_objective(pos, r_target)
            pos_new = pos - 0.001 * grad
            pos_new = _project(pos_new)
            obj_new = _objective(pos_new, r_target)
            if obj_new < _objective(pos, r_target):
                pos = pos_new
            else:
                break
        new_r = _score(pos)
        if new_r > best_r:
            best_r = new_r
            best_pos = pos.copy()

    return best_pos

# EVOLVE-BLOCK-END
