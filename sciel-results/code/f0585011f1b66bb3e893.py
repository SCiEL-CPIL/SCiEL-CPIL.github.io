"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (evaluator kills at 150s)
    
    def _time_left():
        return max(0.0, deadline - time.time())
    
    best_score = -1.0
    best_points = None
    
    def _compute_radius(pts):
        """Compute the common radius for given points (N,3)."""
        if pts is None or not np.all(np.isfinite(pts)):
            return 0.0
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 0.0
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        return min(min_pair / 2.0, min_boundary)
    
    def _repulsion_energy(pts):
        """
        Compute a repulsion-based energy that correlates with maximizing min distance.
        Uses inverse-square repulsion plus boundary repulsion.
        Lower energy = better packing.
        """
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Inverse-square repulsion: sum of 1/d^2 over all pairs
        # This strongly penalizes close pairs
        inv_sq = np.where(dists > 1e-8, 1.0 / np.maximum(dists, 1e-8)**2, 0.0)
        pair_energy = np.sum(inv_sq)
        
        # Boundary repulsion: penalize points near walls
        b_dists = 0.5 - np.abs(pts)
        min_b = np.min(b_dists, axis=1)
        # Strong penalty for being close to boundary
        bound_energy = np.sum(1.0 / np.maximum(min_b, 1e-8)**2)
        
        return pair_energy + bound_energy
    
    def _repulsion_grad(pts):
        """
        Gradient of repulsion energy.
        For pair repulsion: grad_i = sum_{j != i} 2 * (pts[i] - pts[j]) / d_ij^3
        For boundary: grad_i[k] = 2 * sign(pts[i,k]) / min_b_i^3 for the closest axis
        """
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Weight matrix: 2 / d^3
        W = np.where(dists > 1e-8, 2.0 / np.maximum(dists, 1e-8)**3, 0.0)
        np.fill_diagonal(W, 0.0)
        
        # Pair gradient: sum_j W_ij * (pts_i - pts_j)
        # = sum_j W_ij * pts_i - sum_j W_ij * pts_j
        row_sums = np.sum(W, axis=1)
        grad_pair = row_sums[:, np.newaxis] * pts - (W @ pts)
        
        # Boundary gradient
        b_dists = 0.5 - np.abs(pts)
        min_idx = np.argmin(b_dists, axis=1)
        min_b_vals = np.min(b_dists, axis=1)
        min_b_safe = np.maximum(min_b_vals, 1e-8)
        
        grad_bound = np.zeros_like(pts)
        for i in range(n):
            k = min_idx[i]
            s = np.sign(pts[i, k])
            if s == 0:
                s = 1.0
            grad_bound[i, k] = 2.0 * s / min_b_safe[i]**3
        
        return grad_pair + grad_bound
    
    def _simulated_annealing(pts, t_start, t_end, max_iter):
        """
        Simulated annealing to minimize repulsion energy.
        Returns improved points.
        """
        current = pts.copy()
        current_energy = _repulsion_energy(current)
        best = current.copy()
        best_energy = current_energy
        
        for it in range(max_iter):
            if time.time() > deadline - 1.0:
                break
            
            # Temperature
            alpha = it / max(max_iter - 1, 1)
            T = t_start * (t_end / t_start) ** alpha
            
            # Generate perturbation: random displacement
            # Step size proportional to temperature
            step_scale = 0.05 * (T / t_start)
            delta = np.random.normal(0, step_scale, size=current.shape)
            proposed = current + delta
            proposed = np.clip(proposed, -0.499, 0.499)
            
            proposed_energy = _repulsion_energy(proposed)
            
            # Accept if better or with probability exp(-(dE)/T)
            dE = proposed_energy - current_energy
            if dE < 0 or np.random.random() < np.exp(-dE / max(T, 1e-10)):
                current = proposed
                current_energy = proposed_energy
                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy
        
        return best
    
    def _polish_steepest(pts, max_iter=500):
        """
        Steepest descent polishing on repulsion energy with adaptive step.
        """
        current = pts.copy()
        current_energy = _repulsion_energy(current)
        best = current.copy()
        best_energy = current_energy
        
        lr = 0.01
        for it in range(max_iter):
            if time.time() > deadline - 1.0:
                break
            
            grad = _repulsion_grad(current)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            
            # Adaptive step size
            step = lr / (grad_norm + 1e-8)
            proposed = current - step * grad
            proposed = np.clip(proposed, -0.499, 0.499)
            
            proposed_energy = _repulsion_energy(proposed)
            
            if proposed_energy < current_energy:
                current = proposed
                current_energy = proposed_energy
                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy
                # Increase learning rate slightly
                lr = min(lr * 1.05, 0.05)
            else:
                # Decrease learning rate
                lr *= 0.7
        
        return best
    
    def _init_bcc():
        """Body-centered cubic lattice initialization."""
        # BCC lattice: 8 corners + 1 center per cell
        # For 100 points, use a 5x4x4 grid of BCC cells = 5*4*4*2 = 160, take 100
        # Or use a denser arrangement
        pts = []
        # 5x4x4 grid of cells, each with corner and center
        for ix in range(5):
            for iy in range(4):
                for iz in range(4):
                    # Cell center at ((ix+0.5)/5, (iy+0.5)/4, (iz+0.5)/4) scaled to [-0.5,0.5]
                    cx = (ix + 0.5) / 5.0 - 0.5
                    cy = (iy + 0.5) / 4.0 - 0.5
                    cz = (iz + 0.5) / 4.0 - 0.5
                    # Add center point
                    pts.append([cx, cy, cz])
                    # Add corner points (only some to avoid duplicates)
                    if len(pts) < N:
                        # Offset for BCC
                        pts.append([cx + 0.05, cy + 0.05, cz + 0.05])
        
        pts = np.array(pts[:N])
        return pts.flatten()
    
    def _init_halton():
        """Halton sequence initialization for low-discrepancy sampling."""
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        pts = np.array([[halton(i, 2), halton(i, 3), halton(i, 5)] for i in range(1, N+1)])
        # Map [0,1] to [-0.5, 0.5]
        pts = pts - 0.5
        return pts.flatten()
    
    def _init_grid_perturbed(scale=0.02):
        """Perturbed 5x5x4 grid."""
        m_x, m_y, m_z = 5, 5, 4
        pts = []
        for ix in range(m_x):
            for iy in range(m_y):
                for iz in range(m_z):
                    x = (ix + 0.5) / m_x - 0.5
                    y = (iy + 0.5) / m_y - 0.5
                    z = (iz + 0.5) / m_z - 0.5
                    pts.append([x, y, z])
        pts = np.array(pts[:N])
        pts += np.random.normal(0, scale, size=pts.shape)
        pts = np.clip(pts, -0.49, 0.49)
        return pts.flatten()
    
    # Generate diverse initializations
    np.random.seed(42)  # Reproducibility
    inits = [
        _init_bcc(),
        _init_halton(),
        _init_grid_perturbed(0.015),
        _init_grid_perturbed(0.03),
        np.random.uniform(-0.4, 0.4, size=(N, 3)).flatten(),
        np.random.uniform(-0.4, 0.4, size=(N, 3)).flatten() + np.random.normal(0, 0.02, size=(N*3,)),
    ]
    
    # For each initialization: SA + polish
    for i, x0 in enumerate(inits):
        if _time_left() < 5.0:
            break
        
        x0 = x0.reshape(N, 3)
        
        # Phase 1: Simulated annealing
        sa_iters = min(3000, int(_time_left() * 30))
        sa_iters = max(sa_iters, 500)
        pts_sa = _simulated_annealing(x0, t_start=1.0, t_end=0.01, max_iter=sa_iters)
        
        if _time_left() < 2.0:
            score = _compute_radius(pts_sa)
            if score > best_score:
                best_score = score
                best_points = pts_sa.copy()
            continue
        
        # Phase 2: Steepest descent polish
        polish_iters = min(300, int(_time_left() * 20))
        polish_iters = max(polish_iters, 100)
        pts_polished = _polish_steepest(pts_sa, max_iter=polish_iters)
        
        score = _compute_radius(pts_polished)
        if score > best_score:
            best_score = score
            best_points = pts_polished.copy()
    
    if best_points is None:
        best_points = _init_grid_perturbed(0.01).reshape(N, 3)
    
    return best_points
# EVOLVE-BLOCK-END
