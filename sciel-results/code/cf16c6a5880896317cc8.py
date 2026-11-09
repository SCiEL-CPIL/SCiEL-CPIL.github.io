# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points in a convex region to maximize
    A_min / A_hull, where A_min is the smallest triangle area among all
    286 triples and A_hull is the convex hull area.
    
    Uses soft-minimum annealing (log-sum-exp) to push all triangles up
    simultaneously, avoiding local minima where only the single smallest
    triangle is addressed.
    """
    n = 13
    deadline = time.time() + 600.0  # safety margin before 630s kill
    deadline_check_interval = 5.0
    last_check = time.time()
    
    # Precompute all triangle index triples
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    triples = np.array(triples)
    m = len(triples)  # 286
    
    def compute_areas(pts):
        """Compute areas of all 286 triangles efficiently."""
        i, j, k = triples[:, 0], triples[:, 1], triples[:, 2]
        pi, pj, pk = pts[i], pts[j], pts[k]
        # Area = 0.5 * |cross(pj-pi, pk-pi)|
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
                (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        return 0.5 * np.abs(cross)
    
    def compute_hull_area(pts):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 0.0
    
    def score(pts):
        """Compute A_min / A_hull."""
        areas = compute_areas(pts)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        hull_a = compute_hull_area(pts)
        if hull_a <= 0:
            return 0.0
        return a_min / hull_a
    
    def soft_min_score(pts, beta):
        """
        Soft-minimum objective using log-sum-exp.
        Maximizes a smoothed version of the minimum area, normalized by hull area.
        Higher beta -> closer to true minimum.
        """
        areas = compute_areas(pts)
        hull_a = compute_hull_area(pts)
        if hull_a <= 0 or np.any(~np.isfinite(areas)):
            return -1e10
        # Normalize areas by hull area for scale invariance
        norm_areas = areas / hull_a
        # Soft minimum: -(1/beta) * log(sum(exp(-beta * a)))
        # This is a smooth approximation of min(a)
        max_val = np.max(norm_areas)
        log_sum = max_val + np.log(np.sum(np.exp(-beta * (norm_areas - max_val))))
        soft_min_val = -(1.0 / beta) * log_sum + max_val / beta
        # Actually: soft_min = -(1/beta)*log(sum(exp(-beta*a)))
        # Let me redo: soft_min(a, beta) = -(1/beta) * log(sum(exp(-beta * a)))
        exp_vals = np.exp(-beta * norm_areas)
        log_sum_exp = np.log(np.sum(exp_vals))
        soft_min_val = -(1.0 / beta) * log_sum_exp
        return soft_min_val
    
    def soft_min_gradient(pts, beta):
        """
        Gradient of the soft-minimum objective w.r.t. point coordinates.
        Returns gradient of the objective (to maximize).
        """
        areas = compute_areas(pts)
        hull_a = compute_hull_area(pts)
        if hull_a <= 0:
            return None
        norm_areas = areas / hull_a
        # Weights for soft-min: w_i = exp(-beta * a_i) / sum(exp(-beta * a_j))
        exp_vals = np.exp(-beta * norm_areas)
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient of soft_min = (1/beta) * sum(w_i * grad(a_i))
        # But we want to maximize soft_min, and a_i = 0.5*|cross|
        # For simplicity, use the sign-aware gradient
        grad_pts = np.zeros_like(pts)
        
        i, j, k = triples[:, 0], triples[:, 1], triples[:, 2]
        pi, pj, pk = pts[i], pts[j], pts[k]
        
        # cross = (pj_x - pi_x)(pk_y - pi_y) - (pj_y - pi_y)(pk_x - pi_x)
        dx_pj_pi = pj[:, 0] - pi[:, 0]
        dy_pj_pi = pj[:, 1] - pi[:, 1]
        dx_pk_pi = pk[:, 0] - pi[:, 0]
        dy_pk_pi = pk[:, 1] - pi[:, 1]
        cross = dx_pj_pi * dy_pk_pi - dy_pj_pi * dx_pk_pi
        sign_cross = np.sign(cross)
        
        # d(cross)/d(pi_x) = -(dy_pk_pi) + dy_pj_pi = dy_pj_pi - dy_pk_pi
        # d(cross)/d(pi_y) = dx_pk_pi - dx_pj_pi
        # d(cross)/d(pj_x) = dy_pk_pi
        # d(cross)/d(pj_y) = -dx_pk_pi
        # d(cross)/d(pk_x) = -dy_pj_pi
        # d(cross)/d(pk_y) = dx_pj_pi
        
        w = weights * sign_cross * (1.0 / hull_a)
        w_beta = w / beta
        
        d_pi_x = np.zeros(n)
        d_pi_y = np.zeros(n)
        d_pj_x = np.zeros(n)
        d_pj_y = np.zeros(n)
        d_pk_x = np.zeros(n)
        d_pk_y = np.zeros(n)
        
        np.add.at(d_pi_x, i, w_beta * (dy_pj_pi - dy_pk_pi))
        np.add.at(d_pi_y, i, w_beta * (dx_pk_pi - dx_pj_pi))
        np.add.at(d_pj_x, j, w_beta * dy_pk_pi)
        np.add.at(d_pj_y, j, w_beta * (-dx_pk_pi))
        np.add.at(d_pk_x, k, w_beta * (-dy_pj_pi))
        np.add.at(d_pk_y, k, w_beta * dx_pj_pi)
        
        grad_pts[:, 0] = d_pi_x + d_pj_x + d_pk_x
        grad_pts[:, 1] = d_pi_y + d_pj_y + d_pk_y
        
        return grad_pts
    
    # Initial configuration: regular polygon + center + random perturbations
    def make_initial_points(seed):
        rng = np.random.default_rng(seed)
        # Start with a regular 13-gon
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        # Add small random perturbation
        pts += rng.normal(0, 0.05, size=pts.shape)
        return pts
    
    best_pts = None
    best_score = -np.inf
    
    # Multiple restarts with different initial configurations
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            pts = make_initial_points(42)
        elif restart == 1:
            pts = make_initial_points(123)
        elif restart == 2:
            # Square grid-like arrangement
            rng = np.random.default_rng(456)
            pts = np.array([
                [0, 0], [1, 0], [2, 0], [3, 0],
                [0, 1], [1, 1], [2, 1], [3, 1],
                [0, 2], [1, 2], [2, 2], [3, 2],
                [1.5, 3]
            ], dtype=float)
            pts += rng.normal(0, 0.03, size=pts.shape)
        elif restart == 3:
            # Hexagonal-like arrangement
            rng = np.random.default_rng(789)
            pts = np.array([
                [0, 0], [1, 0], [2, 0],
                [0.5, 0.866], [1.5, 0.866], [2.5, 0.866],
                [0, 1.732], [1, 1.732], [2, 1.732],
                [0.5, 2.598], [1.5, 2.598],
                [1, 3.464], [2, 3.464]
            ], dtype=float)
            pts += rng.normal(0, 0.03, size=pts.shape)
        else:
            pts = make_initial_points(1000 + restart * 137)
        
        # Annealing: start with low beta (smooth), increase to high beta (sharp)
        beta_start = 5.0
        beta_end = 500.0
        n_anneal_steps = 200
        step_size_start = 0.01
        step_size_end = 0.0001
        
        current_pts = pts.copy()
        
        for step in range(n_anneal_steps):
            if time.time() > deadline - 30:
                break
            
            progress = step / max(n_anneal_steps - 1, 1)
            beta = beta_start + (beta_end - beta_start) * progress
            step_size = step_size_start + (step_size_end - step_size_start) * progress
            
            # Compute gradient
            grad = soft_min_gradient(current_pts, beta)
            if grad is None:
                break
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            # Update
            current_pts = current_pts + step_size * grad
            
            # Keep points in a bounded region (normalize to unit scale)
            # Center and scale
            center = np.mean(current_pts, axis=0)
            current_pts -= center
            scale = np.max(np.abs(current_pts))
            if scale > 0:
                current_pts /= scale
            current_pts += center
            
            # Check score periodically
            if step % 20 == 0 or step == n_anneal_steps - 1:
                s = score(current_pts)
                if s > best_score:
                    best_score = s
                    best_pts = current_pts.copy()
            
            # Time check
            if time.time() - last_check > deadline_check_interval:
                last_check = time.time()
                if time.time() > deadline - 30:
                    break
        
        # Final score check for this restart
        s = score(current_pts)
        if s > best_score:
            best_score = s
            best_pts = current_pts.copy()
    
    # Fallback: if something went wrong, return a valid configuration
    if best_pts is None or best_score <= 0:
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts

# EVOLVE-BLOCK-END
