# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of (x, y, angle) placements for the L-shaped corridor that
    maximizes the area of the intersection of all corridor placements.
    
    Strategy:
    1. Precompute the "sofa" grid points.
    2. Use a Beam Search in the configuration space (x, y, theta).
    3. State: (current_x, current_y, current_theta, intersection_mask).
    4. Goal: Maximize the number of points in the final intersection mask.
    5. Since the intersection shrinks, we prioritize moves that keep the intersection large.
    6. We discretize x, y, theta into a coarse grid to make the search tractable.
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Precompute the sofa grid
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    gx = X.ravel()
    gy = Y.ravel()
    n_pts = len(gx)

    def _densify_path(path):
        """
        Densify path exactly as evaluator: prepend (0,0,0), append (-2,0,90),
        then interpolate between consecutive points in m = max(1, ceil(|da|)) steps.
        Returns array of shape (N, 3).
        """
        full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
        result = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(math.ceil(da)))
            for k in range(1, m + 1):
                t = k / m
                interp = p1 + t * (p2 - p1)
                result.append(interp)
        return np.array(result)

    def _evaluate_path_vectorized(path):
        """
        Vectorized evaluation of the final path.
        """
        dense = _densify_path(path)
        n_placements = len(dense)
        
        cx = dense[:, 0]
        cy = dense[:, 1]
        a_deg = dense[:, 2]
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = gx[None, :] - cx[:, None]
        dy = gy[None, :] - cy[:, None]
        u = cos_a[:, None] * dx + sin_a[:, None] * dy
        v = -sin_a[:, None] * dx + cos_a[:, None] * dy
        
        in_arm1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_arm2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        in_base = in_arm1 | in_arm2
        
        mask = in_base.all(axis=0)
        
        return float(np.sum(mask)) * 0.0001

    def _get_mask(x, y, a_deg):
        """
        Compute the boolean mask of grid points inside the L-shaped corridor
        placed at (x, y) with rotation a_deg.
        """
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = gx - x
        dy = gy - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        in_arm1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        in_arm2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        
        return in_arm1 | in_arm2

    # --- Beam Search Setup ---
    
    # We search in the space of (x, y, theta).
    # Start: (0, 0, 0)
    # End: (-2, 0, 90)
    
    # We will define a set of candidate "steps" or "nodes".
    # To keep it simple and robust, we'll generate a grid of possible intermediate states.
    # However, the path must be continuous in the evaluator's interpolation sense.
    # The evaluator interpolates linearly between our returned points.
    # So we just need to return a sequence of (x, y, theta).
    
    # Let's use a beam search where each state is (x, y, theta, mask).
    # We want to maximize the popcount of the final mask.
    
    # Initial state
    init_mask = _get_mask(0.0, 0.0, 0.0)
    
    # We will perform a search over a fixed number of steps (e.g., 10-20 steps).
    # At each step, we propose moves towards the target.
    
    # Target: (-2, 0, 90)
    # Let's discretize the path into N segments.
    N_steps = 15
    
    # Beam width
    beam_width = 50
    
    # Current beam: list of (x, y, theta, mask, path_so_far)
    # path_so_far is a list of (x, y, theta)
    
    current_beam = [(0.0, 0.0, 0.0, init_mask, [(0.0, 0.0, 0.0)])]
    
    # We will iterate through steps.
    # At step i (0 to N_steps-1), we are at some position.
    # We want to move towards the target.
    # Let's define candidate next positions based on the current position and the target.
    # To ensure we reach the target exactly at the end, we can parameterize the "ideal" path
    # and add perturbations.
    
    # Ideal linear path:
    # x(t) = 0 + t * (-2 - 0) = -2t
    # y(t) = 0
    # theta(t) = 0 + t * (90 - 0) = 90t
    
    # We will explore perturbations around this ideal path.
    # Perturbation in x, y, theta.
    
    # Let's try a few different "perturbation scales" and "step counts" to be robust.
    # But 15 steps seems reasonable.
    
    best_final_path = None
    best_final_score = -1.0
    
    # To speed up, we can pre-compute masks for a grid of candidate positions.
    # But the number of candidates might be large.
    # Let's generate candidates dynamically.
    
    # Candidate generation:
    # For each state in the beam, generate K neighbors.
    # Neighbors are defined by:
    # 1. Moving along the ideal path by 1/N_steps (with small noise).
    # 2. Moving to a random point closer to the target.
    
    K_neighbors = 20
    
    for step in range(N_steps):
        if time.time() > deadline - 10:
            break
            
        next_beam = []
        t_curr = step / N_steps
        t_next = (step + 1) / N_steps
        
        # Ideal positions at t_next
        x_ideal = -2.0 * t_next
        y_ideal = 0.0
        theta_ideal = 90.0 * t_next
        
        for (x, y, theta, mask, path) in current_beam:
            # Generate K neighbors
            # We want to stay close to the ideal path but explore variations.
            # Noise scales:
            # x_noise, y_noise, theta_noise
            
            # Let's use a mix of "small steps" and "jumps".
            # Actually, since the evaluator interpolates, large jumps in theta create many intermediate placements.
            # We must ensure the path is valid (no collisions) in the evaluator's dense representation.
            # Our mask check only checks the *discrete* points we output.
            # We MUST ensure the dense path is valid!
            # This is the critical constraint.
            
            # To ensure validity, we should check the dense path for every candidate.
            # This is expensive.
            # Alternative: Keep the steps small enough that the linear interpolation
            # stays within the "safe" zone.
            # Or, simply check the dense path for the top candidates.
            
            # Let's generate candidates.
            candidates = []
            
            # 1. Ideal step
            candidates.append((x_ideal, y_ideal, theta_ideal))
            
            # 2. Perturbed steps
            # Noise in x, y, theta
            # x range: [-0.5, 0.5] relative to ideal?
            # y range: [-0.5, 0.5]
            # theta range: [-10, 10] degrees
            
            np.random.seed(step * 100 + 42) # Deterministic noise for reproducibility in search
            for _ in range(K_neighbors - 1):
                dx = np.random.uniform(-0.5, 0.5)
                dy = np.random.uniform(-0.5, 0.5)
                dtheta = np.random.uniform(-15.0, 15.0)
                
                cx = x_ideal + dx
                cy = y_ideal + dy
                ctheta = theta_ideal + dtheta
                
                # Clamp theta to [0, 90] to avoid backtracking?
                # The problem allows any angle, but we start at 0 and end at 90.
                # Backtracking might be useful? Probably not for maximizing intersection.
                # Let's keep it monotonic in theta for now?
                # Actually, the evaluator just interpolates.
                # Let's allow any theta.
                
                candidates.append((cx, cy, ctheta))
            
            # Evaluate candidates
            valid_candidates = []
            for (cx, cy, ctheta) in candidates:
                new_mask = mask & _get_mask(cx, cy, ctheta)
                score = np.sum(new_mask)
                
                # Heuristic: Penalize if score drops too fast?
                # We just want the highest score at the end.
                # But we need to ensure we can REACH the target.
                # If the mask becomes empty, we're stuck.
                
                # Check if we can still reach the target?
                # Approximation: If the mask is not empty, assume we can.
                # Better: Check if the target placement is compatible with the current mask?
                # No, the mask is the intersection of ALL placements so far.
                # The final mask will be this mask & (intersection of future placements).
                # So we just need the mask to be non-empty.
                
                if score > 0:
                    # Check dense path validity?
                    # This is the expensive part.
                    # Let's only check for the top few candidates per state later,
                    # or assume small steps are safe.
                    # Given the 600s budget, we can afford to check dense validity for the top beam.
                    
                    valid_candidates.append((score, cx, cy, ctheta, new_mask, path + [(cx, cy, ctheta)]))
            
            # Sort by score
            valid_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Add top candidates to next_beam
            next_beam.extend(valid_candidates[:beam_width // 2]) # Limit per parent
            
        # Deduplicate and select top beam_width
        if not next_beam:
            # Fallback: use ideal path
            break
            
        # Sort next_beam by score
        next_beam.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top beam_width
        current_beam = next_beam[:beam_width]
        
        # Optional: Prune based on diversity?
        # For now, just keep top scores.
        
    # After search, we have a set of candidate paths in current_beam.
    # We need to complete the path to the target (-2, 0, 90).
    # The search ended at step N_steps, which is t=1.0, so x=-2, y=0, theta=90.
    # Wait, my loop goes from 0 to N_steps-1.
    # t_next = (step+1)/N_steps.
    # At the last step (step=N_steps-1), t_next = 1.0.
    # So the last point in the path is exactly the target.
    
    # So the paths in current_beam are complete.
    
    # Now, we must verify the dense path validity for the top candidates.
    # And compute the true score.
    
    top_candidates = current_beam[:10]
    
    best_val = -1.0
    best_path = None
    
    for (score, x, y, theta, mask, path) in top_candidates:
        if time.time() > deadline - 5:
            break
            
        # path is a list of tuples. Convert to array.
        path_arr = np.array(path)
        
        # Check dense validity
        # The evaluator adds (0,0,0) and (-2,0,90) and interpolates.
        # Our path already includes (0,0,0) as the first point (from init) 
        # and ends at (-2,0,90) (from the last step).
        # Wait, init was (0,0,0).
        # The first step added (x_ideal, ...).
        # So path[0] is (0,0,0).
        # path[-1] is (-2,0,90).
        
        # The evaluator does:
        # full = [(0,0,0)] + path + [(-2,0,90)]
        # If path already contains endpoints, we might have duplicates.
        # The contract says: "The evaluator puts (0, 0, 0) before your rows and (-2, 0, 90) after them."
        # So if my path is [(0,0,0), ..., (-2,0,90)], the evaluator makes:
        # [(0,0,0), (0,0,0), ..., (-2,0,90), (-2,0,90)]
        # Interpolation between identical points is fine (m=1, t=1, same point).
        # So it's safe.
        
        val = _evaluate_path_vectorized(path_arr)
        
        if val > best_val:
            best_val = val
            best_path = path_arr

    # If beam search failed or didn't find a good path, fallback to a simple parametric path
    if best_path is None or best_val < 0.01:
        # Simple cubic path
        t_vals = np.linspace(0.05, 0.95, 10)
        x_vals = -2.0 * t_vals**3
        y_vals = np.zeros_like(t_vals)
        theta_vals = 90.0 * t_vals**3
        fallback_path = np.column_stack((x_vals, y_vals, theta_vals))
        
        # Try to optimize this slightly? No, just use it.
        # Or use the previous best from a simpler optimizer if available?
        # Let's just return this.
        return fallback_path

    return best_path

# EVOLVE-BLOCK-END
