import numpy as np
import time
from scipy.optimize import least_squares

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Precompute pair indices for efficiency
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def compute_score(pts):
        """Compute the packing radius score efficiently."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        # Boundary distances
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = np.sqrt(sq_dists[i_idx, j_idx]) / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        return np.min(all_vals)

    def hex_lattice_init(spacing):
        """Generate hexagonal lattice points inside unit circle."""
        y_step = spacing * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / spacing) + 2
        
        pts = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else spacing / 2
            for i in range(-max_i, max_i + 1):
                x = i * spacing + offset
                if np.hypot(x, y) < 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) > N:
            dists = np.linalg.norm(pts, axis=1)
            idx = np.argsort(dists)[:N]
            pts = pts[idx]
        elif len(pts) < N:
            while len(pts) < N:
                p = np.random.uniform(-0.95, 0.95, 2)
                if np.linalg.norm(p) < 0.95:
                    pts = np.vstack([pts, p])
            pts = pts[:N]
        
        return pts

    def random_init():
        """Random initialization with rejection sampling."""
        pts = []
        while len(pts) < N:
            p = np.random.uniform(-0.98, 0.98, 2)
            if np.linalg.norm(p) < 0.98:
                pts.append(p)
        return np.array(pts)

    def simulate_dynamics(pts_init, target_r, time_budget, dt=0.001, friction=0.98, force_power=4, force_scale=100.0):
        """
        Rigid-body collision dynamics simulation.
        Uses Verlet integration and Hertzian-like repulsive forces.
        """
        pts = pts_init.copy()
        velocities = np.zeros_like(pts)
        
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        end_time = start_time + time_budget
        steps = 0
        
        # Precompute indices for force calculation
        # We need to calculate forces for all pairs
        
        while time.time() < end_time:
            # Check time frequently to avoid overshooting
            if steps % 100 == 0 and time.time() > end_time:
                break
                
            # Calculate forces
            forces = np.zeros_like(pts)
            
            # Pairwise forces
            # We want to push apart if distance < 2 * target_r
            # Force magnitude ~ (overlap)^force_power
            # Direction: away from each other
            
            # Vectorized difference
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            sq_dists = np.sum(diff**2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            
            dists = np.sqrt(sq_dists)
            
            # Target distance is 2 * target_r
            target_dist = 2.0 * target_r
            
            # Overlap is positive if dist < target_dist
            overlap = target_dist - dists
            mask = overlap > 0
            
            if np.any(mask):
                # Compute force magnitudes
                # F = force_scale * overlap^force_power
                # We only care about pairs in mask
                # But we need to sum them up.
                
                # To be efficient, we can just compute for all pairs and mask out, 
                # or use sparse operations. Given N=300, dense is fine (90k pairs).
                
                # Gradient of distance w.r.t position is (p_i - p_j) / dist
                # Force on i is in direction (p_i - p_j)
                # Force on j is in direction (p_j - p_i)
                
                # We compute the vector part
                # vecs = diff / dist
                # Avoid division by zero
                safe_dists = np.where(dists > 1e-8, dists, 1.0)
                unit_vecs = diff / safe_dists[:, :, np.newaxis]
                
                # Force magnitudes
                # Only non-zero where mask is true
                # We use a large power to make it stiff
                force_mags = np.zeros_like(sq_dists)
                force_mags[mask] = force_scale * (overlap[mask] ** force_power)
                
                # Total force vector for each particle
                # F_i = sum_j F_ij * unit_vec_ij
                # Note: unit_vec_ij points from j to i (since diff is i-j)
                # So force on i is +F_ij * unit_vec_ij
                # Force on j is -F_ij * unit_vec_ij
                
                # Sum over j for each i
                # forces = np.sum(unit_vecs * force_mags[:, :, np.newaxis], axis=1)
                
                # This creates a 300x300x2 array, which is ~720KB, acceptable.
                force_vectors = unit_vecs * force_mags[:, :, np.newaxis]
                forces += np.sum(force_vectors, axis=1)
            
            # Boundary forces
            # Push inward if rad > 1 - target_r
            rad = np.linalg.norm(pts, axis=1)
            target_rad = 1.0 - target_r
            bound_overlap = target_rad - rad
            bound_mask = bound_overlap > 0
            
            if np.any(bound_mask):
                # Force direction: inward (towards origin)
                # Unit vector from p to 0 is -p/rad
                # Force magnitude ~ overlap^power
                safe_rad = np.where(rad > 1e-8, rad, 1.0)
                bound_unit_vecs = -pts / safe_rad[:, np.newaxis]
                bound_force_mags = force_scale * (bound_overlap ** force_power)
                bound_forces = bound_unit_vecs * bound_force_mags[:, np.newaxis]
                bound_forces[~bound_mask] = 0
                forces += bound_forces
            
            # Verlet Integration
            # v_new = (v + a*dt) * friction
            # p_new = p + v_new*dt
            
            # Acceleration = Force / Mass (Mass=1)
            acc = forces
            
            velocities = (velocities + acc * dt) * friction
            pts = pts + velocities * dt
            
            # Project back inside circle strictly to prevent escape
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
                velocities[mask_out] = 0 # Stop them at boundary for stability
            
            # Update best
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
            
            steps += 1
            
            # Adaptive target_r increase
            # If we are stable (low kinetic energy or high score), increase target_r
            # Simple heuristic: increase target_r slowly
            # We want to find the maximum r, so we can try to increase it if we are improving
            # But we need to be careful not to destabilize.
            # Let's just run for a fixed time and let the forces push as hard as they can.
            # The force_scale and power determine the stiffness.
            
        return best_pts, best_score

    # Generate multiple initializations
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Hexagonal lattice dynamics
    # Try a few spacings close to optimal
    spacing_multipliers = [0.98, 1.0, 1.02]
    
    time_per_init = min(20.0, time_limit * 0.25)
    
    for mult in spacing_multipliers:
        if time.time() - start_time > time_limit * 0.70:
            break
            
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        
        # Start with a target_r slightly below current best possible for this init
        # Estimate initial score
        init_score = compute_score(pts)
        
        # Run dynamics
        # We use a target_r that we want to achieve. 
        # If we set target_r too high, forces are huge and unstable.
        # If too low, we don't improve.
        # Let's start with init_score * 0.95 and let it find its way up?
        # Actually, the dynamics will push them apart until forces balance.
        # The equilibrium is where the repulsive force is zero, i.e., dist >= 2*target_r.
        # So if we set target_r high, it will try to push them apart until they hit the boundary or each other.
        # This effectively maximizes the minimum distance.
        
        # Let's set target_r to a value slightly higher than the current best score found so far, 
        # or just a reasonable estimate.
        # For the first run, use the theoretical max ~ 0.053
        # But to be safe, let's start with 0.05
        
        target_r_start = 0.05
        
        pts_opt, score_opt = simulate_dynamics(pts, target_r_start, time_budget=time_per_init, dt=0.0005, friction=0.95, force_power=4, force_scale=50.0)
        
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt.copy()
            
    # Phase 2: Random init dynamics
    if time.time() - start_time < time_limit * 0.80:
        for _ in range(2):
            if time.time() - start_time > time_limit * 0.80:
                break
            pts = random_init()
            pts_opt, score_opt = simulate_dynamics(pts, 0.05, time_budget=10.0, dt=0.0005, friction=0.95, force_power=4, force_scale=50.0)
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt.copy()

    # Phase 3: Polish with higher target_r and smaller dt
    if best_pts is not None and time.time() - start_time < time_limit * 0.90:
        remaining_time = time_limit * 0.97 - time.time()
        if remaining_time > 5.0:
            # Try to push higher
            target_r_high = best_score * 1.01
            pts_polished, score_polished = simulate_dynamics(best_pts, target_r_high, time_budget=remaining_time * 0.5, dt=0.0002, friction=0.90, force_power=6, force_scale=100.0)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
            
            # Final safety check
            rad = np.linalg.norm(best_pts, axis=1)
            mask_out = rad > 1.0
            if np.any(mask_out):
                best_pts[mask_out] /= rad[mask_out, np.newaxis]

    if best_pts is None:
        best_pts = hex_lattice_init(base_spacing)

    return best_pts
