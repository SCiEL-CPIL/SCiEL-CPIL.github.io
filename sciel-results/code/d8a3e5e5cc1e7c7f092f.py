"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def construct():
    """
    Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum distance.
    Strategy:
    1. Generate multiple high-quality initial guesses (Fibonacci lattice, Halton, random).
    2. Refine each using L-BFGS-B with a smoothed log-sum-exp objective.
    3. Polish the best result using Nelder-Mead on the exact non-smooth objective.
    4. Local search: perturb individual points to escape local minima.
    """
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 10.0
    effective_limit = time_limit - safety_margin
    
    def check_time():
        return (time.time() - start_time) >= effective_limit
    
    def score_centres(centres):
        """Compute the radius (min of pairwise/2 and boundary distances)."""
        if not np.all(np.isfinite(centres)):
            return 0.0
        if np.any(centres < -0.5) or np.any(centres > 0.5):
            return 0.0
        
        # Boundary distances
        boundary_dists = 0.5 - np.abs(centres)
        min_boundary = np.min(boundary_dists)
        
        # Pairwise distances
        if N > 1:
            # Using cdist is often faster than manual broadcasting for moderate N
            dists = cdist(centres, centres)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            min_pair_half = min_pair / 2.0
        else:
            min_pair_half = np.inf
        
        return min(min_boundary, min_pair_half)

    def physics_relax(centres, n_steps=200, dt=0.01, damping=0.95, k_repel=10.0, k_wall=100.0):
        """
        Perform a force-directed relaxation step.
        Forces:
        1. Repulsive force between spheres: F ~ 1/d^2 (or similar)
        2. Repulsive force from walls: F ~ (0.5 - |x|)^2 or linear
        """
        c = centres.copy()
        v = np.zeros_like(c)
        
        for step in range(n_steps):
            if check_time():
                break
                
            forces = np.zeros_like(c)
            
            # 1. Pairwise repulsion
            # Calculate distances
            dists = cdist(c, c)
            # Avoid division by zero
            dists[~np.isfinite(dists)] = 1e-10
            np.fill_diagonal(dists, 1.0) # Dummy value for diagonal
            
            # Force magnitude: k / d^2 (inverse square law approx)
            # We want to push them apart.
            # Only consider pairs close to each other to save time? No, N=50 is small.
            inv_d2 = 1.0 / (dists**2)
            
            # Vector differences
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            
            # Force on i from j is k * diff_ij / d^3 (direction of diff, magnitude 1/d^2)
            # F_ij = k * (c_i - c_j) / |c_i - c_j|^3
            # We sum forces on i.
            # diff is (N, N, 3).
            # inv_d2 is (N, N). We need inv_d3 for the vector part if we use 1/d^2 magnitude?
            # F = k * diff / d^3.
            inv_d3 = 1.0 / (dists**3)
            
            # Zero out diagonal
            inv_d3[np.diag_indices_from(inv_d3)] = 0.0
            
            # forces[i] += sum_j k * (c_i - c_j) / d_ij^3
            # (N, N, 1) * (N, N, 3) -> (N, N, 3)
            force_terms = k_repel * inv_d3[:, :, np.newaxis] * diff
            forces += np.sum(force_terms, axis=1)
            
            # 2. Wall repulsion
            # Distance to nearest wall
            dist_to_wall = 0.5 - np.abs(c)
            # If inside, dist > 0. Force pushes away from wall (towards center).
            # F_wall = k_wall * dist_to_wall (linear spring)
            # Direction: away from wall, i.e., towards 0.
            # c is positive -> wall at 0.5, force negative.
            # c is negative -> wall at -0.5, force positive.
            # So force is -sign(c) * k_wall * dist_to_wall?
            # Actually, potential V = 0.5 * k * dist^2. Force = -dV/dx.
            # If x > 0, dist = 0.5 - x. d(dist)/dx = -1. F = -k*dist*(-1) = k*dist.
            # Wait, if x is close to 0.5, dist is small. Force should push x down (negative).
            # Let's just use a simple penalty: if dist < margin, push back.
            # Simpler: F_wall = k_wall * (0.5 - |c|) * sign(c) ? No.
            # Let's use: F_wall = k_wall * (0.5 - |c|) * np.sign(c) is wrong direction?
            # If c=0.49, dist=0.01. We want force to decrease c. So force should be negative.
            # sign(0.49) = 1. So F = -k * dist * sign(c).
            # Let's check: c=0.49. F = -k * 0.01 * 1 = -0.01k. Correct.
            # c=-0.49. sign=-1. F = -k * 0.01 * (-1) = +0.01k. Correct.
            
            wall_forces = -k_wall * dist_to_wall * np.sign(c)
            # Add a small constant push to center if very close to wall to prevent sticking?
            # The linear force goes to 0 at the wall, which might cause sticking.
            # Let's add a constant repulsion if dist < 0.1
            close_to_wall = dist_to_wall < 0.1
            wall_forces[close_to_wall] += -np.sign(c[close_to_wall]) * k_wall * 0.1
            
            forces += wall_forces
            
            # Update velocity and position (Velocity Verlet / Langevin)
            v = v * damping + forces * dt
            c = c + v * dt
            
            # Clip to cube to ensure validity
            c = np.clip(c, -0.4999, 0.4999)
            
            # Optional: re-centering to prevent drift?
            # c -= np.mean(c, axis=0)
            # c = np.clip(c, -0.4999, 0.4999)
            
        return c
    
    def soft_min_score(centres_flat, beta=100.0):
        """
        Smooth approximation of the negative radius.
        Objective to minimize: -soft_min(r_i)
        where r_i are the constraints (pairwise/2, boundary).
        """
        centres = centres_flat.reshape(N, 3)
        if not np.all(np.isfinite(centres)):
            return 1e10
        
        # Boundary terms: 0.5 - |c_ik|
        boundary_terms = 0.5 - np.abs(centres)  # (N, 3)
        
        # Pairwise distance terms: |c_i - c_j| / 2
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        iu = np.triu_indices(N, k=1)
        pair_terms = dists[iu] / 2.0
        
        all_terms = np.concatenate([boundary_terms.flatten(), pair_terms])
        
        # Soft-min: -1/beta * log(sum(exp(-beta * x)))
        # To be stable, shift by max
        max_term = np.max(all_terms)
        shifted = -beta * (all_terms - max_term)
        # Clip to avoid overflow in exp
        shifted = np.clip(shifted, -500, 500)
        exp_terms = np.exp(shifted)
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0 or not np.isfinite(sum_exp):
            return 1e10
        soft_min_val = max_term - np.log(sum_exp) / beta
        
        return -soft_min_val

    def neg_score(centres_flat):
        """Negative exact score for direct methods."""
        centres = centres_flat.reshape(N, 3)
        if not np.all(np.isfinite(centres)):
            return 1e10
        if np.any(centres < -0.5) or np.any(centres > 0.5):
            return 1e10
        return -score_centres(centres)

    def halton_sequence(n, base):
        """Generate n points in the Halton sequence with given base."""
        result = []
        f = 1.0
        for i in range(1, n + 1):
            x = 0.0
            i_val = i
            while i_val > 0:
                f /= base
                x += f * (i_val % base)
                i_val //= base
            result.append(x)
        return np.array(result)
    
    def generate_halton_init(seed_offset=0):
        """Generate a Halton-based initial configuration."""
        bases = [(2, 3, 5), (3, 5, 7), (5, 7, 11), (2, 5, 7)]
        b = bases[seed_offset % len(bases)]
        h1 = halton_sequence(N, b[0])
        h2 = halton_sequence(N, b[1])
        h3 = halton_sequence(N, b[2])
        
        scale = 0.9
        centres = np.column_stack((
            h1 * scale - scale/2,
            h2 * scale - scale/2,
            h3 * scale - scale/2
        ))
        
        rng = np.random.default_rng(42 + seed_offset)
        centres += rng.normal(0, 0.015, size=(N, 3))
        centres = np.clip(centres, -0.49, 0.49)
        return centres
    
    def generate_fibonacci_init(seed_offset=0):
        """Generate a Fibonacci lattice-based initial configuration for maximal separation."""
        # Golden ratio conjugate
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        inv_phi = 1.0 / phi
        
        n = N
        i = np.arange(n)
        
        # Fibonacci lattice in [0,1]^3
        # x: uniform
        # y: fractional part of i * inv_phi
        # z: fractional part of i * inv_phi^2
        x = (i + 0.5) / n
        y = np.mod(i * inv_phi, 1.0)
        z = np.mod(i * inv_phi**2, 1.0)
        
        centres = np.column_stack((x, y, z))
        
        # Scale to fit within cube with some margin
        scale = 0.95
        centres = centres * scale - scale/2
        
        # Add small perturbation to break symmetry
        rng = np.random.default_rng(42 + seed_offset)
        centres += rng.normal(0, 0.01, size=(N, 3))
        centres = np.clip(centres, -0.49, 0.49)
        return centres
    
    def generate_random_init(seed):
        """Generate a random initial configuration."""
        rng = np.random.default_rng(seed)
        centres = rng.uniform(-0.45, 0.45, size=(N, 3))
        return centres
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_centres = None
    best_score = 0.0
    
    # Try multiple initializations - Fibonacci lattice first for better initial separation
    initializers = [
        ("fib0", lambda: generate_fibonacci_init(0)),
        ("fib1", lambda: generate_fibonacci_init(1)),
        ("halton0", lambda: generate_halton_init(0)),
        ("halton1", lambda: generate_halton_init(1)),
        ("halton2", lambda: generate_halton_init(2)),
        ("rand1", lambda: generate_random_init(1)),
        ("rand2", lambda: generate_random_init(2)),
    ]
    
    for name, gen_func in initializers:
        if check_time():
            break
        
        try:
            centres = gen_func()
        except Exception:
            continue
        
        current_score = score_centres(centres)
        current_centres = centres.copy()
        
        # Stage 1: L-BFGS-B with increasing beta
        for beta in [5.0, 15.0, 40.0, 100.0]:
            if check_time():
                break
            
            def smooth_obj(cf, b=beta):
                return soft_min_score(cf, beta=b)
            
            try:
                result = minimize(
                    smooth_obj,
                    current_centres.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 15}
                )
            except Exception:
                continue
            
            new_centres = result.x.reshape(N, 3)
            if not np.all(np.isfinite(new_centres)):
                continue
            new_score = score_centres(new_centres)
            if new_score > current_score:
                current_centres = new_centres
                current_score = new_score
        
        # Stage 2: High-beta polish
        for beta in [300.0, 800.0]:
            if check_time():
                break
            
            def smooth_obj_high(cf, b=beta):
                return soft_min_score(cf, beta=b)
            
            try:
                result = minimize(
                    smooth_obj_high,
                    current_centres.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 15}
                )
            except Exception:
                continue
            
            new_centres = result.x.reshape(N, 3)
            if not np.all(np.isfinite(new_centres)):
                continue
            new_score = score_centres(new_centres)
            if new_score > current_score:
                current_centres = new_centres
                current_score = new_score
        
        # Stage 2.5: Physics-based relaxation
        if not check_time():
            try:
                phys_centres = physics_relax(current_centres, n_steps=300, dt=0.005, damping=0.98, k_repel=5.0, k_wall=50.0)
                phys_score = score_centres(phys_centres)
                if phys_score > current_score:
                    current_centres = phys_centres
                    current_score = phys_score
            except Exception:
                pass
            
            # Re-run L-BFGS-B after physics to clean up
            if not check_time():
                try:
                    result = minimize(
                        lambda cf: soft_min_score(cf, beta=100.0),
                        current_centres.flatten(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 15}
                    )
                    new_centres = result.x.reshape(N, 3)
                    if np.all(np.isfinite(new_centres)):
                        new_score = score_centres(new_centres)
                        if new_score > current_score:
                            current_centres = new_centres
                            current_score = new_score
                except Exception:
                    pass

        # Stage 3: Nelder-Mead polish
        if not check_time():
            try:
                result = minimize(
                    neg_score,
                    current_centres.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                new_centres = result.x.reshape(N, 3)
                if np.all(np.isfinite(new_centres)):
                    new_score = score_centres(new_centres)
                    if new_score > current_score:
                        current_centres = new_centres
                        current_score = new_score
            except Exception:
                pass
        
        if current_score > best_score:
            best_centres = current_centres.copy()
            best_score = current_score
    
    # Local search: try perturbing individual points and relaxing
    if best_centres is not None and not check_time():
        rng = np.random.default_rng(99)
        for iteration in range(100):
            if check_time():
                break
            
            idx = rng.integers(0, N)
            perturb_scale = 0.02
            new_point = best_centres[idx] + rng.normal(0, perturb_scale, size=3)
            new_point = np.clip(new_point, -0.495, 0.495)
            
            trial = best_centres.copy()
            trial[idx] = new_point
            
            trial_score = score_centres(trial)
            
            # If perturbation improved score, accept
            if trial_score > best_score:
                best_centres = trial
                best_score = trial_score
            else:
                # Try physics relaxation on the perturbed state
                if not check_time():
                    try:
                        phys_trial = physics_relax(trial, n_steps=50, dt=0.005, damping=0.98, k_repel=5.0, k_wall=50.0)
                        phys_score = score_centres(phys_trial)
                        if phys_score > best_score:
                            best_centres = phys_trial
                            best_score = phys_score
                    except Exception:
                        pass
    
    # Final validation
    if best_centres is None:
        # Fallback
        best_centres = generate_fibonacci_init(0)
    
    final_score = score_centres(best_centres)
    if not np.isfinite(final_score) or final_score <= 0:
        best_centres = generate_fibonacci_init(0)
    
    return best_centres

# EVOLVE-BLOCK-END
