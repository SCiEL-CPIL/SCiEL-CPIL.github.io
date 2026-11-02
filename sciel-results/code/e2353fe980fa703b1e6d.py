"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 disjoint circles.
Strategy: Joint force-directed soft-body relaxation with smooth softplus penalties for 
overlaps and wall violations. Optimizes positions, radii, and aspect ratio simultaneously.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 148.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    rng = np.random.RandomState(12345)
    
    def softplus(x, beta=100.0):
        """Numerically stable softplus approximation."""
        return np.logaddexp(0, beta * x) / beta
    
    def compute_energy(params, beta=100.0):
        """Compute total energy: negative sum of radii + smooth penalties."""
        w = params[0]
        h = 2.0 - w
        
        if w < 0.3 or h < 0.3:
            return 1000.0
            
        centers = params[1:1+2*N].reshape(N, 2)
        radii = params[1+2*N:1+3*N]
        
        # Objective: maximize sum of radii (minimize negative)
        energy = -np.sum(radii)
        
        # Wall penalties (smooth)
        for i in range(N):
            x, y = centers[i]
            r = max(radii[i], 0.001)
            # Left wall
            energy += 10.0 * softplus(-x + r, beta)
            # Right wall
            energy += 10.0 * softplus(x - (w - r), beta)
            # Bottom wall
            energy += 10.0 * softplus(-y + r, beta)
            # Top wall
            energy += 10.0 * softplus(y - (h - r), beta)
        
        # Overlap penalties (smooth)
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                ri = max(radii[i], 0.001)
                rj = max(radii[j], 0.001)
                overlap = ri + rj - dist
                energy += 10.0 * softplus(overlap, beta)
        
        # Radius positivity penalty
        energy += 1.0 * np.sum(np.maximum(0, -radii))
        
        return energy
    
    def check_validity(w, centers, radii, tol=1e-7):
        """Check if configuration is strictly valid."""
        h = 2.0 - w
        if w < 0.1 or h < 0.1:
            return False
            
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            if r < 0:
                return False
            if x - r < -tol or x + r > w + tol:
                return False
            if y - r < -tol or y + r > h + tol:
                return False
        
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - tol:
                    return False
        
        return True
    
    def grid_init(w, h, seed=0):
        """Create initial grid configuration."""
        rng_local = np.random.RandomState(seed)
        side = int(np.ceil(np.sqrt(N)))
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        
        for idx in range(N):
            i = idx % side
            j = idx // side
            cx = (i + 0.5) * w / side + rng_local.randn() * 0.01
            cy = (j + 0.5) * h / side + rng_local.randn() * 0.01
            centers[idx] = [np.clip(cx, 0.05, w-0.05), np.clip(cy, 0.05, h-0.05)]
            radii[idx] = 0.5 * min(w/side, h/side) * 0.95
        
        return centers, radii
    
    def run_force_directed(w_init, centers_init, radii_init, max_iter=5000, beta_start=50.0, beta_end=500.0):
        """Run force-directed relaxation with annealing of softness parameter."""
        w = w_init
        centers = centers_init.copy()
        radii = radii_init.copy()
        
        best_energy = compute_energy(np.concatenate([[w], centers.flatten(), radii]), beta_start)
        best_w, best_centers, best_radii = w, centers.copy(), radii.copy()
        
        # Adaptive step size
        step = 0.01
        no_improve = 0
        max_no_improve = 50
        
        for it in range(max_iter):
            if time.time() > DEADLINE:
                break
            
            # Anneal beta
            progress = it / max_iter
            beta = beta_start * (beta_end / beta_start) ** progress
            
            # Compute forces (numerical gradient approximation for efficiency)
            eps = 1e-6
            forces_w = np.zeros(1)
            forces_c = np.zeros((N, 2))
            forces_r = np.zeros(N)
            
            # Only compute gradient for a random subset if too slow
            use_subset = (it > 1000 and N > 15)
            if use_subset:
                subset = rng.choice(N, size=min(N, 7), replace=False)
            else:
                subset = range(N)
            
            # Gradient for w
            base_params = np.concatenate([[w], centers.flatten(), radii])
            base_energy = compute_energy(base_params, beta)
            
            p_plus = base_params.copy()
            p_plus[0] += eps
            e_plus = compute_energy(p_plus, beta)
            forces_w[0] = (e_plus - base_energy) / eps
            
            # Gradients for centers (subset)
            for i in subset:
                for d in range(2):
                    p_plus = base_params.copy()
                    p_plus[1 + 2*i + d] += eps
                    e_plus = compute_energy(p_plus, beta)
                    forces_c[i, d] = (e_plus - base_energy) / eps
            
            # Gradients for radii (subset)
            for i in subset:
                p_plus = base_params.copy()
                p_plus[1 + 2*N + i] += eps
                e_plus = compute_energy(p_plus, beta)
                forces_r[i] = (e_plus - base_energy) / eps
            
            # Update parameters with adaptive step
            w -= step * forces_w[0]
            w = np.clip(w, 0.3, 1.7)
            
            for i in subset:
                centers[i, 0] -= step * forces_c[i, 0]
                centers[i, 1] -= step * forces_c[i, 1]
                centers[i, 0] = np.clip(centers[i, 0], 0.01, w - 0.01)
                centers[i, 1] = np.clip(centers[i, 1], 0.01, (2.0-w) - 0.01)
                radii[i] -= step * forces_r[i]
                radii[i] = max(radii[i], 0.001)
            
            # Evaluate
            current_params = np.concatenate([[w], centers.flatten(), radii])
            current_energy = compute_energy(current_params, beta)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_w, best_centers, best_radii = w, centers.copy(), radii.copy()
                no_improve = 0
                step = min(step * 1.05, 0.05)
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    step *= 0.7
                    no_improve = 0
                    if step < 1e-5:
                        break
            
            # Add small random perturbation to escape local minima
            if it % 100 == 0 and it > 0:
                pert_idx = rng.randint(N)
                centers[pert_idx, 0] += rng.randn() * 0.02
                centers[pert_idx, 1] += rng.randn() * 0.02
                centers[pert_idx, 0] = np.clip(centers[pert_idx, 0], 0.01, w - 0.01)
                centers[pert_idx, 1] = np.clip(centers[pert_idx, 1], 0.01, (2.0-w) - 0.01)
        
        return best_w, best_centers, best_radii
    
    # Try multiple aspect ratios with force-directed relaxation
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.7, 2.0, 0.5, 1.3, 0.9, 1.1]
    
    best_score = 0.0
    best_result = None
    
    for ar_idx, ar in enumerate(aspect_ratios):
        if time.time() > DEADLINE - 5:
            break
        
        w_init = ar * 2.0 / (ar + 1.0)
        h_init = 2.0 - w_init
        w_init = np.clip(w_init, 0.3, 1.7)
        h_init = 2.0 - w_init
        
        centers_init, radii_init = grid_init(w_init, h_init, seed=ar_idx * 13 + 7)
        
        # Run force-directed relaxation
        w_opt, centers_opt, radii_opt = run_force_directed(
            w_init, centers_init, radii_init, 
            max_iter=3000, beta_start=30.0, beta_end=300.0
        )
        
        # Validate
        if check_validity(w_opt, centers_opt, radii_opt):
            score = np.sum(radii_opt)
            if score > best_score:
                best_score = score
                best_result = (w_opt, centers_opt.copy(), radii_opt.copy())
        
        # Also try SLSQP polish on the best found so far
        if best_result is not None and time.time() < DEADLINE - 15:
            w_best, c_best, r_best = best_result
            h_best = 2.0 - w_best
            
            def slsqp_objective(params):
                w = params[0]
                h = 2.0 - w
                if w < 0.3 or h < 0.3:
                    return 1000.0
                centers = params[1:1+2*N].reshape(N, 2)
                radii = params[1+2*N:1+3*N]
                
                energy = -np.sum(radii)
                
                # Hard constraint penalties (smooth)
                for i in range(N):
                    x, y = centers[i]
                    r = max(radii[i], 0.001)
                    energy += 50.0 * softplus(-x + r, 50.0)
                    energy += 50.0 * softplus(x - (w - r), 50.0)
                    energy += 50.0 * softplus(-y + r, 50.0)
                    energy += 50.0 * softplus(y - (h - r), 50.0)
                
                for i in range(N):
                    for j in range(i+1, N):
                        dx = centers[i, 0] - centers[j, 0]
                        dy = centers[i, 1] - centers[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        overlap = radii[i] + radii[j] - dist
                        energy += 50.0 * softplus(overlap, 50.0)
                
                return energy
            
            x0 = np.concatenate([[w_best], c_best.flatten(), r_best.flatten()])
            bounds = [(0.3, 1.7)]
            for i in range(N):
                bounds.append((0.0, w_best))
                bounds.append((0.0, h_best))
                bounds.append((0.001, min(w_best, h_best)/2))
            
            try:
                result = minimize(slsqp_objective, x0, method='SLSQP', bounds=bounds,
                                 options={'maxiter': 200, 'ftol': 1e-8})
                if result.fun < 100.0:
                    w_slq = result.x[0]
                    c_slq = result.x[1:1+2*N].reshape(N, 2)
                    r_slq = result.x[1+2*N:1+3*N]
                    
                    if check_validity(w_slq, c_slq, r_slq):
                        score = np.sum(r_slq)
                        if score > best_score:
                            best_score = score
                            best_result = (w_slq, c_slq.copy(), r_slq.copy())
            except Exception:
                pass
    
    # Final validation and construction
    if best_result is not None:
        w, centers, radii = best_result
        h = 2.0 - w
        
        # Final clip to ensure strict validity
        for i in range(N):
            radii[i] = max(radii[i], 0.001)
            centers[i, 0] = np.clip(centers[i, 0], radii[i], w - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], h - radii[i])
        
        # Re-validate after clipping
        if not check_validity(w, centers, radii, tol=1e-6):
            # Fallback: reduce radii slightly
            for i in range(N):
                radii[i] *= 0.999
                centers[i, 0] = np.clip(centers[i, 0], radii[i], w - radii[i])
                centers[i, 1] = np.clip(centers[i, 1], radii[i], h - radii[i])
        
        result_array = np.zeros((N, 3))
        for i in range(N):
            result_array[i] = [centers[i, 0], centers[i, 1], radii[i]]
        return result_array
    else:
        # Fallback: grid packing
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        result_array = np.zeros((N, 3))
        for idx in range(N):
            i = idx % side
            j = idx // side
            r = 0.5 / side - 0.001
            x = (i + 0.5) * w / side
            y = (j + 0.5) * h / side
            result_array[idx] = [x, y, r]
        return result_array
