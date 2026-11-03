"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def compute_hard_radii(centers):
        """Compute exact maximum radii for given centers."""
        centers = np.asarray(centers)
        n = len(centers)
        radii = np.zeros(n)
        
        for i in range(n):
            radii[i] = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = dist / 2.0
                if max_r < radii[i]:
                    radii[i] = max_r
                if max_r < radii[j]:
                    radii[j] = max_r
        return radii

    def compute_potential_and_grad(centers, k_rep=50.0, k_boundary=100.0):
        """
        Compute the potential energy and gradient for the Hamiltonian system.
        Potential includes:
        1. Pairwise repulsion (soft overlap penalty)
        2. Boundary springs
        """
        centers = np.asarray(centers)
        n = len(centers)
        potential = 0.0
        grad = np.zeros_like(centers)
        
        # Pairwise repulsion: for each pair, if they're too close, apply repulsive force
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dist = 1e-10
                    dx = 1e-10
                    dy = 0.0
                
                # Desired minimum distance is related to radii
                # Use a soft potential: V = k * max(0, r_min - dist)^2
                # where r_min is estimated from current configuration
                # For now, use a fixed reference distance
                r_ref = 0.08  # approximate minimum distance
                if dist < r_ref:
                    overlap = r_ref - dist
                    force_mag = k_rep * overlap / dist
                    grad[i, 0] += force_mag * dx
                    grad[i, 1] += force_mag * dy
                    grad[j, 0] -= force_mag * dx
                    grad[j, 1] -= force_mag * dy
                    potential += 0.5 * k_rep * overlap * overlap
        
        # Boundary springs
        for i in range(n):
            # Left boundary
            if centers[i, 0] < 0.05:
                force = k_boundary * (0.05 - centers[i, 0])
                grad[i, 0] += force
                potential += 0.5 * k_boundary * (0.05 - centers[i, 0])**2
            # Right boundary
            if centers[i, 0] > 0.95:
                force = k_boundary * (centers[i, 0] - 0.95)
                grad[i, 0] -= force
                potential += 0.5 * k_boundary * (centers[i, 0] - 0.95)**2
            # Bottom boundary
            if centers[i, 1] < 0.05:
                force = k_boundary * (0.05 - centers[i, 1])
                grad[i, 1] += force
                potential += 0.5 * k_boundary * (0.05 - centers[i, 1])**2
            # Top boundary
            if centers[i, 1] > 0.95:
                force = k_boundary * (centers[i, 1] - 0.95)
                grad[i, 1] -= force
                potential += 0.5 * k_boundary * (centers[i, 1] - 0.95)**2
        
        return potential, grad

    def hamiltonian_dynamics(centers_init, velocities_init, dt=0.001, steps=5000, damping=0.999):
        """
        Run Hamiltonian dynamics with symplectic (leapfrog) integrator.
        """
        centers = centers_init.copy()
        velocities = velocities_init.copy()
        n = len(centers)
        
        # Initial potential and gradient
        _, grad = compute_potential_and_grad(centers)
        
        for step in range(steps):
            # Leapfrog: half-step velocity, full-step position, half-step velocity
            # v += 0.5 * dt * F
            velocities += 0.5 * dt * (-grad)
            
            # Apply damping
            velocities *= damping
            
            # x += dt * v
            centers += dt * velocities
            
            # Clamp to [0.01, 0.99] to keep circles inside
            centers = np.clip(centers, 0.01, 0.99)
            
            # New gradient
            _, grad_new = compute_potential_and_grad(centers)
            
            # v += 0.5 * dt * F
            velocities += 0.5 * dt * (-grad_new)
            
            grad = grad_new
        
        return centers

    def local_refinement(centers, max_iter=200):
        """
        Local refinement using gradient-based optimization.
        """
        def objective_and_grad(x_flat):
            centers = x_flat.reshape(-1, 2)
            # Use a smooth objective: negative sum of min radii
            radii = compute_hard_radii(centers)
            # Add small penalty for non-uniformity to encourage packing
            obj = -np.sum(radii)
            
            # Numerical gradient
            eps = 1e-6
            grad = np.zeros_like(x_flat)
            for i in range(len(x_flat)):
                x_plus = x_flat.copy()
                x_plus[i] += eps
                obj_plus = objective_and_grad._eval(x_plus)
                x_minus = x_flat.copy()
                x_minus[i] -= eps
                obj_minus = objective_and_grad._eval(x_minus)
                grad[i] = (obj_plus - obj_minus) / (2 * eps)
            
            return obj, grad
        
        def eval_obj(x_flat):
            centers = x_flat.reshape(-1, 2)
            radii = compute_hard_radii(centers)
            return -np.sum(radii)
        
        objective_and_grad._eval = eval_obj
        
        x = centers.flatten()
        for _ in range(max_iter):
            obj, grad = objective_and_grad(x)
            # Simple gradient descent with adaptive step
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-8:
                break
            step_size = 0.001
            x_new = x - step_size * grad
            x_new = np.clip(x_new, 0.01, 0.99)
            
            obj_new = eval_obj(x_new)
            if obj_new < obj:
                x = x_new
            else:
                step_size *= 0.5
        
        return x.reshape(-1, 2)

    def initial_grid_centers():
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])

    def random_centers(seed=0):
        rng = np.random.RandomState(seed)
        centers = rng.uniform(0.1, 0.9, size=(N, 2))
        return centers

    best_result = None
    best_score = -np.inf
    best_centers = None

    # Multiple restarts with Hamiltonian dynamics
    seeds = [0, 1, 7, 13, 23, 37, 53, 71, 89, 42, 99, 123]
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        
        if seed == 0:
            centers = initial_grid_centers()
        else:
            centers = random_centers(seed)
        
        # Initialize velocities randomly
        rng = np.random.RandomState(seed + 1000)
        velocities = rng.normal(0, 0.01, size=centers.shape)
        
        # Run Hamiltonian dynamics
        t_start = time.time()
        centers_dyn = hamiltonian_dynamics(centers, velocities, dt=0.002, steps=3000, damping=0.998)
        t_dyn = time.time() - t_start
        
        # Local refinement
        if time.time() < deadline - 15:
            centers_ref = local_refinement(centers_dyn, max_iter=100)
        else:
            centers_ref = centers_dyn
        
        # Evaluate
        radii = compute_hard_radii(centers_ref)
        final_x = np.column_stack([centers_ref, radii])
        
        if is_valid(final_x, tol=1e-6):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_centers = centers_ref
                best_result = final_x
    
    # Final polish with more iterations if time allows
    if best_centers is not None and time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            rng = np.random.RandomState(int(time.time() * 1000) % 10000)
            perturbed = best_centers + rng.normal(0, 0.003, size=best_centers.shape)
            perturbed = np.clip(perturbed, 0.01, 0.99)
            
            velocities = rng.normal(0, 0.005, size=perturbed.shape)
            centers_dyn = hamiltonian_dynamics(perturbed, velocities, dt=0.001, steps=2000, damping=0.999)
            centers_ref = local_refinement(centers_dyn, max_iter=50)
            
            radii = compute_hard_radii(centers_ref)
            final_x = np.column_stack([centers_ref, radii])
            
            if is_valid(final_x, tol=1e-6):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_centers = centers_ref
                    best_result = final_x

    if best_result is None:
        centers = initial_grid_centers()
        radii = compute_hard_radii(centers)
        best_result = np.column_stack([centers, radii])
        
    return best_result

# EVOLVE-BLOCK-END
