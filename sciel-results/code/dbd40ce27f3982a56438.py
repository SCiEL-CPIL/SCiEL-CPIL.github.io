import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # ---------- helpers ----------

    def compute_energy(pts):
        """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) = -sum log(d)"""
        # Use pairwise distances via broadcasting
        # pts shape (N,3)
        dist_sq = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2.0 * (pts @ pts.T)
        dist_sq = np.maximum(dist_sq, 1e-14)
        dist = np.sqrt(dist_sq)
        i_upper, j_upper = np.triu_indices(N, k=1)
        d_pairs = dist[i_upper, j_upper]
        # Guard against zero distances
        d_pairs = np.maximum(d_pairs, 1e-14)
        return -np.sum(np.log(d_pairs))

    def compute_gradient(pts):
        """
        Gradient of E = -sum_{i<j} log(d_ij) w.r.t. each point on R^3.
        grad_k = sum_{j != k} (x_k - x_j) / d_kj^2
        """
        dist_sq = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2.0 * (pts @ pts.T)
        dist_sq = np.maximum(dist_sq, 1e-14)
        inv_d2 = 1.0 / dist_sq
        np.fill_diagonal(inv_d2, 0.0)
        row_sums = np.sum(inv_d2, axis=1)
        grad = pts * row_sums[:, None] - inv_d2 @ pts
        return grad

    def project_to_tangent(pts, grad):
        """Project gradient onto tangent space of sphere at each point."""
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        return grad - dot * pts

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return pts / norms

    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = normalize(pts)
        energy = compute_energy(pts)
        grad = compute_gradient(pts)
        grad_tangent = project_to_tangent(pts, grad)
        return energy, grad_tangent.flatten()

    # ---------- Spherical Harmonic Helpers ----------
    
    def spherical_harmonic_gradient(pts, coeffs, l_max):
        """
        Compute the gradient of a spherical harmonic field on the sphere.
        coeffs: array of shape (l_max+1, 2*l_max+1) or flattened.
        We use a simplified version: random harmonic perturbations.
        """
        # For simplicity and speed, we will generate random harmonic perturbations
        # by summing a few random gradients of low-degree harmonics.
        # This avoids implementing full SH basis and gradients.
        pass

    def harmonic_perturbation(pts, scale, l_max=8):
        """
        Perturb points using a random low-degree spherical harmonic field.
        This creates a structured, global deformation.
        """
        # Generate random coefficients for l=1 to l_max
        # We approximate the effect by summing random directions weighted by random harmonic-like patterns.
        # A true implementation would use SH basis. Here we use a simpler "coherent" noise.
        # Let's stick to a simpler, effective heuristic: 
        # Sum of a few random "dipole" and "quadrupole" fields.
        
        # Random dipole
        d1 = np.random.randn(3)
        d1 /= np.linalg.norm(d1)
        # Random quadrupole (symmetric traceless)
        q1 = np.random.randn(3, 3)
        q1 = (q1 + q1.T) / 2
        trace = np.trace(q1)
        q1 -= (trace / 3.0) * np.eye(3)
        
        # Perturbation: P' = P + alpha * ( (P . d1) d1 - P ) + beta * ( q1 @ P )
        # The first term is a dipole field gradient-like effect.
        # The second term is a quadrupole field.
        
        p_dipole = np.sum(pts * d1, axis=1, keepdims=True) * d1 - pts
        p_quad = pts @ q1.T
        
        noise = scale * (p_dipole + 0.5 * p_quad)
        
        new_pts = pts + noise
        return normalize(new_pts)

    # ---------- initializations ----------

    def fibonacci_spiral(offset=0.0):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + offset)
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return normalize(pts)

    def random_points():
        pts = np.random.randn(N, 3)
        return normalize(pts)

    def ring_initialization(num_rings=10):
        pts_list = []
        for ring in range(num_rings):
            z_val = -1.0 + 2.0 * ring / (num_rings - 1)
            r_val = np.sqrt(max(1.0 - z_val**2, 0.0))
            if r_val < 0.01:
                r_val = 0.01
            num_per_ring = N // num_rings
            for j in range(num_per_ring):
                angle = 2.0 * np.pi * j / num_per_ring + ring * 0.3
                pts_list.append([r_val * np.cos(angle), r_val * np.sin(angle), z_val])
        while len(pts_list) < N:
            pts_list.append(np.random.randn(3).tolist())
        pts = np.array(pts_list[:N])
        return normalize(pts)

    # ---------- L-BFGS optimization ----------

    from scipy.optimize import minimize

    def lbfgs_optimize(x0, maxiter=500):
        """Run L-BFGS-B and return (best_points, best_energy)."""
        try:
            result = minimize(
                lambda x: objective_and_grad(x)[0],
                x0,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
            )
            new_pts = normalize(result.x.reshape(N, 3))
            new_energy = compute_energy(new_pts)
            return new_pts, new_energy
        except Exception:
            return None, None

    # ---------- gradient descent with line search ----------

    def gradient_descent(pts, max_iters=5000, initial_step=0.005):
        """Adaptive-step gradient descent in tangent space."""
        current_energy = compute_energy(pts)
        step_size = initial_step
        min_step = 1e-12
        max_step = 0.05
        no_improve = 0

        for it in range(max_iters):
            if time.time() > deadline:
                break

            grad = compute_gradient(pts)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break

            grad_tangent = project_to_tangent(pts, grad)

            candidate = pts - step_size * grad_tangent
            candidate = normalize(candidate)
            cand_energy = compute_energy(candidate)

            if cand_energy < current_energy:
                pts = candidate
                current_energy = cand_energy
                no_improve = 0
                step_size = min(step_size * 1.1, max_step)
            else:
                step_size *= 0.7
                no_improve += 1
                if no_improve > 100:
                    if step_size < min_step:
                        break
                    step_size = 0.005
                    no_improve = 0

        return pts, current_energy

    # ---------- simulated annealing refinement ----------

    def simulated_annealing(pts, max_iters=3000, T_start=0.05, T_end=1e-6):
        """Simulated annealing on the sphere."""
        current_energy = compute_energy(pts)
        best_pts = pts.copy()
        best_energy = current_energy
        current_pts = pts.copy()

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Annealing schedule
            progress = it / max_iters
            T = T_start * (T_end / T_start) ** progress

            # Random perturbation
            n_perturb = np.random.randint(1, 5)  # perturb 1-4 points
            indices = np.random.choice(N, n_perturb, replace=False)
            noise = np.random.randn(n_perturb, 3) * 0.1
            candidate = current_pts.copy()
            candidate[indices] += noise
            candidate = normalize(candidate)

            cand_energy = compute_energy(candidate)
            delta = cand_energy - current_energy

            if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-12)):
                current_pts = candidate
                current_energy = cand_energy
                if cand_energy < best_energy:
                    best_energy = cand_energy
                    best_pts = candidate.copy()

        return best_pts, best_energy

    # ---------- main search ----------

    best_points = None
    best_energy = float('inf')

    # Initial set of starting points
    initializations = [
        fibonacci_spiral(0.0),
        fibonacci_spiral(0.3),
        fibonacci_spiral(0.7),
        ring_initialization(10),
        ring_initialization(14),
        random_points(),
        random_points(),
        random_points(),
    ]

    # Phase 1: L-BFGS from multiple starts
    for init_pts in initializations:
        if time.time() > deadline - 30:
            break

        pts, energy = lbfgs_optimize(init_pts.flatten(), maxiter=500)
        if pts is not None and energy < best_energy:
            best_energy = energy
            best_points = pts.copy()

    # Phase 2: Gradient descent polish on best
    if best_points is not None:
        if time.time() < deadline - 20:
            pts, energy = gradient_descent(best_points, max_iters=5000, initial_step=0.003)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Phase 3: More L-BFGS restarts from perturbed best
    if best_points is not None:
        for restart in range(8):
            if time.time() > deadline - 15:
                break

            # Use harmonic perturbation for structured diversity
            if restart % 2 == 0:
                noise_scale = 0.05
                perturbed = harmonic_perturbation(best_points, noise_scale)
            else:
                noise_scale = 0.02 + 0.01 * restart
                noise = np.random.randn(N, 3) * noise_scale
                perturbed = normalize(best_points + noise)
            
            pts, energy = lbfgs_optimize(perturbed.flatten(), maxiter=300)
            if pts is not None and energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Phase 4: Simulated annealing on best
    if best_points is not None:
        if time.time() < deadline - 10:
            pts, energy = simulated_annealing(best_points, max_iters=2000, T_start=0.03, T_end=1e-7)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Phase 5: Final gradient descent polish
    if best_points is not None:
        if time.time() < deadline - 3:
            pts, energy = gradient_descent(best_points, max_iters=3000, initial_step=0.001)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()

    # Final normalization and validity check
    if best_points is None:
        best_points = fibonacci_spiral(0.0)

    best_points = normalize(best_points)

    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_spiral(0.0)
        best_points = normalize(best_points)

    return best_points

# EVOLVE-BLOCK-END
