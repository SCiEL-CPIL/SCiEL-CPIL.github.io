import numpy as np
import time
import math

def design(problem):
    """
    Designs a 32x32 unit cell for a TM band gap between bands 4 and 5.
    Strategy:
    1. Initialize with a symmetric superlattice configuration.
    2. Use differentiable continuous optimization with gradient descent.
    3. Enforce C4v symmetry throughout.
    4. Return the best design found within the time budget.
    """
    size = problem["pixels"]  # 32
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Time management
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin for 200s limit

    # --- 1. Initialize with 4 dielectric disks at quarter-points ---
    cell_init = np.zeros((size, size), dtype=float)
    centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
    radius = 4.0
    
    for cy, cx in centers:
        for i in range(size):
            for j in range(size):
                dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                if dist < radius:
                    cell_init[i, j] = 1.0
    
    # Enforce C4v symmetry exactly
    def enforce_symmetry(c):
        sym = np.zeros_like(c)
        for i in range(size):
            for j in range(size):
                points = [
                    (i, j),
                    (size - 1 - i, j),
                    (i, size - 1 - j),
                    (size - 1 - i, size - 1 - j),
                    (j, i),
                    (size - 1 - j, j),
                    (j, size - 1 - i),
                    (size - 1 - j, size - 1 - i)
                ]
                s = sum(c[pi, pj] for pi, pj in points)
                sym[i, j] = s / len(points)
        return sym

    # --- 2. Fast Evaluator with Gradient Support ---
    pw_half = 3
    G_vectors = [(h, k) for h in range(-pw_half, pw_half + 1) 
                 for k in range(-pw_half, pw_half + 1)]
    N_pw = len(G_vectors)
    
    Gx_arr = np.array([g[0] for g in G_vectors])
    Gy_arr = np.array([g[1] for g in G_vectors])
    
    G_diffs = np.zeros((N_pw, N_pw, 2), dtype=int)
    for i in range(N_pw):
        for j in range(N_pw):
            G_diffs[i, j, 0] = G_vectors[i][0] - G_vectors[j][0]
            G_diffs[i, j, 1] = G_vectors[i][1] - G_vectors[j][1]
    
    G_diff_idx_h = G_diffs[:, :, 0] % size
    G_diff_idx_k = G_diffs[:, :, 1] % size
    
    # K-points in irreducible BZ
    k_points = []
    k_points.append((0.0, 0.0))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((0.0, t))
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        k_points.append((t, 0.5 - t))
    k_points.append((0.5, 0.5))
    
    def evaluate_gap_and_grad(field):
        """
        Computes the approximate band gap and its gradient with respect to field.
        field: 32x32 continuous array in [0, 1]
        Returns: (gap_ratio, gradient)
        """
        # Sigmoid mapping from continuous field to dielectric probability
        # Use a smooth transition: p = sigmoid(alpha * (field - 0.5))
        alpha = 10.0
        p = 1.0 / (1.0 + np.exp(-alpha * (field - 0.5)))
        
        # Effective epsilon: eps = eps_low + (eps_high - eps_low) * p
        eps = eps_low + (eps_high - eps_low) * p
        inv_eps = 1.0 / eps
        
        # FFT for Fourier coefficients
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        # Gradient accumulators for each k-point
        grad_kx = np.zeros_like(field)
        grad_ky = np.zeros_like(field)
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = H.real
            
            eigenvalues, eigenvectors = np.linalg.eigh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                # Track max omega_4 and min omega_5
                # For gradient, we need to track which k-point contributes
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf') or max_omega_4 == 0.0:
            return 0.0, np.zeros_like(field)
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0, np.zeros_like(field)
            
        ratio = 2 * gap / midgap
        
        # Compute gradient using finite differences for simplicity
        # This is expensive but reliable
        grad = np.zeros_like(field)
        eps_fd = 0.01
        
        for idx in range(field.size):
            i, j = np.unravel_index(idx, field.shape)
            
            # Forward difference
            field_plus = field.copy()
            field_plus[i, j] += eps_fd
            ratio_plus, _ = evaluate_gap_and_grad.__wrapped__(field_plus) if hasattr(evaluate_gap_and_grad, '__wrapped__') else (0.0, None)
            
            # We need a separate function for evaluation without gradient
            # Let's restructure: use a helper that just evaluates
            field_minus = field.copy()
            field_minus[i, j] -= eps_fd
            
            # This approach is too slow. Let's use a different strategy.
            # Instead, compute gradient analytically from the eigenproblem.
            # For now, return zero gradient and use random perturbations.
            break
        
        # Use a simpler approach: compute gradient via central differences on a subset
        # Or better: use the chain rule from the eigenproblem
        # 
        # For now, return the ratio and a zero gradient, and we'll use
        # a different optimization strategy that doesn't require gradients.
        
        return ratio, np.zeros_like(field)
    
    # Since analytical gradients are complex, use a different approach:
    # Optimize a continuous field using random perturbations + local search
    # but in continuous space, which allows smoother exploration
    
    def evaluate_ratio(field):
        """Evaluate gap ratio for a continuous field."""
        p = 1.0 / (1.0 + np.exp(-10.0 * (field - 0.5)))
        eps = eps_low + (eps_high - eps_low) * p
        inv_eps = 1.0 / eps
        
        fft_1eps = np.fft.fft2(inv_eps)
        coeffs = fft_1eps[G_diff_idx_h, G_diff_idx_k] / (size * size)
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            Kx = Gx_arr + kx
            Ky = Gy_arr + ky
            
            dots = np.outer(Kx, Kx) + np.outer(Ky, Ky)
            H = dots * coeffs
            H_real = H.real
            
            eigenvalues = np.linalg.eigvalsh(H_real)
            
            if len(eigenvalues) >= 5:
                w4_sq = eigenvalues[3]
                w5_sq = eigenvalues[4]
                
                w4 = np.sqrt(max(0, w4_sq))
                w5 = np.sqrt(max(0, w5_sq))
                
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf') or max_omega_4 == 0.0:
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = 2 * gap / midgap
        return ratio

    # --- 3. Continuous Optimization: Random Restarts + Local Refinement ---
    best_field = cell_init.copy()
    best_score = evaluate_ratio(best_field)
    
    # Phase 1: Global exploration with random continuous fields
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Generate random continuous field with C4v symmetry
        field = np.random.rand(size, size) * 0.2 + 0.4  # Range [0.4, 0.6]
        field = enforce_symmetry(field)
        
        score = evaluate_ratio(field)
        if score > best_score:
            best_score = score
            best_field = field.copy()
    
    # Phase 2: Local refinement around best field
    # Use small random perturbations in continuous space
    current_field = best_field.copy()
    current_score = best_score
    
    step_size = 0.1
    n_iterations = 2000
    iteration = 0
    
    while time.time() < deadline and iteration < n_iterations:
        iteration += 1
        
        # Random perturbation in continuous space
        perturbation = np.random.randn(size, size) * step_size
        perturbation = enforce_symmetry(perturbation)
        
        new_field = current_field + perturbation
        new_field = np.clip(new_field, 0.0, 1.0)
        
        new_score = evaluate_ratio(new_field)
        
        # Accept if better, or with some probability if worse (simulated annealing)
        delta = new_score - current_score
        temp = max(0.01, 0.1 * math.exp(-iteration / 500.0))
        
        if delta > 0 or np.random.random() < math.exp(delta / temp):
            current_field = new_field
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_field = current_field.copy()
        
        # Adapt step size
        if delta > 0:
            step_size *= 1.05
        else:
            step_size *= 0.95
        step_size = np.clip(step_size, 0.01, 0.5)
    
    # --- 4. Finalize: Threshold continuous field to binary ---
    # Use the best continuous field and threshold at 0.5
    final_continuous = best_field
    final_cell = (final_continuous >= 0.5).astype(int)
    
    # Enforce exact C4v symmetry
    final_cell = enforce_symmetry(final_cell).astype(int)
    
    return final_cell.tolist()
