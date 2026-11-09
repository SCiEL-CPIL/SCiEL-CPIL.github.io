"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    N = 24
    
    # Precompute evaluation grid for fast sup-norm estimation
    # Use a sufficiently dense grid to distinguish near-optimal candidates
    num_points = 65536  # Power of 2 for fast FFT
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    # Z[k, j] = exp(1j * theta_j * k) for k=0..N-1, j=0..num_points-1
    # Shape: (N, num_points)
    k_indices = np.arange(N)
    Z = np.exp(1j * np.outer(k_indices, theta))  # (N, num_points)
    
    def sup_norm_fast(coeffs):
        """Compute sup norm of P(z) on the unit circle using dense grid."""
        # coeffs: shape (N,)
        # P(z_j) = sum_k coeffs[k] * z_j^k = coeffs @ Z[:, j]
        vals = Z.T @ coeffs  # shape (num_points,)
        return np.max(np.abs(vals))
    
    def refine_peak(coeffs, peak_theta, num_fine=200):
        """Refine the peak value near peak_theta using golden-section-like search."""
        best_val = 0.0
        best_theta = peak_theta
        
        # Multi-scale refinement
        scales = [1e-3, 1e-4, 1e-5, 1e-6]
        current_center = peak_theta
        current_radius = 1e-3
        
        for scale in scales:
            t_range = np.linspace(current_center - scale, current_center + scale, num_fine)
            z_fine = np.exp(1j * t_range)
            # Evaluate P at these points using Horner's method for speed
            # P(z) = c_0 + c_1*z + c_2*z^2 + ... + c_{N-1}*z^{N-1}
            # Build powers incrementally
            powers = np.ones((num_fine, N), dtype=complex)
            powers[:, 0] = 1.0
            for k in range(1, N):
                powers[:, k] = powers[:, k-1] * z_fine
            vals_fine = powers @ coeffs
            abs_fine = np.abs(vals_fine)
            local_max_idx = np.argmax(abs_fine)
            if abs_fine[local_max_idx] > best_val:
                best_val = abs_fine[local_max_idx]
                best_theta = t_range[local_max_idx]
            current_center = best_theta
            current_radius = scale * 0.5
        
        return best_val, best_theta
    
    def evaluate_refined(coeffs):
        """Evaluate score with peak refinement."""
        vals = Z.T @ coeffs
        abs_vals = np.abs(vals)
        max_idx = np.argmax(abs_vals)
        max_theta = theta[max_idx]
        refined_val, _ = refine_peak(coeffs, max_theta)
        return refined_val / np.sqrt(N)
    
    def evaluate_fast(coeffs):
        """Fast evaluation without refinement (for initial screening)."""
        return sup_norm_fast(coeffs) / np.sqrt(N)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Golay-Rudin-Shapiro polynomial
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    gsr = np.array(p[:N])
    candidates.append(gsr)
    
    # 2. Several random candidates with different seeds
    np.random.seed(42)
    for _ in range(50):
        r = np.array([1 if np.random.rand() < 0.5 else -1 for _ in range(N)])
        candidates.append(r)
    
    best_score = float('inf')
    best_coeffs = None
    
    # First pass: fast screening to find good candidates
    screened = []
    for cand in candidates:
        if time.time() > deadline - 10.0:
            break
        s = evaluate_fast(cand)
        screened.append((s, cand))
    
    # Sort by fast score and take top ones for detailed refinement
    screened.sort(key=lambda x: x[0])
    top_candidates = [cand for _, cand in screened[:15]]
    
    # Local Search (Hill Climbing) on top candidates with refined evaluation
    for i, cand in enumerate(top_candidates):
        if time.time() > deadline - 5.0:
            break
            
        current = cand.copy().astype(float)
        current_score = evaluate_refined(current)
        
        improved = True
        max_iterations = 30
        iteration = 0
        while improved and time.time() < deadline - 5.0 and iteration < max_iterations:
            improved = False
            iteration += 1
            # Try flipping each bit in random order
            indices = np.random.permutation(N)
            for k in indices:
                if time.time() > deadline - 5.0:
                    break
                # Flip bit k
                old_val = current[k]
                current[k] = -old_val
                new_score = evaluate_fast(current)  # Use fast eval during search
                
                if new_score < current_score - 1e-10:
                    current_score = new_score
                    improved = True
                else:
                    current[k] = old_val
            
        # Final refined evaluation
        final_score = evaluate_refined(current)
        if final_score < best_score:
            best_score = final_score
            best_coeffs = current.copy()
    
    # If we have time, do more random restarts from the best found
    if best_coeffs is not None and time.time() < deadline - 10.0:
        for restart in range(30):
            if time.time() > deadline - 5.0:
                break
            # Start from a perturbation of the best
            current = best_coeffs.copy().astype(float)
            # Flip 2-4 random bits
            num_flips = np.random.randint(2, 5)
            flip_indices = np.random.choice(N, num_flips, replace=False)
            for k in flip_indices:
                current[k] = -current[k]
            
            current_score = evaluate_fast(current)
            improved = True
            max_iterations = 20
            iteration = 0
            while improved and time.time() < deadline - 5.0 and iteration < max_iterations:
                improved = False
                iteration += 1
                indices = np.random.permutation(N)
                for k in indices:
                    if time.time() > deadline - 5.0:
                        break
                    old_val = current[k]
                    current[k] = -old_val
                    new_score = evaluate_fast(current)
                    
                    if new_score < current_score - 1e-10:
                        current_score = new_score
                        improved = True
                    else:
                        current[k] = old_val
                
                # Check if better than current best
                if current_score < best_score - 1e-10:
                    refined = evaluate_refined(current)
                    if refined < best_score:
                        best_score = refined
                        best_coeffs = current.copy()
    
    # If best_coeffs is None (shouldn't happen), return GRS
    if best_coeffs is None:
        return [int(x) for x in gsr]
        
    return [int(x) for x in best_coeffs]
# EVOLVE-BLOCK-END
