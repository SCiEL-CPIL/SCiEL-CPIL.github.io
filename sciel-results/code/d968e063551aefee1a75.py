import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 115.0  # 5s margin for final refinement and return
    
    # Precompute evaluation matrix for fast sup norm approximation
    # N=4096 is a good balance between speed and accuracy for search
    N = 4096
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    k = np.arange(24)
    E = np.exp(1j * np.outer(theta, k))  # Shape (N, 24)
    
    def eval_batch(coeffs_batch):
        """
        Evaluate sup norm for a batch of polynomials.
        coeffs_batch: (M, 24) array of coefficients.
        Returns: (M,) array of sup norms.
        """
        P = coeffs_batch @ E.T
        mags = np.abs(P)
        return np.max(mags, axis=1)

    def refine_sup_norm(coeffs):
        """
        High-precision sup norm calculation for the final candidate.
        """
        # Use a denser grid for initial peak finding
        N_ref = 20000
        theta_ref = np.linspace(0, 2 * np.pi, N_ref, endpoint=False)
        E_ref = np.exp(1j * np.outer(theta_ref, k))
        P = E_ref @ coeffs
        mags = np.abs(P)
        
        # Find top 10 peaks
        top_indices = np.argsort(mags)[-10:]
        max_val = 0.0
        
        for idx in top_indices:
            center = theta_ref[idx]
            width = 2 * np.pi / N_ref
            # Refine around the peak with a very fine grid
            # 200 points in the width of one coarse grid cell
            fine_theta = np.linspace(center - width, center + width, 200)
            fine_E = np.exp(1j * np.outer(fine_theta, k))
            fine_P = fine_E @ coeffs
            fine_mags = np.abs(fine_P)
            local_max = np.max(fine_mags)
            if local_max > max_val:
                max_val = local_max
        return max_val

    best_coeffs = None
    best_score = float('inf')
    
    # Generate initial candidates
    candidates = []
    
    # 1. Random starts
    for _ in range(20):
        candidates.append(np.random.choice([-1, 1], 24))
        
    # 2. Golay-Rudin-Shapiro (Seed)
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    candidates.append(np.array(p[:24]))
    
    # 3. Other structured patterns
    candidates.append(np.array([1] * 12 + [-1] * 12))
    candidates.append(np.array([1 if i % 2 == 0 else -1 for i in range(24)]))
    candidates.append(np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]))
    
    for c in candidates:
        if time.time() - start_time > deadline - 5:
            break
            
        current_c = c.copy()
        current_score = eval_batch(current_c.reshape(1, -1))[0]
        
        # Local Search with Kicks
        max_steps = 3000
        steps = 0
        
        while steps < max_steps and (time.time() - start_time < deadline - 5):
            steps += 1
            
            # Generate 24 single-bit flips
            flips = np.tile(current_c, (24, 1))
            flips[np.arange(24), np.arange(24)] *= -1
            
            scores = eval_batch(flips)
            min_score = np.min(scores)
            
            if min_score < current_score:
                # Accept improvement
                best_idx = np.argmin(scores)
                current_c = flips[best_idx]
                current_score = min_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_coeffs = current_c.copy()
            else:
                # Stuck in local minimum. Perform a "kick" (random perturbation)
                # Flip 2 random bits
                if steps % 10 == 0: # Do this occasionally
                    kick_indices = np.random.choice(24, 2, replace=False)
                    current_c[kick_indices] *= -1
                    current_score = eval_batch(current_c.reshape(1, -1))[0]
                    
                    # If the kick was very bad, maybe revert? 
                    # For now, just accept and continue. The search will find its way back if possible.
                    # To be safer, we can store the best local state and revert if stuck for too long.
                    # But simple SA-like acceptance is often enough for this size.
        
    # Final Refinement
    if best_coeffs is not None:
        # The evaluator will compute the score, but we ensure our internal best is the one returned.
        # We don't need to return the score, just the coefficients.
        return list(best_coeffs)
    else:
        # Fallback (should not happen)
        return [1] * 24

