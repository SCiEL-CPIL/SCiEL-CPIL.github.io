import numpy as np
import time

def _evaluate_sup_norm(coeffs, n_grid=10000):
    """
    Compute the sup norm of the polynomial P(z) = sum c_k z^k for |z|=1.
    Uses dense sampling and golden-section refinement on the highest peaks.
    """
    n = len(coeffs)
    coeffs = np.array(coeffs, dtype=np.float64)
    
    # Dense sampling
    thetas = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    z = np.exp(1j * thetas)
    
    # Vectorized evaluation: P(z_j) = sum_k c_k z_j^k
    # We can use matrix multiplication: coeffs @ z_powers
    # z_powers[k, j] = z_j^k
    # To save memory and time for small n (24), we can compute iteratively or use outer product logic.
    # z_j^k = exp(i * k * theta_j)
    # Let's build the power matrix. n=24, n_grid=10000. 240k complex numbers. Fine.
    k_vals = np.arange(n)
    # Shape: (n, n_grid)
    powers = np.exp(1j * np.outer(k_vals, thetas))
    p_vals = coeffs @ powers
    magnitudes = np.abs(p_vals)
    
    # Identify top peaks for refinement
    # We want to find the max. Let's take the top 5 indices.
    top_indices = np.argsort(magnitudes)[-5:]
    
    max_val = 0.0
    for idx in top_indices:
        theta_start = thetas[idx]
        # Refine around theta_start
        # Golden section search on [theta_start - dtheta, theta_start + dtheta]
        dtheta = 2 * np.pi / n_grid
        a = theta_start - dtheta
        b = theta_start + dtheta
        
        # Function to evaluate magnitude at a single theta
        def f_mag(theta):
            z_theta = np.exp(1j * theta)
            # Horner's rule is efficient for single point, but coeffs are small.
            # Or just sum c_k * z_theta^k
            # Since we only do this a few times, speed is less critical than accuracy.
            p_val = 0.0
            z_power = 1.0
            for c in coeffs:
                p_val += c * z_power
                z_power *= z_theta
            return np.abs(p_val)
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2  # 0.618...
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        f_c = f_mag(c_pt)
        f_d = f_mag(d_pt)
        
        for _ in range(50):  # ~50 iterations gives very high precision
            if f_c < f_d:
                b = d_pt
                d_pt = c_pt
                f_d = f_c
                c_pt = b - gr * (b - a)
                f_c = f_mag(c_pt)
            else:
                a = c_pt
                c_pt = d_pt
                f_c = f_d
                d_pt = a + gr * (b - a)
                f_d = f_mag(d_pt)
                
            if abs(f_c - f_d) < 1e-12:
                break
        
        current_max = max(f_c, f_d)
        if current_max > max_val:
            max_val = current_max
            
    return max_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n_coeffs = 24
    sqrt_n = np.sqrt(n_coeffs)
    
    # Initial best solution: Golay-Rudin-Shapiro truncated
    def generate_grs(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    best_coeffs = np.array(generate_grs(n_coeffs), dtype=np.float64)
    # Calculate initial score
    # Use a faster evaluation for the search loop, reserve high-precision for final check
    # For search, a grid of 2048 is usually sufficient to guide the search.
    n_grid_search = 2048
    thetas_search = np.linspace(0, 2 * np.pi, n_grid_search, endpoint=False)
    z_search = np.exp(1j * thetas_search)
    k_vals = np.arange(n_coeffs)
    powers_search = np.exp(1j * np.outer(k_vals, thetas_search))
    
    def fast_score(coeffs):
        # coeffs is a numpy array of shape (n_coeffs,)
        p_vals = coeffs @ powers_search
        return np.max(np.abs(p_vals))
    
    best_score = fast_score(best_coeffs)
    
    # Simulated Annealing
    # Parameters
    T_start = 5.0
    T_end = 0.01
    cooling_rate = 0.999
    T = T_start
    
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    # To speed up, we pre-compute the powers matrix.
    # Flipping a coefficient c_i by 2 (from -1 to 1 or vice versa) changes P(z) by 2 * c_i * z^i.
    # So we can update the polynomial values incrementally.
    
    # Let's maintain the current polynomial values on the grid
    current_p_vals = current_coeffs @ powers_search
    
    while time.time() < deadline and T > T_end:
        T *= cooling_rate
        
        # Random flip
        idx = np.random.randint(0, n_coeffs)
        old_c = current_coeffs[idx]
        new_c = -old_c
        delta = new_c - old_c  # +/- 2
        
        # Update p_vals incrementally
        # p_vals_new = p_vals_old + delta * z^idx
        z_idx = z_search ** idx
        new_p_vals = current_p_vals + delta * z_idx
        new_score = np.max(np.abs(new_p_vals))
        
        delta_score = new_score - current_score
        
        if delta_score < 0 or np.random.random() < np.exp(-delta_score / T):
            # Accept
            current_coeffs[idx] = new_c
            current_p_vals = new_p_vals
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs.copy()
        else:
            # Reject
            pass
            
    # Final high-precision evaluation of the best candidate found
    # We should check if we have time for a few high-precision checks of the top candidates
    # But we only kept one best. Let's do one final precise check.
    
    # It's possible the fast score is slightly off from the true sup norm due to grid resolution.
    # The task asks for the score computed by dense sampling and golden-section refinement.
    # The evaluator does this. We just need to return the coefficients.
    # However, to ensure we are returning the *best* coefficients, we might want to do a local polish
    # using the high-precision metric if time permits.
    
    if time.time() < deadline - 10.0:
        # Do a quick local search using the high-precision metric on the best candidate
        # This is expensive, so we only do a few iterations.
        precise_coeffs = best_coeffs.copy()
        precise_score = _evaluate_sup_norm(precise_coeffs)
        
        # Try flipping each coefficient once to see if it improves the precise score
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(n_coeffs):
                trial = precise_coeffs.copy()
                trial[i] = -trial[i]
                trial_score = _evaluate_sup_norm(trial)
                if trial_score < precise_score:
                    precise_coeffs = trial
                    precise_score = trial_score
                    improved = True
    
    # Return as list of ints (+1 or -1)
    return [int(c) for c in best_coeffs]

if __name__ == "__main__":
    # Self-test
    coeffs = construct()
    print(coeffs)
    score = _evaluate_sup_norm(coeffs) / np.sqrt(24)
    print(f"Score: {score}")
