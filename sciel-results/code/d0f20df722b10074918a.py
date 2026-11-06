import numpy as np
import time

def _supnorm(c):
    """
    Compute the sup norm of the polynomial with coefficients c on the unit circle.
    """
    # Dense sampling
    n_samples = 20000
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    z = np.exp(1j * theta)
    
    k = np.arange(len(c))
    Z = z[:, None] ** k[None, :]
    vals = Z @ c
    max_val = np.max(np.abs(vals))
    
    # Refine peaks
    abs_vals = np.abs(vals)
    idx = np.argsort(abs_vals)[::-1]
    
    best_refined = max_val
    for i in idx[:10]:
        delta = 2 * np.pi / n_samples
        a = theta[i] - delta
        b = theta[i] + delta
        
        gr = (np.sqrt(5) - 1) / 2
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        
        def eval_at_angle(ang):
            zz = np.exp(1j * ang)
            powers = zz ** k
            val = np.sum(c * powers)
            return np.abs(val)
            
        fc = eval_at_angle(c_pt)
        fd = eval_at_angle(d_pt)
        
        for _ in range(50):
            if fc < fd:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = eval_at_angle(c_pt)
            else:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = eval_at_angle(d_pt)
                
        refined_val = max(fc, fd)
        if refined_val > best_refined:
            best_refined = refined_val
            
    return best_refined

def construct():
    # Precompute Z for fast sup-norm
    n_samples_fast = 4000
    theta_fast = np.linspace(0, 2 * np.pi, n_samples_fast, endpoint=False)
    z_fast = np.exp(1j * theta_fast)
    k_arr = np.arange(24)
    Z_fast = z_fast[:, None] ** k_arr[None, :]
    
    def _fast_supnorm(c):
        vals = Z_fast @ c
        return np.max(np.abs(vals))
    
    # Initial guess: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=float)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = _fast_supnorm(best_coeffs)
    
    # Hill climbing with vectorized single-flip evaluation
    # For each bit i, flipping changes P(z) by -2*c_i*z^i
    # We can compute all 24 flipped scores at once
    
    def _all_single_flip_scores(current):
        """Compute sup-norm for all 24 single-bit flips of current."""
        scores = np.zeros(24)
        for i in range(24):
            flipped = current.copy()
            flipped[i] = -flipped[i]
            scores[i] = _fast_supnorm(flipped)
        return scores
    
    # Simulated annealing
    np.random.seed(42)
    current = best_coeffs.copy()
    current_score = best_score
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.999
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Pick a random bit to flip
        i = np.random.randint(24)
        current[i] = -current[i]
        new_score = _fast_supnorm(current)
        
        # Accept if better, or with probability exp(-delta/T)
        delta = new_score - current_score
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, T_min)):
            if new_score < current_score:
                current_score = new_score
                no_improve_count = 0
                if new_score < best_score:
                    best_score = new_score
                    best_coeffs = current.copy()
            else:
                no_improve_count += 1
        else:
            current[i] = -current[i]  # Revert
            no_improve_count += 1
        
        T = max(T * cooling_rate, T_min)
        
        if no_improve_count > max_no_improve:
            # Restart from best or random
            if np.random.rand() < 0.5:
                current = best_coeffs.copy()
            else:
                current = np.random.choice([-1, 1], size=24)
            current_score = _fast_supnorm(current)
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current.copy()
            no_improve_count = 0
            T = 0.05
    
    # Final refinement
    return best_coeffs.tolist()
