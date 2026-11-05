import numpy as np
import time

# Precompute for evaluation
N_SAMPLES = 1024
THETAS = np.arange(N_SAMPLES) / N_SAMPLES * 2 * np.pi
Z = np.exp(1j * THETAS)
# Powers matrix: powers[k, j] = Z[j]^k
# We want coeffs @ powers.T to get P(Z)
POWERS = np.vander(Z, N=24, increasing=True).T # Shape (1024, 24)
ARANGE_24 = np.arange(24)

def eval_sup_norm(coeffs):
    """
    Evaluate the sup norm of the polynomial with given coefficients.
    coeffs: array of 24 floats (+1 or -1)
    """
    # Fast evaluation on grid
    vals = POWERS @ coeffs
    mags = np.abs(vals)
    
    # Find top 5 peaks
    # np.argpartition is faster for finding top k
    top_indices = np.argpartition(mags, -5)[-5:]
    
    max_val = 0.0
    for idx in top_indices:
        # Refine around this peak
        # Interval is [theta[idx-1], theta[idx+1]]
        # Handle wrap-around
        if idx == 0:
            left = THETAS[-1] - 2*np.pi
            right = THETAS[1]
        elif idx == N_SAMPLES - 1:
            left = THETAS[-2]
            right = THETAS[0] + 2*np.pi
        else:
            left = THETAS[idx-1]
            right = THETAS[idx+1]
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        a = left
        b = right
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        # Function to evaluate magnitude at angle theta
        def mag_at(theta):
            z_powers = np.exp(1j * theta * ARANGE_24)
            p_val = np.dot(coeffs, z_powers)
            return np.abs(p_val)
            
        fc = mag_at(c)
        fd = mag_at(d)
        
        for _ in range(50): # 50 iterations is plenty for 1e-9 accuracy
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = mag_at(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = mag_at(d)
                
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val

def local_search(start_coeffs):
    current = start_coeffs.copy()
    current_val = eval_sup_norm(current)
    improved = True
    while improved:
        improved = False
        for i in range(24):
            new_coeffs = current.copy()
            new_coeffs[i] *= -1
            new_val = eval_sup_norm(new_coeffs)
            if new_val < current_val - 1e-12:
                current = new_coeffs
                current_val = new_val
                improved = True
    return current, current_val

def construct():
    start_time = time.time()
    deadline = 110.0
    
    best_coeffs = None
    best_val = float('inf')
    
    # Initial guess: Golay-Rudin-Shapiro
    p, q = np.array([1.0]), np.array([1.0])
    while len(p) < 24:
        p, q = np.concatenate([p, q]), np.concatenate([p, -q])
    grs = p[:24].astype(float)
    
    # Try GRS first
    coeffs, val = local_search(grs)
    if val < best_val:
        best_val = val
        best_coeffs = coeffs
    
    # Random restarts
    while time.time() - start_time < deadline:
        rand_coeffs = np.random.choice([-1.0, 1.0], 24)
        coeffs, val = local_search(rand_coeffs)
        if val < best_val:
            best_val = val
            best_coeffs = coeffs
            # If we find the theoretical optimum, we can break
            # Optimum sup_norm is approx 6.164
            if best_val < 6.165:
                break
                
    return best_coeffs.astype(int).tolist()
