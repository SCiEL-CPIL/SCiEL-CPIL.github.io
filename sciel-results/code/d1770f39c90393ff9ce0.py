import time
import numpy as np

N = 600
DEADLINE = 140.0  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of autoconvolution."""
    conv = np.convolve(f, f, mode='full')
    s = np.sum(f)
    if s == 0:
        return np.inf
    return np.max(conv) / (s * s)

def parametric_to_step(coeffs, n_harmonics):
    """Convert cosine coefficients to non-negative step function.
    
    f[k] = base + sum_{j=1}^{n_harmonics} coeffs[j] * cos(2*pi*j*k/N)
    Clipped to non-negative.
    """
    k = np.arange(N)
    f = np.full(N, coeffs[0])
    for j in range(1, n_harmonics + 1):
        f += coeffs[j] * np.cos(2 * np.pi * j * k / N)
    return np.maximum(f, 0.0)

def objective_parametric(coeffs, n_harmonics):
    """Objective: normalized max autoconvolution of parametric function."""
    f = parametric_to_step(coeffs, n_harmonics)
    return compute_score(f)

def local_refinement(f, max_iter=200, tol=1e-8):
    """Local refinement via coordinate descent on individual bins."""
    f = f.copy()
    current_score = compute_score(f)
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE:
            break
        improved = False
        for i in range(N):
            old_val = f[i]
            # Try a few perturbations
            for delta in [0.1, 0.01, 0.001, -0.1, -0.01, -0.001]:
                new_val = old_val + delta
                if new_val < 0:
                    continue
                f[i] = new_val
                new_score = compute_score(f)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                f[i] = old_val
        if not improved or iteration % 20 == 19:
            if not improved:
                break
    return f, current_score

def spectral_coordinate_descent(n_harmonics=10, max_iter=500):
    """Optimize using coordinate descent on cosine coefficients."""
    # Initialize with constant function
    coeffs = np.zeros(n_harmonics + 1)
    coeffs[0] = 1.0
    
    best_coeffs = coeffs.copy()
    best_score = objective_parametric(coeffs, n_harmonics)
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        improved = False
        for j in range(n_harmonics + 1):
            old_coeff = coeffs[j]
            # Try perturbations
            for delta in [0.1, 0.01, 0.001, 0.0001, -0.1, -0.01, -0.001, -0.0001]:
                new_coeff = old_coeff + delta
                coeffs[j] = new_coeff
                new_score = objective_parametric(coeffs, n_harmonics)
                if new_score < best_score:
                    best_score = new_score
                    best_coeffs = coeffs.copy()
                    improved = True
                    break
                coeffs[j] = old_coeff
        
        if not improved:
            break
    
    f = parametric_to_step(best_coeffs, n_harmonics)
    return f, best_score

def construct():
    start_time = time.time()
    
    # Start with constant function
    f = np.ones(N)
    best_score = compute_score(f)
    best_f = f.copy()
    
    # Try spectral coordinate descent with increasing harmonics
    for n_harm in [5, 10, 15]:
        if time.time() > DEADLINE:
            break
        f_spectral, score_spectral = spectral_coordinate_descent(n_harm, max_iter=300)
        if score_spectral < best_score:
            best_score = score_spectral
            best_f = f_spectral.copy()
    
    # Local refinement on best found
    if time.time() < DEADLINE:
        f_refined, score_refined = local_refinement(best_f, max_iter=100)
        if score_refined < best_score:
            best_score = score_refined
            best_f = f_refined.copy()
    
    return list(best_f)

if __name__ == "__main__":
    f = construct()
    score = compute_score(np.array(f))
    print(f"Score: {score}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
