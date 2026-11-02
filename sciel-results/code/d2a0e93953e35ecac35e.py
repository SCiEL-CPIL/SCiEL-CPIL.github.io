import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 600
dx = (1.0 / 4.0 - (-1.0 / 4.0)) / N  # 1/1200

def _autoconv_max(heights, dx):
    """
    Compute the maximum of the autoconvolution of a step function.
    Uses FFT for efficiency.
    """
    heights = np.asarray(heights, dtype=float)
    n_fft = 2 * N
    h_padded = np.zeros(n_fft)
    h_padded[:N] = heights
    H = np.fft.fft(h_padded, n=n_fft)
    conv = np.fft.ifft(H * H).real[:2*N-1] * dx
    # The autoconvolution is symmetric. The maximum for non-negative functions 
    # is typically at lag 0, but we check the first half to be safe.
    return np.max(conv[:N])

def _build_symmetric_heights(params):
    """
    Construct a symmetric step function from parameters.
    params: [h_center, w_center, h_side, w_side, gap]
    
    Structure:
    - Central block of width w_center and height h_center at the middle.
    - Two side blocks of width w_side and height h_side, placed symmetrically.
    - 'gap' is the distance from the edge of the central block to the inner edge of the side blocks.
    
    This ensures the function is symmetric around 0.
    """
    h_c = max(0.0, params[0])
    w_c = int(np.clip(round(params[1]), 1, N))
    h_s = max(0.0, params[2])
    w_s = int(np.clip(round(params[3]), 0, N))
    gap = int(np.clip(round(params[4]), 0, N))
    
    heights = np.zeros(N)
    
    # Place central block
    # Center index is N/2. 
    # To make it symmetric, we need to handle odd/even widths carefully.
    # Let's define the center of the interval array as (N-1)/2.
    # Simpler approach: define positions relative to start.
    
    # Total width used by center + 2*gap + 2*side
    # We want the whole structure centered.
    # Let's calculate the start index for the central block.
    # If N is even, center is between N/2-1 and N/2.
    # Let's just place the central block centered at N/2.
    start_c = N // 2 - w_c // 2
    end_c = start_c + w_c
    # Clip to bounds
    start_c = max(0, start_c)
    end_c = min(N, end_c)
    if end_c > start_c:
        heights[start_c:end_c] += h_c
    
    # Place side blocks
    if w_s > 0:
        # Inner edge of central block is at start_c (left) and end_c (right)
        # Left side block:
        # Its right edge should be at start_c - gap.
        # So its left edge is at (start_c - gap) - w_s.
        end_l = start_c - gap
        start_l = end_l - w_s
        
        # Right side block:
        # Its left edge should be at end_c + gap.
        # So its right edge is at (end_c + gap) + w_s.
        start_r = end_c + gap
        end_r = start_r + w_s
        
        # Clip and assign
        if end_l > 0 and start_l >= 0:
            heights[start_l:end_l] += h_s
        elif start_l < 0 and end_l > 0:
            heights[0:end_l] += h_s # Truncated, though symmetry might be slightly broken by boundary, 
                                    # but for optimal solutions, they should fit inside.
            
        if end_r <= N and start_r < N:
            heights[start_r:end_r] += h_s
        elif start_r < N and end_r > N:
            heights[start_r:N] += h_s
            
    return heights

def _objective_sym(params):
    """
    Objective function for the symmetric block family.
    Returns max_autoconv / (int_f)^2
    """
    heights = _build_symmetric_heights(params)
    int_f = np.sum(heights) * dx
    if int_f < 1e-10:
        return 1e10
    max_conv = _autoconv_max(heights, dx)
    return max_conv / (int_f ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Parameters: [h_center, w_center, h_side, w_side, gap]
    # Bounds:
    # h: [0, 10]
    # w: [1, N]
    # gap: [0, N]
    
    # Define multiple diverse initial guesses
    # 1. Single central block (w_side=0)
    # 2. Two blocks separated (h_center=0)
    # 3. Three blocks (center + sides)
    # 4. Wide center, small sides
    
    initial_guesses = [
        [1.0, N//2, 0.0, 0, 0],       # Single central block
        [1.0, N//4, 1.0, N//4, N//10], # Three blocks, moderate gap
        [1.0, N//10, 1.0, N//3, N//20],# Small center, large sides
        [0.5, N//2, 2.0, N//4, N//5],  # High sides, lower center
        [1.0, N//3, 0.5, N//6, N//10], # Moderate all
    ]
    
    best_val = float('inf')
    best_params = initial_guesses[0].copy()
    
    for x0 in initial_guesses:
        if time.time() > deadline - 10:
            break
            
        try:
            # Nelder-Mead is good for this small dimensional space (5D) 
            # and handles the non-smoothness of integer rounding reasonably well
            # if the step sizes are small enough or if we use adaptive simplex.
            
            res = minimize(
                _objective_sym,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-5,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
                
        except Exception as e:
            # If optimization fails, keep the initial guess value
            val_x0 = _objective_sym(x0)
            if val_x0 < best_val:
                best_val = val_x0
                best_params = x0.copy()
            
    # Final polish: try a few random perturbations around the best found
    # to escape any tiny local minima caused by rounding
    if time.time() < deadline - 5:
        for _ in range(5):
            noise = np.random.normal(0, 0.1, size=5)
            # Scale noise for widths/gaps to be integer-ish
            noise[1] *= 5.0
            noise[3] *= 5.0
            noise[4] *= 5.0
            perturbed = best_params + noise
            # Ensure non-negative heights
            perturbed[0] = max(0, perturbed[0])
            perturbed[2] = max(0, perturbed[2])
            
            val_pert = _objective_sym(perturbed)
            if val_pert < best_val:
                best_val = val_pert
                best_params = perturbed
    
    # Construct final heights from best parameters
    heights = _build_symmetric_heights(best_params)
    
    # Ensure non-negative (should be by construction, but safe)
    heights = np.maximum(heights, 0.0)
    
    return list(heights)

if __name__ == "__main__":
    t0 = time.time()
    h = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    # Calculate value for debugging
    int_f = np.sum(h) * dx
    max_c = _autoconv_max(h, dx)
    val = max_c / (int_f**2)
    print(f"Value: {val}")
