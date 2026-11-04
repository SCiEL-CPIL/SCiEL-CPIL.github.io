import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit
    
    # Grid points
    # The problem states "equal intervals of [-1/4, 1/4]".
    # This usually means the domain is divided into N intervals, or there are N points.
    # "Choose 50 nonnegative heights" implies 50 values.
    # Let's assume the heights correspond to bins or points in that interval.
    # The convolution of discrete heights usually implies the heights are samples.
    # If we treat them as a piecewise constant function on N sub-intervals of [-1/4, 1/4],
    # the convolution will be on a grid of 2N-1 points.
    
    # Let's define the shape parameter alpha
    # We want to maximize R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    best_alpha = 1.0
    best_ratio = 0.0
    best_h = np.ones(N)
    
    # Search space for alpha
    # Alpha controls the concentration. 
    # Alpha=0 is uniform. Alpha -> inf is a delta function.
    # We know uniform is 1.0. Delta function: h=[0,..,1,0..]. 
    # ||h||_2^2 = 1, ||h||_1 = 1. 
    # Convolution of delta is delta. ||f||_inf = 1. R = 1/1 = 1.
    # So extremes are 1.0. There must be a sweet spot in between.
    
    alphas = np.linspace(0.1, 10.0, 200)
    
    for alpha in alphas:
        if time.time() - start_time > timeout:
            break
        
        # Create shape concentrated on one side
        # t_i from 0 to 1
        t = np.arange(N, dtype=float) / (N - 1)
        h = t ** alpha
        
        # Ensure non-negative and handle potential zeros if alpha > 0 and t[0]=0
        # If alpha > 0, h[0] = 0. This is allowed (nonnegative).
        
        # Compute norms
        l1 = np.sum(h)
        if l1 == 0:
            continue
        l2_sq = np.sum(h**2)
        
        # Compute autoconvolution
        # np.convolve with mode='full' gives length 2N-1
        f = np.convolve(h, h, mode='full')
        f_inf = np.max(f)
        
        if f_inf == 0:
            continue
            
        ratio = l2_sq / (l1 * f_inf)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_alpha = alpha
            best_h = h.copy()
            
    # Refine around the best alpha found
    if best_alpha > 0:
        refinement_range = np.linspace(max(0.01, best_alpha - 0.5), best_alpha + 0.5, 100)
        for alpha in refinement_range:
            if time.time() - start_time > timeout:
                break
            t = np.arange(N, dtype=float) / (N - 1)
            h = t ** alpha
            l1 = np.sum(h)
            if l1 == 0: continue
            l2_sq = np.sum(h**2)
            f = np.convolve(h, h, mode='full')
            f_inf = np.max(f)
            if f_inf == 0: continue
            ratio = l2_sq / (l1 * f_inf)
            if ratio > best_ratio:
                best_ratio = ratio
                best_alpha = alpha
                best_h = h.copy()
                
    # The score is the raw value. 
    # We can scale h to any value, it doesn't change the ratio.
    # However, the task asks for "heights". 
    # Returning the shape as is is fine.
    
    return list(best_h)
