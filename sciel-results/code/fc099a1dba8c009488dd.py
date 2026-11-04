import numpy as np

N = 600

def construct():
    # The seed is uniform [1.0]*N with score 1.0.
    # We assume the metric is max_conv(f) / max_conv(uniform)
    # with the constraint sum(f) == sum(uniform) = N.
    
    # Reference peak for uniform [1]*N
    # Convolution of [1]*N with itself is a triangle peaking at N.
    ref_peak = float(N)
    
    best_score = float('inf')
    best_heights = [1.0] * N  # Fallback to seed
    
    # Strategy: Double Box
    # Symmetric. Center gap of 2*m zeros.
    # Two blocks of length (N - 2*m) / 2.
    # Height H = N / (N - 2*m).
    
    # m ranges from 0 to N//2 - 1
    # If m=0, it's a single box (uniform), score 1.0.
    # We want to find m > 0 that lowers the peak.
    
    for m in range(0, N // 2):
        gap = 2 * m
        non_zero_count = N - gap
        
        # Ensure non_zero_count is even for symmetric double box
        if non_zero_count % 2 != 0:
            # If odd, we can't have two equal integer blocks.
            # We can have blocks of size k and k+1.
            # Or just skip this m, or adjust.
            # Let's handle general case:
            # Left block size, Right block size.
            # To maintain symmetry and sum, we can distribute.
            # But "Double Box" implies two equal blocks.
            # Let's only consider m where non_zero_count is even.
            continue
            
        block_len = non_zero_count // 2
        
        if block_len == 0:
            continue
            
        H = float(N) / float(non_zero_count)
        
        # Construct heights
        heights = np.zeros(N)
        # Indices for left block: 0 to block_len-1
        # Indices for right block: N-block_len to N-1
        heights[0:block_len] = H
        heights[N-block_len:N] = H
        
        # Compute autoconvolution
        # np.convolve does linear convolution
        conv = np.convolve(heights, heights, mode='full')
        
        current_peak = np.max(conv)
        
        score = current_peak / ref_peak
        
        if score < best_score:
            best_score = score
            best_heights = heights.tolist()
            
    # If no improvement found, return seed
    # But we can also try a "smoothed" version or different shapes if time permits.
    # For now, the grid search over m is the primary strategy.
    
    # Additional Refinement: 
    # The optimal shape for minimizing L_inf norm of autoconvolution 
    # with fixed L_1 norm is often a "sinc"-like or specific trigonometric shape,
    # but for step functions, the "double box" is a strong candidate.
    # We can also try "Four Boxes" if the double box is not enough.
    # However, the double box search is fast.
    
    # Let's also try a "Tapered" double box to see if it helps.
    # Tapered: H * (1 - alpha * |x|) within the blocks?
    # This adds complexity. Let's stick to the constant height double box first.
    # If best_score is close to 1.0, we might need to refine.
    
    # One more check: The problem allows ANY nonnegative step function.
    # The double box is a subset.
    # Is there a better subset?
    # "Hollow" structures are known to reduce the central peak.
    
    # Let's add a quick optimization for the "Four Box" case as well, 
    # just in case, but keep it simple.
    # Actually, let's just return the best double box found.
    
    return best_heights
