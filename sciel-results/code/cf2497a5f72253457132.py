import time
import numpy as np

N = 600

def _autoconvolve(heights):
    """Compute autoconvolution via FFT."""
    n = len(heights)
    size = 2 * n
    f = np.fft.rfft(heights, n=size)
    conv = np.fft.irfft(f * f, n=size)
    return conv[:2*n-1]

def _score(heights):
    """
    Compute the normalized maximum of autoconvolution.
    Normalization: max(autoconv) / (sum(heights))^2
    This is a common normalization for autoconvolution bounds.
    """
    if len(heights) == 0:
        return float('inf')
    total = sum(heights)
    if total == 0:
        return float('inf')
    conv = _autoconvolve(np.array(heights, dtype=np.float64))
    max_conv = np.max(conv)
    return max_conv / (total * total)

def construct():
    """
    Constructs a nonnegative step function with 600 equal intervals on [-1/4, 1/4].
    
    Strategy: Represent the step function as a sum of two disjoint rectangular blocks
    with optimized widths and positions. The autoconvolution of a multi-rectangle function
    is a sum of triangles, and by choosing appropriate widths and separations, we can
    suppress the peak via destructive interference of their support overlaps.
    
    This is a structural change from continuous gradient or convex formulations,
    leveraging the analytic structure of rectangle autoconvolutions.
    """
    start_time = time.time()
    deadline = 145.0
    
    best_score = float('inf')
    best_heights = [1.0] * N  # fallback: uniform
    
    # Try two disjoint rectangular blocks
    # Block 1: width w1, starting at position p1
    # Block 2: width w2, starting at position p2
    # Constraint: blocks are disjoint (p1 + w1 <= p2 or p2 + w2 <= p1)
    
    # We'll search over widths and positions
    # For efficiency, fix block 1 at position 0, and vary block 2's position and widths
    
    # First, try a coarse grid over widths
    width_range = range(50, 551, 10)  # widths from 50 to 550
    
    for w1 in width_range:
        for w2 in width_range:
            if w1 + w2 > N:
                continue
            
            # Place block 1 at start, block 2 at end (maximal separation)
            heights = [0.0] * N
            for i in range(w1):
                heights[i] = 1.0
            for i in range(w2):
                heights[N - w2 + i] = 1.0
            
            s = _score(heights)
            if s < best_score:
                best_score = s
                best_heights = heights[:]
            
            # Check time
            if time.time() > deadline - 10:
                return best_heights
    
    # Now refine around the best configuration found
    if best_score < float('inf'):
        # Find the best w1, w2 from the coarse search
        best_w1 = None
        best_w2 = None
        for w1 in width_range:
            for w2 in width_range:
                if w1 + w2 > N:
                    continue
                heights = [0.0] * N
                for i in range(w1):
                    heights[i] = 1.0
                for i in range(w2):
                    heights[N - w2 + i] = 1.0
                s = _score(heights)
                if s < best_score + 1e-9:
                    best_w1 = w1
                    best_w2 = w2
                    break
            if best_w1 is not None:
                break
        
        # Refine around the best widths
        if best_w1 is not None and best_w2 is not None:
            for dw1 in range(-5, 6):
                for dw2 in range(-5, 6):
                    w1_new = best_w1 + dw1
                    w2_new = best_w2 + dw2
                    if w1_new < 10 or w2_new < 10 or w1_new + w2_new > N:
                        continue
                    
                    heights = [0.0] * N
                    for i in range(w1_new):
                        heights[i] = 1.0
                    for i in range(w2_new):
                        heights[N - w2_new + i] = 1.0
                    
                    s = _score(heights)
                    if s < best_score:
                        best_score = s
                        best_heights = heights[:]
                    
                    # Check time
                    if time.time() > deadline - 5:
                        return best_heights
    
    # Also try three blocks for potentially better results
    if time.time() < deadline - 20:
        for w1 in range(100, 301, 20):
            for w2 in range(100, 301, 20):
                for w3 in range(100, 301, 20):
                    if w1 + w2 + w3 > N:
                        continue
                    
                    # Place three blocks: at start, middle, end
                    heights = [0.0] * N
                    for i in range(w1):
                        heights[i] = 1.0
                    mid_start = N // 2 - w2 // 2
                    for i in range(w2):
                        if mid_start + i < N:
                            heights[mid_start + i] = 1.0
                    for i in range(w3):
                        heights[N - w3 + i] = 1.0
                    
                    s = _score(heights)
                    if s < best_score:
                        best_score = s
                        best_heights = heights[:]
                    
                    # Check time
                    if time.time() > deadline - 10:
                        return best_heights
    
    return best_heights
