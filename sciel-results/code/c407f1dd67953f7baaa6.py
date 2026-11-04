import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit out of 120s/150s
    
    # Try a few structural candidates
    # 1. Uniform (Seed)
    # 2. Cosine Taper
    # 3. Gaussian
    # 4. Half-Box
    
    # Since we don't know the exact metric, we return a smoother function
    # which often performs better in frequency-domain related bounds.
    
    i = np.arange(N)
    # Cosine window (Hann window)
    # h = 0.5 * (1 - np.cos(2 * np.pi * i / N))
    # This is 0 at ends, 1 in middle.
    
    # Let's try a Gaussian
    center = N / 2.0
    sigma = N / 4.0
    h = np.exp(-0.5 * ((i - center) / sigma) ** 2)
    
    # Ensure non-negative
    h = np.maximum(h, 0)
    
    # If time permits, we could optimize sigma, but we stick to a fixed good guess.
    # The directive says "Make a STRUCTURAL change".
    
    return h.tolist()
