import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    
    # The seed program used a uniform distribution, which corresponds to a single
    # large "pulse" in the convolution sense, yielding a normalized peak of 1.0.
    # By using a sparse pattern, such as alternating 1s and 0s, we create multiple
    # disjoint pulses. The peak of the autoconvolution is determined by the
    # self-convolution of a single pulse (height 1, width 1 interval), while the
    # normalization (L2 norm or L1/Linf product) scales with the number of pulses.
    # This significantly reduces the ratio.
    
    # Alternating pattern: 1, 0, 1, 0, ...
    # This creates 300 disjoint pulses.
    h = np.zeros(N)
    h[::2] = 1.0
    
    # Optional: Verify time
    if time.time() - start_time > 10:
        return list(h)
        
    return list(h)
