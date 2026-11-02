import numpy as np
import time

def construct():
    # The problem asks to minimize the maximum overlap of h with shifted 1-h.
    # Overlap at shift k is O_k = sum_i h_i (1 - h_{(i+k)%N}).
    # O_k = sum_i h_i - sum_i h_i h_{(i+k)%N} = N/2 - A_k.
    # Minimizing max(O_k) is equivalent to maximizing min(A_k).
    # For h_i in [0,1] with sum N/2, the maximum possible value for min(A_k) is N/4,
    # achieved by the constant sequence h_i = 0.5.
    # This results in a maximum overlap of N/4.
    # Any deviation from h_i = 0.5 will cause some A_k to be less than N/4,
    # thus increasing the maximum overlap.
    # Therefore, the constant sequence is the global optimum.
    
    N = 100
    h = np.full(N, 0.5)
    
    return h
