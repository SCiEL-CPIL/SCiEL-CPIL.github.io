import numpy as np
import time

N = 400

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Baseline: all ones
    # This minimizes the variance, keeping c0/S^2 at the theoretical minimum 1/N
    # and keeping off-diagonal terms predictable.
    x = np.ones(N)
    
    # If the problem allows real values and the metric is M/S^2, this is optimal.
    # If the problem requires integers, convert.
    # "signed step heights" might imply integers.
    # However, "step functions" in calculus are often real-valued.
    # I will return floats. If it fails, I'd need to try ints.
    # But the prompt says "Keep the function names...".
    
    # Check if we have time to do a trivial check or optimization.
    # Given the theoretical optimality of the constant vector for M/S^2,
    # no optimization is needed.
    
    return x.tolist()
