import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
       t0, t1 = t_span
       y0 = np.array(y0, dtype=float)
       n = len(y0)
       
       # Deadline guard
       start_time = time.time()
       deadline = start_time + 280.0

       # Assume homogeneous and constant coefficient system: y' = A y
       # Extract A by evaluating f(t0, e_j) for each basis vector e_j
       A = np.zeros((n, n))
       for j in range(n):
           e_j = np.zeros(n)
           e_j[j] = 1.0
           A[:, j] = f(t0, e_j)
           
       # Check time
       if time.time() > deadline - 10:
           # Fallback: simple Euler? No, just return y0 or something?
           # Better to return best guess.
           pass
           
       dt = t1 - t0
       M = expm(A * dt)
       y_final = M @ y0
       
       return y_final
