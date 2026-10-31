import time
import numpy as np

_start_time = time.time()
_DEADLINE = 290.0

def priority(vector, n):
       if time.time() - _start_time > _DEADLINE:
           return 0.0
       
       v_arr = np.array(vector, dtype=np.int64)
       
       # Count occurrences of 0, 1, 2
       n_0 = np.sum(v_arr == 0)
       n_1 = np.sum(v_arr == 1)
       n_2 = np.sum(v_arr == 2)
       
       # Ideal counts for uniform distribution
       ideal = n / 3.0
       
       # Quadratic deviation from uniformity
       # We want to MINIMIZE this deviation (favor random-like vectors)
       # So Priority = - Deviation
       deviation = (n_0 - ideal)**2 + (n_1 - ideal)**2 + (n_2 - ideal)**2
       
       # To make it a "smooth quadratic surrogate" as per findings,
       # we can also consider the "spread" of the vector in the embedding space.
       # But let's stick to the coordinate balance first.
       
       # Add a small random tie-breaker to avoid identical scores for symmetric vectors
       # Use a hash of the vector for deterministic tie-breaking
       # h = zlib.crc32(bytes(v_arr))
       
       # Return negative deviation (higher is better)
       # Scale to float
       return float(-deviation)
