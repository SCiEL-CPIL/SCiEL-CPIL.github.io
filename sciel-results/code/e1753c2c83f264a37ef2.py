import numpy as np

# ... sieve as before ...

a_vals = np.arange(1, TARGET)
b_vals = TARGET - a_vals
costs = np.maximum(lp[a_vals], lp[b_vals])
min_cost = np.min(costs)
min_idx = np.argmin(costs)
best_a = a_vals[min_idx]
best_b = b_vals[min_idx]
