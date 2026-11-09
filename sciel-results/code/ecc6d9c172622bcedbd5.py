# Try Powell
res = minimize(wrapped_func, start_point, method='Powell', options={'maxiter': 100, 'xtol': 1e-3, 'ftol': 1e-3})
