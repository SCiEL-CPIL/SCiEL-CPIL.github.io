        # Handle remaining prime factors (primes > sqrt(end))
        for i in range(n):
            if remaining[i] > 1:
                phi_vals[i] -= phi_vals[i] // remaining[i]
