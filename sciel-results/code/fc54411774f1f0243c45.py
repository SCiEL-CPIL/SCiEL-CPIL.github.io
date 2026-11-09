        g_coeffs = np.zeros(51)
        for k in range(26):
            if c[k] == 0: continue
            # lag_coeffs[k] has length 2k+1
            # g_coeffs has length 51
            # Add c[k] * lag_coeffs[k] to g_coeffs[0:2k+1]
            g_coeffs[:2*k+1] += c[k] * lag_coeffs[k]
