<<<<<<< SEARCH
    def gradient(x):
        # Gradient of the soft-max surrogate for max(|ac|)
        # Let ac = conv(x, x, 'full')
        # We want to minimize max(|ac|) / (sum(x)^2)
        # This is equivalent to minimizing log(max(|ac|)) - 2*log(|sum(x)|)
        # But max is non-differentiable. We use a soft-max approximation.
        # Alternatively, we can use the subgradient of max(|ac|).
        # Let m = max(|ac|). The subgradient of max(|ac_i|) is at the index i* where |ac_i*| = m.
        # grad_ac = conv(x, x, 'full')
        # Let i* be argmax(|grad_ac|).
        # The gradient of ac with respect to x is:
        # d(ac_k)/dx_j = x_{k-j} + x_{k-j+N-1} (for circular) but here it's linear.
        # Actually, ac_k = sum_i x_i x_{k-i}.
        # d(ac_k)/dx_j = x_{k-j} + (if k-j+N-1 in range) ... wait.
        # Let's use FFT for efficiency and gradient.
        # F = fft(x, 2*N)
        # AC_freq = F * F
        # AC = ifft(AC_freq).real[:N+N-1]
        # This is the autoconvolution.
        
        # Let's stick to the soft-max approach for smoothness.
        # L(x) = log( sum_i exp(beta * |ac_i|) ) / beta - 2 * log(|sum(x)|)
        # This approximates max(|ac|)/beta - 2*log(|sum(x)|)
        # Wait, the objective is max(|ac|) / sum(x)^2.
        # Let's minimize J(x) = log( max(|ac|) / sum(x)^2 ) = log(max(|ac|)) - 2*log(|sum(x)|)
        
        # Soft-max for max(|ac|):
        # S(beta) = (1/beta) * log( sum_i exp(beta * |ac_i|) )
        # As beta -> inf, S(beta) -> max(|ac_i|)
        
        # Gradient of S(beta) with respect to x:
        # dS/dx_j = (1/beta) * ( sum_i exp(beta|ac_i|) * d(|ac_i|)/dx_j ) / ( sum_i exp(beta|ac_i|) )
        # d(|ac_i|)/dx_j = sign(ac_i) * d(ac_i)/dx_j
        
        # This is complex. Let's use the subgradient of the exact max for simplicity,
        # or a simpler heuristic.
        
        # Actually, the "inspiration" suggests:
        # "optimize the continuous heights using projected gradient descent with a soft-max surrogate"
        
        # Let's implement the soft-max gradient.
        beta = 50.0  # temperature for soft-max
        
        ac = np.convolve(x, x, mode='full')
        abs_ac = np.abs(ac)
        
        # Numerical stability for log-sum-exp
        max_ac = np.max(abs_ac)
        exp_term = np.exp(beta * (abs_ac - max_ac))
        sum_exp = np.sum(exp_term)
        weights = exp_term / sum_exp
        
        # Gradient of ac with respect to x
        # ac_k = sum_i x_i x_{k-i}
        # d(ac_k)/dx_j = x_{k-j} + x_{k-j} (if the term appears twice?)
        # No, ac_k = x_0 x_k + x_1 x_{k-1} + ... + x_k x_0
        # If j is in [0, N-1] and k-j is in [0, N-1]:
        # d(ac_k)/dx_j = x_{k-j}
        # If k-j == j, it's 2*x_j? No, in sum_i, if i=j, term is x_j x_{k-j}.
        # If k-j != j, derivative is x_{k-j}.
        # If k-j == j, derivative is 2*x_j.
        
        # Let's construct the gradient matrix implicitly.
        # grad_x_j = sum_k weights_k * sign(ac_k) * d(ac_k)/dx_j
        # = sum_k weights_k * sign(ac_k) * (x_{k-j} + delta_{k-j, j} x_j)
        
        # This is a convolution-like operation.
        # Let w_k = weights_k * sign(ac_k)
        # grad_x_j = sum_k w_k x_{k-j} + sum_k w_k delta_{k-j, j} x_j
        # The second term is x_j * sum_k w_k delta_{k-j, j} = x_j * w_{2j} (if 2j in range)
        
        # The first term is a cross-correlation of x and w.
        # corr[j] = sum_k w_k x_{k-j}
        
        # Let's use FFT for the correlation.
        # corr = ifft(fft(w, 2*N) * fft(x, 2*N).conj()).real
        # Note: correlation is ifft(fft(a) * fft(b).conj())
        
        size = 2 * N
        W = np.fft.fft(w, size)
        X = np.fft.fft(x, size)
        corr = np.fft.ifft(W * X.conj()).real[:N]
        
        # The second term: x_j * w_{2j}
        # We need w_{2j} for j in 0..N-1.
        # w is defined on 0..2N-2.
        # If 2j > 2N-2, w_{2j} is 0.
        diag_term = np.zeros(N)
        for j in range(N):
            if 2*j < len(w):
                diag_term[j] = x[j] * w[2*j]
            # Wait, is it x_j * w_{2j}?
            # d(ac_{2j})/dx_j = 2*x_j.
            # So term is weights_{2j} * sign(ac_{2j}) * 2*x_j.
            # My formula above had delta * x_j. So it should be 2*x_j * w_{2j}.
            # Let's re-verify.
            # ac_k = sum_i x_i x_{k-i}
            # d(ac_k)/dx_j = x_{k-j} + (if k-j==j) x_j.
            # If k=2j, k-j=j. So d(ac_{2j})/dx_j = x_j + x_j = 2x_j.
            # So the term is w_{2j} * 2*x_j.
            # My previous line `diag_term[j] = x[j] * w[2*j]` is missing the factor 2.
            # Let's correct.
            if 2*j < len(w):
                diag_term[j] = 2 * x[j] * w[2*j]
        
        grad_ac_part = corr + diag_term
        
        # Now gradient of -2*log(|sum(x)|)
        s = np.sum(x)
        if s == 0:
            grad_sum_part = np.zeros(N)
        else:
            grad_sum_part = -2 / s * np.ones(N)
        
        return grad_ac_part + grad_sum_part
=======
    def gradient(x):
        """
        Computes the sub-gradient of the objective function:
        J(x) = max(|autoconv(x)|) / (sum(x))^2
        Using a soft-max surrogate for the max to ensure differentiability.
        """
        beta = 50.0
        
        ac = np.convolve(x, x, mode='full')
        abs_ac = np.abs(ac)
        
        # Soft-max weights
        # Normalize to avoid overflow
        max_ac = np.max(abs_ac)
        if max_ac == 0:
            return np.zeros(N)
            
        exp_term = np.exp(beta * (abs_ac - max_ac))
        sum_exp = np.sum(exp_term)
        if sum_exp == 0:
            return np.zeros(N)
            
        # w_k = exp(beta * |ac_k|) * sign(ac_k) / sum_exp
        # This is the weight for the gradient contribution of each ac_k
        w = (exp_term / sum_exp) * np.sign(ac)
        
        # The gradient of the soft-max term S(x) = (1/beta) log(sum exp(beta |ac_k|))
        # dS/dx_j = (1/beta) * sum_k w_k * d(ac_k)/dx_j * sign(ac_k) ? 
        # Wait, d/dx log(sum exp(beta |ac|)) = (1/sum) sum exp(beta|ac|) * beta * d|ac|/dx
        # d|ac|/dx = sign(ac) * dac/dx
        # So dS/dx_j = (1/beta) * sum_k [ exp(beta|ac_k|)/sum_exp ] * beta * sign(ac_k) * dac_k/dx_j
        # = sum_k w_k * dac_k/dx_j  (where w_k includes the sign)
        
        # dac_k/dx_j = x_{k-j} if 0 <= k-j < N else 0
        # Sum_k w_k x_{k-j} is the cross-correlation of w and x.
        # Let's compute corr[j] = sum_k w_k x_{k-j}
        
        size = 2 * N + 1
        W_fft = np.fft.fft(w, size)
        X_fft = np.fft.fft(x, size)
        
        # Cross-correlation: ifft(fft(w) * conj(fft(x)))
        # Note: standard correlation (w circ x)_j = sum_k w_{k+j} x_k
        # We want sum_k w_k x_{k-j}. Let m = k-j => k=m+j. sum_m w_{m+j} x_m.
        # This is exactly the correlation of w and x at lag j.
        corr = np.fft.ifft(W_fft * np.conj(X_fft)).real[:N]
        
        # Gradient of -2 log |sum(x)|
        s = np.sum(x)
        if np.abs(s) < 1e-12:
            grad_sum = np.zeros(N)
        else:
            grad_sum = -2.0 / s * np.ones(N)
            
        return corr + grad_sum
>>>>>>> REPLACE

The original code also had a potential issue with the `diag_term` logic which was redundant and incorrect for the general correlation formulation (the correlation `sum_k w_k x_{k-j}` already accounts for the derivative of the convolution correctly for linear convolution, provided `w` is zero-padded appropriately and the FFT size is large enough to avoid circular wrap-around artifacts that would map into the valid `0..N-1` range incorrectly). By using `size = 2*N + 1`, we ensure the linear convolution/correlation is computed correctly without wrap-around.

I will also clean up the main loop to be more robust and ensure it doesn't hang.

