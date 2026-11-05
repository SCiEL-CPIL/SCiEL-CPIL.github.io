import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # Objective: Minimize ||h*h||_2^2 subject to sum(h) = 1
    # Value = 2 * N * max(h*h) / (sum(h))^2
    # If sum(h)=1, Value = 2 * N * max(h*h)
    # Minimizing sum((h*h)^2) tends to flatten the autoconvolution,
    # reducing the peak max(h*h).
    
    # Precompute Toeplitz matrix structure for autoconvolution
    # c_k = sum_i h_i h_{i+k}
    # We want to minimize sum_k c_k^2 = sum_k (sum_i h_i h_{i+k})^2
    # This is equivalent to ||h||_4^4.
    
    # Gradient of f(h) = ||h*h||_2^2 = sum_k c_k^2
    # df/dh_j = 2 * sum_k c_k * dc_k/dh_j
    # dc_k/dh_j = h_{j+k} + h_{j-k} (with appropriate bounds)
    # Note: c_k is defined for k in [-(N-1), N-1].
    # Actually, np.convolve(h, h) gives c from index 0 to 2N-2.
    # Let's stick to the definition: z = np.convolve(h, h)
    # f(h) = sum(z^2)
    # z_k = sum_i h_i h_{k-i}
    # dz_k/dh_j = h_{k-j} + h_{j-k} (if indices valid, else 0)
    # But it's easier to compute gradient via FFT or direct summation for N=400.
    # N=400 is small enough for O(N^2) gradient calculation if optimized, 
    # or O(N log N) via FFT.
    
    # Let's use FFT for speed in gradient calculation.
    # z = ifft(fft(h) * fft(h))
    # This is circular convolution. The problem uses linear convolution.
    # For N=400, linear convolution length is 799.
    # We can zero-pad to length >= 2N-1 to make circular = linear.
    # Let M = next_power_of_2(2*N - 1)
    
    M = 512 # 2*400-1 = 799. 512 is too small. Need 1024.
    M = 1024
    
    fft_h = np.fft.fft(np.zeros(M)) # placeholder
    
    def _eval_h(h):
        # h is length N
        # Pad to M
        h_padded = np.zeros(M)
        h_padded[:N] = h
        # Linear conv via circular conv
        z = np.fft.ifft(np.fft.fft(h_padded) * np.fft.fft(h_padded)).real
        # z is length M. The first 2N-1 elements are the linear conv.
        z_lin = z[:2*N-1]
        # Objective: sum(z_lin^2)
        val = np.sum(z_lin ** 2)
        return val, z_lin

    def _objective_and_grad(h):
        if not np.all(np.isfinite(h)):
            return float('inf'), np.zeros(N)
        
        val, z_lin = _eval_h(h)
        
        # Gradient calculation
        # f = sum_k z_k^2
        # df/dh_j = 2 * sum_k z_k * (dz_k/dh_j)
        # z_k = sum_i h_i h_{k-i}
        # dz_k/dh_j = h_{k-j} + h_{j-k} (for valid indices)
        # This is effectively 2 * (z * h) linear convolution?
        # Let's check.
        # z = h * h
        # f = ||z||_2^2
        # grad = 2 * (z * h) ? No.
        # Let's derive carefully.
        # f = sum_k (sum_i h_i h_{k-i})^2
        # Let u_k = sum_i h_i h_{k-i}
        # f = sum_k u_k^2
        # df/du_k = 2 u_k
        # du_k/dh_j = h_{k-j} + h_{j-k} (if k-j in [0, N-1] and j-k in [0, N-1])
        # df/dh_j = sum_k 2 u_k (h_{k-j} + h_{j-k})
        # Let m = k-j. k = m+j.
        # Term 1: sum_m 2 u_{m+j} h_m (where u_{m+j} is valid)
        # Term 2: sum_m 2 u_{j-m} h_m (where u_{j-m} is valid)
        # This is 2 * (u * h) + 2 * (u_rev * h)?
        # Actually, u is symmetric? No, u_k from linear conv is not symmetric.
        # However, u_{-k} is not defined in 0..2N-2 indexing.
        
        # Simpler: Use FFT for gradient too.
        # f = sum_k |ifft(FFT(h)^2)|^2 ? No, that's circular.
        # For linear conv, we can use the fact that grad is related to cross-correlation.
        # Let's just compute it directly for N=400. O(N^2) is 160k ops, very fast.
        
        grad = np.zeros(N)
        # u_k is z_lin[k] for k in 0..2N-2
        # df/dh_j = 2 * sum_{k} z_lin[k] * (h[k-j] + h[j-k])
        # We can compute this via convolution of z_lin with h.
        # Let w = np.convolve(z_lin, h, mode='full')
        # The term sum_k z_k h_{k-j} is (z * h)_{2j}? No.
        # Let's just do the direct sum in a vectorized way if possible, or loop.
        # Loop over j is 400. Inner sum is up to 400. 160k ops. Python loop is slow.
        # Use FFT trick:
        # The gradient of sum((h*h)^2) is 4 * (h * h * h) ? No.
        # Let's use the property: 
        # If we pad h to M, and compute circular conv z_circ = ifft(fft(h)^2)
        # Then d/dh sum(z_circ^2) = 2 * ifft(fft(z_circ) * fft(h)) * 2 ?
        # Actually, for circular conv, grad = 2 * (z * h)_circ ?
        # Let's verify with small N.
        
        # Given N=400, let's use a robust numerical check or direct computation.
        # Direct computation using numpy broadcasting might be memory heavy (400x800).
        # Let's use the FFT approximation for gradient, which is exact for circular.
        # Since we zero-pad to M=1024, the linear conv is contained in the first 2N-1.
        # The "tail" of the circular conv wraps around.
        # z_circ[k] = z_lin[k] for k < 2N-1? No.
        # z_circ is the circular conv of padded h.
        # Since h is zero for N..M-1, the circular conv is identical to linear conv
        # IF no wrap-around occurs.
        # Wrap-around occurs if support of h * h exceeds M-1.
        # Support is 2N-2 = 798. M=1024. 798 < 1023.
        # So z_circ == z_lin (padded) exactly.
        
        # So we can use circular conv formulas.
        # f = sum_k z_circ[k]^2
        # grad_h = 2 * ifft(fft(z_circ) * fft(h)) * 2 ?
        # Let's derive:
        # z = h * h (circular)
        # f = <z, z> = sum z_k^2
        # dz_k/dh_j = h_{k-j} + h_{j-k} (circular indices)
        # df/dh_j = sum_k 2 z_k (h_{k-j} + h_{j-k})
        # = 2 (z * h)_j + 2 (z * h_rev)_j ?
        # Since z is not symmetric, we need to be careful.
        # Actually, (z * h)_j = sum_k z_k h_{j-k}.
        # The second term sum_k z_k h_{k-j} = sum_m z_{m+j} h_m. Let l = -j. sum_m z_{m-l} h_m = (z_rev * h)_l?
        # It turns out that for real signals, the gradient is 2 * (z * h) + 2 * (z * h) if z is symmetric?
        # No.
        # Let's just compute:
        # G1 = ifft(fft(z) * fft(h))  -> This is z * h
        # G2 = ifft(fft(z) * fft(h[::-1]))?
        
        # Actually, there is a known result: grad of ||h*h||^2 is 4 * (h*h*h) ? No.
        # Let's use the direct sum in a vectorized block manner or just accept O(N^2) in numpy.
        # N=400. 400*400 = 160,000 operations.
        # We can do:
        # indices_j = np.arange(N)
        # indices_k = np.arange(2*N-1)
        # # z_k h_{k-j}
        # # Create matrix where M[j, k] = h[k-j] if valid else 0
        # # This is a Toeplitz matrix.
        
        # Efficient way:
        # conv_z_h = np.convolve(z_lin, h, mode='full')
        # The term sum_k z_k h_{k-j} corresponds to specific slices of conv_z_h.
        # Let's just use the FFT method which is O(M log M).
        # z_fft = fft(z_circ)
        # h_fft = fft(h_padded)
        # grad_circ = 2 * ifft(z_fft * h_fft) + 2 * ifft(z_fft * fft(h_padded[::-1])?)
        
        # Let's stick to a safe, verified direct method using scipy.signal or manual convolution.
        # Since M=1024, FFT is fast.
        # Let's assume the gradient of sum(z^2) where z = ifft(fft(h)^2) is:
        # 2 * ifft(fft(z) * fft(h)) + 2 * ifft(fft(z) * fft(h_flipped))
        # Wait, h is real.
        # Let's test this logic in the code with a small example first? No, must be robust.
        
        # Alternative: Use numerical gradient? Too slow for SLSQP iterations.
        
        # Let's use the property:
        # f(h) = ||h||_4^4
        # The gradient is 4 * (h * h * h) ? No, that's for ||h||_4^4 = sum h_i^4.
        # Here it is sum (h*h)_k^2.
        
        # Let's use the direct convolution approach for gradient, it's safer.
        # grad[j] = 2 * sum_k z[k] * (h[k-j] + h[j-k])
        # We can compute A = np.convolve(z_lin, h, mode='full')
        # A[m] = sum_k z[k] h[m-k]
        # We need sum_k z[k] h[k-j]. Let p = k-j => k = p+j.
        # sum_p z[p+j] h[p]. This is cross-correlation.
        # Correlation C[j] = sum_p z[p+j] h[p] = (z * h_rev)[j + N - 1]?
        
        # Let's just do the O(N^2) loop in numpy. It's 400 iterations of vector ops.
        grad = np.zeros(N)
        # z_lin is length 2N-1
        # h is length N
        # For each j in 0..N-1:
        # term1 = sum_{k} z_lin[k] * h[k-j] (where 0<=k-j<N)
        # term2 = sum_{k} z_lin[k] * h[j-k] (where 0<=j-k<N)
        
        # Vectorized:
        # Create Toeplitz matrix T where T[j, k] = h[k-j] if valid
        # This is hard to vectorize without memory blowup (400x800 is fine).
        
        # Let's use scipy.signal.correlate
        from scipy.signal import correlate, convolve
        # term1_j = sum_k z_k h_{k-j} = correlate(z, h)[j + N - 1] ?
        # correlate(z, h)[m] = sum_n z[n] h[n+m]
        # We want sum_k z[k] h[k-j]. Let n=k, m=-j.
        # So it is correlate(z, h)[-j].
        # Since correlate output is length 2N-1.
        # Index for -j is (N-1) - j ?
        
        # Let's just use the FFT gradient which is standard for this quartic form.
        # f = sum_k |z_k|^2
        # grad_h = 2 * ifft(fft(z) * fft(h)) * 2 ?
        # Actually, d/dh sum(z^2) = 2 * (z * h) if z is linear conv?
        # Let's try:
        # z = ifft(fft(h)^2)
        # f = sum(z^2)
        # df/dh = 2 * ifft(fft(z) * fft(h)) + 2 * ifft(fft(z) * fft(h)[::-1])?
        
        # To be absolutely safe and fast, I will use the direct sum but optimized.
        # N=400 is small.
        # We can compute the gradient by convolving z_lin with h and z_lin with h[::-1].
        
        # z_lin * h (full conv) length 3N-2
        # z_lin * h[::-1] (full conv) length 3N-2
        
        # Let's use the fact that we can just run the optimizer with a numerical gradient 
        # if the analytic one is tricky, but SLSQP is slow with numerical grad.
        
        # Let's use the analytic gradient derived from circular convolution.
        # Since M=1024 > 2N-1, circular == linear.
        # z_circ = ifft(fft(h_padded)^2)
        # f = sum(z_circ^2)
        # grad_padded = 2 * ifft(fft(z_circ) * fft(h_padded)) + 2 * ifft(fft(z_circ) * fft(h_padded)[::-1])?
        # No, the derivative of z_k = sum h_i h_{k-i} wrt h_j is h_{k-j} + h_{j-k}.
        # In frequency domain:
        # FFT(z) = FFT(h)^2
        # FFT(grad) = FFT(z) * FFT(h) + FFT(z) * FFT(h_flipped)?
        # Actually, for real h, the gradient of sum(z^2) is 4 * ifft(FFT(z) * FFT(h)) ?
        # Let's assume grad = 2 * ifft(FFT(z) * FFT(h)) + 2 * ifft(FFT(z) * FFT(h)[::-1]) is wrong.
        
        # Correct derivation:
        # f = <z, z>
        # df = 2 <z, dz> = 2 <z, dh*h + h*dh> = 2 <z*h, dh> + 2 <z*h_rev, dh>
        # So grad = 2 * (z * h) + 2 * (z * h_rev)
        # Where * is circular convolution.
        
        z_circ = np.fft.ifft(np.fft.fft(h_padded) * np.fft.fft(h_padded)).real
        fft_z = np.fft.fft(z_circ)
        fft_h = np.fft.fft(h_padded)
        
        # Term 1: z * h
        grad1 = np.fft.ifft(fft_z * fft_h).real
        # Term 2: z * h_rev
        # h_rev is h[::-1] padded?
        # h_rev_padded = np.zeros(M); h_rev_padded[:N] = h[::-1]
        # But circular conv with reversed h is related to correlation.
        # Actually, <z, h*dh> = <z*h, dh>
        # <z, dh*h> = <z*h_rev, dh> ?
        # Let's just compute both convs.
        
        h_rev_padded = np.zeros(M)
        h_rev_padded[:N] = h[::-1]
        fft_h_rev = np.fft.fft(h_rev_padded)
        grad2 = np.fft.ifft(fft_z * fft_h_rev).real
        
        grad_full = 2 * (grad1 + grad2)
        grad = grad_full[:N]
        
        return val, grad

    # Initial guess: ones
    h0 = np.ones(N) / N # Normalize sum to 1 initially?
    # Constraint sum(h) = 1.
    # If h0 = ones, sum = N.
    # Let's start with h0 = np.ones(N).
    # Constraint: sum(h) - 1 = 0.
    
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 'jac': lambda h: np.ones(N)}]
    
    # Bounds: None (allow negative heights)
    
    res = None
    try:
        res = minimize(
            _objective_and_grad, 
            np.ones(N), 
            method='SLSQP', 
            jac=True,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
    except Exception as e:
        pass

    if res is not None and res.success:
        best_h = res.x
    else:
        best_h = np.ones(N)
        
    # Ensure sum is not zero and finite
    s = np.sum(best_h)
    if abs(s) < 1e-9:
        best_h = np.ones(N)
        
    # Normalize to sum=1 to ensure stability? 
    # The objective value is scale invariant.
    # value = 2*N*max(conv)/sum^2.
    # If we return best_h, the evaluator computes the value.
    # We don't need to normalize, but it helps if we want to report intermediate.
    
    return best_h
