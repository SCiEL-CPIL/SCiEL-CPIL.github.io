import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 600
DEADLINE = 142  # safety margin before 150s hard limit
NFFT = 2048  # next power of 2 >= 2N-1 = 1199

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    # ---- FFT-based autoconvolution of a (sum-normalized) height vector ----
    nfft = NFFT

    def conv(h):
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        return C[:2 * N - 1]

    def exact_metric(h):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        hn = h / s
        return np.max(conv(hn))

    def softmax_max(c, temp):
        c = c / temp
        c = c - np.max(c)
        return temp * (np.log(np.sum(np.exp(c))))

    def smooth_metric(h, temp):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        hn = h / s
        return softmax_max(conv(hn), temp)

    # ---- Epigraph formulation: minimize t s.t. t >= smoothed_max(conv(h)) ----
    # Variables: [h_0, ..., h_{N-1}, t]
    # Objective: t
    # Constraint: smooth_max(conv(h/s)) - t <= 0  (smooth surrogate)
    # We use the log-sum-exp form: log-sum-exp((c - t)/temp) <= 0
    # This gives gradient wrt t = -1, and wrt h_k = (1/s) * sum_j w_j * (dc/dh_k)
    # where w_j = exp((c_j - t)/temp) / sum exp((c_j - t)/temp)

    def epigraph_obj_and_grad(x, temp):
        """x = [h (length N), t (scalar)]. Returns (f, grad)."""
        h = np.maximum(x[:N], 0.0)
        t = x[N]
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(N + 1)
        hn = h / s
        c = conv(hn)
        # Smooth constraint: log-sum-exp((c - t)/temp) should be <= 0
        # Objective: maximize the violation to push t up, i.e., minimize -log-sum-exp
        # Actually, we want: minimize t subject to log-sum-exp((c-t)/temp) <= 0
        # Use penalty: f = t + penalty * max(0, log-sum-exp((c-t)/temp))^2
        # Better: use the smooth max directly as the objective with t as a free var
        # Simplest: f = t + lambda * (smooth_max(c, temp) - t)_+^2
        # But cleanest for L-BFGS-B: just use smooth_max as objective on h,
        # and let t be tracked separately. For now, use the direct smooth metric.
        
        # Direct approach: f = smooth_max(c, temp), grad wrt h
        c_shifted = (c - t) / temp
        c_shifted = c_shifted - np.max(c_shifted)
        weights = np.exp(c_shifted)
        w_sum = np.sum(weights)
        w = weights / w_sum
        f = t + temp * np.log(w_sum)  # = smooth_max(c, temp) when t=0, but we want just smooth_max
        
        # Actually let's just compute smooth_max and its gradient wrt h
        # smooth_max = temp * log(sum(exp(c/temp)))
        # d(smooth_max)/d(h_k) = (1/s) * sum_j w_j * d(conv(hn)_j)/d(h_k)
        # conv(hn)_j = sum_i hn_i * hn_{i+j-N+1} (circular, truncated)
        # d(conv_j)/d(h_k) = (1/s) * (hn_{j-k} + hn_{j+k}) approximately for linear conv
        # For FFT-based: dC/dh = 2 * ifft(fft(h) * conj(fft(h))) ... no
        # C = ifft(fft(h)^2), dC_j/dh_k = sum_m h_m * delta... 
        # Use the chain rule via FFT:
        # C = ifft(fft(h)^2), so dC = ifft(2*fft(h)*fft(dh))
        # d(smooth_max)/d(h_k) = sum_j w_j * dC_j/dh_k = sum_j w_j * [ifft(2*fft(h)*fft(e_k))]_j
        # = [ifft(2*fft(h)*fft(w_rev))]_k where w_rev is the reversed/shifted weight vector
        
        # More carefully: C_j = sum_{a+b=j} hn_a * hn_b (for linear conv, 0-indexed)
        # dC_j/dh_k = (1/s) * (sum_a hn_a * delta_{a+k,j} + sum_b hn_b * delta_{b+k,j})
        #            = (1/s) * (hn_{j-k} + hn_{j+k})  (with appropriate bounds)
        # For the full linear convolution of length 2N-1:
        # dC_j/dh_k = (1/s) * (hn_{j-k} + hn_{j+k}) for valid indices
        
        # The gradient of smooth_max wrt h is:
        # grad_h[k] = (1/s) * sum_j w_j * (hn_{j-k} + hn_{j+k})
        # This is a convolution-like operation.
        
        # Using FFT: let W = fft(w) padded to nfft, H = fft(h) padded to nfft
        # Then grad = (1/s) * ifft(2 * H * conj(W)) ... need to be careful with indexing
        
        # Actually, the autoconvolution C = ifft(|fft(h)|^2) gives circular conv.
        # For linear conv of h with h, we use zero-padded FFT.
        # dC/dh_k: C_j = sum_i h_i h_{j-i}, dC_j/dh_k = h_{j-k} + h_{j+k}
        # grad_h[k] = sum_j w_j (h_{j-k} + h_{j+k}) / s
        #           = (1/s) * [ (w * h)_k + (w * h)_{-k} ]  (circular, with proper padding)
        
        # Let's compute it via FFT for efficiency:
        # Define W_full of length nfft: W_full[0:2N-1] = w, rest = 0
        # H_full of length nfft: H_full[0:N] = h, rest = 0
        # Then grad_circ = ifft(fft(W_full) * fft(H_full))
        # grad_h[k] = (1/s) * (grad_circ[k] + grad_circ[(-k) mod nfft])
        # But we need to be careful about the linear vs circular convolution.
        
        # Simpler: just compute it directly for correctness, it's O(N^2) but N=600 so 360k ops
        # Actually let's use the FFT approach properly.
        
        W_padded = np.zeros(nfft)
        W_padded[:2*N-1] = w
        H_padded = np.zeros(nfft)
        H_padded[:N] = h
        
        FW = np.fft.rfft(W_padded, n=nfft)
        FH = np.fft.rfft(H_padded, n=nfft)
        
        # grad_circ[k] = ifft(FW * FH)[k] = sum_j w_j * h_{(k-j) mod nfft}
        # We want sum_j w_j * h_{j-k} and sum_j w_j * h_{j+k}
        # sum_j w_j * h_{j-k} = ifft(FW * FH)[k] ... let me verify
        # ifft(FW*FH)[k] = sum_j W_padded[j] * H_padded[(k-j) mod nfft]
        # = sum_j w_j * h_{(k-j) mod nfft}
        # For j-k in [0, N-1]: this gives h_{j-k} when j >= k
        # sum_j w_j * h_{j+k} = ifft(FW * conj(FH) shifted)... 
        
        # Let me just do it the direct way for now, it's fast enough
        # grad_h[k] = (1/s) * sum_j w_j * (h[max(0,j-k):N-1] + h[min(N-1,j+k)])
        # But this is O(N^2) = 360k, which is fine.
        
        # Actually, let me use a cleaner FFT approach:
        # The linear autoconvolution C_j = sum_{i=max(0,j-N+1)}^{min(N-1,j)} h_i * h_{j-i}
        # dC_j/dh_k = h_{j-k} + h_{j+k} (with h_m = 0 for m outside [0,N-1])
        # grad_h[k] = (1/s) * sum_j w_j * (h_{j-k} + h_{j+k})
        
        # Let A = w (length 2N-1), B = h (length N)
        # sum_j w_j * h_{j-k} = (w * h)_k where * is cross-correlation
        # = convolve(w, h)[k] in the appropriate sense
        
        # Use FFT: 
        # corr1[k] = sum_j w_j * h_{j-k} = ifft(fft(w) * conj(fft(h)))... 
        # Actually: sum_j w_j h_{j-k} = sum_j w_j h_{j-k}
        # Let i = j-k, j = i+k: sum_i w_{i+k} h_i = (w_shifted * h)_0
        # This is the cross-correlation of w and h at lag k.
        # Cross-correlation can be computed as ifft(fft(w) * conj(fft(h)))
        
        W_full = np.zeros(nfft)
        W_full[:2*N-1] = w
        H_full = np.zeros(nfft)
        H_full[:N] = h
        
        FW = np.fft.rfft(W_full, n=nfft)
        FH = np.fft.rfft(H_full, n=nfft)
        
        # cross_corr[k] = sum_j W_full[j] * H_full[(j-k) mod nfft]
        # = ifft(FW * conj(FH))[k]... let me check
        # ifft(FW * conj(FH))[k] = sum_j W_full[j] * H_full[(k-j) mod nfft]
        # = sum_j w_j * h_{(k-j) mod nfft}
        # For k in [0, N-1] and j in [0, 2N-2]:
        #   (k-j) mod nfft: if k-j >= 0, it's h_{k-j}; if k-j < 0, it wraps around
        # This isn't quite what we want.
        
        # Let me just use direct computation - it's only 600*1199 ~ 720k multiply-adds
        # grad_h[k] = (1/s) * sum_j w_j * (h[j-k] if 0<=j-k<N else 0 + h[j+k] if 0<=j+k<N else 0)
        
        # Vectorized direct computation:
        # For each k, we need sum over j of w_j * h_{j-k} and w_j * h_{j+k}
        # This is a Toeplitz-like matrix-vector product.
        
        # Build the Toeplitz matrix? No, too memory intensive.
        # Use FFT-based convolution properly:
        
        # sum_j w_j * h_{j-k} for j=0..2N-2, h defined on 0..N-1
        # = sum_{i=0}^{N-1} h_i * w_{i+k}  (where w_m = 0 if m outside [0,2N-2])
        # This is the cross-correlation of h and w at lag k.
        # cross_corr(k) = sum_i h_i * w_{i+k}
        # = ifft(fft(h) * conj(fft(w)))[k]  ... let me verify
        # ifft(FFT(h) * conj(FFT(w)))[k] = sum_i h_i * w_{(i-k) mod nfft}
        # That's w_{i-k}, not w_{i+k}. So we need:
        # cross_corr_plus(k) = sum_i h_i * w_{i+k} = ifft(FFT(h) * conj(FFT(w_rev)))[k]
        # where w_rev is w reversed... 
        
        # This is getting complicated. Let me just use the direct O(N^2) approach
        # since N=600, it's about 720k operations per gradient evaluation, which is fine.
        
        # Actually, for L-BFGS-B we'll do maybe 1000-2000 gradient evals,
        # so 720k * 2000 = 1.4 billion ops. That might be too slow.
        
        # Let me use a smarter FFT approach:
        # sum_j w_j * h_{j-k} = (w * h)_{k} where * denotes convolution with one argument reversed
        # = convolve(w, flip(h))[2N-2-k] ... 
        
        # OK, let me just use the standard trick:
        # corr[k] = sum_j w_j h_{j-k} = ifft(fft(w) * conj(fft(h)))[k] for circular
        # For linear, we zero-pad to nfft >= (2N-1) + N - 1 = 3N-2, but nfft=2048 > 1800, OK.
        
        # Wait, I already have nfft=2048. Let me use it.
        # corr_plus[k] = sum_j w_j * h_{j+k} for j=0..2N-2, h on 0..N-1
        # = sum_i h_i * w_{i-k}  (substituting j = i-k, but this changes bounds)
        # Hmm, the indices are tricky.
        
        # Let me just compute it directly with numpy broadcasting for the first version.
        # grad_h[k] = (1/s) * (sum_j w_j * h_{j-k} + sum_j w_j * h_{j+k})
        
        # For sum_j w_j * h_{j+k}: j ranges 0..2N-2, j+k ranges k..2N-2+k
        # h_{j+k} is nonzero when 0 <= j+k <= N-1, i.e., -k <= j <= N-1-k
        # So sum = sum_{j=max(0,-k)}^{min(2N-2, N-1-k)} w_j * h_{j+k}
        
        # This is a sliding window sum, which can be done with FFT.
        # Let me define:
        # A = w (length 2N-1)
        # B = h (length N)
        # C1[k] = sum_j A[j] * B[j+k] = cross-correlation at lag k
        # C2[k] = sum_j A[j] * B[j-k] = cross-correlation at lag -k
        
        # Cross-correlation: C1[k] = ifft(fft(A) * conj(fft(B)))[k] for circular
        # With zero-padding to nfft, this gives linear cross-correlation for valid k.
        
        FA = np.fft.rfft(W_full, n=nfft)
        FB = np.fft.rfft(H_full, n=nfft)
        
        # C1[k] = sum_j A[j] B[j-k] = ifft(FA * conj(FB))[k]
        corr1 = np.fft.irfft(FA * np.conj(FB), n=nfft)[:2*N-1]
        # C2[k] = sum_j A[j] B[j+k] = ifft(FA * conj(FB))[2N-1-k]... 
        # Actually C2[k] = sum_j A[j] B[j+k] = sum_i B[i] A[i-k] = C1[-k] = C1[2N-1-k] (by symmetry of A? No)
        # Let me just compute both:
        # C2[k] = sum_j A[j] B[j+k]
        # Let i = j+k, j = i-k: sum_i A[i-k] B[i] = sum_i B[i] A[i-k] = corr1[k]... 
        # Wait, that's the same as C1[k]! No:
        # C1[k] = sum_j A[j] B[j-k]
        # C2[k] = sum_j A[j] B[j+k]
        # These are different. C2[k] = C1[-k] if A and B were infinite, but with finite support they differ.
        
        # C2[k] = sum_j A[j] B[j+k] = sum_i B[i] A[i-k] where i = j+k
        # = sum_i B[i] A[i-k]
        # C1[k] = sum_j A[j] B[j-k] = sum_i A[i] B[i+k] where i = j
        # So C1[k] = sum_i A[i] B[i+k] and C2[k] = sum_i B[i] A[i-k]
        # C2[k] = C1[-k] only if A and B have the same support structure.
        
        # Let me just compute C2 separately:
        # C2[k] = sum_i B[i] A[i-k] = ifft(fft(B) * conj(fft(A)))[k]
        corr2 = np.fft.irfft(FB * np.conj(FA), n=nfft)[:2*N-1]
        
        # So grad_h[k] = (1/s) * (C1[k] + C2[k]) for k = 0..N-1
        grad_h = (corr1[:N] + corr2[:N]) / s
        
        # Gradient wrt t: d(smooth_max)/dt = 0 if we're just computing smooth_max(c)
        # But in the epigraph formulation, we'd have a different setup.
        # For now, just return the smooth metric gradient.
        
        f = temp * (np.log(np.sum(np.exp((c - np.max(c)) / temp))))
        grad = np.concatenate([grad_h, [0.0]])
        return f, grad

    # ---- Evaluate structured candidates ----
    best_val = np.inf
    best_h = np.ones(N)

    candidates = []
    candidates.append(np.ones(N))
    x = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(x)))
    n = np.arange(N)
    center = N / 2 - 0.5
    for w in [0.2, 0.3, 0.4]:
        sh = np.sinc((n - center) / (N * w))
        candidates.append(np.clip(sh, 0, None))
    xg = np.linspace(-3, 3, N)
    candidates.append(np.exp(-0.5 * xg**2))
    blackman = 0.42 - 0.5 * np.cos(2*np.pi*n/(N-1)) + 0.08*np.cos(4*np.pi*n/(N-1))
    candidates.append(np.clip(blackman, 0, None))
    hamming = 0.54 - 0.46*np.cos(2*np.pi*n/(N-1))
    candidates.append(np.clip(hamming, 0, None))
    hanning = 0.5 - 0.5*np.cos(2*np.pi*n/(N-1))
    candidates.append(np.clip(hanning, 0, None))
    flat_top = (0.21557895 + 0.41663158*np.cos(2*np.pi*n/(N-1)) +
                0.277263158*np.cos(4*np.pi*n/(N-1)) + 0.083578947*np.cos(6*np.pi*n/(N-1)) +
                0.006947368*np.cos(8*np.pi*n/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))
    # Two-plateau variants
    for a, b in [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3), (0.2, 0.8), (0.8, 0.2)]:
        tp = np.ones(N)
        tp[N//4:3*N//4] = a
        tp[:N//4] = b
        tp[3*N//4:] = b
        candidates.append(tp)
    # Sine bump
    candidates.append(np.sin(np.pi * n / (N - 1)))
    # Plateau in middle, zero edges
    for e in [0.1, 0.2]:
        pp = np.zeros(N)
        pp[int(e*N):N-int(e*N)] = 1.0
        candidates.append(pp)

    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = exact_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # ---- Gaussian mixture parameterization with differential evolution ----
    # This combines the structural idea from inspiration 1 with the refinement
    # machinery from the current program.
    
    DT = (1/4 - (-1/4)) / N  # interval width = 1/600
    X_GRID = -1/4 + (np.arange(N) + 0.5) * DT  # midpoints of intervals

    def gaussian_mixture_heights(params, n_components):
        """Convert parameter vector to step heights using Gaussian mixture.
        
        params layout: [amp_1, center_1, width_1, amp_2, center_2, width_2, ...]
        """
        h = np.zeros(N)
        for k in range(n_components):
            amp = params[3*k]
            c = params[3*k+1]
            w = params[3*k+2]
            if w < 1e-6:
                w = 1e-6
            h += amp * np.exp(-0.5 * ((X_GRID - c) / w) ** 2)
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-15:
            return np.ones(N) / N
        return h / s

    def gm_objective(params, n_components):
        """Objective for Gaussian mixture optimization."""
        if elapsed() > DEADLINE:
            return 1e10
        h = gaussian_mixture_heights(params, n_components)
        return exact_metric(h)

    # Run differential evolution on Gaussian mixture parameters
    # Use 6 components first (faster), then 10 if time permits
    de_configs = [
        (6, 80, 10, 42),
        (10, 60, 12, 123),
    ]
    
    for n_components, maxiter, popsize, seed in de_configs:
        if elapsed() > DEADLINE - 45:
            break
        n_params = 3 * n_components
        
        bounds = []
        for _ in range(n_components):
            bounds.append((0.0, 2.0))   # amplitude
            bounds.append((-0.25, 0.25)) # center
            bounds.append((0.01, 0.5))   # width
        
        try:
            de_result = differential_evolution(
                lambda p: gm_objective(p, n_components),
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=seed,
                polish=True
            )
            
            if de_result.x is not None:
                h_de = gaussian_mixture_heights(de_result.x, n_components)
                v_de = exact_metric(h_de)
                if v_de < best_val:
                    best_val = v_de
                    best_h = h_de.copy()
        except Exception:
            pass

    # ---- Smooth gradient-based optimization on the simplex ----
    def objective_and_grad(x, temp):
        # x = shape (nonnegative, not necessarily normalized). Normalize internally.
        h = np.maximum(x, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10, np.zeros(N)
        hn = h / s
        c = conv(hn)
        # Smooth max via log-sum-exp
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        weights = np.exp(c_shifted)
        w_sum = np.sum(weights)
        w = weights / w_sum
        f = temp * np.log(w_sum)
        
        # Gradient: grad_h[k] = (1/s) * sum_j w_j * (h_{j-k} + h_{j+k})
        # Using FFT cross-correlation
        W_full = np.zeros(nfft)
        W_full[:2*N-1] = w
        H_full = np.zeros(nfft)
        H_full[:N] = h
        
        FW = np.fft.rfft(W_full, n=nfft)
        FH = np.fft.rfft(H_full, n=nfft)
        
        # corr1[k] = sum_j w_j * h_{j-k}
        corr1 = np.fft.irfft(FW * np.conj(FH), n=nfft)[:2*N-1]
        # corr2[k] = sum_j w_j * h_{j+k}  
        corr2 = np.fft.irfft(FH * np.conj(FW), n=nfft)[:2*N-1]
        
        grad_h = (corr1[:N] + corr2[:N]) / s
        
        mask = x > 0
        grad = np.where(mask, grad_h, 0.0)
        return f, grad

    def objective_only(x, temp):
        return objective_and_grad(x, temp)[0]

    # Start from best candidate shape
    x0 = best_h.copy()
    x0 = x0 / np.sum(x0)  # normalize

    # Refined temperature schedule: start higher for global smoothing, then decrease
    # More iterations at low temperatures for precise local optimization
    temp_schedule = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    iter_schedule = [100, 200, 300, 400, 500, 600, 800, 1000]
    
    for idx, (temp, maxiter) in enumerate(zip(temp_schedule, iter_schedule)):
        if elapsed() > DEADLINE - 20:
            break
        try:
            res = minimize(
                lambda x: objective_only(x, temp),
                x0,
                jac=lambda x: objective_and_grad(x, temp)[1],
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-10, 'maxfun': maxiter * 3},
            )
            h_opt = np.maximum(res.x, 0.0)
            if np.sum(h_opt) > 1e-10:
                v = exact_metric(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
                    x0 = h_opt / np.sum(h_opt)
        except Exception:
            pass

    # ---- Final exact coordinate-descent polish ----
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(7)
    no_improve = 0
    max_no_improve = 8000
    mean_h = np.mean(current_h)

    # Use a finer grid of step sizes for more precise local search
    step_factors = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    
    while elapsed() < DEADLINE - 10 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]
        step_sizes = step_factors * mean_h
        cand_vals = np.concatenate([orig + step_sizes, orig - step_sizes, [0.0, orig]])
        cand_vals = np.array([max(0.0, v) for v in cand_vals])

        best_local_val = current_val
        best_local_v = orig
        for cv in cand_vals:
            if abs(cv - orig) < 1e-12:
                continue
            current_h[idx] = cv
            if np.sum(current_h) < 1e-15:
                current_h[idx] = orig
                continue
            v = exact_metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv

        if best_local_val < current_val - 1e-14:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            mean_h = np.mean(current_h)
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1

        # periodic global perturbation with varying strength
        if no_improve % 150 == 0:
            strength = 0.05 + 0.1 * (no_improve // 150) % 3  # 0.05, 0.15, 0.25
            pert = current_h.copy()
            pert += strength * np.random.randn(N) * mean_h
            pert = np.clip(pert, 0, None)
            if pert.sum() > 0:
                pv = exact_metric(pert)
                if pv < current_val:
                    current_h = pert
                    current_val = pv
                    no_improve = 0
                    mean_h = np.mean(current_h)
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()

    # ---- Two-coordinate fine tune with adaptive step size ----
    if elapsed() < DEADLINE - 5:
        step_frac = 0.002
        mean_best = np.mean(best_h)
        for _ in range(5000):
            if elapsed() > DEADLINE - 2:
                break
            i, j = np.random.choice(N, 2, replace=False)
            # Adaptive step: smaller when close to optimum
            delta = step_frac * np.random.uniform(-1, 1) * mean_best
            new_h = best_h.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] >= 0 and new_h[j] >= 0 and np.sum(new_h) > 1e-15:
                v = exact_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h
                    mean_best = np.mean(best_h)

    # Ensure nonnegative and valid
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    # Final micro-polish: try tiny perturbations on the top 50 indices by value
    if elapsed() < DEADLINE - 1:
        top_idx = np.argsort(best_h)[-50:]
        for idx in top_idx:
            if elapsed() > DEADLINE - 0.5:
                break
            orig = best_h[idx]
            for factor in [0.01, -0.01, 0.005, -0.005]:
                new_v = orig + factor * np.mean(best_h)
                if new_v < 0:
                    continue
                best_h[idx] = new_v
                v = exact_metric(best_h)
                if v < best_val:
                    best_val = v
                else:
                    best_h[idx] = orig
                    break

    return best_h.tolist()
