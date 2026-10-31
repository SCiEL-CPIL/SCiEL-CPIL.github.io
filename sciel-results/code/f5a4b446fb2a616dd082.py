import time
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_metric(h):
    """Compute normalized maximum autoconvolution."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s < 1e-15:
        return 1e10
    hn = h / s
    # Linear autoconvolution via FFT
    fft_size = 2 * N
    F = np.fft.rfft(hn, fft_size)
    conv = np.fft.irfft(F * F, fft_size)[:2 * N - 1]
    return conv.max()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def elapsed():
        return time.time() - start_time

    # Grid definition
    # Intervals are equal on [-1/4, 1/4]
    # Bin centers:
    # width = 0.5 / 600
    # centers = -1/4 + (i + 0.5) * width
    bin_width = 0.5 / N
    centers = -0.25 + (np.arange(N) + 0.5) * bin_width

    h_best = np.ones(N)
    best_val = compute_metric(h_best)

    def gmm_to_h(params, K):
        """
        Convert GMM parameters to a height vector on the grid.
        Symmetric GMM: 
        K components. 
        If K is odd, one at 0, (K-1)/2 pairs.
        If K is even, K/2 pairs.
        For simplicity, we always use an odd number of components to ensure a central peak.
        params: [w0, w1, ..., w_{M-1}, mu1, ..., mu_{M-1}, log_sig0, log_sig1, ..., log_sig_{M-1}]
        where M = (K+1)//2.
        w0 is weight for center. wi (i>0) is weight for pair at +/- mu_i.
        """
        M = (K + 1) // 2
        w = params[:M]
        mus = params[M:2*M-1] # length M-1
        log_sigs = params[2*M-1:] # length M
        
        h = np.zeros(N)
        
        # Center component
        w0 = w[0]
        sig0 = np.exp(log_sigs[0])
        h += w0 * np.exp(-0.5 * (centers / sig0)**2)
        
        # Paired components
        for i in range(1, M):
            wi = w[i]
            mui = mus[i-1]
            sigi = np.exp(log_sigs[i])
            
            # Ensure mean is positive for the pair definition
            mui = abs(mui)
            
            # Component at +mui
            h += wi * np.exp(-0.5 * ((centers - mui) / sigi)**2)
            # Component at -mui
            h += wi * np.exp(-0.5 * ((centers + mui) / sigi)**2)
            
        return h

    def objective(params, K):
        if elapsed() > deadline:
            return 1e10
        h = gmm_to_h(params, K)
        return compute_metric(h)

    # Try different numbers of components
    K_candidates = [3, 5, 7]
    
    for K in K_candidates:
        if elapsed() > deadline - 20:
            break
            
        M = (K + 1) // 2
        n_params = 3 * M - 1 # M weights, M-1 means, M sigs
        
        # Initial guesses
        # 1. Single broad Gaussian (K=1 effectively, but we use K=3 with others weight 0)
        # 2. Two peaks
        # 3. Multi-peak
        
        inits = []
        
        # Init 1: Uniform-ish (very broad Gaussian)
        p = np.zeros(n_params)
        p[0] = 1.0 # w0
        # small weights for others
        for i in range(1, M):
            p[i] = 0.01
            p[M + i - 1] = 0.1 # mus
            p[2*M - 1 + i] = np.log(0.1) # sigs
        p[2*M - 1] = np.log(0.2) # sig0
        inits.append(p)
        
        # Init 2: Two distinct peaks
        if M >= 2:
            p = np.zeros(n_params)
            p[0] = 0.1
            p[1] = 0.9
            p[M] = 0.1 # mu1
            p[2*M - 1] = np.log(0.05) # sig0
            p[2*M] = np.log(0.05) # sig1
            inits.append(p)
            
        # Init 3: Three peaks (if M>=3)
        if M >= 3:
            p = np.zeros(n_params)
            p[0] = 0.2
            p[1] = 0.4
            p[2] = 0.4
            p[M] = 0.05 # mu1
            p[M+1] = 0.15 # mu2
            p[2*M - 1] = np.log(0.05)
            p[2*M] = np.log(0.05)
            p[2*M+1] = np.log(0.05)
            inits.append(p)

        for p0 in inits:
            if elapsed() > deadline - 10:
                break
                
            try:
                res = minimize(
                    objective, 
                    p0, 
                    args=(K,), 
                    method='L-BFGS-B', 
                    options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < best_val:
                    h_new = gmm_to_h(res.x, K)
                    val_new = compute_metric(h_new)
                    if val_new < best_val:
                        best_val = val_new
                        h_best = h_new.copy()
            except Exception:
                pass

    # Optional: Local refinement with coordinate descent on the best GMM result
    # This can help if the GMM is slightly suboptimal due to discretization effects
    if elapsed() < deadline - 15:
        h_curr = h_best.copy()
        curr_val = best_val
        
        # Simple coordinate descent
        step_sizes = [0.01, 0.05, 0.1]
        for _ in range(2): # 2 passes
            for i in range(N):
                if elapsed() > deadline - 5:
                    break
                for step in step_sizes:
                    for sign in [1, -1]:
                        h_trial = h_curr.copy()
                        h_trial[i] += sign * step
                        h_trial = np.maximum(h_trial, 0)
                        if h_trial.sum() < 1e-15:
                            continue
                        val_trial = compute_metric(h_trial)
                        if val_trial < curr_val:
                            h_curr = h_trial
                            curr_val = val_trial
                            if curr_val < best_val:
                                best_val = curr_val
                                h_best = h_trial.copy()

    h_best = np.maximum(h_best, 0.0)
    if h_best.sum() < 1e-15:
        h_best = np.ones(N)
        
    return list(h_best)
