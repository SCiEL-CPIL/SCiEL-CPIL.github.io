import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _legendre_sequence(n):
    """Generate Legendre symbol sequence for first n odd primes' residues mod p."""
    # Use quadratic residues mod a large prime as a deterministic low-autocorrelation sequence
    p = 401  # prime > N
    # Legendre symbol: (a/p) = a^((p-1)/2) mod p, giving +1 if QR, -1 if non-QR, 0 if a=0
    seq = np.zeros(n)
    for i in range(n):
        a = i + 1
        ls = pow(a, (p - 1) // 2, p)
        if ls == 1:
            seq[i] = 1.0
        elif ls == p - 1:
            seq[i] = -1.0
        else:
            seq[i] = 1.0  # a=0 mod p won't happen for i+1 < p
    return seq

def _mobius_like_sequence(n):
    """Generate a sequence based on Möbius function for first n integers."""
    seq = np.zeros(n)
    for i in range(n):
        m = i + 1
        # Compute Möbius function
        mu = 1
        x = m
        for p in range(2, int(np.sqrt(x)) + 1):
            if x % p == 0:
                x //= p
                if x % p == 0:
                    mu = 0
                    break
                else:
                    mu = -mu
        if x > 1:
            mu = -mu
        if mu == 0:
            seq[i] = 1.0
        else:
            seq[i] = float(mu)
    return seq

def _autocorr_from_flip(h, ac, i):
    """
    Compute new autocorrelation after flipping bit i in h.
    ac[t] = sum_k h[k]*h[k+t] (circular, length 2N via FFT)
    Flipping h[i] -> -h[i] changes ac by:
    For circular autocorrelation of length N:
    ac_new[t] = ac[t] - 2*h[i]*h[i+t mod N] - 2*h[i]*h[i-t mod N] + 2*h[i]*h[i] ... 
    Actually for linear aperiodic autocorrelation computed via FFT of length 2N:
    ac[t] = sum_{k=0}^{N-1-t} h[k]*h[k+t] for t>=0
    Flipping h[i]: ac_new[t] = ac[t] - 2*h[i]*h[i+t] (if i+t < N) 
                      - 2*h[i]*h[i-t] (if i-t >= 0, for the symmetric part)
    But since we use FFT-based linear autocorr, let's just do the O(N) update.
    """
    # For linear aperiodic autocorrelation: ac[t] = sum_{k=max(0,t-N+1)}^{min(N-1, N-1+t)} h[k]*h[k-t]
    # More simply: ac[t] for t in [0, N-1] = sum_{k=t}^{N-1} h[k]*h[k-t]
    # Flipping h[i] affects:
    #   ac[t] for t where i >= t and i-t < N: term h[i]*h[i-t] changes
    #   ac[t] for t where i+t < N: term h[i+t]*h[i] changes (this is ac[t] with k=i+t, k-t=i)
    # So for each t in [0, N-1]:
    #   if i >= t: ac[t] -= 2*h[i]*h[i-t]  (the h[i] in position k=i)
    #   if i+t < N: ac[t] -= 2*h[i+t]*h[i]  (the h[i] in position k-t=i, i.e., k=i+t)
    #   Also ac[N+t] = ac[N-t] by symmetry for linear autocorr
    
    h_old_i = h[i]
    h_new_i = -h_old_i
    
    # For t in 0..N-1:
    # Contribution from h[i] at position i: pairs with h[i-t] for t<=i, and h[i+t] for i+t<N
    # ac[t] includes h[k]*h[k-t] for k from t to N-1
    # When k=i (if i>=t): term is h[i]*h[i-t], changes by -2*h_old_i*h[i-t]
    # When k-t=i i.e. k=i+t (if i+t<N): term is h[i+t]*h[i], changes by -2*h[i+t]*h_old_i
    
    delta_ac = np.zeros(2 * N)
    # For t = 0 to N-1:
    # t where i >= t: t in [0, i]
    # t where i+t < N: t in [0, N-1-i]
    
    t_range = np.arange(0, N)
    
    # First term: k=i, valid when i >= t, i.e., t <= i
    mask1 = t_range <= i
    if np.any(mask1):
        t1 = t_range[mask1]
        delta_ac[t1] -= 2 * h_old_i * h[i - t1]
    
    # Second term: k=i+t, valid when i+t < N, i.e., t < N-i
    mask2 = t_range < (N - i)
    if np.any(mask2):
        t2 = t_range[mask2]
        delta_ac[t2] -= 2 * h[i + t2] * h_old_i
    
    # For the negative lags: ac[N+t] = ac[N-t] in the FFT representation
    # Actually in the rfft/irfft representation, ac has length 2N
    # ac[0..N-1] are lags 0..N-1, ac[N..2N-1] are lags -(N-1)..-1
    # ac[N+t] corresponds to lag -(t+1) for t=0..N-2, and ac[N-1]... 
    # Let me just recompute: for linear autocorr of length N,
    # ac_fft[t] for t=0..N-1: sum_{k=t}^{N-1} h[k]*h[k-t]  (lag t)
    # ac_fft[N+t] for t=0..N-1: sum_{k=0}^{N-1-t-1} h[k+t]*h[k] = ac_fft[N-t] ... no
    # Actually irfft gives: ac[t] for t=0..2N-1 where ac[t] = sum_k h[k]*h[(k+t) mod 2N]... no
    # The FFT method gives circular autocorrelation of the zero-padded signal.
    # h_padded = [h, 0, ..., 0] of length 2N
    # ac = ifft(fft(h_padded)^2)
    # ac[t] = sum_{k=0}^{2N-1} h_padded[k]*h_padded[(k+t) mod 2N]
    # Since h_padded[k]=0 for k>=N, ac[t] = sum_{k=max(0,t-N+1)}^{min(N-1, 2N-1-t)} h[k]*h[k+t-2N if k+t>=2N else k+t]
    # This is getting complex. Let me just do O(N) direct computation for the delta.
    
    # Simpler: just recompute ac from scratch but it's O(N log N). 
    # For speed, let's use the fact that flipping one element changes ac by a rank-1 update.
    # ac = ifft(|fft(h)|^2). If we flip h[i], fft(h_new) = fft(h) - 2*h[i]*fft(e_i)
    # where e_i is the unit vector at position i.
    # |F_new|^2 = |F - 2*h[i]*E|^2 = |F|^2 - 4*h[i]*(F*conj(E) + conj(F)*E) + 4*h[i]^2*|E|^2
    # This is O(N log N) for the FFTs. Not better.
    
    # Let's just return None to signal we should recompute.
    return None

def _compute_autocorr(h):
    """
    Compute linear aperiodic autocorrelation using FFT.
    Returns array of length 2*N.
    """
    n_fft = 2 * N
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    return ac

def _max_sidelobe(ac):
    """
    Compute max absolute value of sidelobes.
    """
    # Due to symmetry, max |ac[1:N]| is sufficient
    return np.max(np.abs(ac[1:N]))

def _soft_max_sidelobe(ac, beta=50.0):
    """
    Smooth approximation of max |ac[1:N]| using log-sum-exp.
    """
    sidelobes = np.abs(ac[1:N])
    max_val = np.max(sidelobes)
    shifted = beta * (sidelobes - max_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_val + log_sum / beta

def _spectral_objective_and_gradient(p, beta=50.0):
    """
    Objective: Optimize power spectrum p (real, non-negative) to minimize
    soft max of autocorrelation sidelobes.
    
    p is the real-valued power spectrum of length N_fft//2 + 1.
    The autocorrelation is the inverse FFT of p.
    We optimize p directly in the frequency domain.
    
    Returns (objective_value, gradient_wrt_p)
    """
    n_fft = 2 * N
    # p has length n_fft//2 + 1 (rfft output size)
    # The autocorrelation ac = ifft(p) where p is the power spectrum
    
    # Compute autocorrelation from power spectrum
    ac = np.fft.irfft(p, n=n_fft)
    
    # Sidelobes are ac[1:N] (and by symmetry ac[N+1:] = ac[N:0:-1])
    sidelobes_abs = np.abs(ac[1:N])
    
    # Soft max
    max_val = np.max(sidelobes_abs)
    shifted = beta * (sidelobes_abs - max_val)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    obj = max_val + np.log(sum_exp) / beta
    
    # Gradient: d obj / d p
    # ac = ifft(p), so d ac[k] / d p[m] involves the FFT matrix
    # For rfft/irfft: ac = (1/n_fft) * ifft(p_extended)
    # where p_extended is the full spectrum with conjugate symmetry
    
    # Let's compute gradient analytically
    # ac[t] = sum_m p[m] * exp(2*pi*i*m*t/n_fft) / n_fft (for real case)
    # Actually, irfft(p, n) gives ac[t] = (2/n) * sum_{m=1}^{n/2-1} p[m] cos(2*pi*m*t/n) + p[0]/n + p[n/2]/n * (-1)^t
    
    # More precisely for n_fft even:
    # ac[t] = (1/n_fft) * (p[0] + 2*sum_{m=1}^{n_fft/2-1} p[m] cos(2*pi*m*t/n_fft) + p[n_fft/2] * (-1)^t)
    
    # So d ac[t] / d p[m] = (2/n_fft) * cos(2*pi*m*t/n_fft) for 1 <= m < n_fft/2
    #                        (1/n_fft) for m = 0
    #                        (1/n_fft) * (-1)^t for m = n_fft/2
    
    # Gradient of soft max w.r.t. ac[1:N]:
    # w_k = beta * exp_shifted[k-1] / sum_exp for k=1..N-1
    # sign_k = sign(ac[k])
    # d obj / d ac[k] = w_k * sign_k
    
    # Then d obj / d p[m] = sum_{k=1}^{N-1} (d obj / d ac[k]) * (d ac[k] / d p[m])
    
    # Build gradient
    grad_p = np.zeros_like(p)
    
    # Precompute cos terms for efficiency
    # We need sum_{k=1}^{N-1} w_k * sign_k * cos(2*pi*m*k/n_fft) for each m
    
    # This is essentially an FFT-like operation
    # Let f[k] = w_k * sign_k for k=1..N-1
    # Then grad_p[m] = (2/n_fft) * sum_{k=1}^{N-1} f[k] * cos(2*pi*m*k/n_fft)
    
    f = np.zeros(N)
    f[1:N] = beta * exp_shifted / sum_exp * np.sign(ac[1:N])
    
    # Compute the cosine transform
    # grad_p[m] = (2/n_fft) * real(fft(f_padded)[m]) for m=1..n_fft/2-1
    # where f_padded has f[k] at positions k and n_fft-k
    
    f_padded = np.zeros(n_fft)
    f_padded[1:N] = f[1:N]
    f_padded[n_fft-N+1:n_fft] = f[1:N][::-1]
    
    F_f = np.fft.rfft(f_padded)
    
    # grad_p[m] = (2/n_fft) * F_f[m] for m=1..n_fft/2-1
    # grad_p[0] = (1/n_fft) * sum f[k] = (1/n_fft) * F_f[0]
    # grad_p[n_fft/2] = (1/n_fft) * sum f[k] * (-1)^k
    
    if len(p) > 0:
        grad_p[0] = (1.0 / n_fft) * F_f[0].real
    if len(p) > 1:
        grad_p[1:-1] = (2.0 / n_fft) * F_f[1:-1].real
    if len(p) > n_fft // 2:
        # For the Nyquist term
        alt = np.ones(N)
        alt[1::2] = -1.0
        grad_p[n_fft // 2] = (1.0 / n_fft) * np.sum(f[1:N] * alt[1:N])
    
    return obj, grad_p

def _local_search(h, max_iterations=10000, time_limit=5.0):
    """
    Fast local search using FFT for evaluation.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try flipping a random bit
        i = np.random.randint(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        
        if new_val < current_val:
            current_val = new_val
            ac = ac_new
        else:
            # Revert
            h[i] = old_h_i
            # ac remains unchanged (old ac)
            
    return h, current_val

def _simulated_annealing(h, max_iterations=50000, time_limit=10.0, T0=0.01, T_end=0.0001):
    """
    Simulated annealing with single-bit flips.
    """
    start_time = time.time()
    ac = _compute_autocorr(h)
    current_val = _max_sidelobe(ac)
    best_h = h.copy()
    best_val = current_val
    
    n_iter = max_iterations
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Annealing schedule
        progress = it / max(1, n_iter)
        T = T0 * (T_end / T0) ** progress
        
        # Try flipping a random bit
        i = np.random.randint(N)
        old_h_i = h[i]
        h[i] = -old_h_i
        
        ac_new = _compute_autocorr(h)
        new_val = _max_sidelobe(ac_new)
        
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_val = new_val
            ac = ac_new
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            h[i] = old_h_i
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Spectral Shaping Optimization
    # Optimize the power spectrum directly to minimize sidelobe peak
    n_fft = 2 * N
    p_size = n_fft // 2 + 1
    
    num_spectral_starts = 8
    betas = [20.0, 50.0, 100.0]
    
    # Also try number-theoretic sequences as direct initializations
    for i in range(num_spectral_starts):
        if time.time() >= deadline - 25.0:  # Reserve time for local search
            break
            
        # Initialize power spectrum
        # Start with a reasonable initial guess
        if i == 0:
            # Flat spectrum (corresponds to random signal)
            p0 = np.ones(p_size)
        elif i == 1:
            # Low-pass shaped
            freqs = np.linspace(0, 1, p_size)
            p0 = 1.0 + 0.5 * np.cos(np.pi * freqs)
        elif i == 2:
            # High-pass shaped
            freqs = np.linspace(0, 1, p_size)
            p0 = 1.0 + 0.5 * np.sin(np.pi * freqs)
        else:
            # Random positive spectrum
            p0 = np.random.uniform(0.5, 1.5, size=p_size)
        
        beta = betas[i % len(betas)]
        
        # Define objective with fixed beta
        def obj_grad(p, beta=beta):
            return _spectral_objective_and_gradient(p, beta)
        
        # Use L-BFGS-B with non-negativity constraint on p
        try:
            res = minimize(obj_grad, p0, method='L-BFGS-B', 
                           bounds=[(0.0, None)] * p_size,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30})
        except Exception:
            continue
        
        # Get the optimized power spectrum
        p_opt = np.maximum(res.x, 0.0)
        
        # Compute the corresponding signal
        # The autocorrelation is ac = ifft(p_opt)
        # We need to find h such that |FFT(h)|^2 ≈ p_opt
        # This is a phase retrieval problem
        # Simple heuristic: take the square root and use random phases
        
        # Compute the desired magnitude spectrum
        # p_opt is the power spectrum, so |F[m]| = sqrt(p_opt[m])
        mag_spec = np.sqrt(p_opt)
        
        # Create a complex spectrum with random phases
        # But we need h to be real, so F must have conjugate symmetry
        # F[m] = mag_spec[m] * exp(i * phase[m])
        # For real h: F[n_fft-m] = conj(F[m])
        
        # Use a simple approach: random phases for m=1..n_fft/2-1
        # F[0] and F[n_fft/2] are real
        
        phase = np.random.uniform(0, 2 * np.pi, size=p_size)
        phase[0] = 0.0  # DC term is real
        if p_size > n_fft // 2:
            phase[n_fft // 2] = 0.0  # Nyquist is real
        
        F_complex = mag_spec * np.exp(1j * phase)
        
        # Make it conjugate symmetric
        F_full = np.zeros(n_fft, dtype=complex)
        F_full[:p_size] = F_complex
        for m in range(1, n_fft // 2):
            F_full[n_fft - m] = np.conj(F_full[m])
        if n_fft % 2 == 0:
            F_full[n_fft // 2] = np.conj(F_full[n_fft // 2])
        
        # Inverse FFT to get h
        h_cont = np.fft.ifft(F_full).real
        
        # Threshold to ±1
        h_bin = np.sign(h_cont)
        zero_mask = (h_bin == 0)
        if np.any(zero_mask):
            h_bin[zero_mask] = np.random.choice([-1.0, 1.0], size=np.sum(zero_mask))
        
        # Evaluate binary solution
        ac = _compute_autocorr(h_bin)
        val = _max_sidelobe(ac)
        
        if val < best_val:
            best_val = val
            best_h = h_bin.copy()
    
    # Phase 1b: Number-theoretic initializations
    if time.time() < deadline - 20.0:
        for seq_gen in [_legendre_sequence, _mobius_like_sequence]:
            if time.time() >= deadline - 20.0:
                break
            h_nt = seq_gen(N)
            ac = _compute_autocorr(h_nt)
            val = _max_sidelobe(ac)
            if val < best_val:
                best_val = val
                best_h = h_nt.copy()
    
    # Phase 2: Refine best solution with local search
    if best_h is not None and time.time() < deadline - 15.0:
        h_ref, val_ref = _local_search(best_h.copy(), max_iterations=10000, time_limit=10.0)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Phase 3: Simulated Annealing
    if best_h is not None and time.time() < deadline - 10.0:
        remaining_sa = min(15.0, deadline - time.time() - 5.0)
        if remaining_sa > 2.0:
            h_sa, val_sa = _simulated_annealing(best_h.copy(), max_iterations=30000, time_limit=remaining_sa, T0=0.005, T_end=0.00005)
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa
    
    # Phase 4: Final Discrete Search
    remaining_time = deadline - time.time()
    if remaining_time > 3.0 and best_h is not None:
        h_final, val_final = _local_search(best_h.copy(), max_iterations=100000, time_limit=remaining_time - 2.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
            
    if best_h is None:
        return np.ones(N)
        
    return best_h
