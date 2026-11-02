"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

SCALE = 2.0 / N

def _compute_O(h):
    """
    Compute the overlap vector O for all shifts k=0..N-1.
    O[k] = SCALE * sum_{j} h[j] * (1 - h[(j-k) % N])
    This is the circular cross-correlation-based overlap.
    """
    # For shift k, overlap = sum_j h[j] * (1 - h[(j-k) mod N])
    # = sum_j h[j] - sum_j h[j]*h[(j-k) mod N]
    # sum_j h[j] = N/2 (by constraint)
    # So O[k] = SCALE * (N/2 - corr[k]) where corr[k] = sum_j h[j]*h[(j-k)%N]
    
    # Use FFT for efficient circular cross-correlation
    H = np.fft.rfft(h, n=2*N)
    corr = np.fft.irfft(H * np.conj(H), n=2*N)[:N]
    
    # corr[k] = sum_j h[j] * h[(j-k) mod N] for k=0..N-1
    # Actually let me verify: if H = FFT(h), then IFFT(|H|^2) gives circular autocorrelation
    # corr[k] = sum_j h[j] * h[(j-k) mod N]
    
    total = np.sum(h)  # should be N/2
    O = SCALE * (total - corr)
    return O

def _max_overlap(h):
    """Compute maximum overlap."""
    return float(np.max(_compute_O(h)))

def _project(z):
    """
    Project z onto {h in [0,1]^N : sum(h) = N/2}
    Using bisection on the Lagrange multiplier.
    """
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = N / 2.0
    
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)

def _softmax_grad(h, tau):
    """
    Compute gradient of softmax-smoothed max overlap.
    L = max_k O[k] + (1/tau) * log(sum_k exp(tau*(O[k]-max)))
    dL/dh_i = (1/tau) * sum_k w_k * dO_k/dh_i
    where w_k = exp(tau*(O[k]-max)) / sum_j exp(tau*(O[j]-max))
    
    O[k] = SCALE * (N/2 - corr[k])
    corr[k] = sum_j h[j] * h[(j-k)%N]
    dO[k]/dh[i] = -SCALE * (h[(i-k)%N] + h[(i+k)%N]) for k != 0
    dO[0]/dh[i] = -SCALE * 2*h[i]
    
    Actually: d(corr[k])/dh[i] = h[(i-k)%N] + h[(i+k)%N] for k != 0
    and d(corr[0])/dh[i] = 2*h[i]
    
    So dO[k]/dh[i] = -SCALE * (h[(i-k)%N] + h[(i+k)%N]) for k != 0
    dO[0]/dh[i] = -SCALE * 2*h[i]
    """
    O = _compute_O(h)
    Omax = float(O.max())
    
    # Softmax weights
    z = tau * (O - Omax)
    ez = np.exp(z - z.max())  # numerical stability
    w = ez / ez.sum()
    
    # Compute gradient: grad[i] = sum_k w[k] * dO[k]/dh[i]
    # dO[k]/dh[i] = -SCALE * (h[(i-k)%N] + h[(i+k)%N]) for k != 0
    # dO[0]/dh[i] = -SCALE * 2*h[i]
    
    # Vectorized computation
    # For each i, grad[i] = -SCALE * sum_k w[k] * (h[(i-k)%N] + h[(i+k)%N])
    # = -SCALE * (sum_k w[k]*h[(i-k)%N] + sum_k w[k]*h[(i+k)%N])
    # = -SCALE * (conv(w, h)[i] + conv(w, h)[i])  [since w is symmetric-ish]
    # Actually conv(w,h)[i] = sum_k w[k]*h[(i-k)%N]
    
    # Use FFT for circular convolution
    W = np.fft.rfft(w, n=2*N)
    H = np.fft.rfft(h, n=2*N)
    conv_wh = np.fft.irfft(W * H, n=2*N)[:N]
    # conv_wh[i] = sum_k w[k] * h[(i-k)%N]
    
    # Similarly, sum_k w[k]*h[(i+k)%N] = conv(w_rev, h)[i] where w_rev[k]=w[-k]
    # Or: sum_k w[k]*h[(i+k)%N] = sum_j w[(i-j)%N]*h[j] = conv(w, h)[i] with w reversed
    # Actually: let's define w2[k] = w[-k % N], then conv(w2, h)[i] = sum_k w2[k]*h[(i-k)%N] = sum_k w[-k]*h[(i-k)%N]
    # Let j = -k, = sum_j w[j]*h[(i+j)%N] which is what we want.
    
    w2 = np.zeros(N)
    w2[0] = w[0]
    w2[1:] = w[:0:-1]  # w2[k] = w[N-k] for k=1..N-1
    
    W2 = np.fft.rfft(w2, n=2*N)
    conv_w2h = np.fft.irfft(W2 * H, n=2*N)[:N]
    
    grad = -SCALE * (conv_wh + conv_w2h)
    
    # The softmax gradient adds a factor: the actual gradient of the smooth max is
    # dL/dh[i] = (1/tau) * sum_k w[k] * dO[k]/dh[i] ... wait no.
    # L = Omax + (1/tau)*log(sum exp(tau*(O-Omax)))
    # dL/dh[i] = sum_k [delta_{k,argmax} + (1/tau)*w_k*tau] * dO[k]/dh[i]
    # = sum_k (delta_{k,argmax} + w_k) * dO[k]/dh[i]
    # But when tau is large, w concentrates on argmax, so it's approximately the subgradient.
    # For the gradient descent, we use: grad_smooth = sum_k w_k * dO[k]/dh[i]
    # and the step is h_new = project(h - eta * grad_smooth)
    # The (1/tau) factor is absorbed into eta.
    
    return Omax, grad

def _optimize_from_init(init, deadline, max_iters=300):
    """Run projected gradient descent from an initial point."""
    h = _project(np.asarray(init, dtype=float).copy())
    current_val = _max_overlap(h)
    best_h_local = h.copy()
    best_val_local = current_val
    
    tau_schedule = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for tau in tau_schedule:
        eta = 0.3 / (1.0 + 0.05 * tau)
        for _ in range(max_iters):
            if time.time() > deadline - 3.0:
                return best_h_local, best_val_local
            
            Omax, grad = _softmax_grad(h, tau)
            if not np.all(np.isfinite(grad)):
                break
            
            h_new = _project(h - eta * grad)
            val_new = _max_overlap(h_new)
            
            if val_new < best_val_local - 1e-12:
                best_val_local = val_new
                best_h_local = h_new.copy()
            
            h = h_new
    
    return best_h_local, best_val_local

def _polish(h, best_val, deadline):
    """Fine-tune with very small steps and high tau."""
    h = h.copy()
    current_val = best_val
    
    for tau in [500.0, 1000.0, 2000.0]:
        eta = 0.05
        for _ in range(500):
            if time.time() > deadline - 1.0:
                return h, current_val
            
            Omax, grad = _softmax_grad(h, tau)
            if not np.all(np.isfinite(grad)):
                break
            
            h_new = _project(h - eta * grad)
            val_new = _max_overlap(h_new)
            
            if val_new < current_val - 1e-12:
                current_val = val_new
                h = h_new
            else:
                eta *= 0.9
                if eta < 1e-7:
                    break
    
    # Very fine final steps
    tau = 5000.0
    eta = 0.01
    for _ in range(1000):
        if time.time() > deadline - 0.5:
            break
        
        Omax, grad = _softmax_grad(h, tau)
        if not np.all(np.isfinite(grad)):
            break
        
        h_new = _project(h - eta * grad)
        val_new = _max_overlap(h_new)
        
        if val_new < current_val - 1e-12:
            current_val = val_new
            h = h_new
        else:
            eta *= 0.95
            if eta < 1e-8:
                break
    
    return h, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    idx = np.arange(N)
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.full(N, 0.5))
    
    # 2. Bang-bang random (exactly N/2 ones)
    for _ in range(10):
        bb = np.zeros(N)
        ones_idx = rng.choice(N, N // 2, replace=False)
        bb[ones_idx] = 1.0
        inits.append(bb)
    
    # 3. Periodic bang-bang patterns
    for period in [2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 17, 20]:
        bb = np.zeros(N)
        ones_per_period = period // 2
        pattern = np.zeros(period)
        pattern[:ones_per_period] = 1.0
        bb = np.tile(pattern, N // period + 1)[:N]
        
        # Adjust to have exactly N/2 ones
        num_ones = int(np.sum(bb == 1))
        if num_ones > N // 2:
            ones_idx = np.where(bb == 1)[0]
            to_remove = rng.choice(ones_idx, num_ones - N // 2, replace=False)
            bb[to_remove] = 0
        elif num_ones < N // 2:
            zeros_idx = np.where(bb == 0)[0]
            to_add = rng.choice(zeros_idx, N // 2 - num_ones, replace=False)
            bb[to_add] = 1
        inits.append(bb)
    
    # 4. Continuous structured (cosine)
    for f in [1, 2, 3, 5, 7, 11, 13]:
        for amp in [0.2, 0.3, 0.4, 0.5]:
            h = 0.5 + amp * np.cos(2 * np.pi * f * idx / N + rng.uniform(0, 2*np.pi))
            inits.append(h)
    
    # 5. Random continuous
    for _ in range(10):
        inits.append(rng.random(N))
    
    # 6. Alternating patterns with noise
    for _ in range(5):
        base = np.where(idx % 2 == 0, 0.8, 0.2)
        h = base + rng.normal(0, 0.1, N)
        inits.append(h)
    
    # Optimize each initialization
    best_h = inits[0].copy()
    best_val = _max_overlap(best_h)
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        try:
            h, val = _optimize_from_init(init, deadline, max_iters=200)
            if np.all(np.isfinite(h)) and val < best_val - 1e-10:
                best_val = val
                best_h = h.copy()
        except Exception:
            continue
    
    # Final polish
    if time.time() < deadline - 5.0:
        best_h, best_val = _polish(best_h, best_val, deadline)
    
    # Ensure exact sum and bounds
    best_h = _project(best_h)
    for _ in range(20):
        d = N / 2.0 - best_h.sum()
        if abs(d) < 1e-12:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
