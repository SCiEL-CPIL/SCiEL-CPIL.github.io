import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _ac_fft(x):
    """Full linear autoconvolution via FFT (real)."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    ac = np.fft.irfft(X * X, fsize)
    return ac[:size]

def _compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    Assumes x is a 1D numpy array.
    """
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    conv = np.fft.irfft(X * X, fsize)
    conv = conv[:size]
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_conv / (integral * integral)

def _soft_max_abs(ac, beta=50.0):
    """Smooth approximation of max(|ac|) using log-sum-exp."""
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _smooth_obj(h, beta=50.0):
    """Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _ac_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _local_search_fast(h, max_passes=50, deadline=None):
    """
    Fast local search using O(N) update rule for autoconvolution.
    h is a binary array {-1, 1}.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    ac = _ac_fft(h)
    s = np.sum(h)
    best_val = np.max(np.abs(ac)) / (s * s)
    
    # Precompute indices for fast updates
    # For flip at i:
    # delta_ac[1:N-i] -= 2*h[i]*h[i+1:]
    # delta_ac[1:i+1] -= 2*h[i]*h[:i][::-1]
    # Mirror to 2N-1
    
    for p in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try flipping each bit
        # We can randomize the order to avoid bias
        order = np.random.permutation(N)
        for i in order:
            if deadline is not None and time.time() > deadline:
                break
            
            # Calculate delta_ac for flipping h[i]
            delta_ac = np.zeros(2 * N - 1)
            hi = h[i]
            
            # Positive lags k > 0 where i+k < N
            # k ranges from 1 to N-1-i
            if N - 1 - i > 0:
                # delta_ac[1 : N-i]
                delta_ac[1:N-i] -= 2 * hi * h[i+1:]
            
            # Positive lags k > 0 where i-k >= 0
            # k ranges from 1 to i
            if i > 0:
                # delta_ac[1 : i+1]
                # h[:i] is h[0]...h[i-1]. We need h[i-1]...h[0] for k=1...i
                delta_ac[1:i+1] -= 2 * hi * h[:i][::-1]
            
            # Mirror the delta to the second half
            # ac is symmetric: ac[k] = ac[2N-2-k]
            # So delta_ac[k] = delta_ac[2N-2-k]
            # We computed for k=0..N-1 (implicitly 0 is 0, 1..N-1 are filled)
            # Need to fill N..2N-2
            # delta_ac[N:] should be reverse of delta_ac[:N-1]
            delta_ac[N:] = delta_ac[:N-1][::-1]
            
            # Update ac
            new_ac = ac - delta_ac
            new_s = s - 2 * hi
            if abs(new_s) < 1e-12:
                continue
            new_val = np.max(np.abs(new_ac)) / (new_s * new_s)
            
            if new_val < best_val:
                # Accept flip
                h[i] = -hi
                ac = new_ac
                s = new_s
                best_val = new_val
                improved = True
        
        if not improved:
            break
            
    return h, best_val

def _simulated_annealing(h, max_iters=5000, deadline=None):
    """
    Simple simulated annealing using fast update.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    ac = _ac_fft(h)
    s = np.sum(h)
    current_val = np.max(np.abs(ac)) / (s * s)
    best_h = h.copy()
    best_val = current_val
    
    T = 0.1
    cooling = 0.99
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        i = np.random.randint(N)
        hi = h[i]
        
        delta_ac = np.zeros(2 * N - 1)
        if N - 1 - i > 0:
            delta_ac[1:N-i] -= 2 * hi * h[i+1:]
        if i > 0:
            delta_ac[1:i+1] -= 2 * hi * h[:i][::-1]
        delta_ac[N:] = delta_ac[:N-1][::-1]
        
        new_ac = ac - delta_ac
        new_s = s - 2 * hi
        if abs(new_s) < 1e-12:
            continue
        new_val = np.max(np.abs(new_ac)) / (new_s * new_s)
        
        delta = new_val - current_val
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            h[i] = -hi
            ac = new_ac
            s = new_s
            current_val = new_val
            
            if current_val < best_val:
                best_val = current_val
                best_h = h.copy()
        
        T *= cooling
        if T < 1e-5:
            break
            
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Baseline
    best_x = np.ones(N, dtype=float)
    best_val = _compute_raw_value(best_x)
    
    # 2. Generate diverse candidates
    candidates = [best_x.copy()]
    t = np.arange(N)
    
    # Chirps
    for freq in [1.0, 2.0, 0.5, 3.0, 5.0]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1
        candidates.append(x_chirp)
        
    # Linear
    for freq in [1.0, 2.0, 0.5]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1
        candidates.append(x_lin)
        
    # Random
    rng = np.random.default_rng(42)
    for _ in range(20):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
        
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    x_gold = np.sign(np.cos(2 * np.pi * phi * t))
    x_gold[x_gold == 0] = 1
    candidates.append(x_gold)
    
    # 3. Evaluate candidates
    scored_candidates = []
    for x in candidates:
        val = _compute_raw_value(x)
        scored_candidates.append((val, x))
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    scored_candidates.sort(key=lambda t: t[0])
    
    # 4. Smooth Optimization from top starts
    top_starts = scored_candidates[:5]
    for init_val, h_init in top_starts:
        if time.time() > deadline - 30:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        
        for beta in [50, 100]:
            if time.time() > deadline - 15:
                break
            try:
                def obj(x):
                    return _smooth_obj(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': 200, 'disp': False, 'gtol': 1e-8})
                h_opt = res.x
                
                val = _compute_raw_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_x = h_opt.copy()
                    
                h_bin = np.sign(h_opt)
                h_bin[h_bin == 0] = 1
                val_bin = _compute_raw_value(h_bin)
                if val_bin < best_val:
                    best_val = val_bin
                    best_x = h_bin.copy()
                    
            except Exception:
                pass
                
    # 5. Fast Local Search
    if time.time() < deadline - 10:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        h_work, val_work = _local_search_fast(h_work, max_passes=100, deadline=deadline - 5)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
            
    # 6. Simulated Annealing
    if time.time() < deadline - 5:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        h_work, val_work = _simulated_annealing(h_work, max_iters=5000, deadline=deadline - 2)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
            
    # 7. Final Local Search
    if time.time() < deadline - 2:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        h_work, val_work = _local_search_fast(h_work, max_passes=50, deadline=deadline - 1)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
            
    return best_x

if __name__ == "__main__":
    x = construct()
    val = _compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
