import numpy as np
import time

N = 400

def _autoconv(f):
    """Compute full autoconvolution via FFT."""
    n = len(f)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    F = np.fft.rfft(f, size)
    power = F * np.conj(F)
    auto = np.fft.irfft(power, size)[:2 * n - 1]
    return auto

def _score(auto, s):
    """Score = max|auto| / s^2. Lower is better."""
    if abs(s) < 1e-12:
        return float('inf')
    return float(np.max(np.abs(auto)) / (s * s))

def _update_autoconv_incremental(auto, f, i, delta):
    """
    Update autoconvolution when f[i] changes by delta.
    New auto = old auto + delta * (f shifted right by i + f shifted left by (n-1-i)) + delta^2 at lag 2i
    where the shifts are in the autoconvolution index space.
    """
    n = len(f)
    new_auto = auto.copy()
    
    # The autoconvolution at lag k is sum_j f[j]*f[j+k] for k>=0, and symmetric for k<0.
    # Changing f[i] by delta affects:
    # For lag k = i - j (i.e., j = i - k): the term f[j]*f[i] changes by delta * f[j]
    # For lag k = j - i (i.e., j = i + k): the term f[i]*f[j] changes by delta * f[j]
    # Plus the diagonal term at lag 0 when j=i: delta^2
    # More precisely, in the full autoconvolution of length 2n-1:
    # auto[lag] where lag ranges from -(n-1) to (n-1), stored as index 0..2n-2
    # Actually our auto is the full linear convolution f*f, which has length 2n-1.
    # auto[k] = sum_j f[j] * f[k-j] for k=0..2n-2
    # Changing f[i] by delta:
    # auto[k] += delta * f[k-i] (for the term where first factor is at i) 
    #         + delta * f[i - k + ... ] hmm this is getting confusing.
    # 
    # Let me think again. auto = conv(f, f) where conv(a,b)[k] = sum_j a[j]*b[k-j]
    # So auto[k] = sum_j f[j] * f[k-j]
    # If f[i] changes to f[i]+delta:
    # auto_new[k] = auto[k] + delta * f[k-i] (from j=i in the sum, first factor)
    #                        + delta * f[i - k + ... ] 
    # Wait, the second factor f[k-j] is affected when k-j = i, i.e., j = k-i.
    # So auto_new[k] = auto[k] + delta * f[k-i] (when j=i, second factor is f[k-i])
    #                       + delta * f[k-i] (when k-j=i, i.e., j=k-i, first factor is f[k-i])
    # Wait no. Let me be more careful.
    # auto[k] = sum_j f[j] * f[k-j]
    # Term 1: j = i contributes f[i] * f[k-i]. After change: (f[i]+delta) * f[k-i] = f[i]*f[k-i] + delta*f[k-i]
    # Term 2: k-j = i, i.e., j = k-i, contributes f[k-i] * f[i]. After change: f[k-i] * (f[i]+delta) = f[k-i]*f[i] + delta*f[k-i]
    # So if i != k-i (i.e., 2i != k), both terms exist and auto[k] += 2*delta*f[k-i]
    # If 2i == k, only one term (j=i=k-i) and auto[k] += delta*f[i] + delta*f[i]... 
    # Wait, when j=i and k-j=i, that's the same term. So auto[k] += delta * f[k-i] + delta * f[k-i] but 
    # when 2i=k, f[k-i]=f[i], so it's delta*f[i] + delta*f[i] = 2*delta*f[i]... 
    # Actually no. When 2i=k, the term j=i appears only once in the sum. So:
    # auto_new[k] - auto[k] = delta * f[k-i] (from j=i) + delta * f[k-i] (from j=k-i) if j != k-i
    # = 2*delta*f[k-i] if j != k-i (i.e., 2i != k)
    # = delta*f[i] + delta*f[i] = 2*delta*f[i]... but wait, f[k-i] = f[i] when 2i=k.
    # So in all cases: auto[k] += 2 * delta * f[k-i] for valid k-i
    # BUT when k-i = i (i.e., 2i=k), we have a double count? No:
    # The sum is over j. The term where j=i is f[i]*f[k-i]. The term where k-j=i is the same term when j=i.
    # So if 2i != k, there are two distinct terms: j=i and j=k-i.
    # If 2i == k, there is one term: j=i.
    # 
    # For 2i != k: auto[k] += delta*f[k-i] + delta*f[k-i] = 2*delta*f[k-i] (if both j=i and j=k-i are in [0,n-1])
    # For 2i == k: auto[k] += delta*f[i] (just one term, but f[i] changes so the term becomes (f[i]+delta)*f[i] = f[i]^2 + delta*f[i], change is delta*f[i])
    # Wait, I need to also account for the delta^2 term when the "new" f[i] interacts with itself.
    # 
    # Let me redo: auto_new[k] = sum_j f_new[j] * f_new[k-j]
    # f_new[i] = f[i] + delta, f_new[j] = f[j] for j != i
    # auto_new[k] = sum_{j != i} f[j]*f_new[k-j] + f_new[i]*f_new[k-i]
    # = sum_{j != i} f[j]*(f[k-j] + delta*I(k-j==i)) + (f[i]+delta)*(f[k-i]+delta*I(k-i==i))
    # 
    # This is getting complicated. Let me just use the linearity approach:
    # auto_new = auto + delta * (conv(e_i, f) + conv(f, e_i)) + delta^2 * conv(e_i, e_i)
    # where e_i is a unit impulse at position i.
    # conv(e_i, f)[k] = f[k-i] (shifted f)
    # conv(f, e_i)[k] = f[k-i] (same shift, since convolution is commutative for real signals... wait no)
    # Actually conv(a,b)[k] = sum_j a[j]*b[k-j]. conv(e_i, f)[k] = sum_j e_i[j]*f[k-j] = f[k-i].
    # conv(f, e_i)[k] = sum_j f[j]*e_i[k-j] = f[k-i]... no, e_i[k-j] = 1 when k-j=i, i.e., j=k-i. So conv(f,e_i)[k] = f[k-i].
    # So both are the same: shift of f.
    # conv(e_i, e_i)[k] = 1 when k-i = i, i.e., k=2i.
    # 
    # So: auto_new[k] = auto[k] + 2*delta*f[k-i] + delta^2 * (1 if k==2i else 0)
    # where f[k-i] is 0 if k-i is outside [0, n-1].

    # Create the shift
    shift = np.zeros(2 * n - 1)
    lo = max(0, i)
    hi = min(2 * n - 1, i + n)
    # f[k-i] is nonzero when 0 <= k-i <= n-1, i.e., i <= k <= i+n-1
    shift[i:i+n] = f
    
    new_auto = auto + 2 * delta * shift
    if 0 <= 2 * i < 2 * n - 1:
        new_auto[2 * i] += delta * delta
    
    return new_auto

def construct():
    deadline = time.time() + 145
    
    best_f = np.ones(N, dtype=np.float64)
    best_auto = _autoconv(best_f)
    best_s = float(np.sum(best_f))
    best_score = _score(best_auto, best_s)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Spectral shaping to get a good starting point
    size = 1
    while size < 2 * N - 1:
        size <<= 1
    
    current_f = best_f.copy()
    for _ in range(200):
        if time.time() > deadline - 60:
            break
        F = np.fft.rfft(current_f, size)
        mag = np.abs(F)
        phase = np.angle(F)
        target_mag = np.percentile(mag[1:], 30)
        new_mag = mag.copy()
        new_mag[1:] = target_mag
        new_F = new_mag * np.exp(1j * phase)
        new_f = np.fft.irfft(new_F, size)
        new_f_rounded = np.sign(new_f[:N])
        zero_mask = (new_f_rounded == 0)
        new_f_rounded[zero_mask] = current_f[zero_mask]
        current_f = new_f_rounded
        
        auto = _autoconv(current_f)
        s = float(np.sum(current_f))
        sc = _score(auto, s)
        if sc < best_score:
            best_score = sc
            best_f = current_f.copy()
            best_auto = auto.copy()
            best_s = s
    
    # Phase 2: Fast greedy local search with incremental updates
    if time.time() < deadline - 30:
        f = best_f.copy()
        auto = best_auto.copy()
        s = best_s
        current_score = best_score
        rng2 = np.random.default_rng(123)
        
        improved = True
        while improved and time.time() < deadline - 25:
            improved = False
            order = rng2.permutation(N)
            for i in order:
                if time.time() > deadline - 25:
                    break
                old_val = f[i]
                delta = -2.0 * old_val
                new_auto = _update_autoconv_incremental(auto, f, i, delta)
                new_s = s + delta
                new_score = _score(new_auto, new_s)
                if new_score < current_score - 1e-14:
                    f[i] = old_val + delta
                    auto = new_auto
                    s = new_s
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_f = f.copy()
                        best_auto = auto.copy()
                        best_s = s
                    improved = True
    
    # Phase 3: Simulated annealing for further refinement
    if time.time() < deadline - 5:
        f = best_f.copy()
        auto = best_auto.copy()
        s = best_s
        current_score = best_score
        rng3 = np.random.default_rng(789)
        
        start_temp = 0.05
        end_temp = 0.001
        max_iters = 200000
        it = 0
        while it < max_iters and time.time() < deadline - 2:
            it += 1
            progress = it / max_iters
            temp = start_temp * (end_temp / start_temp) ** progress
            
            i = int(rng3.integers(N))
            old_val = f[i]
            delta = -2.0 * old_val
            new_auto = _update_autoconv_incremental(auto, f, i, delta)
            new_s = s + delta
            new_score = _score(new_auto, new_s)
            
            if new_score < current_score:
                f[i] = old_val + delta
                auto = new_auto
                s = new_s
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
                    best_auto = auto.copy()
                    best_s = s
            else:
                delta_score = new_score - current_score
                if delta_score > 0 and temp > 1e-8:
                    if rng3.random() < np.exp(-delta_score / temp):
                        f[i] = old_val + delta
                        auto = new_auto
                        s = new_s
                        current_score = new_score
    
    # Final greedy polish
    if time.time() < deadline - 1:
        f = best_f.copy()
        auto = best_auto.copy()
        s = best_s
        current_score = best_score
        rng4 = np.random.default_rng(1111)
        order = rng4.permutation(N)
        for i in order:
            if time.time() > deadline - 1:
                break
            old_val = f[i]
            delta = -2.0 * old_val
            new_auto = _update_autoconv_incremental(auto, f, i, delta)
            new_s = s + delta
            new_score = _score(new_auto, new_s)
            if new_score < current_score - 1e-14:
                f[i] = old_val + delta
                auto = new_auto
                s = new_s
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
                    best_auto = auto.copy()
                    best_s = s
    
    return best_f
