import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin under 150s

def _max_overlap(h):
    """Maximum overlap of h with shifted (1-h)."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(np.abs(H)**2)) / N
    return 0.5 - np.min(autocorr)

def _soft_max_overlap(h, temperature=0.05):
    """Smooth surrogate for max overlap using log-sum-exp."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(np.abs(H)**2)) / N
    overlap = 0.5 - autocorr
    # log-sum-exp for numerical stability
    max_o = np.max(overlap)
    log_sum = max_o + np.log(np.sum(np.exp((overlap - max_o) / temperature))) * temperature
    return log_sum

def _soft_max_overlap_grad(h, temperature=0.05):
    """Gradient of soft-max overlap w.r.t. h."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(np.abs(H)**2)) / N
    overlap = 0.5 - autocorr
    max_o = np.max(overlap)
    weights = np.exp((overlap - max_o) / temperature)
    weights /= np.sum(weights)
    # d overlap[m] / d h[j] = -(2/N) * autocorr contribution... 
    # Actually: overlap[m] = 0.5 - (1/N) sum_j h_j h_{j+m}
    # d overlap[m] / d h[j] = -(1/N)(h_{j+m} + h_{j-m}) for m != 0
    # For m=0: d overlap[0] / d h[j] = -(2/N) h_j
    # Use FFT: grad_h = -2 * ifft(weights * H) / N ... let me derive properly.
    # overlap[m] = 0.5 - (1/N) sum_j h_j h_{(j+m) mod N}
    # d overlap[m] / d h[j] = -(1/N)(h_{(j+m) mod N} + h_{(j-m) mod N}) for m>0
    #                        = -(2/N) h_j for m=0
    # Sum over m: grad[j] = -sum_m weights[m] * (1/N)(h_{j+m} + h_{j-m})/2 - weights[0]*(1/N)*h_j
    # = -(1/N) sum_m weights[m] * (h_{j+m} + h_{j-m})/2 - weights[0]/N * h_j
    # Note: for m=0, (h_j + h_j)/2 = h_j, so it's consistent.
    # Using circular convolution: sum_m weights[m] h_{j+m} = (weights * h)[j] circular
    # = ifft(fft(weights) * fft(h))
    conv = np.real(np.fft.ifft(np.fft.fft(weights) * np.fft.fft(h)))
    grad = -(1.0/N) * conv - (weights[0]/N) * h
    # Wait, let me re-derive. 
    # overlap[m] = 0.5 - (1/N) sum_j h_j h_{(j+m)%N}
    # Let's define autocorr[m] = (1/N) sum_j h_j h_{(j+m)%N}
    # d autocorr[m] / d h[j] = (1/N)(h_{(j+m)%N} + h_{(j-m)%N}) for m != 0
    #                         = (2/N) h_j for m = 0
    # In general: d autocorr[m]/d h[j] = (1/N)(h_{(j+m)%N} + h_{(j-m)%N})
    # (For m=0 this gives (1/N)(h_j + h_j) = 2h_j/N, correct)
    # So d overlap[m]/d h[j] = -d autocorr[m]/d h[j] = -(1/N)(h_{(j+m)%N} + h_{(j-m)%N})
    # grad[j] = sum_m weights[m] * d overlap[m]/d h[j]
    #         = -(1/N) sum_m weights[m] (h_{(j+m)%N} + h_{(j-m)%N})
    #         = -(1/N) [sum_m weights[m] h_{(j+m)%N} + sum_m weights[m] h_{(j-m)%N}]
    # The first sum is circular cross-correlation of weights with h at j.
    # sum_m weights[m] h_{(j+m)%N} = ifft(fft(weights) * fft(h))[j] (circular conv)
    # sum_m weights[m] h_{(j-m)%N} = ifft(fft(weights) * conj(fft(h)))[j]... 
    # Actually for real h: sum_m weights[m] h_{(j-m)%N} = ifft(fft(weights) * fft(h))[j] too? No.
    # Let me use: sum_m w[m] h[(j+m)%N]. Let n = j+m, m = n-j. sum_n w[n-j] h[n].
    # This is (w * h)[j] where * is circular convolution: (w*h)[j] = sum_n w[(j-n)%N] h[n].
    # Hmm, let me just compute directly with FFT.
    # sum_m w[m] h[(j+m)%N]: define g[k] = h[(k)%N]. Then sum_m w[m] g[j+m].
    # This is the circular cross-correlation. In FFT: ifft(fft(w) * conj(fft(g)))? 
    # Let's just do: conv1[j] = sum_m w[m] h[(j+m)%N]
    # = sum_m w[m] h[(j+m)%N]. Let l = j+m, m = l-j. = sum_l w[l-j] h[l] = (w * h)[j] circ.
    # circular conv: (w*h)[j] = sum_l w[(j-l)%N] h[l]. So sum_l w[l-j] h[l] = sum_l w[(j-l)%N] h[l] if w is even? No.
    # w[l-j] vs w[(j-l)%N]: these are w at index (l-j) mod N and (j-l) mod N. They're different unless w is symmetric.
    # Let me just compute: conv1 = ifft(fft(w) * fft(h)) gives sum_l w[(j-l)%N] h[l].
    # I want sum_l w[(l-j)%N] h[l] = sum_l w[-(j-l)%N] h[l].
    # If I define w_rev[l] = w[-l % N], then sum_l w_rev[(j-l)%N] h[l] = ifft(fft(w_rev)*fft(h))[j].
    # This is getting complicated. Let me just use the fact that for real sequences:
    # sum_m w[m] h[(j+m)%N] = ifft(fft(w) * conj(fft(h)))[j] ... no.
    # Let me just compute it directly for N=100, it's fast enough.
    # Actually, let me use a simpler approach: compute grad numerically or use the autocorr relation.
    # 
    # Alternative: overlap[m] = 0.5 - autocorr[m]
    # autocorr = ifft(|H|^2)/N where H = fft(h)
    # d autocorr[m]/d h = ... this is messy.
    # 
    # Let me just do finite differences for the gradient in the optimizer, or use a simpler parameterization.
    
    # Simpler: just return a placeholder and use numerical gradient via scipy
    return None  # We'll use numerical gradient

def _objective_and_grad(x, temperature=0.05):
    """Objective: soft-max overlap of h = sigmoid(x) scaled to satisfy constraints.
    h = sigmoid(x), then clip to [0,1] and renormalize to sum = N/2."""
    h = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    h = np.clip(h, 0, 1)
    s = np.sum(h)
    if s > 1e-10:
        h *= (N/2.0) / s
        h = np.clip(h, 0, 1)
    val = _soft_max_overlap(h, temperature)
    return val

def _try_binary_sequences():
    """Try to find good binary sequences (h in {0,1}, sum = N/2) with low max overlap."""
    best_h = None
    best_score = float('inf')
    
    # For binary h, overlap[m] = 0.5 - (1/N) * (number of positions where h_j=1 and h_{j+m}=1)
    # = 0.5 - (1/N) * (N/4 - (number of 1-0 or 0-1 pairs at shift m))... 
    # Actually: sum_j h_j h_{j+m} = number of j where both h_j and h_{j+m} are 1.
    # For balanced binary (N/2 ones), autocorr[m] = (1/N) * count_ones_at_shift_m.
    # We want to minimize max overlap = 0.5 - min_m autocorr[m], so maximize min_m autocorr[m].
    # min_m autocorr[m] should be as high as possible, ideally close to N/4 (the average).
    
    # Try random balanced binary sequences
    np.random.seed(123)
    for trial in range(500):
        h = np.zeros(N)
        indices = np.random.choice(N, N//2, replace=False)
        h[indices] = 1.0
        score = _max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Try structured sequences: periodic patterns
    for period in range(2, 11):
        # Create a pattern of length period with period//2 ones
        pattern = np.zeros(period)
        pattern[:period//2] = 1.0
        h = np.tile(pattern, N//period)[:N]
        # Adjust to get exactly N/2 ones
        s = np.sum(h)
        if s != N//2:
            # Add or remove ones
            if s < N//2:
                zeros = np.where(h == 0)[0]
                if len(zeros) > 0:
                    for idx in zeros[:N//2 - int(s)]:
                        h[idx] = 1.0
            elif s > N//2:
                ones = np.where(h == 1)[0]
                for idx in ones[:int(s) - N//2]:
                    h[idx] = 0.0
        score = _max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Try m-sequence-like patterns
    for shift in range(1, 10):
        h = np.zeros(N)
        for j in range(N):
            if (j * shift) % N < N//2:
                h[j] = 1.0
        s = np.sum(h)
        if s != N//2:
            continue
        score = _max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    return best_h, best_score

def _polish(h, time_budget=30.0):
    """Polish a starting h using coordinate descent and local optimization."""
    start = time.time()
    h = h.copy()
    h = np.clip(h, 0, 1)
    s = np.sum(h)
    if s > 0:
        h *= (N/2.0) / s
        h = np.clip(h, 0, 1)
    best_score = _max_overlap(h)
    
    # Coordinate descent: try adjusting individual elements
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        for j in range(N):
            if time.time() - start > time_budget:
                break
            # Try different values for h[j]
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1, 0.2, -0.2]:
                h_test = h.copy()
                h_test[j] = np.clip(h_test[j] + delta, 0, 1)
                # Renormalize
                s_test = np.sum(h_test)
                if s_test > 0:
                    h_test *= (N/2.0) / s_test
                    h_test = np.clip(h_test, 0, 1)
                score_test = _max_overlap(h_test)
                if score_test < best_score - 1e-8:
                    h = h_test
                    best_score = score_test
                    improved = True
                    break
    
    # Try L-BFGS-B on a smoothed version
    if time.time() - start < time_budget - 5:
        # Parameterize h = sigmoid(x) / sum(sigmoid(x)) * N/2
        x0 = np.log(h / (1 - h + 1e-10) + 1e-10)  # inverse sigmoid
        x0 = np.clip(x0, -10, 10)
        
        def obj(x):
            h_s = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
            h_s = np.clip(h_s, 0, 1)
            s_s = np.sum(h_s)
            if s_s > 1e-10:
                h_s *= (N/2.0) / s_s
                h_s = np.clip(h_s, 0, 1)
            return _soft_max_overlap(h_s, 0.03)
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            h_opt = 1.0 / (1.0 + np.exp(-np.clip(result.x, -50, 50)))
            h_opt = np.clip(h_opt, 0, 1)
            s_opt = np.sum(h_opt)
            if s_opt > 0:
                h_opt *= (N/2.0) / s_opt
                h_opt = np.clip(h_opt, 0, 1)
            score_opt = _max_overlap(h_opt)
            if score_opt < best_score:
                h = h_opt
                best_score = score_opt
        except Exception:
            pass
    
    return h, best_score

def construct():
    start = time.time()
    
    # Get initial candidates
    # 1. Constant
    h_const = np.full(N, 0.5)
    score_const = _max_overlap(h_const)
    
    # 2. Binary sequences
    h_bin, score_bin = _try_binary_sequences()
    
    # 3. Random continuous
    np.random.seed(42)
    h_rand = np.random.uniform(0, 1, N)
    h_rand *= (N/2.0) / np.sum(h_rand)
    h_rand = np.clip(h_rand, 0, 1)
    score_rand = _max_overlap(h_rand)
    
    # Pick best starting point
    candidates = [(h_const, score_const), (h_bin, score_bin), (h_rand, score_rand)]
    candidates = [c for c in candidates if c[0] is not None]
    candidates.sort(key=lambda x: x[1])
    best_h, best_score = candidates[0]
    
    # Polish the best candidate
    if time.time() - start < DEADLINE - 10:
        best_h, best_score = _polish(best_h, time_budget=min(60, DEADLINE - 10 - (time.time() - start)))
    
    # Try polishing other candidates if time allows
    for h_cand, score_cand in candidates[1:]:
        if time.time() - start > DEADLINE - 15:
            break
        if score_cand >= best_score:
            continue
        h_polished, score_polished = _polish(h_cand, time_budget=min(20, DEADLINE - 15 - (time.time() - start)))
        if score_polished < best_score:
            best_h = h_polished
            best_score = score_polished
    
    # Final: ensure best_h is valid
    h_final = np.clip(best_h, 0, 1)
    s = np.sum(h_final)
    if s > 0:
        h_final *= (N/2.0) / s
        h_final = np.clip(h_final, 0, 1)
    
    # Ensure sum is exactly N/2
    diff = N/2.0 - np.sum(h_final)
    if abs(diff) > 1e-10:
        if diff > 0:
            idx = np.argmin(h_final)
            h_final[idx] = min(1.0, h_final[idx] + diff)
            remaining = N/2.0 - np.sum(h_final)
            if abs(remaining) > 1e-10:
                mask = h_final < 1.0
                if np.any(mask):
                    room = 1.0 - h_final[mask]
                    total_room = np.sum(room)
                    if total_room > 0:
                        h_final[mask] += remaining * room / total_room
        else:
            idx = np.argmax(h_final)
            h_final[idx] = max(0.0, h_final[idx] + diff)
            remaining = N/2.0 - np.sum(h_final)
            if abs(remaining) > 1e-10:
                mask = h_final > 0.0
                if np.any(mask):
                    total = np.sum(h_final[mask])
                    if total > 0:
                        h_final[mask] += remaining * h_final[mask] / total
    
    # Final check
    if not (np.all(h_final >= -1e-9) and np.all(h_final <= 1.0 + 1e-9) and abs(np.sum(h_final) - N/2.0) < 1e-6):
        return np.full(N, 0.5)
    
    return h_final
