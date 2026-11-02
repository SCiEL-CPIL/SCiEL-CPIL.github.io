"""Search for a low-autocorrelation signed step function (400 equal intervals).

Minimise the peak aperiodic autocorrelation sidelobe  max_{k>=1} |sum_i h[i] h[i+k]|
for a +-1 sequence of length 400; this drives the C3 autoconvolution upper bound.

Design: continuous gradient descent on L2 autocorrelation norm (flat spectrum) to
generate high-quality binary seeds, followed by simulated annealing with variable-
length block flips, interleaved with greedy single-flip and adjacent-pair-flip
polish for exploitation. Each evaluation is O(N log N) via FFT.
"""
import numpy as np
import time

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i]*h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))


def _continuous_seed(seed, time_budget):
    """Generate a good binary seed via continuous gradient descent on L2 autocorrelation norm.
    
    Minimises sum_k |ac[k]|^2 (flat spectrum) in continuous space, then rounds to ±1.
    This provides a much better starting point for binary local search than random.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N)
    start = time.time()
    
    # Gradient descent with adaptive learning rate
    lr = 0.01
    momentum = np.zeros(N)
    best_x = x.copy()
    best_l2 = _l2_autocorr(x)
    
    while time.time() - start < time_budget:
        # Compute gradient of sum_k |ac[k]|^2
        H = np.fft.fft(x, _FFT_SIZE)
        ac = np.fft.ifft(H * np.conj(H)).real[:N]
        l2_val = float(np.sum(ac[1:] ** 2))
        
        if l2_val < best_l2:
            best_l2 = l2_val
            best_x = x.copy()
        
        # Gradient: d/dx_i sum_k ac[k]^2 = 2 * sum_k 2*ac[k] * d(ac[k])/d(x_i)
        # d(ac[k])/d(x_i) = x[i+k] for i+k < N (and x[i-k] for the symmetric part)
        # Using FFT: grad = 4 * ifft(FFT(x) * conj(FFT(x)) * FFT(x)) restricted to [1:N]
        # Simpler: compute autocorrelation gradient directly
        # ac[k] = sum_i x[i]*x[i+k], so d(ac[k])/d(x_i) = x[i+k] + x[i-k] (with boundary handling)
        
        # Efficient gradient via FFT: 
        # The L2 norm of autocorrelation is related to the power spectrum
        # sum_k |ac[k]|^2 = sum_f |S(f)|^4 / N where S is the spectrum
        # Gradient w.r.t. x: 4 * ifft(|S|^2 * S) but we need to be careful
        
        # Direct approach: gradient of sum_k ac[k]^2 where ac[k] = sum_i x[i]x[i+k]
        # d/dx_i sum_k ac[k]^2 = 2 * sum_k ac[k] * (x[i+k] + x[i-k]) (with appropriate bounds)
        
        # Using FFT-based computation:
        # Let's compute the gradient numerically via FFT
        # ac_full[k] for k in [-N+1, N-1]
        # sum_k ac[k]^2 where we sum over k=1..N-1 (one-sided)
        # Actually for real signals, ac[-k] = ac[k], so sum_{k=1}^{N-1} ac[k]^2 
        # = 0.5 * sum_{k=-N+1}^{N-1, k!=0} ac[k]^2
        # = 0.5 * (sum_{k=-N+1}^{N-1} ac[k]^2 - ac[0]^2)
        # = 0.5 * N * sum_f |S(f)|^4 - ac[0]^2 (by Parseval for the autocorrelation)
        
        # Gradient: d/dx_i [0.5 * sum_{k!=0} ac[k]^2] 
        # = sum_{k!=0} ac[k] * d(ac[k])/d(x_i)
        # d(ac[k])/d(x_i) = x[i+k] if 0<=i+k<N, plus x[i-k] if 0<=i-k<N
        # This is equivalent to: (x * x_reversed)_circular but with aperiodic boundaries
        
        # FFT approach for the gradient:
        # The quantity sum_{k} ac[k]^2 * delta(k - something) 
        # Let me just compute it directly with the autocorrelation
        grad = np.zeros(N)
        # For each lag k, ac[k] contributes ac[k] * x[i+k] to grad[i] and ac[k] * x[i-k] to grad[i]
        # Equivalently: grad[i] = 2 * sum_{k=1}^{N-1-i} ac[k] * x[i+k] + 2 * sum_{k=1}^{i} ac[k] * x[i-k]
        # (factor of 2 from the square, and we're summing k>=1)
        # Wait, let me be more careful:
        # f = sum_{k=1}^{N-1} ac[k]^2
        # df/dx_i = 2 * sum_{k=1}^{N-1} ac[k] * d(ac[k])/d(x_i)
        # ac[k] = sum_{j=0}^{N-1-k} x[j] * x[j+k]
        # d(ac[k])/d(x_i) = x[i+k] (if i+k < N) + x[i-k] (if i-k >= 0)
        # So df/dx_i = 2 * sum_{k=1}^{N-1} ac[k] * (x[i+k]*1[i+k<N] + x[i-k]*1[i-k>=0])
        
        # This is O(N^2) which is too slow. Let me use FFT.
        # sum_{k=1}^{N-1} ac[k] * x[i+k] = (ac_conv_x)[i] where ac_conv is circular convolution
        # But ac is aperiodic... 
        # For the FFT approach, let's pad and use circular convolution as an approximation,
        # or compute the gradient using the spectrum.
        
        # Spectrum approach:
        # sum_{k} |ac[k]|^2 (all k) = N * sum_f |S(f)|^4 (by Parseval applied to autocorrelation)
        # Actually: ac[k] = ifft(|S|^2), so sum_k |ac[k]|^2 = N * sum_f ||S(f)|^2|^2 = N * sum_f |S(f)|^4
        # df/dx_i where f = sum_{k=1}^{N-1} ac[k]^2 (real, so ac[k] is real for real x)
        # = 2 * sum_{k=1}^{N-1} ac[k] * d(ac[k])/d(x_i)
        # 
        # Let me use a different approach: compute gradient via the spectrum
        # S(f) = FFT(x)
        # ac[k] = ifft(|S|^2)[k]
        # f = sum_{k=1}^{N-1} ac[k]^2
        # df/dS(f) = 2 * ifft_conj... this is getting complex.
        
        # Let me just do it in O(N^2) but with vectorization for N=400, it's 160k ops per step
        # which is fine.
        # grad[i] = 2 * sum_{k=1}^{N-1} ac[k] * (x[min(i+k, N-1)] * (i+k < N) + x[max(i-k, 0)] * (i-k >= 0))
        # Actually let me be precise:
        # d(ac[k])/d(x_i) = x[i+k] if i+k <= N-1, else 0, PLUS x[i-k] if i-k >= 0, else 0
        
        # Vectorized computation:
        # term1[i] = sum_{k=1}^{N-1} ac[k] * x[i+k] (where i+k < N)
        # term2[i] = sum_{k=1}^{N-1} ac[k] * x[i-k] (where i-k >= 0)
        
        # term1: for each i, sum over k where i+k < N, i.e., k < N-i
        # term1[i] = sum_{k=1}^{min(N-1, N-1-i)} ac[k] * x[i+k]
        # term2[i] = sum_{k=1}^{min(N-1, i)} ac[k] * x[i-k]
        
        # This is a convolution-like operation. Let me use the fact that:
        # term1 = (ac_reversed * x)_aperiodic shifted... 
        # Actually, let me just compute it with a loop over lags, which is O(N^2) but vectorized over i
        
        # For N=400, this is 400 iterations of vector ops of length ~400, so ~160k mults total
        # That's fast enough.
        
        # Alternative: use the full circular autocorrelation and accept boundary effects
        # For a good approximation:
        # grad ≈ 4 * ifft(FFT(x) * |FFT(x)|^2) ... let me check
        # 
        # Let S = FFT(x), then circular autocorrelation r[k] = ifft(|S|^2)[k]
        # sum_k r[k]^2 = N * sum_f |S(f)|^4
        # d/dx_i sum_k r[k]^2 = 4 * (r * x)[i] (circular convolution) = 4 * ifft(FFT(r) * conj(FFT(x)))... 
        # Hmm, let me think again.
        # r[k] = sum_j x[j] x[j-k] (circular)
        # sum_k r[k]^2 = sum_k (sum_j x[j]x[j-k])(sum_l x[l]x[l-k])
        # This is getting complicated. Let me just do the O(N^2) computation.
        
        # Actually, for N=400, let me just compute the gradient directly:
        # grad[i] = 2 * (sum_{k=1}^{N-1-i} ac[k]*x[i+k] + sum_{k=1}^{i} ac[k]*x[i-k])
        
        # Use np.correlate or manual vectorization
        # term1[i] = sum_{k=1}^{N-1-i} ac[k] * x[i+k]
        # This is the cross-correlation of ac[1:] with x, evaluated at specific points
        # Let ac_pos = ac[1:]  (length N-1)
        # term1[i] = sum_{k=0}^{N-2-i} ac_pos[k] * x[i+1+k]
        # = (ac_pos * x)[i+1] in the aperiodic sense
        
        # Let me just do it with a simple loop for clarity, N=400 is small
        for k in range(1, N):
            # ac[k] contributes ac[k]*x[i+k] to grad[i] for i=0..N-1-k
            # and ac[k]*x[i-k] to grad[i] for i=k..N-1
            grad[:N-k] += 2.0 * ac[k] * x[k:]
            grad[k:] += 2.0 * ac[k] * x[:N-k]
        
        # Update with momentum
        momentum = 0.9 * momentum + 0.1 * grad
        x = x - lr * momentum
        
        # Adaptive learning rate: reduce if loss increases
        # (we already track best, so if we're stuck, reduce lr)
        if l2_val > best_l2 * 1.01:
            lr *= 0.9
        
        # Normalize to keep values reasonable
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm * np.sqrt(N)
    
    # Round to ±1
    h = np.where(best_x >= 0, 1.0, -1.0)
    return h, _peak_sidelobe(h)


def _anneal(seed, time_budget):
    """Simulated annealing with variable-length block flips.

    Block length shrinks over time: longer blocks early for exploration,
    shorter blocks late for fine-tuning.
    """
    rng = np.random.default_rng(seed)
    h = rng.choice([-1.0, 1.0], size=N)
    cur = _peak_sidelobe(h)
    best_h = h.copy()
    best = cur
    T = 4.0
    T_min = 1e-7
    start = time.time()
    while time.time() - start < time_budget:
        # Variable block length: longer early, shorter late
        progress = min(1.0, (time.time() - start) / max(time_budget, 1e-9))
        max_len = max(1, int(24.0 * (1.0 - 0.5 * progress)))
        L = int(rng.integers(1, max_len + 1))
        i = int(rng.integers(0, N - L + 1))
        h[i:i + L] *= -1.0
        new = _peak_sidelobe(h)
        d = new - cur
        if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            cur = new
            if cur < best:
                best = cur
                best_h = h.copy()
        else:
            h[i:i + L] *= -1.0
        T = max(T_min, T * 0.99997)
    return best_h, best


def _greedy_polish(h, time_budget, start):
    """Greedy polish with single-flip and adjacent-pair-flip sweeps.

    Alternates between flipping one bit and flipping two adjacent bits.
    Pair flips help escape flat plateaus that single flips cannot cross.
    """
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        improved = False

        # --- single-flip sweep ---
        best_i = -1
        best_c = cur
        for i in range(N):
            h[i] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c:
                best_c = c
                best_i = i
            h[i] *= -1.0
        if best_i >= 0:
            h[best_i] *= -1.0
            cur = best_c
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True

        # --- adjacent-pair-flip sweep ---
        best_pair = -1
        best_c2 = cur
        for i in range(N - 1):
            h[i] *= -1.0
            h[i + 1] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c2:
                best_c2 = c
                best_pair = i
            h[i] *= -1.0
            h[i + 1] *= -1.0
        if best_pair >= 0:
            h[best_pair] *= -1.0
            h[best_pair + 1] *= -1.0
            cur = best_c2
            if cur < best:
                best = cur
                best_h = h.copy()
            improved = True

        if not improved:
            break
    return best_h, best


def _greedy_single_only(h, time_budget, start):
    """Tighter single-flip polish for the final pass."""
    h = h.copy()
    cur = _peak_sidelobe(h)
    best = cur
    best_h = h.copy()
    while time.time() - start < time_budget:
        best_i = -1
        best_c = cur
        for i in range(N):
            h[i] *= -1.0
            c = _peak_sidelobe(h)
            if c < best_c:
                best_c = c
                best_i = i
            h[i] *= -1.0
        if best_i < 0:
            break
        h[best_i] *= -1.0
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()
    return best_h, best


def _l2_autocorr(x):
    """L2 norm of aperiodic autocorrelation sidelobes: sum_{k=1}^{N-1} ac[k]^2."""
    H = np.fft.fft(x, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.sum(ac[1:] ** 2))


def construct():
    try:
        hard_deadline = 140.0
        start = time.time()

        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        def _remaining():
            return hard_deadline - (time.time() - start)

        # Phase 0: Generate high-quality seeds via continuous gradient descent.
        # This provides much better starting points than random ±1.
        seed_budget = min(15.0, _remaining() - 15.0)
        if seed_budget > 3.0:
            h_seed, c_seed = _continuous_seed(42, seed_budget)
            if c_seed < global_best:
                global_best = c_seed
                global_best_h = h_seed
            # Polish the continuous seed
            sub = min(8.0, _remaining() - 15.0)
            if sub > 1.0:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Also try a second continuous seed with different initialization
        seed_budget2 = min(10.0, _remaining() - 15.0)
        if seed_budget2 > 2.0:
            h_seed2, c_seed2 = _continuous_seed(999, seed_budget2)
            if c_seed2 < global_best:
                global_best = c_seed2
                global_best_h = h_seed2
            sub = min(5.0, _remaining() - 15.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 1: Anneal from the best seed found so far.
        budget = min(25.0, _remaining() - 12.0)
        if budget > 3.0:
            h, c = _anneal(12345, budget)
            if c < global_best:
                global_best = c
                global_best_h = h
            # Polish right after to lock in gains
            sub = min(6.0, _remaining() - 12.0)
            if sub > 0.5:
                h2, c2 = _greedy_polish(global_best_h, sub, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2

        # Phase 2: multiple shorter restarts for diversity, with polish after each.
        restart_id = 1
        while _remaining() > 14.0:
            sub = min(10.0, _remaining() - 14.0)
            if sub < 1.5:
                break
            h, c = _anneal(100000 + restart_id, sub)
            if c < global_best:
                global_best = c
                global_best_h = h
            # Quick polish after each restart
            sub2 = min(3.0, _remaining() - 14.0)
            if sub2 > 0.3:
                h2, c2 = _greedy_polish(global_best_h, sub2, time.time())
                if c2 < global_best:
                    global_best = c2
                    global_best_h = h2
            restart_id += 1

        # Phase 3: final high-precision single-flip polish.
        remaining = _remaining() - 2.0
        if remaining > 0.5:
            h, c = _greedy_single_only(global_best_h, remaining, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 4: one more combined polish pass if time remains.
        remaining2 = _remaining() - 1.0
        if remaining2 > 0.5:
            h, c = _greedy_polish(global_best_h, remaining2, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return np.ascontiguousarray(global_best_h, dtype=np.float64)
    except Exception:
        return np.ones(N)
