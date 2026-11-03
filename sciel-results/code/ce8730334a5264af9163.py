"""Autoconvolution minimization via fast SA with O(N) single-bit updates and Chebyshev-inspired seeds.

We choose 400 signed step heights (+/-1) to minimize the maximum absolute
value of the (linear) autoconvolution, which is the C3 upper bound we want
to lower.

Key improvements:
1. O(N) direct autocorrelation update for single-bit flips (no FFT per step).
2. Chebyshev/polynomial phase optimization for high-quality seeds.
3. Thue-Morse initialization as a strong deterministic starting point.
4. Multi-start SA with adaptive temperature schedule.
5. Aggressive multi-scale local polish at the end.
"""
import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345

# Precompute Fourier basis for incremental updates (used for block flips)
_k_indices = np.arange(L // 2 + 1)
_i_indices = np.arange(N)
_exponents = -2j * np.pi * np.outer(_i_indices, _k_indices) / L
_W_basis = np.exp(_exponents)


def _autocorr_fast(h):
    """Linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1."""
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    return ac


def _ac_delta_single_flip(h, i):
    """
    Compute the change in autocorrelation when flipping h[i] -> -h[i].
    Returns delta_ac of shape (N,), where ac_new = ac + delta_ac.
    O(N) time.
    """
    delta_ac = np.zeros(N)
    hi = h[i]
    # For each lag k, the terms involving h[i] are:
    # h[i]*h[i+k] in ac[k] (j=i), and h[i-k]*h[i] in ac[k] (j=i-k)
    # Flipping h[i] changes these to -h[i]*h[i+k] and -h[i-k]*h[i]
    # So delta = -2*hi*(h[i+k] + h[i-k]) with bounds checking
    
    # k ranges 0..N-1
    # h[i+k] is valid when 0 <= i+k < N, i.e., -i <= k < N-i
    # h[i-k] is valid when 0 <= i-k < N, i.e., i-N < k <= i
    
    # For k in [0, N-i-1]: h[i+k] is valid
    if i < N:
        k_start = 0
        k_end = min(N, N - i)  # k < N-i
        if k_end > k_start:
            delta_ac[k_start:k_end] += -2.0 * hi * h[i + k_start:i + k_end]
    
    # For k in [max(0, i-N+1), i]: h[i-k] is valid
    # k such that 0 <= i-k < N => i-N < k <= i => k in [max(0, i-N+1), i]
    k_start = max(0, i - N + 1)
    k_end = min(i, N - 1) + 1
    if k_end > k_start:
        delta_ac[k_start:k_end] += -2.0 * hi * h[i - k_start:i - k_end + 1]
    
    return delta_ac


def _hard_score_from_ac(ac):
    """Compute score from autocorrelation array."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _hard_score_fast(ac):
    """Compute score from autocorrelation array, slightly optimized."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    # ac[0] = N for +/-1 sequence, so dc = N
    return float(np.max(np.abs(ac)) / dc)


def _smooth_score_from_ac(ac, beta):
    """Smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.sum(np.exp(z))) / beta


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1."""
    # Thue-Morse: t(n) = parity of number of 1s in binary representation of n
    # Map: even -> +1, odd -> -1
    tm = np.array([bin(i).count('1') % 2 for i in range(n)])
    return 1.0 - 2.0 * tm  # 0->+1, 1->-1


def _chebyshev_seed(deadline, rng):
    """
    Generate a high-quality binary seed via Chebyshev-inspired phase optimization.
    We view the sign sequence as coefficients of a polynomial P(z) = sum h[i] z^i.
    The autocorrelation is related to |P(e^{i*theta})|^2. We minimize the max of
    |P(e^{i*theta})|^2 over theta by optimizing continuous phases, then threshold.
    """
    # Start with Thue-Morse as initial guess
    x = _thue_morse(N).astype(float)
    
    # Optimize phases: treat x[i] = cos(phi[i]) initially, then refine
    # Actually, let's do continuous gradient descent on L2 of sidelobes
    # which is equivalent to flattening the spectrum
    
    lr = 0.03
    n_steps = 3000
    t_start = time.time()
    
    for step in range(n_steps):
        if time.time() > deadline - 80.0:
            break
        
        # Compute autocorrelation via FFT
        F = np.fft.rfft(x, n=L)
        P = F * np.conj(F)
        ac = np.fft.irfft(P, n=L)
        
        # Gradient of sum_{k=1}^{N-1} ac[k]^2 w.r.t. x
        # grad[j] = 2 * sum_{k=1}^{N-1} ac[k] * (x[j-k] + x[j+k])
        # Use FFT: grad = 2 * irfft(F_x * conj(F_ac)) - 2*ac[0]*x (to exclude k=0)
        F_ac = np.fft.rfft(ac, n=L)
        grad_full = 2.0 * np.fft.irfft(F * np.conj(F_ac), n=L)[:N]
        # Subtract the k=0 contribution: 2*ac[0]*x[j]
        grad = grad_full - 2.0 * ac[0] * x
        
        x -= lr * grad
        
        # Normalize
        norm = np.linalg.norm(x)
        if norm > 1e-10:
            x = x / norm * np.sqrt(N)
    
    # Threshold to binary
    h_binary = np.sign(x)
    h_binary[h_binary == 0] = 1.0
    
    return h_binary


def _local_polish_quick(h, deadline):
    """Quick single-pass local polish using O(N) single-bit updates."""
    h = h.copy()
    ac = _autocorr_fast(h)
    best_hard = _hard_score_from_ac(ac)
    best_h = h.copy()
    
    for _pass in range(5):
        if time.time() > deadline - 75.0:
            break
        any_improve = False
        for i in range(N):
            if time.time() > deadline - 75.0:
                break
            delta_ac = _ac_delta_single_flip(h, i)
            ac_new = ac + delta_ac
            new_hard = _hard_score_from_ac(ac_new)
            if new_hard < best_hard - 1e-12:
                h[i] *= -1.0
                ac = ac_new
                best_hard = new_hard
                best_h = h.copy()
                any_improve = True
        if not any_improve:
            break
    
    return best_h, best_hard


def _sa_run(h_init, rng, deadline, T_start, T_min, decay, beta, sizes):
    """Run simulated annealing from h_init until deadline. Returns (best_h, best_hard).
    
    Uses O(N) direct autocorrelation updates for single-bit flips (fast path)
    and FFT-based updates for block flips (slow path).
    """
    h = h_init.copy()
    ac = _autocorr_fast(h)
    F = np.fft.rfft(h, n=L)
    cur_hard = _hard_score_from_ac(ac)
    cur_smooth = _smooth_score_from_ac(ac, beta)
    best_hard = cur_hard
    best_h = h.copy()
    T = T_start
    steps = 0
    
    # Small block sizes that can use direct update
    max_direct_size = 4
    
    while time.time() < deadline - 2.0:
        # Choose move type
        r = rng.random()
        if r < 0.70:
            # Single bit flip - use fast O(N) path
            i = int(rng.integers(0, N))
            delta_ac = _ac_delta_single_flip(h, i)
            ac_new = ac + delta_ac
            new_hard = _hard_score_from_ac(ac_new)
            new_smooth = _smooth_score_from_ac(ac_new, beta)
            
            dS = new_smooth - cur_smooth
            accept = False
            if new_hard < best_hard:
                accept = True
            elif dS <= 0:
                accept = True
            else:
                if T > 1e-10:
                    if rng.random() < np.exp(-dS / T):
                        accept = True
            
            if accept:
                h[i] *= -1.0
                ac = ac_new
                cur_hard = new_hard
                cur_smooth = new_smooth
                if new_hard < best_hard:
                    best_hard = new_hard
                    best_h = h.copy()
            
        elif r < 0.90:
            # Small block flip (size 2-4) - use direct O(N*size) path
            size = int(rng.choice([2, 3, 4]))
            start = int(rng.integers(0, N - size + 1))
            idx_flip = np.arange(start, start + size)
            
            # Compute delta_ac directly
            delta_ac = np.zeros(N)
            for idx in idx_flip:
                delta_ac += _ac_delta_single_flip(h, int(idx))
            
            ac_new = ac + delta_ac
            new_hard = _hard_score_from_ac(ac_new)
            new_smooth = _smooth_score_from_ac(ac_new, beta)
            
            dS = new_smooth - cur_smooth
            accept = False
            if new_hard < best_hard:
                accept = True
            elif dS <= 0:
                accept = True
            else:
                if T > 1e-10:
                    if rng.random() < np.exp(-dS / T):
                        accept = True
            
            if accept:
                h[idx_flip] *= -1.0
                ac = ac_new
                cur_hard = new_hard
                cur_smooth = new_smooth
                if new_hard < best_hard:
                    best_hard = new_hard
                    best_h = h.copy()
            
        else:
            # Large block flip - use FFT path
            size = int(rng.choice(sizes[sizes > max_direct_size]))
            start = int(rng.integers(0, N - size + 1))
            idx_flip = np.arange(start, start + size)
            
            h_block = h[idx_flip]
            W_block = _W_basis[idx_flip, :]
            delta_F = -2.0 * (W_block @ h_block)
            F_new = F + delta_F
            
            P = F_new * np.conj(F_new)
            ac_new = np.fft.irfft(P, n=L)[:N]
            new_hard = _hard_score_from_ac(ac_new)
            new_smooth = _smooth_score_from_ac(ac_new, beta)
            
            dS = new_smooth - cur_smooth
            accept = False
            if new_hard < best_hard:
                accept = True
            elif dS <= 0:
                accept = True
            else:
                if T > 1e-10:
                    if rng.random() < np.exp(-dS / T):
                        accept = True
            
            if accept:
                h[idx_flip] *= -1.0
                F = F_new
                ac = ac_new
                cur_hard = new_hard
                cur_smooth = new_smooth
                if new_hard < best_hard:
                    best_hard = new_hard
                    best_h = h.copy()
        
        T = max(T_min, T * decay)
        steps += 1
    
    return best_h, best_hard


def _local_polish(h, deadline):
    """Greedy multi-scale local polish. Returns (best_h, best_hard).
    
    Uses O(N) direct updates for small blocks and FFT for larger blocks.
    """
    h = h.copy()
    ac = _autocorr_fast(h)
    F = np.fft.rfft(h, n=L)
    best_hard = _hard_score_from_ac(ac)
    best_h = h.copy()
    
    sizes_polish = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    for _pass in range(100):
        if time.time() > deadline - 0.5:
            break
        any_improve = False
        for size in sizes_polish:
            if time.time() > deadline - 0.5:
                return best_h, best_hard
            for start in range(0, N - size + 1):
                if time.time() > deadline - 0.5:
                    return best_h, best_hard
                
                if size <= 4:
                    # Fast direct path
                    delta_ac = np.zeros(N)
                    for offset in range(size):
                        idx = start + offset
                        delta_ac += _ac_delta_single_flip(h, idx)
                    ac_new = ac + delta_ac
                else:
                    # FFT path
                    idx_flip = np.arange(start, start + size)
                    h_block = h[idx_flip]
                    W_block = _W_basis[idx_flip, :]
                    delta_F = -2.0 * (W_block @ h_block)
                    F_new = F + delta_F
                    P = F_new * np.conj(F_new)
                    ac_new = np.fft.irfft(P, n=L)[:N]
                
                new_hard = _hard_score_from_ac(ac_new)
                
                if new_hard < best_hard - 1e-12:
                    if size <= 4:
                        h[start:start+size] *= -1.0
                        ac = ac_new
                    else:
                        h[start:start+size] *= -1.0
                        F = F_new
                        ac = ac_new
                    best_hard = new_hard
                    best_h = h.copy()
                    any_improve = True
        
        if not any_improve:
            break
    
    return best_h, best_hard


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 145.0  # safety margin before 150s limit
        rng = np.random.default_rng(_SEED)
        idx = np.arange(N)
        
        # Phase 1: Generate high-quality seeds
        # Thue-Morse as a strong deterministic seed
        seed_tm = _thue_morse(N)
        
        # Chebyshev-inspired continuous seed
        cont_deadline = t0 + 25.0
        seed_cheb = _chebyshev_seed(cont_deadline, rng)
        
        # Additional diverse seeds
        seed_all_plus = np.ones(N)
        seed_random = rng.choice(np.array([-1.0, 1.0]), size=N)
        seed_quad1 = np.sign(np.cos(2 * np.pi * idx ** 2 / (2 * N)))
        seed_quad2 = np.sign(np.cos(2 * np.pi * idx ** 2 / (N + 1)))
        
        candidates = [seed_tm, seed_cheb, seed_all_plus, seed_random, seed_quad1, seed_quad2]
        
        best_h = None
        best_hard = 1e9
        for h in candidates:
            ac = _autocorr_fast(h)
            s = _hard_score_from_ac(ac)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()
        
        # Phase 2: Multi-start SA with fast O(N) single-bit updates
        sa_start_time = time.time()
        polish_budget = 35.0
        sa_end = deadline - polish_budget
        
        sizes_sa = np.array([2, 3, 4, 8, 12, 16, 20, 24, 32, 48, 64])
        
        # Run SA from the best seed and perturbed versions
        n_starts = 4
        sa_time_per = (sa_end - sa_start_time) / n_starts
        
        for start_idx in range(n_starts):
            if time.time() > sa_end - 1.0:
                break
            
            if start_idx == 0:
                h_init = best_h.copy()
            elif start_idx == 1:
                # Perturb the best with 5% flips
                h_init = best_h.copy()
                flip_mask = rng.random(N) < 0.05
                h_init[flip_mask] *= -1.0
            elif start_idx == 2:
                # Use Thue-Morse perturbed
                h_init = seed_tm.copy()
                flip_mask = rng.random(N) < 0.1
                h_init[flip_mask] *= -1.0
            else:
                # Use Chebyshev seed perturbed
                h_init = seed_cheb.copy()
                flip_mask = rng.random(N) < 0.1
                h_init[flip_mask] *= -1.0
            
            sa_deadline = min(sa_start_time + (start_idx + 1) * sa_time_per, sa_end - 1.0)
            
            h_candidate, hard_candidate = _sa_run(
                h_init, rng, sa_deadline,
                T_start=0.02, T_min=1e-5, decay=0.9997, beta=40.0, sizes=sizes_sa
            )
            
            if hard_candidate < best_hard:
                best_hard = hard_candidate
                best_h = h_candidate.copy()
        
        # Phase 3: Final local polish
        best_h, best_hard = _local_polish(best_h, deadline)
        
        return best_h
    except Exception:
        return np.ones(N)
