"""Autoconvolution minimization via fast SA with O(N) single-bit updates and refined search.

Combines efficient direct autocorrelation updates with multi-phase SA and high-precision polish.
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
    Returns delta_ac of shape (N,). O(N) time.
    """
    delta_ac = np.zeros(N)
    hi = h[i]
    
    # For k in [0, N-i-1]: h[i+k] is valid
    if i < N:
        k_start = 0
        k_end = min(N, N - i)
        if k_end > k_start:
            delta_ac[k_start:k_end] += -2.0 * hi * h[i + k_start:i + k_end]
    
    # For k in [max(0, i-N+1), i]: h[i-k] is valid
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
    """Compute score from autocorrelation array, optimized for ±1 sequences."""
    return float(np.max(np.abs(ac)) / N)


def _smooth_score_from_ac(ac, beta):
    """Smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.sum(np.exp(z))) / beta


def _smooth_score_fast(ac, beta):
    """Optimized smooth score for ±1 sequences where ac[0]=N."""
    y = (ac[1:] / N) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.sum(np.exp(z))) / beta


def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as +/-1."""
    tm = np.array([bin(i).count('1') % 2 for i in range(n)])
    return 1.0 - 2.0 * tm


def _chebyshev_seed(deadline, rng):
    """
    Generate a high-quality binary seed via continuous phase optimization.
    Minimize L2 of sidelobes via gradient descent, then threshold.
    """
    x = _thue_morse(N).astype(float)
    
    lr = 0.03
    n_steps = 3000
    t_start = time.time()
    
    for step in range(n_steps):
        if time.time() > deadline - 80.0:
            break
        
        F = np.fft.rfft(x, n=L)
        P = F * np.conj(F)
        ac = np.fft.irfft(P, n=L)
        
        F_ac = np.fft.rfft(ac, n=L)
        grad_full = 2.0 * np.fft.irfft(F * np.conj(F_ac), n=L)[:N]
        grad = grad_full - 2.0 * ac[0] * x
        
        x -= lr * grad
        
        norm = np.linalg.norm(x)
        if norm > 1e-10:
            x = x / norm * np.sqrt(N)
    
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


def _sa_run(h_init, rng, deadline, T_start, T_min, decay, beta, sizes, phase):
    """Run simulated annealing from h_init until deadline. Returns (best_h, best_hard).
    
    phase: 0 = exploration (larger blocks, higher T), 1 = exploitation (single-bit, lower T)
    """
    h = h_init.copy()
    ac = _autocorr_fast(h)
    F = np.fft.rfft(h, n=L)
    cur_hard = _hard_score_fast(ac)
    cur_smooth = _smooth_score_fast(ac, beta)
    best_hard = cur_hard
    best_h = h.copy()
    T = T_start
    no_improve_steps = 0
    improve_threshold = 3000 if phase == 0 else 5000
    
    max_direct_size = 4
    
    while time.time() < deadline - 2.0:
        # Adaptive beta: increase as T decreases for sharper surrogate
        beta_adapt = beta * (1.0 + 0.5 * (1.0 - T / T_start)) if T_start > 0 else beta
        
        if phase == 0:
            # Exploration phase: mix of block sizes
            r = rng.random()
            if r < 0.50:
                # Single bit flip
                i = int(rng.integers(0, N))
                delta_ac = _ac_delta_single_flip(h, i)
                ac_new = ac + delta_ac
                new_hard = _hard_score_fast(ac_new)
                new_smooth = _smooth_score_fast(ac_new, beta_adapt)
                
                dS = new_smooth - cur_smooth
                dH = new_hard - cur_hard
                accept = False
                if new_hard < best_hard:
                    accept = True
                elif dH <= 0:
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
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
            
            elif r < 0.80:
                # Small block flip (size 2-8)
                size = int(rng.choice([2, 4, 6, 8]))
                start = int(rng.integers(0, N - size + 1))
                idx_flip = np.arange(start, start + size)
                
                delta_ac = np.zeros(N)
                for idx in idx_flip:
                    delta_ac += _ac_delta_single_flip(h, int(idx))
                
                ac_new = ac + delta_ac
                new_hard = _hard_score_fast(ac_new)
                new_smooth = _smooth_score_fast(ac_new, beta_adapt)
                
                dS = new_smooth - cur_smooth
                dH = new_hard - cur_hard
                accept = False
                if new_hard < best_hard:
                    accept = True
                elif dH <= 0:
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
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
            
            else:
                # Large block flip
                size = int(rng.choice(sizes[sizes > max_direct_size]))
                start = int(rng.integers(0, N - size + 1))
                idx_flip = np.arange(start, start + size)
                
                h_block = h[idx_flip]
                W_block = _W_basis[idx_flip, :]
                delta_F = -2.0 * (W_block @ h_block)
                F_new = F + delta_F
                
                P = F_new * np.conj(F_new)
                ac_new = np.fft.irfft(P, n=L)[:N]
                new_hard = _hard_score_fast(ac_new)
                new_smooth = _smooth_score_fast(ac_new, beta_adapt)
                
                dS = new_smooth - cur_smooth
                dH = new_hard - cur_hard
                accept = False
                if new_hard < best_hard:
                    accept = True
                elif dH <= 0:
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
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
        else:
            # Exploitation phase: primarily single-bit flips
            r = rng.random()
            if r < 0.85:
                # Single bit flip
                i = int(rng.integers(0, N))
                delta_ac = _ac_delta_single_flip(h, i)
                ac_new = ac + delta_ac
                new_hard = _hard_score_fast(ac_new)
                new_smooth = _smooth_score_fast(ac_new, beta_adapt)
                
                dS = new_smooth - cur_smooth
                dH = new_hard - cur_hard
                accept = False
                if new_hard < best_hard:
                    accept = True
                elif dH <= 0:
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
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
            else:
                # Small block flip (size 2-4)
                size = int(rng.choice([2, 3, 4]))
                start = int(rng.integers(0, N - size + 1))
                idx_flip = np.arange(start, start + size)
                
                delta_ac = np.zeros(N)
                for idx in idx_flip:
                    delta_ac += _ac_delta_single_flip(h, int(idx))
                
                ac_new = ac + delta_ac
                new_hard = _hard_score_fast(ac_new)
                new_smooth = _smooth_score_fast(ac_new, beta_adapt)
                
                dS = new_smooth - cur_smooth
                dH = new_hard - cur_hard
                accept = False
                if new_hard < best_hard:
                    accept = True
                elif dH <= 0:
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
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
        
        # Adaptive temperature: boost if stuck
        if no_improve_steps > improve_threshold:
            T = min(T_start, T * 1.3)
            no_improve_steps = 0
        
        T = max(T_min, T * decay)
    
    return best_h, best_hard


def _local_polish(h, deadline):
    """Greedy multi-scale local polish with high-precision final pass. Returns (best_h, best_hard)."""
    h = h.copy()
    ac = _autocorr_fast(h)
    F = np.fft.rfft(h, n=L)
    best_hard = _hard_score_fast(ac)
    best_h = h.copy()
    
    sizes_polish = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    for _pass in range(200):
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
                    delta_ac = np.zeros(N)
                    for offset in range(size):
                        idx = start + offset
                        delta_ac += _ac_delta_single_flip(h, idx)
                    ac_new = ac + delta_ac
                else:
                    idx_flip = np.arange(start, start + size)
                    h_block = h[idx_flip]
                    W_block = _W_basis[idx_flip, :]
                    delta_F = -2.0 * (W_block @ h_block)
                    F_new = F + delta_F
                    P = F_new * np.conj(F_new)
                    ac_new = np.fft.irfft(P, n=L)[:N]
                
                new_hard = _hard_score_fast(ac_new)
                
                if new_hard < best_hard - 1e-14:
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
    
    # High-precision final pass: only single-bit flips with tighter tolerance
    ac = _autocorr_fast(h)
    best_hard_hp = _hard_score_fast(ac)
    best_h_hp = h.copy()
    
    for _pass in range(50):
        if time.time() > deadline - 0.2:
            break
        any_improve = False
        for i in range(N):
            if time.time() > deadline - 0.2:
                break
            delta_ac = _ac_delta_single_flip(h, i)
            ac_new = ac + delta_ac
            new_hard = _hard_score_fast(ac_new)
            if new_hard < best_hard_hp - 1e-15:
                h[i] *= -1.0
                ac = ac_new
                best_hard_hp = new_hard
                best_h_hp = h.copy()
                any_improve = True
        if not any_improve:
            break
    
    if best_hard_hp < best_hard:
        return best_h_hp, best_hard_hp
    return best_h, best_hard


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 145.0
        rng = np.random.default_rng(_SEED)
        idx = np.arange(N)
        
        # Phase 1: Generate high-quality seeds
        seed_tm = _thue_morse(N)
        
        cont_deadline = t0 + 20.0
        seed_cheb = _chebyshev_seed(cont_deadline, rng)
        
        # More diverse structured seeds
        seed_all_plus = np.ones(N)
        seed_random = rng.choice(np.array([-1.0, 1.0]), size=N)
        seed_quad1 = np.sign(np.cos(2 * np.pi * idx ** 2 / (2 * N)))
        seed_quad2 = np.sign(np.cos(2 * np.pi * idx ** 2 / (N + 1)))
        seed_quad3 = np.sign(np.cos(2 * np.pi * idx ** 2 / (3 * N)))
        seed_sin1 = np.sign(np.sin(2 * np.pi * idx / N))
        seed_sin2 = np.sign(np.sin(4 * np.pi * idx / N))
        seed_sin3 = np.sign(np.sin(6 * np.pi * idx / N))
        # Legendre-inspired: use polynomial coefficients
        legendre_coeffs = np.polynomial.legendre.legvander(idx, 10).T
        seed_leg = np.sign(legendre_coeffs[10, :])
        
        candidates = [seed_tm, seed_cheb, seed_all_plus, seed_random, seed_quad1, 
                      seed_quad2, seed_quad3, seed_sin1, seed_sin2, seed_sin3, seed_leg]
        
        best_h = None
        best_hard = 1e9
        for h in candidates:
            ac = _autocorr_fast(h)
            s = _hard_score_fast(ac)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()
        
        # Phase 2: Multi-start SA with two phases
        sa_start_time = time.time()
        polish_budget = 30.0
        sa_end = deadline - polish_budget
        
        sizes_sa = np.array([8, 12, 16, 24, 32, 48, 64])
        
        n_starts = 4
        sa_time_per = (sa_end - sa_start_time) / n_starts
        
        for start_idx in range(n_starts):
            if time.time() > sa_end - 1.0:
                break
            
            if start_idx == 0:
                h_init = best_h.copy()
            elif start_idx == 1:
                h_init = best_h.copy()
                flip_mask = rng.random(N) < 0.08
                h_init[flip_mask] *= -1.0
            elif start_idx == 2:
                h_init = seed_tm.copy()
                flip_mask = rng.random(N) < 0.12
                h_init[flip_mask] *= -1.0
            else:
                h_init = seed_cheb.copy()
                flip_mask = rng.random(N) < 0.12
                h_init[flip_mask] *= -1.0
            
            sa_deadline = min(sa_start_time + (start_idx + 1) * sa_time_per, sa_end - 1.0)
            
            # Two-phase SA: exploration then exploitation
            phase1_end = sa_deadline - (sa_deadline - sa_start_time - start_idx * sa_time_per) * 0.4
            phase1_end = max(sa_start_time + start_idx * sa_time_per + 1.0, phase1_end)
            
            # Phase 1: Exploration
            h_candidate, hard_candidate = _sa_run(
                h_init, rng, phase1_end,
                T_start=0.05, T_min=1e-5, decay=0.999, beta=30.0, sizes=sizes_sa, phase=0
            )
            
            # Phase 2: Exploitation
            h_candidate, hard_candidate = _sa_run(
                h_candidate, rng, sa_deadline,
                T_start=0.01, T_min=1e-7, decay=0.9995, beta=50.0, sizes=sizes_sa, phase=1
            )
            
            if hard_candidate < best_hard:
                best_hard = hard_candidate
                best_h = h_candidate.copy()
        
        # Phase 3: Final local polish
        best_h, best_hard = _local_polish(best_h, deadline)
        
        return best_h
    except Exception:
        return np.ones(N)
