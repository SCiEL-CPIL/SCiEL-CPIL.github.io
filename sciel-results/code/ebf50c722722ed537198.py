import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    n = len(h)
    fsize = 1
    while fsize < 2 * n - 1:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:2*n-1]

def _raw_value(h):
    """Compute the objective: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _compute_autoconv_from_signs(signs):
    """Compute autoconvolution for a sign vector (+1/-1)."""
    return _autoconv_fft(signs.astype(np.float64))

def _update_autoconv_on_flip(ac, signs, j):
    """
    Incrementally update autoconvolution after flipping sign at position j.
    When sign[j] flips from s to -s, the change in autoconv is:
    delta_ac[k] = -2*s * (h[k-j] if k>=j else h[j-k])  for k in 0..2N-2
    where h is the sign vector.
    
    Actually, ac[k] = sum_i h[i]*h[i+k]
    After flipping h[j] -> -h[j], the new ac' = ac - 2*h[j]*g where
    g[k] = h[k-j] if k>=j else h[j-k] ... this is the cross-correlation.
    
    More precisely: ac[k] = sum_i h[i]*h[i+k]
    The terms involving h[j] are: h[j]*h[j+k] (when i=j) and h[j-k]*h[j] (when i+k=j, i.e., i=j-k)
    So ac[k] includes h[j]*h[j+k] + h[j-k]*h[j] (if both indices valid)
    After flip: these become -h[j]*h[j+k] - h[j-k]*h[j]
    Change: -2*(h[j]*h[j+k] + h[j-k]*h[j]) but only for valid indices.
    
    Let's just recompute for simplicity but use the structure.
    """
    n = len(signs)
    s = signs[j]
    new_signs = signs.copy()
    new_signs[j] = -s
    return _autoconv_fft(new_signs)

def _energy_from_signs(signs):
    """Compute energy = max|ac| / S^2 for sign vector."""
    s = np.sum(signs)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(signs.astype(np.float64))
    return np.max(np.abs(ac)) / (s * s)

def _spectral_projection(signs, n_modes=20):
    """
    Project sign vector onto a low-dimensional spectral subspace.
    This helps escape local minima by allowing non-binary intermediate values
    before re-thresholding.
    """
    n = len(signs)
    h = signs.astype(np.float64)
    H = np.fft.rfft(h)
    # Keep only low-frequency components
    keep = min(n_modes, len(H))
    H_proj = np.zeros_like(H)
    H_proj[:keep] = H[:keep]
    h_proj = np.fft.irfft(H_proj, n)
    # Threshold back to signs
    return np.sign(h_proj)

def _gen_random_signs(rng):
    """Random binary signs."""
    return rng.choice([-1.0, 1.0], N)

def _gen_chirp_signs():
    """Chirp-based sign pattern."""
    t = np.arange(N)
    phase = 2 * np.pi * (t / N) ** 2
    return np.sign(np.sin(phase))

def _gen_golden_ratio_signs():
    """Golden ratio based sign pattern."""
    phi = (1 + np.sqrt(5)) / 2
    t = np.arange(N)
    phase = 2 * np.pi * phi * t / N
    return np.sign(np.sin(phase))

def _gen_windowed_signs(rng):
    """Windowed random signs."""
    x = np.arange(N)
    env = np.cos(np.pi * x / N) ** 2
    signs = rng.choice([-1.0, 1.0], N)
    # Weighted threshold
    h = env * signs
    return np.sign(h)

def _simulated_annealing(signs, t_start, t_end, n_sweeps_per_step=1, 
                          spectral_interval=50, n_modes=20):
    """
    Simulated annealing on binary sign configurations.
    
    Energy: max|autoconv| / S^2
    Move: flip a random sign
    Acceptance: Metropolis with cooling schedule
    """
    n = len(signs)
    signs = signs.copy()
    current_energy = _energy_from_signs(signs)
    best_energy = current_energy
    best_signs = signs.copy()
    
    t = t_start
    total_steps = int(n_sweeps_per_step * (t_start - t_end) / (t_start - t_end) * 1000)
    # Simpler: run for a fixed number of iterations with cooling
    n_iters = 2000
    current_temp = t_start
    cooling_rate = (t_end / t_start) ** (1.0 / n_iters)
    
    for it in range(n_iters):
        if time.time() > t_end:
            break
        
        # Periodic spectral projection to escape local minima
        if it % spectral_interval == 0 and it > 0:
            projected = _spectral_projection(signs, n_modes)
            proj_energy = _energy_from_signs(projected)
            if proj_energy < current_energy or np.random.random() < np.exp(-min(50, (proj_energy - current_energy) / current_temp)):
                signs = projected
                current_energy = proj_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_signs = signs.copy()
        
        # Random flip
        j = np.random.randint(0, n)
        old_sign = signs[j]
        signs[j] = -old_sign
        new_energy = _energy_from_signs(signs)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.random() < np.exp(-min(50, delta_e / max(current_temp, 1e-10))):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_signs = signs.copy()
        else:
            signs[j] = old_sign
        
        current_temp *= cooling_rate
    
    return best_signs, best_energy

def _local_flip_search(signs, t_deadline, max_passes=50):
    """
    Deterministic local search: try flipping each sign, keep if improvement.
    """
    signs = signs.copy()
    current_energy = _energy_from_signs(signs)
    best_energy = current_energy
    best_signs = signs.copy()
    
    for _pass in range(max_passes):
        if time.time() > t_deadline:
            break
        
        improved = False
        for j in range(N):
            if time.time() > t_deadline:
                break
            old = signs[j]
            signs[j] = -old
            new_e = _energy_from_signs(signs)
            if new_e < current_energy:
                current_energy = new_e
                improved = True
            else:
                signs[j] = old
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_signs = signs.copy()
        
        if not improved:
            break
    
    return best_signs, best_energy

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin for 150s limit
    
    best_energy = np.inf
    best_signs = np.ones(N)
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse initial candidates
    generators = [
        _gen_random_signs,
        _gen_chirp_signs,
        _gen_golden_ratio_signs,
        _gen_windowed_signs,
    ]
    
    candidates = []
    for i in range(30):
        if time.time() > deadline - 20:
            break
        gen = generators[i % len(generators)]
        if i % len(generators) == 0:
            signs = gen(rng)
        elif i % len(generators) == 3:
            signs = gen(rng)
        else:
            signs = gen()
        
        energy = _energy_from_signs(signs)
        candidates.append((energy, signs))
    
    candidates.sort(key=lambda x: x[0])
    
    # Refine top candidates with simulated annealing + local search
    K = 8
    for i, (init_e, init_signs) in enumerate(candidates[:K]):
        if time.time() > deadline - 5:
            break
        
        # Allocate time based on rank
        remaining = deadline - time.time()
        time_budget = max(2.0, min(10.0, remaining / (K - i)))
        local_deadline = time.time() + time_budget
        
        # Phase 1: Simulated annealing
        sa_signs, sa_energy = _simulated_annealing(
            init_signs, 
            t_start=1.0, 
            t_end=0.01,
            n_sweeps_per_step=1,
            spectral_interval=100,
            n_modes=15
        )
        
        # Phase 2: Local flip search to polish
        if time.time() < local_deadline:
            final_signs, final_energy = _local_flip_search(sa_signs, local_deadline, max_passes=20)
        else:
            final_signs, final_energy = sa_signs, sa_energy
        
        if final_energy < best_energy:
            best_energy = final_energy
            best_signs = final_signs
    
    # Final intensive polish on the best candidate
    if time.time() < deadline - 2 and best_energy < np.inf:
        polished_signs, polished_energy = _local_flip_search(best_signs, deadline - 1, max_passes=100)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_signs = polished_signs
    
    # Return as float array
    if not np.all(np.isfinite(best_signs)):
        return np.ones(N)
        
    return best_signs.astype(np.float64)
# EVOLVE-BLOCK-END
