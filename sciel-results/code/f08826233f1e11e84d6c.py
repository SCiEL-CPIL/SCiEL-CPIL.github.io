"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()


def _c3_value(h):
    """Compute C3 bound: sum of |sidelobe|^3 normalized."""
    F = np.fft.rfft(h, n=_NFFT)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=_NFFT).real[:N]
    sidelobes = acf[1:]
    return np.sum(np.abs(sidelobes) ** 3) / (N ** 3)


def _local_search_polish(h, deadline, max_rounds=200):
    """Deterministic local search: flip single bits to reduce C3."""
    best_h = h.copy()
    best_val = _c3_value(best_h)
    
    for _ in range(max_rounds):
        if time.time() >= deadline:
            break
        improved = False
        for i in range(N):
            if time.time() >= deadline:
                break
            best_h[i] = -best_h[i]
            val = _c3_value(best_h)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_h[i] = -best_h[i]  # revert
        if not improved:
            break
    
    return best_h, best_val


def _simulated_annealing(h, deadline):
    """Simulated annealing to minimize C3."""
    rng = np.random.default_rng(123)
    best_h = h.copy()
    best_val = _c3_value(best_h)
    current_h = h.copy()
    current_val = best_val
    
    T = 5.0
    T_min = 0.001
    cooling = 0.998
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
        
        # Generate candidate by flipping 1-3 random positions
        n_flips = rng.integers(1, 4)
        candidate = current_h.copy()
        indices = rng.choice(N, size=n_flips, replace=False)
        candidate[indices] = -candidate[indices]
        
        new_val = _c3_value(candidate)
        
        if new_val < current_val:
            current_h = candidate
            current_val = new_val
            if new_val < best_val:
                best_h = candidate.copy()
                best_val = new_val
        else:
            delta = new_val - current_val
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_h = candidate
                current_val = new_val
        
        T *= cooling
    
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    # Try multiple initializations and keep the best
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Multiple random starts with SA
    n_starts = 5
    per_start_time = (deadline - start_time - 20.0) / n_starts  # reserve 20s for polishing
    
    for s in range(n_starts):
        phase_deadline = start_time + 20.0 + per_start_time * (s + 1)
        if time.time() >= phase_deadline - 5.0:
            break
        
        # Initialize with random ±1
        h_init = rng.choice([-1.0, 1.0], size=N)
        h_init_val = _c3_value(h_init)
        
        # Run SA
        sa_h, sa_val = _simulated_annealing(h_init, phase_deadline - 5.0)
        
        if sa_val < best_val:
            best_h = sa_h
            best_val = sa_val
    
    # Phase 2: Local search polish on the best result
    if best_h is not None and time.time() < deadline:
        polished_h, polished_val = _local_search_polish(best_h, deadline)
        if polished_val < best_val:
            best_h = polished_h
            best_val = polished_val
    
    if best_h is None:
        # Fallback: return random
        best_h = rng.choice([-1.0, 1.0], size=N)
    
    return best_h
# EVOLVE-BLOCK-END
