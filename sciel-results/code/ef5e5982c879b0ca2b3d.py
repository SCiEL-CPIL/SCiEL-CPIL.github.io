"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _c3_bound(h):
    """
    Compute the C3 upper bound for a signed step function h of length N.
    The C3 bound is related to the maximum aperiodic autocorrelation sidelobe.
    We use the max absolute autocorrelation sidelobe as the metric.
    """
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    # Aperiodic autocorrelation sidelobes are ac[1:N]
    return np.max(np.abs(ac[1:N]))


def _smooth_c3_bound(h, beta=100.0):
    """
    Smooth surrogate for the max sidelobe using log-sum-exp.
    """
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    ac = np.fft.irfft(power, n=n_fft)
    sidelobes = np.abs(ac[1:N])
    # Log-sum-exp for smooth max
    m = np.max(sidelobes)
    return m + np.log(np.sum(np.exp(beta * (sidelobes - m)))) / beta


def _local_search_polish(h, time_limit=20.0):
    """
    Local search to minimize the C3 bound by flipping individual bits.
    """
    start = time.time()
    current_cost = _c3_bound(h)
    h = h.copy()
    
    while time.time() - start < time_limit:
        improved = False
        n_fft = 1
        while n_fft < 2 * N:
            n_fft *= 2
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        ac = np.fft.irfft(power, n=n_fft)
        sidelobes = ac[1:N]
        
        for i in range(N):
            # Try flipping bit i
            old_val = h[i]
            h[i] = -old_val
            
            new_cost = _c3_bound(h)
            
            if new_cost < current_cost - 1e-12:
                current_cost = new_cost
                improved = True
            else:
                h[i] = old_val
        
        if not improved:
            break
    
    return h, current_cost


def _simulated_annealing(h, time_limit=60.0):
    """
    Simulated annealing to find a better signed step function.
    """
    start = time.time()
    h = h.copy()
    current_cost = _c3_bound(h)
    best_h = h.copy()
    best_cost = current_cost
    
    T_start = 10.0
    T_end = 0.1
    n_steps = int(time_limit * 50)  # Approximate number of steps
    
    for step in range(n_steps):
        if time.time() - start > time_limit:
            break
        
        # Temperature schedule
        progress = step / max(n_steps - 1, 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a random position to flip
        i = np.random.randint(N)
        old_val = h[i]
        h[i] = -old_val
        
        new_cost = _c3_bound(h)
        delta = new_cost - current_cost
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
        else:
            h[i] = old_val
    
    return best_h, best_cost


def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    # Start with a random initialization
    np.random.seed(42)
    h = np.random.choice([-1.0, 1.0], size=N)
    
    # First, do a quick local search polish
    h, cost = _local_search_polish(h, time_limit=10.0)
    
    # If we have time, try simulated annealing with different random starts
    if time.time() < deadline - 60:
        h_sa, cost_sa = _simulated_annealing(h, time_limit=40.0)
        if cost_sa < cost:
            h = h_sa
            cost = cost_sa
    
    # Final polish
    if time.time() < deadline - 10:
        h, cost = _local_search_polish(h, time_limit=20.0)
    
    return h
# EVOLVE-BLOCK-END
