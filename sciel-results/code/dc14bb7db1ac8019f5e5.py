"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _compute_overlap_fast(h):
    """Compute maximum overlap of h with shifted 1-h using FFT-based correlation."""
    N = len(h)
    # overlap_k = sum_i h[i] * (1 - h[(i+k) % N]) for k = 1, ..., N-1
    # = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    # = N/2 - autocorrelation of h at lag k
    # Compute autocorrelation via FFT
    h_fft = np.fft.fft(h)
    acf = np.fft.ifft(h_fft * np.conj(h_fft)).real
    # acf[k] = sum_i h[i]*h[(i+k)%N]
    # overlap at lag k = N/2 - acf[k] for k=1,...,N-1
    overlaps = N / 2 - acf[1:]
    return np.max(overlaps)


def _compute_overlap(h):
    """Compute maximum overlap of h with shifted 1-h."""
    return _compute_overlap_fast(h)


def _project_to_constraints(h, target_sum=N/2):
    """Project h onto [0,1]^N with sum constraint using bisection."""
    h = np.clip(h, 0, 1)
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-10:
        return h
    
    lo, hi = -1.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        h_adj = np.clip(h + mid, 0, 1)
        s = np.sum(h_adj)
        if s < target_sum:
            lo = mid
        else:
            hi = mid
    h_adj = np.clip(h + (lo + hi) / 2, 0, 1)
    return h_adj


def _boundary_to_heights(boundaries, N=N):
    """
    Convert boundary positions to a step function.
    boundaries: sorted array of positions where h transitions from 0 to 1 or 1 to 0.
    We use pairs of boundaries: (start, end) for each "on" interval.
    h is 1 in [start, end) and 0 elsewhere (cyclic).
    """
    h = np.zeros(N)
    if len(boundaries) == 0:
        return h
    # boundaries should come in pairs: (start, end)
    # Ensure even number
    if len(boundaries) % 2 != 0:
        boundaries = boundaries[:-1]
    for i in range(0, len(boundaries), 2):
        start = int(boundaries[i]) % N
        end = int(boundaries[i + 1]) % N
        if start <= end:
            h[start:end] = 1.0
        else:
            h[start:] = 1.0
            h[:end] = 1.0
    return h


def _soft_boundary_to_heights(boundaries, temperature=0.5, N=N):
    """
    Differentiable soft version: use sigmoid transitions at boundaries.
    boundaries: array of 2K boundary positions (floats).
    Returns h in [0,1]^N.
    """
    h = np.zeros(N)
    if len(boundaries) == 0:
        return h
    if len(boundaries) % 2 != 0:
        boundaries = boundaries[:-1]
    
    positions = np.arange(N)
    for i in range(0, len(boundaries), 2):
        start = boundaries[i]
        end = boundaries[i + 1]
        # Soft step: sigmoid centered at start and end
        # h_soft(x) = sigmoid((x - start)/T) - sigmoid((x - end)/T)
        # But we need cyclic handling
        # For simplicity, use a smooth approximation
        s_start = 1.0 / (1.0 + np.exp(-(positions - start) / temperature))
        s_end = 1.0 / (1.0 + np.exp(-(positions - end) / temperature))
        segment = s_start - s_end
        # Handle wrap-around
        if start > end:
            # Wrap-around case
            s_start_wrapped = 1.0 / (1.0 + np.exp(-(positions + N - start) / temperature))
            s_end_wrapped = 1.0 / (1.0 + np.exp(-(positions + N - end) / temperature))
            segment = np.maximum(segment, s_start_wrapped - s_end_wrapped)
        h += segment
    
    h = np.clip(h, 0, 1)
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with uniform solution as baseline
    best_result = np.full(N, 0.5)
    best_score = _compute_overlap(best_result)
    
    np.random.seed(42)
    
    # Strategy 1: Optimize boundary positions for a few intervals
    # Try different numbers of intervals (domain walls)
    max_intervals = 8
    max_boundary_params = 2 * max_intervals
    
    # Use differential evolution to optimize boundary positions
    # Each boundary is in [0, N)
    bounds = [(0, N)] * max_boundary_params
    
    def boundary_objective(boundaries):
        if time.time() > deadline - 20:
            return best_score
        # Only use first 2K boundaries where K varies
        # For this run, fix K = 4 (4 intervals)
        K = 4
        b = boundaries[:2*K]
        h = _boundary_to_heights(b)
        h = _project_to_constraints(h)
        return _compute_overlap_fast(h)
    
    try:
        result_de = differential_evolution(
            boundary_objective,
            bounds,
            seed=42,
            maxiter=100,
            tol=1e-6,
            popsize=15,
            polish=True
        )
        if result_de.fun < best_score:
            K = 4
            b = result_de.x[:2*K]
            best_result = _boundary_to_heights(b)
            best_result = _project_to_constraints(best_result)
            best_score = result_de.fun
            print(f"DE with K=4: improved to {best_score:.6f}")
    except Exception as e:
        print(f"DE failed: {e}")
    
    if time.time() > deadline - 40:
        # Finalize and return
        best_result = np.clip(best_result, 0, 1)
        diff = N/2 - np.sum(best_result)
        if abs(diff) > 1e-10:
            best_result += diff / N
            best_result = np.clip(best_result, 0, 1)
            best_result = _project_to_constraints(best_result)
        assert np.all(best_result >= 0) and np.all(best_result <= 1)
        assert abs(np.sum(best_result) - N / 2) < 1e-6
        return best_result
    
    # Strategy 2: Local refinement using Nelder-Mead on the best boundary solution
    # Start from the best DE solution
    K = 4
    if 'result_de' in dir() and result_de.success:
        b_init = result_de.x[:2*K].copy()
    else:
        # Initialize with evenly spaced boundaries
        b_init = np.array([i * N / (2*K) for i in range(2*K)])
    
    def refine_objective(b):
        if time.time() > deadline - 15:
            return best_score
        h = _boundary_to_heights(b)
        h = _project_to_constraints(h)
        return _compute_overlap_fast(h)
    
    try:
        result_nm = minimize(
            refine_objective,
            b_init,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 0.01, 'fatol': 1e-8, 'adaptive': True}
        )
        if result_nm.fun < best_score:
            best_result = _boundary_to_heights(result_nm.x[:2*K])
            best_result = _project_to_constraints(best_result)
            best_score = result_nm.fun
            print(f"Nelder-Mead refinement: improved to {best_score:.6f}")
    except Exception as e:
        print(f"Nelder-Mead refinement failed: {e}")
    
    if time.time() > deadline - 20:
        # Finalize and return
        best_result = np.clip(best_result, 0, 1)
        diff = N/2 - np.sum(best_result)
        if abs(diff) > 1e-10:
            best_result += diff / N
            best_result = np.clip(best_result, 0, 1)
            best_result = _project_to_constraints(best_result)
        assert np.all(best_result >= 0) and np.all(best_result <= 1)
        assert abs(np.sum(best_result) - N / 2) < 1e-6
        return best_result
    
    # Strategy 3: Try different K values
    for K in [2, 3, 5, 6]:
        if time.time() > deadline - 15:
            break
        
        # Initialize with evenly spaced boundaries
        b_init = np.array([i * N / (2*K) for i in range(2*K)])
        
        def obj_K(b, K=K):
            if time.time() > deadline - 10:
                return best_score
            h = _boundary_to_heights(b)
            h = _project_to_constraints(h)
            return _compute_overlap_fast(h)
        
        try:
            result_K = minimize(
                obj_K,
                b_init,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 0.01, 'fatol': 1e-8, 'adaptive': True}
            )
            if result_K.fun < best_score:
                best_result = _boundary_to_heights(result_K.x[:2*K])
                best_result = _project_to_constraints(best_result)
                best_score = result_K.fun
                print(f"K={K}: improved to {best_score:.6f}")
        except Exception as e:
            print(f"K={K} failed: {e}")
    
    # Final verification
    best_result = np.clip(best_result, 0, 1)
    diff = N/2 - np.sum(best_result)
    if abs(diff) > 1e-10:
        best_result += diff / N
        best_result = np.clip(best_result, 0, 1)
        best_result = _project_to_constraints(best_result)
    
    assert np.all(best_result >= 0) and np.all(best_result <= 1)
    assert abs(np.sum(best_result) - N / 2) < 1e-6
    
    return best_result
# EVOLVE-BLOCK-END
