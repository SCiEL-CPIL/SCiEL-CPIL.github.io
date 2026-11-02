"""Combined approach: Nelder-Mead simplex + multi-start for Erdos minimum overlap."""
import numpy as np
import time

N = 100
DEADLINE = 145  # safety margin before 150s kill


def _max_overlap_fft(h):
    """Compute the maximum overlap of h with shifted (1-h) using FFT."""
    one_minus_h = 1.0 - h
    fft_h = np.fft.fft(h)
    fft_omh = np.fft.fft(one_minus_h)
    corr = np.fft.ifft(fft_h * np.conj(fft_omh)).real
    return np.max(corr)


def _objective(h):
    """Objective: max overlap (to minimize). Uses FFT for speed."""
    h = np.asarray(h, dtype=np.float64)
    return _max_overlap_fft(h)


def _objective_smooth(h, alpha=50.0):
    """Smoothed objective using soft-max of overlaps for better gradient info."""
    h = np.asarray(h, dtype=np.float64)
    one_minus_h = 1.0 - h
    fft_h = np.fft.fft(h)
    fft_omh = np.fft.fft(one_minus_h)
    corr = np.fft.ifft(fft_h * np.conj(fft_omh)).real
    # Soft-max: log-sum-exp approximation of max
    m = np.max(corr)
    # If all close, return max; otherwise use smooth approximation
    return m + np.log(np.sum(np.exp((corr - m) * alpha)) / N) / alpha


def _project(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    diff = target - np.sum(h)
    if abs(diff) < 1e-12:
        return h
    # Iteratively adjust
    for _ in range(20):
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-10
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        else:
            mask = h > 1e-10
            if not np.any(mask):
                break
            h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
        diff = target - np.sum(h)
    # Final exact fix
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        # Distribute to elements not at boundary
        interior = (h > 1e-8) & (h < 1.0 - 1e-8)
        if np.any(interior):
            h[interior] += diff / np.sum(interior)
            h = np.clip(h, 0.0, 1.0)
    return h


def _nelder_mead(h0, max_iter=500, time_limit=None, t_start=None):
    """Nelder-Mead optimization with projection and time guard."""
    if t_start is None:
        t_start = time.time()
    
    h0 = np.asarray(h0, dtype=np.float64).copy()
    h0 = _project(h0)
    
    # Initialize simplex: h0 plus perturbations
    n = N
    simplex = [h0.copy()]
    f_values = [_objective_smooth(h0)]
    
    # Create simplex vertices
    step = 0.05
    for i in range(n):
        v = h0.copy()
        v[i] += step
        v = _project(v)
        simplex.append(v)
        f_values.append(_objective_smooth(v))
    
    # Sort by function value
    order = np.argsort(f_values)
    simplex = [simplex[i].copy() for i in order]
    f_values = [f_values[i] for i in order]
    
    best_h = simplex[0].copy()
    best_f = f_values[0]
    best_raw = _objective(best_h)
    
    # Nelder-Mead parameters
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - t_start > time_limit:
            break
        
        # Ensure sorted
        order = np.argsort(f_values)
        simplex = [simplex[i].copy() for i in order]
        f_values = [f_values[i] for i in order]
        
        # Centroid of all but worst
        centroid = np.mean([simplex[i] for i in range(len(simplex) - 1)], axis=0)
        worst = simplex[-1]
        worst_f = f_values[-1]
        
        # Reflection
        xr = centroid + alpha * (centroid - worst)
        xr = _project(xr)
        fr = _objective_smooth(xr)
        
        if fr < f_values[0]:
            # Expansion
            xe = centroid + gamma * (centroid - worst)
            xe = _project(xe)
            fe = _objective_smooth(xe)
            if fe < fr:
                simplex[-1] = xe
                f_values[-1] = fe
            else:
                simplex[-1] = xr
                f_values[-1] = fr
        elif fr < f_values[-2]:
            simplex[-1] = xr
            f_values[-1] = fr
        else:
            # Contraction
            if fr < worst_f:
                # Outside contraction
                xc = centroid + rho * (centroid - worst)
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - worst)
            xc = _project(xc)
            fc = _objective_smooth(xc)
            
            if fc < min(fr, worst_f):
                simplex[-1] = xc
                f_values[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = _project(simplex[i])
                    f_values[i] = _objective_smooth(simplex[i])
        
        # Track best
        min_idx = np.argmin(f_values)
        if f_values[min_idx] < best_f:
            best_f = f_values[min_idx]
            best_h = simplex[min_idx].copy()
            best_raw = _objective(best_h)
        
        # Convergence check
        if max(f_values) - min(f_values) < 1e-10:
            break
    
    return best_h, best_raw


def construct():
    t0 = time.time()
    
    # Generate diverse initializations
    inits = []
    
    # 1. Uniform (baseline)
    inits.append(np.full(N, 0.5))
    
    # 2. Linear ramp
    inits.append(np.array([(2*i + 1) / (2*N) for i in range(N)]))
    
    # 3. Sinusoidal patterns
    for freq in [1, 2, 3, 5]:
        for A in [0.1, 0.2, 0.3, 0.4, 0.5]:
            h = 0.5 + A * np.sin(2 * np.pi * freq * np.arange(N) / N)
            inits.append(np.clip(h, 0, 1))
    
    # 4. Cosine patterns
    for freq in [1, 2, 3]:
        for A in [0.1, 0.2, 0.3, 0.4, 0.5]:
            h = 0.5 + A * np.cos(2 * np.pi * freq * np.arange(N) / N)
            inits.append(np.clip(h, 0, 1))
    
    # 5. Triangular wave
    x = np.linspace(0, 2, N, endpoint=False)
    tri = 1 - np.abs(2*x - 1)
    for A in [0.1, 0.2, 0.3, 0.4, 0.5]:
        h = 0.5 + A * (tri - 0.5)
        inits.append(np.clip(h, 0, 1))
    
    # 6. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(10):
        h = rng.uniform(0, 1, N)
        inits.append(h)
    
    # Project all and evaluate
    best_h = None
    best_val = float('inf')
    
    for h0 in inits:
        h0 = _project(h0)
        val = _objective(h0)
        if val < best_val:
            best_val = val
            best_h = h0.copy()
        
        if time.time() - t0 > DEADLINE - 5:
            break
    
    if best_h is None:
        best_h = np.full(N, 0.5)
        best_val = _objective(best_h)
    
    # Run Nelder-Mead from best and several diverse starts
    # Time budget for optimization
    opt_budget = DEADLINE - 5
    
    # Run NM from best first
    t_nm_start = time.time()
    remaining = opt_budget - (t_nm_start - t0)
    h_opt, val_opt = _nelder_mead(best_h, max_iter=300, time_limit=remaining, t_start=t0)
    if val_opt < best_val:
        best_val = val_opt
        best_h = h_opt.copy()
    
    # Run NM from other diverse starts if time permits
    diverse_starts = [
        np.array([(2*i + 1) / (2*N) for i in range(N)]),  # ramp
        0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N),  # sin freq 1
        0.5 + 0.4 * np.cos(2 * np.pi * 2 * np.arange(N) / N),  # cos freq 2
        np.clip(0.5 + 0.3 * (1 - np.abs(2*np.linspace(0, 2, N, endpoint=False) - 1) - 0.5), 0, 1),  # triangle
    ]
    
    for h_start in diverse_starts:
        if time.time() - t0 > DEADLINE - 3:
            break
        
        h_start = _project(h_start)
        val_start = _objective(h_start)
        
        # Only optimize if this start is competitive
        if val_start < best_val + 5:
            t_nm_start = time.time()
            remaining = opt_budget - (t_nm_start - t0)
            if remaining > 2:
                h_opt, val_opt = _nelder_mead(h_start, max_iter=200, time_limit=remaining, t_start=t0)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
    
    # Final polish: run a short NM from the best found
    if time.time() - t0 < DEADLINE - 2:
        t_nm_start = time.time()
        remaining = DEADLINE - 2 - t0
        if remaining > 1:
            h_opt, val_opt = _nelder_mead(best_h, max_iter=100, time_limit=remaining, t_start=t0)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
    
    # Ensure exact sum constraint
    final_h = best_h.copy()
    final_h = _project(final_h)
    
    return final_h
