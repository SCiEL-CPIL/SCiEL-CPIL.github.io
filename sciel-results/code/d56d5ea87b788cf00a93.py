import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0  # width of each interval

# Fine grid for accurate autoconvolution
FINE_N = 3000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()


def ratio_from_h(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Normalization: max((f*f)(t)) / (integral f)^2
    """
    h = np.asarray(h, dtype=np.float64)
    if h.size != N:
        h = np.resize(h, N)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')

    # Map step heights to fine grid
    f_fine = h[CELL]

    # FFT-based autoconvolution
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX  # scale by dx for continuous integral

    max_conv = np.max(conv)

    # Normalization: (integral f)^2 = (sum(h) * W)^2
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')

    return max_conv / integral_sq


def smooth_objective(h, tau=0.01):
    """
    Smooth surrogate: log-sum-exp of conv / (integral)^2.
    Uses fine-grid FFT for accuracy.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX

    # Log-sum-exp for smooth max
    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    lse = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10

    return lse / integral_sq


def make_double_box(m):
    """Create a symmetric double-box structure with gap 2*m."""
    heights = np.zeros(N)
    gap = 2 * m
    non_zero_count = N - gap
    if non_zero_count <= 0 or non_zero_count % 2 != 0:
        return None
    block_len = non_zero_count // 2
    if block_len == 0:
        return None
    H = float(N) / float(non_zero_count)
    heights[0:block_len] = H
    heights[N - block_len:N] = H
    return heights


def make_multi_box(n_boxes, gap):
    """Create n_boxes equal blocks separated by gaps."""
    heights = np.zeros(N)
    total_gap = (n_boxes - 1) * gap
    non_zero_count = N - total_gap
    if non_zero_count <= 0:
        return None
    block_len = non_zero_count // n_boxes
    if block_len == 0:
        return None
    H = float(N) / float(non_zero_count)
    for i in range(n_boxes):
        start = i * (block_len + gap)
        end = start + block_len
        if end <= N:
            heights[start:end] = H
    return heights


def coordinate_descent_polish(h, max_iters=200, deadline=None):
    """
    Simple coordinate descent to polish the solution.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    best_val = ratio_from_h(h)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Try adjusting each bin
        for i in range(0, N, 2):  # Step by 2 for speed
            if deadline is not None and time.time() > deadline:
                break
            
            orig = h[i]
            
            # Try a few perturbations
            for delta in [-0.1, -0.05, 0.05, 0.1]:
                h[i] = max(0.0, orig + delta)
                val = ratio_from_h(h)
                if val < best_val:
                    best_val = val
                    improved = True
                    break
                h[i] = orig
            
            if not improved and it > 50:
                # Try smaller steps
                for delta in [-0.02, -0.01, 0.01, 0.02]:
                    h[i] = max(0.0, orig + delta)
                    val = ratio_from_h(h)
                    if val < best_val:
                        best_val = val
                        improved = True
                        break
                    h[i] = orig
    
    return h, best_val


def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)

    # Generate diverse initializations from block structures
    inits = []

    # 1. Double-box structures with various gaps
    for m in [10, 20, 30, 50, 75, 100, 150, 200]:
        h = make_double_box(m)
        if h is not None:
            inits.append(h.copy())

    # 2. Multi-box structures
    for n_boxes in [3, 4, 5]:
        for gap in [5, 10, 20, 30]:
            h = make_multi_box(n_boxes, gap)
            if h is not None:
                inits.append(h.copy())

    # 3. Cosine-tapered boxes
    for m in [20, 50, 100]:
        h = make_double_box(m)
        if h is not None:
            gap = 2 * m
            non_zero_count = N - gap
            block_len = non_zero_count // 2
            # Apply cosine taper to each block
            taper = 0.5 * (1 + np.cos(np.pi * np.arange(block_len) / (block_len - 1)))
            h[0:block_len] *= taper
            h[N - block_len:N] *= taper[::-1]
            # Renormalize
            s = h.sum()
            if s > 1e-10:
                h *= N / s
            inits.append(h.copy())

    # 4. Random smooth perturbations of uniform
    np.random.seed(42)
    for _ in range(5):
        h = np.ones(N)
        # Add low-frequency noise
        for k in [1, 2, 3]:
            amp = np.random.uniform(0.1, 0.3)
            phase = np.random.uniform(0, 2 * np.pi)
            h += amp * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
        h = np.clip(h, 0.0, None)
        s = h.sum()
        if s > 1e-10:
            h *= N / s
        inits.append(h.copy())

    # Evaluate all initializations and keep the best few
    scored_inits = []
    for h in inits:
        val = ratio_from_h(h)
        scored_inits.append((val, h))
        if val < best_val:
            best_val = val
            best_h = h.copy()

    scored_inits.sort(key=lambda x: x[0])

    # Optimize from the top initializations
    for idx, (init_val, init_h) in enumerate(scored_inits[:8]):
        if time.time() > deadline - 30.0:
            break

        try:
            # Use L-BFGS-B on the smooth surrogate
            res = minimize(
                smooth_objective,
                init_h,
                args=(0.01,),
                method='L-BFGS-B',
                bounds=[(0.0, None)] * N,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 30}
            )

            if np.isfinite(res.fun):
                h_candidate = np.clip(res.x, 0.0, None)
                # Renormalize to maintain L1 norm
                s = h_candidate.sum()
                if s > 1e-10:
                    h_candidate *= N / s
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Final coordinate descent polish
    if time.time() < deadline - 10.0:
        polished_h, polished_val = coordinate_descent_polish(best_h, max_iters=100, deadline=deadline - 5.0)
        if polished_val < best_val:
            best_val = polished_val
            best_h = polished_h

    # Ensure nonnegative and return as list
    best_h = np.clip(best_h, 0.0, None)
    return best_h.tolist()
