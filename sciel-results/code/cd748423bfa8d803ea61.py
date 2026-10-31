"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

def _overlap(h, s):
    """Compute overlap of h with shifted (1-h) at shift s."""
    shifted = np.roll(h, -s)
    return np.sum(h * (1.0 - shifted))

def _max_overlap(h):
    """Compute maximum overlap over all shifts."""
    # For efficiency, compute all shifts at once using FFT or direct
    # overlap[s] = sum_i h[i] * (1 - h[(i+s)%N])
    # = sum_i h[i] - sum_i h[i]*h[(i+s)%N]
    # = N/2 - autocorrelation of h at lag s
    # autocorr[s] = sum_i h[i] * h[(i+s)%N]
    h_centered = h - np.mean(h)
    # Circular autocorrelation via FFT
    H = np.fft.fft(h)
    acf = np.fft.ifft(H * np.conj(H)).real
    acf /= N  # normalize
    # overlap[s] = N/2 - acf[s]
    overlaps = N / 2.0 - acf
    return np.max(overlaps)

def construct():
    deadline = time.time() + 130  # safety margin
    
    # Start with a structured initial guess: alternating pattern
    # Try a few different initial structures
    best_h = None
    best_score = np.inf
    
    # Initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.full(N, 0.5))
    
    # 2. Block pattern: first half 1, second half 0
    h_block = np.zeros(N)
    h_block[:N//2] = 1.0
    candidates.append(h_block.copy())
    
    # 3. Alternating
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    candidates.append(h_alt)
    
    # 4. Sawtooth-like
    h_saw = np.linspace(0, 1, N)
    h_saw = (h_saw - h_saw.mean()) / h_saw.std() * 0.5 + 0.5
    h_saw = np.clip(h_saw, 0, 1)
    # Adjust to sum to N/2
    h_saw = h_saw - np.mean(h_saw) + 0.5
    h_saw = np.clip(h_saw, 0, 1)
    candidates.append(h_saw.copy())
    
    # 5. Random with mean 0.5
    np.random.seed(42)
    h_rand = np.random.uniform(0, 1, N)
    h_rand = h_rand - np.mean(h_rand) + 0.5
    h_rand = np.clip(h_rand, 0, 1)
    # Adjust sum
    diff = N/2.0 - np.sum(h_rand)
    h_rand += diff / N
    h_rand = np.clip(h_rand, 0, 1)
    candidates.append(h_rand.copy())
    
    # Evaluate initial candidates
    for h0 in candidates:
        # Ensure sum is exactly N/2
        h0 = np.clip(h0, 0, 1)
        s = np.sum(h0)
        if s > 0:
            h0 = h0 * (N/2.0) / s
            h0 = np.clip(h0, 0, 1)
        # Final adjustment
        diff = N/2.0 - np.sum(h0)
        if abs(diff) > 1e-10:
            # Distribute difference
            mask = (h0 < 1) if diff > 0 else (h0 > 0)
            if np.any(mask):
                h0[mask] += diff / np.sum(mask)
                h0 = np.clip(h0, 0, 1)
        
        score = _max_overlap(h0)
        if score < best_score:
            best_score = score
            best_h = h0.copy()
    
    if time.time() > deadline:
        return best_h
    
    # Use optimization to refine the best candidate
    h_current = best_h.copy()
    
    def objective(x):
        x = np.clip(x, 0, 1)
        # Enforce sum constraint by projection
        s = np.sum(x)
        if s > 1e-10:
            x = x * (N/2.0) / s
            x = np.clip(x, 0, 1)
        return _max_overlap(x)
    
    # Try Nelder-Mead on the best candidate
    try:
        result = minimize(
            objective,
            h_current,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        if result.fun < best_score:
            best_h = np.clip(result.x, 0, 1)
            best_score = result.fun
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_h
    
    # Try a few more random restarts with optimization
    for seed in range(5):
        if time.time() > deadline:
            break
        np.random.seed(seed + 100)
        h_rand = np.random.uniform(0, 1, N)
        h_rand = h_rand - np.mean(h_rand) + 0.5
        h_rand = np.clip(h_rand, 0, 1)
        diff = N/2.0 - np.sum(h_rand)
        h_rand += diff / N
        h_rand = np.clip(h_rand, 0, 1)
        
        try:
            result = minimize(
                objective,
                h_rand,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < best_score:
                best_h = np.clip(result.x, 0, 1)
                best_score = result.fun
        except Exception:
            pass
    
    # Final: ensure sum is exactly N/2
    if best_h is not None:
        best_h = np.clip(best_h, 0, 1)
        s = np.sum(best_h)
        if s > 1e-10:
            best_h = best_h * (N/2.0) / s
            best_h = np.clip(best_h, 0, 1)
        # Fine-tune to get exact sum
        diff = N/2.0 - np.sum(best_h)
        if abs(diff) > 1e-12:
            # Adjust one element
            idx = 0
            if diff > 0:
                while idx < N and best_h[idx] >= 1.0:
                    idx += 1
                if idx < N:
                    best_h[idx] += diff
                    best_h[idx] = min(best_h[idx], 1.0)
            else:
                while idx < N and best_h[idx] <= 0.0:
                    idx += 1
                if idx < N:
                    best_h[idx] += diff
                    best_h[idx] = max(best_h[idx], 0.0)
    
    return best_h if best_h is not None else np.full(N, 0.5)
# EVOLVE-BLOCK-END
