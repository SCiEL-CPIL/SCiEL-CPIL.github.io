import numpy as np
import time

N = 50
_start_time = time.time()
_TIMEOUT = 135.0  # safety margin before 150s kill

def _score(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    # Full linear autoconvolution
    conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _random_init():
    """Generate a random nonnegative vector with good structure."""
    # Multi-spike patterns are key
    n_spikes = np.random.choice([2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.2])
    h = np.zeros(N)
    
    # Place spikes at random positions with minimum spacing
    min_spacing = np.random.choice([3, 5, 8, 12])
    positions = []
    for _ in range(n_spikes):
        for _attempt in range(100):
            pos = np.random.randint(0, N)
            if all(abs(pos - p) >= min_spacing for p in positions):
                positions.append(pos)
                break
    
    # Assign random amplitudes
    for pos in positions:
        h[pos] = np.random.exponential(1.0)
    
    # Optionally add small background
    if np.random.random() < 0.3:
        h += np.random.uniform(0, 0.05, size=N)
    
    return h

def _local_optimize(h0, max_iters=3000):
    """Local optimization using coordinate descent + random perturbation."""
    h = h0.copy()
    h = np.maximum(h, 0.0)
    best_score = _score(h)
    best_h = h.copy()
    
    # Phase 1: Coordinate descent
    step_size = 0.1
    for i in range(max_iters):
        if time.time() - _start_time > _TIMEOUT:
            break
        
        improved = False
        for j in range(N):
            # Try increasing
            h_trial = h.copy()
            h_trial[j] += step_size
            s_trial = _score(h_trial)
            if s_trial > best_score:
                h = h_trial
                best_score = s_trial
                best_h = h.copy()
                improved = True
                continue
            
            # Try decreasing
            if h[j] > step_size:
                h_trial = h.copy()
                h_trial[j] -= step_size
                s_trial = _score(h_trial)
                if s_trial > best_score:
                    h = h_trial
                    best_score = s_trial
                    best_h = h.copy()
                    improved = True
                    continue
            
            # Try zeroing
            if h[j] > 0:
                h_trial = h.copy()
                h_trial[j] = 0
                s_trial = _score(h_trial)
                if s_trial > best_score:
                    h = h_trial
                    best_score = s_trial
                    best_h = h.copy()
                    improved = True
        
        if improved:
            step_size = min(step_size * 1.05, 0.5)
        else:
            step_size = max(step_size * 0.7, 0.01)
            if step_size < 0.015:
                break
    
    # Phase 2: Random perturbation refinement
    step_size = 0.05
    for i in range(2000):
        if time.time() - _start_time > _TIMEOUT:
            break
        h_trial = h + np.random.normal(0, step_size, size=N)
        h_trial = np.maximum(h_trial, 0.0)
        s_trial = _score(h_trial)
        if s_trial > best_score:
            h = h_trial
            best_score = s_trial
            best_h = h.copy()
            step_size = max(step_size * 1.03, 0.005)
        else:
            step_size = max(step_size * 0.95, 0.001)
    
    return best_h, best_score

def _try_structural_patterns():
    """Try known good structural patterns - focused on multi-spike configurations."""
    candidates = []
    
    # Two-spike patterns with various spacings and amplitude ratios
    for gap in range(2, 25):
        for r in [0.5, 0.7, 1.0, 1.4, 2.0, 3.0]:
            h = np.zeros(N)
            a = r / (1 + r)
            b = 1 / (1 + r)
            h[0] = a
            h[gap] = b
            candidates.append((h.copy(), _score(h)))
    
    # Three-spike patterns
    for gap1 in range(3, 15):
        for gap2 in range(gap1 + 3, min(gap1 + 15, N)):
            for r in [0.7, 1.0, 1.5]:
                h = np.zeros(N)
                a = r / (1 + r + 1)
                b = 1 / (1 + r + 1)
                c = 1 / (1 + r + 1)
                h[0] = a
                h[gap1] = b
                h[gap2] = c
                candidates.append((h.copy(), _score(h)))
    
    # Four-spike patterns
    for spacing in [5, 8, 10, 12]:
        for r in [0.8, 1.0, 1.3]:
            h = np.zeros(N)
            a = r / (1 + r)
            b = 1 / (1 + r)
            for k in range(4):
                h[k * spacing] = a if k % 2 == 0 else b
            candidates.append((h.copy(), _score(h)))
    
    # Five-spike patterns
    for spacing in [6, 8, 10]:
        h = np.zeros(N)
        for k in range(5):
            h[k * spacing] = 1.0 / 5
        candidates.append((h.copy(), _score(h)))
    
    # Alternating high/low patterns
    for period in [4, 6, 8]:
        for ratio in [0.3, 0.5, 0.7]:
            h = np.zeros(N)
            for i in range(N):
                if i % period < period // 2:
                    h[i] = 1.0
                else:
                    h[i] = ratio
            candidates.append((h.copy(), _score(h)))
    
    return candidates

def construct():
    """Construct 50 nonnegative heights maximizing ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    global _start_time
    _start_time = time.time()
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try structural patterns
    patterns = _try_structural_patterns()
    for h, s in patterns:
        if time.time() - _start_time > _TIMEOUT:
            break
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Local optimization from best structural pattern
    if best_score > 0:
        h_opt, s_opt = _local_optimize(best_h, max_iters=2000)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Multiple random restarts
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() - _start_time > _TIMEOUT:
            break
        h_rand = _random_init()
        h_opt, s_opt = _local_optimize(h_rand, max_iters=1500)
        if s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Final polish
    if time.time() - _start_time < _TIMEOUT - 5:
        h_final, s_final = _local_optimize(best_h, max_iters=2000)
        if s_final > best_score:
            best_score = s_final
            best_h = h_final
    
    return list(best_h)
