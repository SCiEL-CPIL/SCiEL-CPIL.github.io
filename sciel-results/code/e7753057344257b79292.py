import numpy as np
import time

N = 50

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0)  # enforce nonnegativity
    norm1 = np.sum(h)
    if norm1 < 1e-12:
        return 0.0
    norm2_sq = np.sum(h**2)
    # Autoconvolution: h*h
    n_conv = 2 * N
    h_padded = np.zeros(n_conv)
    h_padded[:N] = h
    conv = np.fft.irfft(np.fft.rfft(h_padded) ** 2)
    sup_conv = np.max(np.abs(conv[:N]))
    if sup_conv < 1e-12:
        return 0.0
    return norm2_sq / (norm1 * sup_conv)

def _find_sidon_set(n, max_k):
    """Find a Sidon set of size n in {0, ..., max_k}."""
    # A Sidon set has all pairwise sums distinct
    # For N=50, we need a Sidon set of size 50
    # Try greedy construction
    best = None
    best_score = -1
    
    for attempt in range(20):
        if attempt == 0:
            # Start with a known structure
            s = list(range(0, n))
        else:
            s = [0]
            for i in range(1, n):
                # Greedily add the smallest possible element
                added = False
                for candidate in range(s[-1] + 1, max_k + 1):
                    # Check if adding candidate creates duplicate sums
                    valid = True
                    sums = set()
                    for j in range(len(s)):
                        for k in range(j, len(s)):
                            if s[j] + s[k] in sums:
                                valid = False
                                break
                            sums.add(s[j] + s[k])
                        if not valid:
                            break
                    if not valid:
                        continue
                    sums = set()
                    for j in range(len(s)):
                        sums.add(s[j] + candidate)
                    # Check for conflicts with existing sums
                    for j in range(len(s)):
                        for k in range(j, len(s)):
                            if s[j] + s[k] in sums:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        s.append(candidate)
                        added = True
                        break
                if not added:
                    break
        
        if len(s) == n:
            # Score: prefer smaller max element
            score = -max(s)
            if score > best_score:
                best_score = score
                best = s[:]
    
    if best is None:
        # Fallback: use a simple arithmetic progression (not Sidon but valid)
        best = list(range(n))
    
    return best

def _spectral_optimize(h0, max_iter=500, tol=1e-8):
    """
    Optimize spectral magnitudes to minimize autoconvolution peak.
    h = IDFT(|S|^2 * phase) where we optimize |S|^2 subject to h >= 0.
    """
    h = np.maximum(h0, 0).astype(float)
    s = np.sum(h)
    if s < 1e-12:
        return h0
    
    # Work in normalized space
    h = h / s
    
    # Get current spectrum
    H = np.fft.rfft(h, n=2*N)
    S_mag = np.abs(H)
    S_phase = np.angle(H)
    
    # We optimize S_mag (non-negative) to minimize sup|conv|
    # while keeping h >= 0
    
    best_h = h.copy()
    best_ratio = _ratio(h)
    best_S_mag = S_mag.copy()
    
    lr = 0.01
    for it in range(max_iter):
        # Compute current h from spectrum
        H = S_mag * np.exp(1j * S_phase)
        h = np.fft.irfft(H, n=2*N)[:N]
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-12:
            break
        h = h / s
        
        ratio = _ratio(h)
        if ratio > best_ratio:
            best_ratio = ratio
            best_h = h.copy()
            best_S_mag = S_mag.copy()
        
        # Gradient: d(sup_conv)/d(S_mag)
        # Approximate by finite differences on a few components
        # For efficiency, use a simplified gradient
        # The autoconvolution peak is dominated by low frequencies
        # Increase low-freq magnitudes to spread energy
        
        # Simple heuristic: flatten the spectrum
        # Target: make |H| more uniform
        target = np.ones_like(S_mag) * np.mean(S_mag)
        S_mag = 0.9 * S_mag + 0.1 * target
        
        # Enforce non-negativity of h
        H = S_mag * np.exp(1j * S_phase)
        h_check = np.fft.irfft(H, n=2*N)[:N]
        if np.min(h_check) < -1e-6:
            # Scale back
            S_mag = best_S_mag + 0.5 * (S_mag - best_S_mag)
    
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    best_h = np.ones(N)
    best_ratio = _ratio(best_h)
    
    # Strategy 1: Sidon set initialization
    # A Sidon set minimizes collisions in autoconvolution
    try:
        sidon = _find_sidon_set(N, 2*N)
        h_sidon = np.zeros(N)
        for pos in sidon:
            if pos < N:
                h_sidon[pos] = 1.0
        if np.sum(h_sidon) > 0:
            h_sidon = h_sidon / np.sum(h_sidon)
            r = _ratio(h_sidon)
            if r > best_ratio:
                best_ratio = r
                best_h = h_sidon.copy()
    except Exception:
        pass
    
    # Strategy 2: Sparse random with Sidon-like structure
    np.random.seed(42)
    
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        
        # Create a sparse pattern with controlled spacing
        k = np.random.randint(5, 15)  # number of non-zero elements
        if k <= 1:
            continue
        
        # Try to place elements with roughly equal spacing
        spacing = N // k
        positions = []
        for i in range(k):
            pos = i * spacing + np.random.randint(0, max(1, spacing // 2))
            if pos < N and pos not in positions:
                positions.append(pos)
        
        if len(positions) < 2:
            continue
        
        h = np.zeros(N)
        h[positions] = 1.0
        h = h / np.sum(h)
        
        r = _ratio(h)
        if r > best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Strategy 3: Spectral optimization from best starting point
    if time.time() < deadline - 20:
        h_spec = _spectral_optimize(best_h, max_iter=200)
        r = _ratio(h_spec)
        if r > best_ratio:
            best_ratio = r
            best_h = h_spec.copy()
    
    # Strategy 4: Local refinement with coordinate descent
    if time.time() < deadline - 10:
        h = best_h.copy()
        s = np.sum(h)
        if s > 0:
            h = h / s
        
        improved = True
        max_cd_iter = 100
        cd_iter = 0
        while improved and cd_iter < max_cd_iter and time.time() < deadline - 5:
            improved = False
            cd_iter += 1
            
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                old_val = h[i]
                # Try moving mass from i to other positions
                for j in range(N):
                    if j == i:
                        continue
                    
                    # Move small amount from i to j
                    delta = 0.01 * old_val
                    if delta > h[i]:
                        delta = h[i] * 0.5
                    
                    h_new = h.copy()
                    h_new[i] -= delta
                    h_new[j] += delta
                    h_new = np.maximum(h_new, 0)
                    s_new = np.sum(h_new)
                    if s_new < 1e-12:
                        continue
                    h_new = h_new / s_new
                    
                    r_new = _ratio(h_new)
                    if r_new > best_ratio:
                        best_ratio = r_new
                        best_h = h_new.copy()
                        h = h_new
                        improved = True
    
    # Strategy 5: Try binary patterns (0 or 1)
    if time.time() < deadline - 5:
        # Try various sparsity levels
        for k in range(3, 20):
            if time.time() > deadline - 5:
                break
            
            # Random binary patterns
            for trial in range(20):
                if time.time() > deadline - 5:
                    break
                
                idx = np.random.choice(N, size=k, replace=False)
                h = np.zeros(N)
                h[idx] = 1.0
                h = h / np.sum(h)
                
                r = _ratio(h)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h.copy()
    
    # Final: ensure non-negative and return as list
    best_h = np.maximum(best_h, 0)
    s = np.sum(best_h)
    if s < 1e-12:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    
    return [float(x) for x in best_h]
