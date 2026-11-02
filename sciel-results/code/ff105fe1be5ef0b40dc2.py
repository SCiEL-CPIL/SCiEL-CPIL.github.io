import numpy as np
import time

N = 50

def compute_score(h):
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    l2_sq = np.sum(h**2)
    
    c = np.correlate(h, h, mode='full')
    sup_c = np.max(c)
    
    if sup_c < 1e-15:
        return 0.0
    
    return l2_sq / (l1 * sup_c)

def negative_score(h):
    return -compute_score(h)

def _correlation_peaks(h):
    """Compute all autoconvolution peaks for a given h."""
    return np.correlate(h, h, mode='full')

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    def update_best(h):
        nonlocal best_h, best_score
        h = np.maximum(h, 0)
        s = compute_score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        return s
    
    # Multiplicative update for maximizing ||h||_2^2 / (||h||_1 * sup|autoconv|)
    # This is equivalent to minimizing sup|autoconv| / ||h||_2^2 subject to ||h||_1 = 1
    # or maximizing the ratio directly.
    
    def multiplicative_update(h, n_iters=500):
        """Sinkhorn-Knopp style multiplicative update."""
        h = np.maximum(h, 1e-15)
        for i in range(n_iters):
            if time_left() < 5.0:
                break
            # Normalize to L1 = 1
            l1 = np.sum(h)
            if l1 < 1e-15:
                break
            h = h / l1
            
            # Compute autoconvolution
            c = np.correlate(h, h, mode='full')
            sup_c = np.max(c)
            if sup_c < 1e-15:
                break
            
            # Find which lag achieves the supremum
            # The gradient of sup|autoconv| w.r.t. h is tricky, but we can use
            # the peak location to guide updates
            
            # For now, use a simpler approach: penalize high autoconv peaks
            # by reducing h where it contributes most to the peak
            
            # Compute contribution of each h[i] to the peak autoconv
            peak_lag = np.argmax(c)
            # The autoconv at lag k is sum_i h[i]*h[i+k] (with appropriate indexing)
            # For full mode, lag index corresponds to offset
            
            # Simplified: use the fact that we want to minimize sup|autoconv|
            # while maintaining ||h||_2^2
            # This is like a water-filling problem
            
            # Try a gradient-free approach: randomly perturb and keep improvements
            if np.random.random() < 0.1:
                # Random restart-like perturbation
                mask = np.random.random(N) > 0.5
                h_new = h * (1 + 0.1 * np.random.randn(N) * mask)
                h_new = np.maximum(h_new, 0)
                l1_new = np.sum(h_new)
                if l1_new > 1e-15:
                    h_new = h_new / l1_new
                    s_new = compute_score(h_new)
                    s_old = compute_score(h)
                    if s_new > s_old:
                        h = h_new
        return h
    
    def coordinate_descent_polish(h, n_iters=1000):
        """Polish with coordinate descent."""
        h = np.maximum(h, 0)
        for i in range(n_iters):
            if time_left() < 5.0:
                break
            improved = False
            for j in range(N):
                if time_left() < 5.0:
                    break
                # Try adjusting h[j]
                h_old_j = h[j]
                best_j_val = h_old_j
                best_j_score = compute_score(h)
                
                # Try several values around current
                for factor in [0.5, 0.7, 1.3, 2.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
                    h[j] = h_old_j * factor if factor > 0 else 0.0
                    s = compute_score(h)
                    if s > best_j_score + 1e-12:
                        best_j_score = s
                        best_j_val = h[j]
                        improved = True
                
                h[j] = best_j_val
        
            if not improved:
                break
        return h
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Single spike
    h1 = np.zeros(N)
    h1[0] = 1.0
    candidates.append(h1)
    
    # 2. Two adjacent
    h2 = np.zeros(N)
    h2[0] = 0.7
    h2[1] = 0.3
    candidates.append(h2)
    
    # 3. Two non-adjacent (Sidon-like)
    h3 = np.zeros(N)
    h3[0] = 0.5
    h3[10] = 0.5
    candidates.append(h3)
    
    # 4. Three elements Sidon-like
    h4 = np.zeros(N)
    h4[0] = 1/3
    h4[7] = 1/3
    h4[15] = 1/3
    candidates.append(h4)
    
    # 5. Random sparse
    for _ in range(5):
        hr = np.zeros(N)
        k = np.random.randint(2, 10)
        idx = np.random.choice(N, k, replace=False)
        hr[idx] = 1.0 / k
        candidates.append(hr)
    
    # 6. Structured: low-discrepancy lattice based
    # Use a mask that selects positions with good separation
    for support_size in [3, 5, 7, 10]:
        # Try to find well-separated positions
        best_positions = None
        best_min_dist = 0
        for trial in range(100):
            if time_left() < 10.0:
                break
            positions = np.sort(np.random.choice(N, support_size, replace=False))
            # Check minimum distance
            dists = np.diff(positions)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_positions = positions
        
        if best_positions is not None:
            hs = np.zeros(N)
            hs[best_positions] = 1.0 / support_size
            candidates.append(hs)
    
    # 7. Based on DFT magnitude squared (as per directive)
    # Try to construct h as |IDFT(mask)|^2 for binary masks
    for mask_density in [0.1, 0.2, 0.3, 0.5]:
        if time_left() < 10.0:
            break
        for trial in range(20):
            if time_left() < 10.0:
                break
            mask = (np.random.random(N) < mask_density).astype(float)
            # IDFT of mask
            h_dft = np.abs(np.fft.ifft(mask))**2
            h_dft = np.maximum(h_dft, 0)
            l1 = np.sum(h_dft)
            if l1 > 1e-15:
                h_dft = h_dft / l1
                candidates.append(h_dft)
    
    # Evaluate all candidates and keep the best
    for h in candidates:
        if time_left() < 10.0:
            break
        s = update_best(h)
    
    # Now run multiplicative updates from the best candidate
    if best_score > 0:
        h_work = best_h.copy()
        h_work = multiplicative_update(h_work, n_iters=300)
        update_best(h_work)
    
    # Run coordinate descent polish
    if best_score > 0:
        h_work = best_h.copy()
        h_work = coordinate_descent_polish(h_work, n_iters=200)
        update_best(h_work)
    
    # Try a few more random restarts if time permits
    restarts_done = 0
    while time_left() > 20.0 and restarts_done < 10:
        restarts_done += 1
        # Random sparse initialization
        hr = np.zeros(N)
        k = np.random.randint(2, 15)
        idx = np.random.choice(N, k, replace=False)
        weights = np.random.dirichlet(np.ones(k))
        hr[idx] = weights
        
        # Run multiplicative update
        hr = multiplicative_update(hr, n_iters=100)
        update_best(hr)
        
        # Polish
        hr = coordinate_descent_polish(hr, n_iters=50)
        update_best(hr)
    
    # Final polish with more iterations
    if best_score > 0 and time_left() > 10.0:
        h_work = best_h.copy()
        h_work = coordinate_descent_polish(h_work, n_iters=500)
        update_best(h_work)
    
    # Ensure valid output
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h[0] = 1.0
    
    return list(best_h)
