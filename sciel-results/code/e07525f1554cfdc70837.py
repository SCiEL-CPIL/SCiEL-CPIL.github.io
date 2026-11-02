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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    def update_best(h):
        nonlocal best_h, best_score
        h = np.maximum(h, 0.0)
        s = compute_score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        return s
    
    # --- Coordinate descent with analytic optimal step for each coordinate ---
    # For fixed other coordinates, the problem reduces to a 1D optimization
    # over h_j >= 0 of: (S2 + h_j^2) / (S1 + h_j) * max_k c_k(h)
    # where c_k depends on h_j linearly for each k.
    # We use a grid search over h_j values, which is fast and effective.
    
    def coordinate_descent(h, max_iter=100):
        h = np.maximum(h, 0.0).copy()
        for it in range(max_iter):
            if time_left() < 5.0:
                break
            improved = False
            for j in range(N):
                if time_left() < 5.0:
                    break
                old_j = h[j]
                best_val = old_j
                best_s = compute_score(h)
                
                # Generate candidate values for h[j]
                # Include 0, current, and scaled versions
                candidates = [0.0, old_j]
                if old_j > 1e-15:
                    for factor in [0.1, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0, 5.0]:
                        candidates.append(old_j * factor)
                else:
                    # If zero, try small positive values
                    l1_rest = np.sum(h) - old_j
                    if l1_rest > 1e-15:
                        for frac in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]:
                            candidates.append(frac * l1_rest / N)
                
                for val in candidates:
                    h[j] = val
                    s = compute_score(h)
                    if s > best_s + 1e-12:
                        best_s = s
                        best_val = val
                
                h[j] = best_val
                if best_val != old_j:
                    improved = True
            
            if not improved:
                break
        return h
    
    # --- Multiplicative update inspired by Sinkhorn-Knopp ---
    # Maximizing ||h||_2^2 / (||h||_1 * sup|c|) is equivalent to
    # minimizing sup|c| / ||h||_2^2 subject to ||h||_1 = 1.
    # The gradient of the objective w.r.t. h_i involves the autoconvolution peak.
    
    def multiplicative_update(h_init, n_iter=500):
        h = np.maximum(h_init, 1e-10).copy()
        # Normalize to L1 = 1
        l1 = np.sum(h)
        if l1 < 1e-15:
            return h
        h = h / l1
        
        for it in range(n_iter):
            if time_left() < 5.0:
                break
            
            c = np.correlate(h, h, mode='full')
            c_max = np.max(c)
            if c_max < 1e-15:
                break
            
            # Find peak lag
            peak_idx = np.argmax(c)
            
            # Compute gradient-like update
            # The autoconv at lag k: c[k] = sum_i h[i] * h[i + (k - (N-1))]
            # For full mode, index k corresponds to offset (k - (N-1))
            offset = peak_idx - (N - 1)
            
            # Contribution of each h[i] to c[peak_idx]:
            # c[peak] = sum_i h[i] * h[i - offset] (with bounds)
            # So d(c[peak])/d(h[i]) = h[i - offset] + h[i + offset] (with bounds)
            
            grad = np.zeros(N)
            for i in range(N):
                # h[i] appears in c[peak] as h[i]*h[i-offset] and h[i+offset]*h[i]
                idx1 = i - offset
                idx2 = i + offset
                if 0 <= idx1 < N:
                    grad[i] += h[idx1]
                if 0 <= idx2 < N:
                    grad[i] += h[idx2]
            
            # Objective: maximize ||h||_2^2 / (||h||_1 * c_max)
            # d(log obj)/d(h_i) = 2*h_i/||h||_2^2 - 1/||h||_1 - (d(c_max)/d(h_i))/c_max
            # For multiplicative update: h_i <- h_i * exp(eta * gradient)
            
            l2_sq = np.sum(h**2)
            l1_curr = np.sum(h)
            
            # Approximate gradient of the log-objective
            # Simplified: use the ratio of "good" to "bad" contributions
            # h_i should increase if it contributes more to ||h||_2^2 than to c_max
            
            # Scale factor: we want h_i proportional to h_i^2 / (grad_i * c_max)
            # But this is approximate. Use a simpler heuristic:
            # Reduce h_i where it contributes most to the peak, increase elsewhere
            
            # Normalize gradient
            if np.max(grad) > 1e-15:
                grad_norm = grad / np.max(grad)
            else:
                grad_norm = np.ones(N) / N
            
            # Update: h_i <- h_i * (1 - eta * grad_norm_i + eta * 2*h_i/l2_sq * l1_curr)
            eta = 0.5
            factor = 1.0 - eta * grad_norm
            # Add positive contribution from L2 term
            factor += eta * (2.0 * h / l2_sq) * l1_curr
            factor = np.maximum(factor, 0.01)
            
            h = h * factor
            
            # Renormalize to L1 = 1
            l1_new = np.sum(h)
            if l1_new < 1e-15:
                break
            h = h / l1_new
        
        return h
    
    # --- Generate diverse initial candidates ---
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
    
    # 3. Two non-adjacent
    h3 = np.zeros(N)
    h3[0] = 0.5
    h3[25] = 0.5
    candidates.append(h3)
    
    # 4. Three elements well-separated
    h4 = np.zeros(N)
    h4[0] = 1/3
    h4[17] = 1/3
    h4[34] = 1/3
    candidates.append(h4)
    
    # 5. Random sparse patterns
    np.random.seed(42)
    for _ in range(20):
        hr = np.zeros(N)
        k = np.random.randint(2, 15)
        idx = np.random.choice(N, k, replace=False)
        weights = np.random.dirichlet(np.ones(k))
        hr[idx] = weights
        candidates.append(hr)
    
    # 6. Structured: well-separated positions
    for support_size in [3, 5, 7, 10]:
        best_positions = None
        best_min_dist = 0
        for trial in range(200):
            positions = np.sort(np.random.choice(N, support_size, replace=False))
            dists = np.diff(positions)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_positions = positions
        if best_positions is not None:
            hs = np.zeros(N)
            hs[best_positions] = 1.0 / support_size
            candidates.append(hs)
    
    # 7. Periodic patterns
    for period in [3, 4, 5, 7, 10]:
        for phase in range(period):
            h = np.zeros(N)
            for i in range(N):
                if (i + phase) % period == 0:
                    h[i] = 1.0
            candidates.append(h)
    
    # 8. Cosine windows
    for freq in [1, 2, 3, 4]:
        h = np.cos(np.pi * np.arange(N) * freq / N)
        h = np.maximum(h, 0)
        if np.sum(h) > 0:
            candidates.append(h)
    
    # 9. Power-law decay
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        h = np.zeros(N)
        for i in range(N):
            h[i] = (i+1)**(-alpha)
        candidates.append(h)
    
    # 10. DFT-based: |IDFT(mask)|^2 for binary masks
    for mask_density in [0.1, 0.2, 0.3, 0.5]:
        for trial in range(10):
            mask = (np.random.random(N) < mask_density).astype(float)
            h_dft = np.abs(np.fft.ifft(mask))**2
            h_dft = np.maximum(h_dft, 0)
            l1 = np.sum(h_dft)
            if l1 > 1e-15:
                h_dft = h_dft / l1
                candidates.append(h_dft)
    
    # Evaluate all candidates
    for h in candidates:
        if time_left() < 10.0:
            break
        s = update_best(h)
    
    # Run multiplicative update from best candidate
    if best_score > 0:
        h_work = best_h.copy()
        h_work = multiplicative_update(h_work, n_iter=300)
        update_best(h_work)
    
    # Run coordinate descent polish
    if best_score > 0:
        h_work = best_h.copy()
        h_work = coordinate_descent(h_work, max_iter=200)
        update_best(h_work)
    
    # Try random restarts with multiplicative update + coordinate descent
    restarts_done = 0
    while time_left() > 20.0 and restarts_done < 15:
        restarts_done += 1
        hr = np.zeros(N)
        k = np.random.randint(2, 20)
        idx = np.random.choice(N, k, replace=False)
        weights = np.random.dirichlet(np.ones(k))
        hr[idx] = weights
        
        hr = multiplicative_update(hr, n_iter=150)
        update_best(hr)
        
        hr = coordinate_descent(hr, max_iter=100)
        update_best(hr)
    
    # Final intensive polish
    if best_score > 0 and time_left() > 10.0:
        h_work = best_h.copy()
        h_work = coordinate_descent(h_work, max_iter=500)
        update_best(h_work)
        
        # One more multiplicative update round
        h_work = multiplicative_update(h_work, n_iter=200)
        update_best(h_work)
        
        # Final coordinate descent
        h_work = coordinate_descent(h_work, max_iter=300)
        update_best(h_work)
    
    # Ensure valid output
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h[0] = 1.0
    
    return list(best_h)
