import numpy as np
import time
import heapq

def construct():
    N = 24
    start_time = time.time()
    deadline = start_time + 115
    
    # Coarse grid for initial screening
    M_coarse = 256
    thetas_coarse = np.linspace(0, 2 * np.pi, M_coarse, endpoint=False)
    Z_coarse = np.exp(1j * thetas_coarse[:, None] * np.arange(N)[None, :])
    
    # Medium grid for secondary screening
    M_med = 4096
    thetas_med = np.linspace(0, 2 * np.pi, M_med, endpoint=False)
    Z_med = np.exp(1j * thetas_med[:, None] * np.arange(N)[None, :])
    
    # Fine grid for final evaluation
    M_fine = 10000
    thetas_fine = np.linspace(0, 2 * np.pi, M_fine, endpoint=False)
    Z_fine = np.exp(1j * thetas_fine[:, None] * np.arange(N)[None, :])
    
    # Search all 2^24 combinations
    total_combinations = 1 << 24
    chunk_size = 100000
    
    # Min-heap to keep top K candidates (score, coeffs)
    # We want the SMALLEST scores, so we use a max-heap logic or just store and sort later if K is small enough?
    # K=2000 is small. We can store all top candidates in a list and sort at the end.
    # But storing 2^24 is too much.
    # We can maintain a heap of size K.
    # Python's heapq is a min-heap. To keep the smallest K, we can push -score? No, we want to discard large scores.
    # So we keep a heap of the "worst" of the "best K".
    # i.e., a max-heap of size K.
    # Python doesn't have a max-heap. We can negate scores.
    
    K_coarse = 2000
    heap = [] # Stores (-score, coeffs)
    
    for start in range(0, total_combinations, chunk_size):
        if time.time() > deadline - 10:
            break
            
        end = min(start + chunk_size, total_combinations)
        count = end - start
        
        indices = np.arange(start, end)
        shifts = np.arange(N)
        bits = (indices[:, None] >> shifts[None, :]) & 1
        coeffs = 1 - 2 * bits
        
        # Evaluate on coarse grid
        vals = coeffs @ Z_coarse.T
        max_norms = np.max(np.abs(vals), axis=1)
        
        # Get top K from this chunk
        if count > K_coarse:
            # We need the smallest max_norms
            part_idx = np.argpartition(max_norms, K_coarse)[:K_coarse]
            top_scores = max_norms[part_idx]
            top_coeffs = coeffs[part_idx]
        else:
            top_scores = max_norms
            top_coeffs = coeffs
            
        for i in range(len(top_scores)):
            s = -top_scores[i] # Negate for max-heap behavior in min-heap
            c = top_coeffs[i]
            if len(heap) < K_coarse:
                heapq.heappush(heap, (s, c))
            else:
                # If this candidate is better (smaller score, i.e., larger -score) than the worst in heap
                # heap[0] is the smallest -score, i.e., largest score.
                if s > heap[0][0]:
                    heapq.heapreplace(heap, (s, c))
                    
    # Convert heap to list of (score, coeffs)
    # heap contains (-score, coeffs)
    # Sort by -score descending (which is score ascending)
    candidates_coarse = [(-s, c) for s, c in heap]
    candidates_coarse.sort(key=lambda x: x[0])
    
    best_c = None
    best_score = np.inf
    
    # Medium pass
    K_med = 50
    candidates_med = candidates_coarse[:K_med]
    
    if candidates_med:
        # Batch evaluation for medium grid
        # coeffs: (K, N), Z_med.T: (N, M_med)
        # Result: (K, M_med)
        coeffs_med = np.array([c for _, c in candidates_med])
        vals_med = coeffs_med @ Z_med.T
        max_norms_med = np.max(np.abs(vals_med), axis=1)
        
        # Combine scores
        scores_med = max_norms_med
        # Update best_c
        for i in range(len(scores_med)):
            if scores_med[i] < best_score:
                best_score = scores_med[i]
                best_c = candidates_med[i][1].copy()
                
        # Select top K_med for fine pass
        # We just took the top K_med from coarse, so we can use them all.
        # Or we can re-rank based on med scores.
        # Let's re-rank.
        order = np.argsort(scores_med)
        top_med_indices = order[:K_med]
        candidates_fine = [candidates_med[i] for i in top_med_indices]
    else:
        candidates_fine = []
        if best_c is None:
            best_c = np.ones(N, dtype=int)

    # Fine pass with local refinement
    if candidates_fine:
        for _, c in candidates_fine:
            if time.time() > deadline - 2:
                break
            
            # Evaluate on fine grid
            vals_fine = Z_fine @ c
            abs_vals = np.abs(vals_fine)
            
            # Find global max on fine grid
            max_idx = np.argmax(abs_vals)
            current_max = abs_vals[max_idx]
            
            # Local refinement around max_idx
            # Parabolic interpolation using 3 points
            # We need to handle boundaries
            i = max_idx
            M = M_fine
            # Get neighbors (circular)
            im1 = (i - 1) % M
            ip1 = (i + 1) % M
            
            y0 = abs_vals[im1]
            y1 = abs_vals[i]
            y2 = abs_vals[ip1]
            
            # If peak is flat or at boundary of data, skip refinement or handle carefully
            if y1 > y0 and y1 > y2:
                # Parabolic fit
                # Vertex x = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
                denom = y0 - 2 * y1 + y2
                if abs(denom) > 1e-12:
                    x_vertex = 0.5 * (y0 - y2) / denom
                    # x_vertex is in [-0.5, 0.5] relative to i
                    # True theta = thetas_fine[i] + x_vertex * d_theta
                    d_theta = 2 * np.pi / M
                    theta_peak = thetas_fine[i] + x_vertex * d_theta
                    z_peak = np.exp(1j * theta_peak)
                    
                    # Evaluate P(z_peak)
                    # P(z) = sum c_k z^k
                    # We can compute this directly
                    z_powers = np.power(z_peak, np.arange(N))
                    p_val = np.sum(c * z_powers)
                    refined_max = abs(p_val)
                    
                    if refined_max > current_max:
                        current_max = refined_max
                        
            if current_max < best_score:
                best_score = current_max
                best_c = c.copy()

    if best_c is None:
        best_c = np.ones(N, dtype=int)
    else:
        best_c = best_c.astype(int)
        
    return [int(x) for x in best_c]
