import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _calculate_ratio(h):
    """
    Calculate the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    h is a numpy array of nonnegative heights.
    """
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
        
    l2_sq = np.sum(h**2)
    
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    
    if max_conv == 0:
        return 0.0
        
    return l2_sq / (l1 * max_conv)

def _spectral_design_gradient(h, step_size=0.1, n_iter=2000):
    """
    Spectral design approach:
    1. Compute DFT of h
    2. Treat |H|^2 as a "power spectrum"
    3. Optimize the spectral magnitudes to minimize peak autoconvolution
    4. Reconstruct h via IDFT and project to non-negative
    
    The autoconvolution in time domain corresponds to |H|^2 in frequency domain.
    Specifically, the DFT of the autoconvolution is |H|^2.
    
    We want to minimize the peak of the autoconvolution, which is related to
    the inverse DFT of |H|^2.
    """
    h = h.copy().astype(float)
    h = np.maximum(h, 0)
    
    # Normalize
    if np.sum(h) > 0:
        h = h / np.sum(h)
    
    best_h = h.copy()
    best_score = _calculate_ratio(best_h)
    
    for iteration in range(n_iter):
        # Compute DFT
        H = np.fft.fft(h)
        H_mag = np.abs(H)
        H_phase = np.angle(H)
        
        # The autoconvolution in frequency domain is |H|^2
        # We want to minimize the peak of the time-domain autoconvolution
        # which is the IDFT of |H|^2
        
        # Gradient with respect to H_mag:
        # If we change H_mag by delta, the change in autoconvolution peak
        # can be computed via the sensitivity
        
        # Simpler approach: use the fact that for a fixed total energy,
        # spreading the spectrum more evenly reduces the peak autoconvolution
        # (by uncertainty principle type reasoning)
        
        # Compute current autoconvolution
        conv = np.convolve(h, h, mode='full')
        peak_idx = np.argmax(conv)
        peak_val = conv[peak_idx]
        
        if peak_val < 1e-15:
            break
        
        # The peak of the autoconvolution is related to the IDFT of |H|^2
        # We can compute the gradient of the peak with respect to H_mag
        # by noting that conv[k] = sum_j h[j] h[j+k]
        
        # Alternative: use a soft-max approximation for the peak
        # and compute gradient via chain rule
        
        # Let's use a different approach: 
        # For each frequency bin, compute how much it contributes to the peak
        # The contribution of frequency bin m to conv[k] is proportional to
        # |H[m]|^2 * exp(2*pi*i*m*k/N)
        
        # So the gradient of conv[peak_idx] with respect to |H[m]|^2 is
        # Re[exp(2*pi*i*m*peak_idx/N)]
        
        # We want to minimize the peak, so we should decrease |H[m]|^2 for bins
        # where Re[exp(2*pi*i*m*peak_idx/N)] > 0 and increase where < 0
        
        # But we also want to keep h non-negative, which constrains the phase
        
        # Simplified approach: iteratively adjust the spectral magnitudes
        # to flatten the autoconvolution
        
        # Compute the "sensitivity" of the peak to each frequency bin
        m = np.arange(N)
        sensitivity = np.real(np.exp(2j * np.pi * m * peak_idx / N))
        
        # Normalize sensitivity
        sens_norm = sensitivity - np.mean(sensitivity)
        
        # Update H_mag: decrease where sensitivity is positive, increase where negative
        # Use a step size that adapts
        step = step_size * (1.0 / (1.0 + iteration * 0.01))
        
        H_mag_new = H_mag * np.exp(-step * sens_norm)
        
        # Ensure non-negative H_mag
        H_mag_new = np.maximum(H_mag_new, 1e-10)
        
        # Reconstruct h
        H_new = H_mag_new * np.exp(1j * H_phase)
        h_new = np.real(np.fft.ifft(H_new))
        
        # Project to non-negative
        h_new = np.maximum(h_new, 0)
        
        # Normalize
        if np.sum(h_new) > 1e-10:
            h_new = h_new / np.sum(h_new)
        
        # Check improvement
        score_new = _calculate_ratio(h_new)
        if score_new > best_score:
            best_score = score_new
            best_h = h_new.copy()
        
        h = h_new
    
    return best_h, best_score

def _coordinate_descent(h, n_iter=5000, tol=1e-10):
    """
    Coordinate descent on the non-negative heights.
    For each coordinate, optimize it while keeping others fixed.
    """
    h = h.copy().astype(float)
    h = np.maximum(h, 0)
    
    if np.sum(h) > 0:
        h = h / np.sum(h)
    
    best_h = h.copy()
    best_score = _calculate_ratio(best_h)
    
    for iteration in range(n_iter):
        improved = False
        
        for i in range(N):
            # Optimize h[i] while keeping others fixed
            # The ratio is: sum_j h[j]^2 / (sum_j h[j] * max_k conv[k])
            
            # For fixed other coordinates, this is a 1D optimization in h[i]
            # Let a = sum_{j!=i} h[j]^2, b = sum_{j!=i} h[j]
            # Then ratio = (a + h[i]^2) / ((b + h[i]) * max_k conv[k])
            
            # conv[k] = sum_j h[j] h[j+k]
            # = (sum_{j!=i} h[j] h[j+k]) + h[i] * (h[i+k] + h[i-k]) + h[i]^2 * delta(k=0)
            
            # This is complex, so let's just do a simple line search
            
            h_old = h[i]
            
            # Try a few candidate values
            candidates = [0.0, h_old * 0.5, h_old, h_old * 1.5, h_old * 2.0,
                         1.0 / N, 2.0 / N, 0.1, 0.2, 0.5]
            
            best_candidate = h_old
            best_local_score = _calculate_ratio(h)
            
            for candidate in candidates:
                candidate = max(0.0, candidate)
                h[i] = candidate
                score = _calculate_ratio(h)
                if score > best_local_score:
                    best_local_score = score
                    best_candidate = candidate
            
            h[i] = best_candidate
            
            if best_local_score > best_score + tol:
                improved = True
                best_score = best_local_score
                best_h = h.copy()
        
        if not improved:
            break
    
    return best_h, best_score

def _evolutionary_search(pop_size=30, n_generations=50, mutation_rate=0.3, seed=42):
    """
    Evolutionary search with non-negative constraint.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize population
    population = []
    scores = []
    
    for _ in range(pop_size):
        # Random initialization: mix of sparse and dense
        if rng.random() < 0.5:
            # Sparse: pick a few random indices
            n_spikes = rng.randint(2, 10)
            h = np.zeros(N)
            indices = rng.choice(N, size=n_spikes, replace=False)
            for idx in indices:
                h[idx] = rng.exponential(1.0)
        else:
            # Dense
            h = rng.exponential(1.0, N)
        
        h = np.maximum(h, 0)
        if np.sum(h) > 0:
            h = h / np.sum(h)
        
        score = _calculate_ratio(h)
        population.append(h)
        scores.append(score)
    
    best_h = population[np.argmax(scores)].copy()
    best_score = max(scores)
    
    for generation in range(n_generations):
        # Sort by score
        order = np.argsort(scores)[::-1]
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Keep top half
        elite_size = max(5, pop_size // 2)
        new_population = population[:elite_size].copy()
        new_scores = scores[:elite_size].copy()
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Crossover: pick two parents
            parent1_idx = rng.randint(0, elite_size)
            parent2_idx = rng.randint(0, elite_size)
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Uniform crossover
            mask = rng.random(N) < 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            if rng.random() < mutation_rate:
                # Add noise
                noise = rng.exponential(0.1, N)
                child = child + noise
            
            child = np.maximum(child, 0)
            if np.sum(child) > 0:
                child = child / np.sum(child)
            
            score = _calculate_ratio(child)
            new_population.append(child)
            new_scores.append(score)
        
        population = new_population
        scores = new_scores
        
        # Update best
        if max(scores) > best_score:
            best_score = max(scores)
            best_h = population[np.argmax(scores)].copy()
    
    return best_h, best_score

def _find_sidon_set_greedy(n_elements, max_index, seed=42):
    """
    Find a greedy Sidon-like set: indices where all pairwise sums are distinct.
    This minimizes autoconvolution collisions.
    """
    rng = np.random.RandomState(seed)
    best_set = None
    best_size = 0
    
    for attempt in range(100):
        used_sums = set()
        selected = []
        
        candidates = list(range(max_index))
        rng.shuffle(candidates)
        
        for idx in candidates:
            if len(selected) >= n_elements:
                break
            new_sums = set()
            for j in selected:
                s = idx + j
                if s in used_sums or s in new_sums:
                    break
                new_sums.add(s)
            else:
                s_self = idx + idx
                if s_self in used_sums or s_self in new_sums:
                    break
                new_sums.add(s_self)
                selected.append(idx)
                used_sums.update(new_sums)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = sorted(selected)
    
    return best_set if best_set else [0]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_score = 0.0
    best_h = None
    
    def update_best(h, score):
        nonlocal best_score, best_h
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Candidate 1: Spike
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    update_best(h_spike, _calculate_ratio(h_spike))
    
    # Candidate 2: Uniform
    h_uniform = np.ones(N)
    update_best(h_uniform, _calculate_ratio(h_uniform))
    
    # Candidate 3: Sidon-set based sparse vectors
    for n_sidon in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        if time.time() > deadline:
            break
        sidon = _find_sidon_set_greedy(n_sidon, N, seed=n_sidon)
        if len(sidon) < 2:
            continue
        h_sidon = np.zeros(N)
        for idx in sidon:
            h_sidon[idx] = 1.0
        update_best(h_sidon, _calculate_ratio(h_sidon))
    
    # Candidate 4: Spectral design
    if time.time() < deadline - 20:
        # Start from best so far
        h_init = best_h.copy()
        h_spec, score_spec = _spectral_design_gradient(h_init, step_size=0.05, n_iter=3000)
        update_best(h_spec, score_spec)
        
        # Also try from uniform
        h_spec2, score_spec2 = _spectral_design_gradient(h_uniform, step_size=0.05, n_iter=3000)
        update_best(h_spec2, score_spec2)
    
    # Candidate 5: Coordinate descent on best
    if time.time() < deadline - 15:
        h_cd, score_cd = _coordinate_descent(best_h, n_iter=3000)
        update_best(h_cd, score_cd)
    
    # Candidate 6: Evolutionary search
    if time.time() < deadline - 25:
        h_evo, score_evo = _evolutionary_search(pop_size=25, n_generations=40, seed=42)
        update_best(h_evo, score_evo)
        
        # Polish the evolutionary result with coordinate descent
        if time.time() < deadline - 10:
            h_polished, score_polished = _coordinate_descent(h_evo, n_iter=2000)
            update_best(h_polished, score_polished)
    
    # Candidate 7: Spectral design from Sidon initializations
    if time.time() < deadline - 15:
        for n_sidon in [5, 8, 10, 12]:
            if time.time() > deadline - 10:
                break
            sidon = _find_sidon_set_greedy(n_sidon, N, seed=200 + n_sidon)
            if len(sidon) < 2:
                continue
            h_sidon = np.zeros(N)
            for idx in sidon:
                h_sidon[idx] = 1.0
            h_sidon = h_sidon / np.sum(h_sidon)
            
            h_spec, score_spec = _spectral_design_gradient(h_sidon, step_size=0.03, n_iter=2000)
            update_best(h_spec, score_spec)
    
    # Final polish
    if best_h is not None and time.time() < deadline - 5:
        h_final, score_final = _coordinate_descent(best_h, n_iter=1000)
        update_best(h_final, score_final)
    
    if best_h is None:
        return [1.0] + [0.0] * (N - 1)
        
    return list(best_h)
