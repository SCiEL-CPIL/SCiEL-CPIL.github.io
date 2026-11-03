"""Rewritten approach: use a different representation and optimization strategy."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _max_overlap_fast(h):
    """Compute max overlap of h with shifted (1-h) using FFT."""
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    min_autocorr = np.min(autocorr)
    max_overlap = N / 2.0 - min_autocorr
    return max_overlap


def _project_to_feasible(h):
    """Project h to satisfy: sum = N/2, 0 <= h[i] <= 1."""
    h = h.copy()
    h = np.clip(h, 0, 1)
    target = N / 2.0
    current_sum = np.sum(h)
    diff = target - current_sum
    if abs(diff) > 1e-12:
        if diff > 0:
            room = 1.0 - h
            if np.sum(room) > 1e-12:
                h += diff * room / np.sum(room)
        else:
            if np.sum(h) > 1e-12:
                h += diff * h / np.sum(h)
        h = np.clip(h, 0, 1)
    return h


def _smooth_candidate(t):
    """Generate a smooth candidate using cosine waves with parameter t controlling frequency."""
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Use multiple harmonics with decreasing amplitude
    h = 0.5
    freq1 = max(1, int(round(t * 10)))
    freq2 = max(2, int(round(t * 20 + 5)))
    freq3 = max(3, int(round(t * 30 + 10)))
    h += 0.3 * np.cos(2 * np.pi * freq1 * x / N)
    h += 0.15 * np.cos(2 * np.pi * freq2 * x / N + np.pi / 3)
    h += 0.08 * np.cos(2 * np.pi * freq3 * x / N + 2 * np.pi / 3)
    h = _project_to_feasible(h)
    return h


def _random_candidate():
    """Generate a random feasible candidate."""
    h = np.random.uniform(0, 1, N)
    h = _project_to_feasible(h)
    return h


def _binary_candidate(seed=None):
    """Generate a binary-like candidate with exactly N/2 ones."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    idx = rng.permutation(N)
    h = np.zeros(N)
    h[idx[:N // 2]] = 1.0
    return h


def _coordinate_descent(h0, max_iters=5000, time_limit=25.0):
    """Coordinate descent with adaptive step sizes."""
    h = h0.copy()
    best_score = _max_overlap_fast(h)
    start_time = time.time()
    step = 0.05
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try each coordinate
        for i in range(N):
            if time.time() - start_time > time_limit:
                    break
            
            # Try increasing
            new_h = h.copy()
            new_h[i] = min(1.0, h[i] + step)
            new_h = _project_to_feasible(new_h)
            score = _max_overlap_fast(new_h)
            if score < best_score - 1e-10:
                h = new_h
                best_score = score
            else:
                # Try decreasing
                new_h = h.copy()
                new_h[i] = max(0.0, h[i] - step)
                new_h = _project_to_feasible(new_h)
                score = _max_overlap_fast(new_h)
                if score < best_score - 1e-10:
                    h = new_h
                    best_score = score
        
        # Adaptive step size
        if it % 100 == 0 and it > 0:
            step *= 0.8
            step = max(step, 0.001)
    
    return h, best_score


def _simulated_annealing(h0, max_iters=3000, time_limit=20.0):
    """Simulated annealing with adaptive temperature."""
    h = h0.copy()
    best_h = h.copy()
    best_score = _max_overlap_fast(h)
    start_time = time.time()
    temp = 0.1
    cooling = 0.99
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Generate neighbor
        new_h = h.copy()
        i = np.random.randint(N)
        delta = np.random.uniform(-0.2, 0.2)
        new_h[i] = np.clip(new_h[i] + delta, 0, 1)
        new_h = _project_to_feasible(new_h)
        
        new_score = _max_overlap_fast(new_h)
        
        # Accept if better or with probability exp(-(new_score - best_score) / temp)
        if new_score < best_score:
            h = new_h
            best_h = new_h.copy()
            best_score = new_score
        elif np.random.random() < np.exp(-(new_score - best_score) / temp):
            h = new_h
        
        temp *= cooling
    
    return best_h, best_score


def _nelder_mead(h0, max_iters=1000, time_limit=15.0):
    """Simple Nelder-Mead style optimization on a subset of parameters."""
    # Use a parameterized family: h[i] = 0.5 + a * cos(2*pi*f*i/N + phase)
    # Optimize a, f, phase
    
    best_score = _max_overlap_fast(h0)
    start_time = time.time()
    
    # Initial simplex in parameter space
    # Parameters: [amplitude, frequency, phase]
    params = np.array([0.4, 5.0, 0.0])
    simplex = [params.copy()]
    step = 0.1
    for i in range(3):
        p = params.copy()
        p[i] += step
        simplex.append(p)
    
    def _eval_params(p):
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        h = 0.5 + p[0] * np.cos(2 * np.pi * p[1] * x / N + p[2])
        h = _project_to_feasible(h)
        return _max_overlap_fast(h), h
    
    scores = [_eval_params(p)[0] for p in simplex]
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Sort by score
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        if scores[0] < best_score:
            best_score = scores[0]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        
        # Reflection
        reflected = centroid + (centroid - worst)
        r_score, _ = _eval_params(reflected)
        
        if r_score < scores[0]:
            # Expansion
            expanded = centroid + 2 * (centroid - worst)
            e_score, _ = _eval_params(expanded)
            if e_score < r_score:
                simplex[-1] = expanded
                scores[-1] = e_score
            else:
                simplex[-1] = reflected
                scores[-1] = r_score
        elif r_score < scores[-2]:
            simplex[-1] = reflected
            scores[-1] = r_score
        else:
            # Contraction
            contracted = centroid + 0.5 * (worst - centroid)
            c_score, _ = _eval_params(contracted)
            if c_score < scores[-1]:
                simplex[-1] = contracted
                scores[-1] = c_score
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    scores[i] = _eval_params(simplex[i])[0]
    
    # Get best point
    best_params = simplex[np.argmin(scores)]
    _, best_h = _eval_params(best_params)
    return best_h, scores[np.argmin(scores)]


def construct():
    start_time = time.time()
    deadline = 140.0  # 10 second safety margin
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap_fast(best_h)
    
    # Generate diverse starting candidates
    candidates = []
    
    # Smooth candidates with different parameters
    for t in np.linspace(0, 1, 20):
        candidates.append(_smooth_candidate(t))
    
    # Random candidates
    for _ in range(50):
        candidates.append(_random_candidate())
    
    # Binary candidates
    for seed in range(50):
        candidates.append(_binary_candidate(seed))
    
    # Evaluate all candidates
    scored_candidates = []
    for h in candidates:
        if time.time() > deadline - 60:
            break
        score = _max_overlap_fast(h)
        scored_candidates.append((score, h))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top candidates for optimization
    top_candidates = [h for _, h in scored_candidates[:20]]
    
    # Run coordinate descent from top candidates
    for h0 in top_candidates:
        if time.time() > deadline - 30:
            break
        remaining = deadline - time.time() - 5
        if remaining < 10:
            break
        h_opt, score_opt = _coordinate_descent(
            h0, max_iters=5000,
            time_limit=min(12.0, remaining)
        )
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Run simulated annealing from best
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        if remaining > 10:
            h_sa, score_sa = _simulated_annealing(
                best_h, max_iters=3000,
                time_limit=min(15.0, remaining)
            )
            if score_sa < best_score:
                best_score = score_sa
                best_h = h_sa
    
    # Run Nelder-Mead from best
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 8:
            h_nm, score_nm = _nelder_mead(
                best_h, max_iters=500,
                time_limit=min(10.0, remaining)
            )
            if score_nm < best_score:
                best_score = score_nm
                best_h = h_nm
    
    # Final refinement
    if time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        if remaining > 5:
            h_final, score_final = _coordinate_descent(
                best_h, max_iters=3000,
                time_limit=min(10.0, remaining)
            )
            if score_final < best_score:
                best_score = score_final
                best_h = h_final
    
    # Ensure the result is valid
    best_h = _project_to_feasible(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
