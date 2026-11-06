import numpy as np
import time

def _fast_supnorm(c, n_samples=1500):
    """Fast sup norm evaluation using dense sampling."""
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    z = np.exp(1j * theta)
    k = np.arange(len(c))
    Z = z[:, None] ** k[None, :]
    vals = Z @ c
    return np.max(np.abs(vals))

def _accurate_supnorm(c):
    """Accurate sup norm evaluation with golden-section refinement of peaks."""
    n_samples = 20000
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    z = np.exp(1j * theta)
    k = np.arange(len(c))
    Z = z[:, None] ** k[None, :]
    vals = Z @ c
    max_val = np.max(np.abs(vals))
    
    # Refine top peaks
    abs_vals = np.abs(vals)
    idx = np.argsort(abs_vals)[::-1]
    
    best_refined = max_val
    gr = (np.sqrt(5) - 1) / 2
    
    def eval_at_angle(ang):
        zz = np.exp(1j * ang)
        powers = zz ** k
        val = np.sum(c * powers)
        return np.abs(val)
        
    for i in idx[:15]:
        delta = 2 * np.pi / n_samples
        a = theta[i] - delta
        b = theta[i] + delta
        
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        
        fc = eval_at_angle(c_pt)
        fd = eval_at_angle(d_pt)
        
        for _ in range(50):
            if fc < fd:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = eval_at_angle(c_pt)
            else:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = eval_at_angle(d_pt)
                
        refined_val = max(fc, fd)
        if refined_val > best_refined:
            best_refined = refined_val
            
    return best_refined

def _local_search(coeffs, deadline, rng):
    """Aggressive local search: try flipping each bit, accept improvements."""
    current = coeffs.copy().astype(float)
    current_score = _fast_supnorm(current)
    n = len(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = np.arange(n)
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            old_val = current[i]
            current[i] = -current[i]
            
            new_score = _fast_supnorm(current)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Generate diverse initial seeds
    seeds = []
    
    # Seed 1: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    seeds.append(np.array(p[:24], dtype=float))
    
    # Seed 2: Negative of GR
    seeds.append(-np.array(p[:24], dtype=float))
    
    # Seed 3: Reversed GR
    seeds.append(np.array(p[:24][::-1], dtype=float))
    
    # Seed 4: Random but biased toward low autocorrelation
    rng = np.random.default_rng(42)
    
    # Generate several random seeds
    for _ in range(10):
        seeds.append(rng.choice([-1.0, 1.0], size=24))
    
    # Evaluate all seeds and pick the best few
    seed_scores = []
    for s in seeds:
        score = _fast_supnorm(s)
        seed_scores.append((score, s))
    
    seed_scores.sort(key=lambda x: x[0])
    
    # Start local search from the best seeds
    best_coeffs = seed_scores[0][1].copy()
    best_score = seed_scores[0][0]
    
    # Iterated local search from multiple diverse starts
    num_restarts = min(20, max(5, int((deadline - time.time()) / 3.0)))
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if r < len(seed_scores):
            current = seed_scores[r][1].copy()
        else:
            # Perturb the best found so far
            current = best_coeffs.copy()
            # Flip 2-4 random bits
            n_flips = rng.integers(2, 5)
            flip_idx = rng.choice(24, size=n_flips, replace=False)
            for idx in flip_idx:
                current[idx] = -current[idx]
        
        current, current_score = _local_search(current, deadline, rng)
        
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current.copy()
    
    # Final local search from the best
    if time.time() < deadline - 5.0:
        best_coeffs, best_score = _local_search(best_coeffs, deadline, rng)
    
    # Verify with accurate sup norm
    if time.time() < deadline - 2.0:
        accurate_score = _accurate_supnorm(best_coeffs)
        # The accurate score should be >= fast score; use it for reporting
    
    # Ensure we return integers
    result = np.sign(best_coeffs).astype(int)
    # Ensure no zeros (shouldn't happen, but safety)
    result[result == 0] = 1
    
    return result.tolist()
