"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_score(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    H = np.fft.fft(h)
    H1m = np.fft.fft(1.0 - h)
    corr = np.fft.ifft(H * np.conj(H1m)).real
    max_overlap = np.max(corr[1:])
    return max_overlap

def _objective(h):
    """Smooth max approximation of the overlap objective."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    H = np.fft.fft(h)
    H1m = np.fft.fft(1.0 - h)
    corr = np.fft.ifft(H * np.conj(H1m)).real
    c = corr[1:]
    c = np.clip(c, 1e-12, None)
    beta = 25.0
    log_sum_exp = np.log(np.sum(np.exp(beta * c))) / beta
    return log_sum_exp

def _project_to_box_and_sum(h, target_sum):
    """Project h to [0,1]^N with sum = target_sum."""
    h = np.array(h, dtype=np.float64)
    n = len(h)
    for _ in range(200):
        h = np.clip(h, 0.0, 1.0)
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-12:
            break
        h += diff / n
    # Final adjustment
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-10:
        mask = (h > 1e-8) & (h < 1.0 - 1e-8)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
        else:
            h += diff / n
            h = np.clip(h, 0.0, 1.0)
    return h

def _structured_binary_inits(N, rng):
    """Generate structured binary initializations."""
    inits = []
    
    # Pattern 1: Alternating
    h = np.zeros(N)
    h[::2] = 1.0
    inits.append(h)
    
    # Pattern 2: Block
    h = np.zeros(N)
    h[:N//2] = 1.0
    inits.append(h)
    
    # Pattern 3: Golden ratio based
    h = np.zeros(N)
    golden = (1 + np.sqrt(5)) / 2
    for i in range(N):
        if (i * golden) % 1.0 < 0.5:
            h[i] = 1.0
    # Adjust to get exactly N//2 ones
    ones = np.sum(h)
    if ones > N//2:
        idx = np.where(h == 1.0)[0]
        for i in idx[:int(ones - N//2)]:
            h[i] = 0.0
    elif ones < N//2:
        idx = np.where(h == 0.0)[0]
        for i in idx[:int(N//2 - ones)]:
            h[i] = 1.0
    inits.append(h)
    
    # Pattern 4: Fibonacci-like
    h = np.zeros(N)
    fib_positions = set()
    a, b = 0, 1
    while a < N:
        fib_positions.add(a % N)
        a, b = b, a + b
    for i in range(N):
        if i in fib_positions:
            h[i] = 1.0
    ones = np.sum(h)
    if ones > N//2:
        idx = np.where(h == 1.0)[0]
        for i in idx[:int(ones - N//2)]:
            h[i] = 0.0
    elif ones < N//2:
        idx = np.where(h == 0.0)[0]
        for i in idx[:int(N//2 - ones)]:
            h[i] = 1.0
    inits.append(h)
    
    # Pattern 5: Random with structure
    for _ in range(5):
        h = np.zeros(N)
        # Create blocks of varying sizes
        pos = 0
        while pos < N:
            block_size = rng.integers(1, 5)
            if rng.random() < 0.5:
                h[pos:pos+block_size] = 1.0
            pos += block_size
        ones = np.sum(h)
        if ones > N//2:
            idx = np.where(h == 1.0)[0]
            for i in idx[:int(ones - N//2)]:
                h[i] = 0.0
        elif ones < N//2:
            idx = np.where(h == 0.0)[0]
            for i in idx[:int(N//2 - ones)]:
                h[i] = 1.0
        inits.append(h)
    
    return inits

def _binary_local_search(h_init, deadline, max_iters=5000, rng=None):
    """Local search on binary sequences."""
    if rng is None:
        rng = np.random.default_rng()
    
    h = h_init.copy()
    score = _compute_score(h)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        
        if len(ones) == 0 or len(zeros) == 0:
            break
        
        # Try several random swaps
        best_swap = None
        best_score = score
        
        for _ in range(10):
            i = ones[rng.integers(len(ones))]
            j = zeros[rng.integers(len(zeros))]
            
            h[i] = 0.0
            h[j] = 1.0
            new_score = _compute_score(h)
            
            if new_score < best_score:
                best_score = new_score
                best_swap = (i, j)
            else:
                h[i] = 1.0
                h[j] = 0.0
        
        if best_swap is not None:
            i, j = best_swap
            h[i] = 0.0
            h[j] = 1.0
            score = best_score
        else:
            # No improvement found, try a random perturbation
            if rng.random() < 0.1:
                i = ones[rng.integers(len(ones))]
                j = zeros[rng.integers(len(zeros))]
                h[i] = 0.0
                h[j] = 1.0
                score = _compute_score(h)
    
    return h, score

def _continuous_refine(h_init, deadline):
    """Refine a binary or continuous solution using L-BFGS-B."""
    h = h_init.copy()
    
    try:
        result = minimize(
            _objective,
            h,
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000}
        )
        h_opt = result.x
        h_opt = _project_to_box_and_sum(h_opt, N/2.0)
        score_opt = _compute_score(h_opt)
        
        if score_opt <= _compute_score(h_init) + 1e-9:
            return h_opt
    except Exception:
        pass
    
    return h_init

def construct():
    """Main entry point."""
    start_time = time.time()
    total_deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    best_h = np.full(N, 0.5)
    best_score = _compute_score(best_h)
    
    # Phase 1: Structured binary initializations with local search
    binary_deadline = start_time + 80.0
    inits = _structured_binary_inits(N, rng)
    
    for h_init in inits:
        if time.time() > binary_deadline:
            break
        
        h_local, score_local = _binary_local_search(
            h_init,
            deadline=binary_deadline,
            max_iters=3000,
            rng=rng
        )
        
        if score_local < best_score:
            best_score = score_local
            best_h = h_local.copy()
    
    # Phase 2: More random restarts if time permits
    random_deadline = start_time + 110.0
    n_random = 30
    for _ in range(n_random):
        if time.time() > random_deadline:
            break
        idx = rng.choice(N, size=N//2, replace=False)
        h_rand = np.zeros(N)
        h_rand[idx] = 1.0
        h_local, score_local = _binary_local_search(
            h_rand,
            deadline=random_deadline,
            max_iters=2000,
            rng=rng
        )
        if score_local < best_score:
            best_score = score_local
            best_h = h_local.copy()
    
    # Phase 3: Continuous refinement
    refine_deadline = start_time + 125.0
    if time.time() < refine_deadline:
        h_refined = _continuous_refine(best_h, deadline=refine_deadline)
        score_refined = _compute_score(h_refined)
        if score_refined < best_score:
            best_score = score_refined
            best_h = h_refined
    
    # Final validation
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = N/2.0 - np.sum(best_h)
    if abs(diff) > 1e-8:
        best_h = _project_to_box_and_sum(best_h, N/2.0)
    
    return best_h
# EVOLVE-BLOCK-END
