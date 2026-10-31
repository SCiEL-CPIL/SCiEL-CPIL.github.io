"""Rewritten program: sparse + local sign-flip optimization for C3' minimization."""
import numpy as np
import time

N = 400

def compute_autoconv(a):
    """Compute full autoconvolution of a."""
    return np.correlate(a, a, mode='full')

def compute_objective(a):
    """Compute max abs autoconv / (sum a)^2. Return inf if sum is 0."""
    s = np.sum(a)
    if s == 0:
        return np.inf
    ac = compute_autoconv(a)
    return np.max(np.abs(ac)) / (s * s)

def local_sign_flip_optimize(a, max_iters=200, deadline=None):
    """Iteratively flip signs to reduce max abs autoconv."""
    a = a.copy().astype(np.float64)
    best_obj = compute_objective(a)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        n = len(a)
        # Try random subset of positions to flip (for speed)
        # Use deterministic sweep for reliability
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            # Try flipping position i
            a[i] *= -1
            new_obj = compute_objective(a)
            if new_obj < best_obj:
                best_obj = new_obj
                improved = True
            else:
                a[i] *= -1  # revert
        
        if not improved:
            break
    
    return a, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    best_a = None
    best_obj = np.inf
    
    # Diverse initializations
    rng = np.random.default_rng(42)
    
    # 1. All ones (baseline)
    a1 = np.ones(N)
    obj1 = compute_objective(a1)
    if obj1 < best_obj:
        best_obj = obj1
        best_a = a1.copy()
    
    # 2. Random signs
    for trial in range(20):
        if time.time() > deadline:
            break
        a = rng.choice([-1.0, 1.0], size=N)
        obj = compute_objective(a)
        if obj < best_obj:
            best_obj = obj
            best_a = a.copy()
    
    # 3. Chirp pattern
    t = np.arange(N)
    a_chirp = np.sign(np.cos(2 * np.pi * t / (N // 2)))
    a_chirp[a_chirp == 0] = 1.0
    obj_chirp = compute_objective(a_chirp)
    if obj_chirp < best_obj:
        best_obj = obj_chirp
        best_a = a_chirp.copy()
    
    # 4. Block pattern (alternating blocks)
    block_size = N // 8
    a_block = np.zeros(N)
    for b in range(8):
        a_block[b*block_size:(b+1)*block_size] = 1.0 if b % 2 == 0 else -1.0
    obj_block = compute_objective(a_block)
    if obj_block < best_obj:
        best_obj = obj_block
        best_a = a_block.copy()
    
    # 5. Sparse random subsets with random signs
    for density in [0.1, 0.2, 0.3, 0.5]:
        for trial in range(10):
            if time.time() > deadline:
                break
            mask = rng.random(N) < density
            a = np.zeros(N)
            a[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
            obj = compute_objective(a)
            if obj < best_obj:
                best_obj = obj
                best_a = a.copy()
    
    # 6. Run local optimization on best candidates
    candidates = []
    # Add best so far
    candidates.append(best_a.copy())
    # Add some diverse candidates
    a_rand = rng.choice([-1.0, 1.0], size=N)
    candidates.append(a_rand)
    a_chirp2 = np.sign(np.cos(2 * np.pi * np.arange(N) / (N // 3)))
    a_chirp2[a_chirp2 == 0] = 1.0
    candidates.append(a_chirp2)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        opt_a, opt_obj = local_sign_flip_optimize(cand, max_iters=50, deadline=deadline)
        if opt_obj < best_obj:
            best_obj = opt_obj
            best_a = opt_a
    
    # Final local optimization on the best
    if time.time() < deadline:
        opt_a, opt_obj = local_sign_flip_optimize(best_a, max_iters=100, deadline=deadline)
        if opt_obj < best_obj:
            best_obj = opt_obj
            best_a = opt_a
    
    return best_a
# EVOLVE-BLOCK-END
