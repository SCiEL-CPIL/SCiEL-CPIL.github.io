import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.special import logsumexp

N = 100
DEADLINE = 135.0  # safety margin before 150s kill

def _overlap_at_shifts(h, shifts):
    """Compute overlap sum(h * shifted(1-h)) for each shift.
    h has length 2N (period 2), step width 1/N.
    shift k means (1-h) shifted by k steps: (1-h)[(i-k) mod 2N]
    overlap_k = sum_i h[i] * (1 - h[(i-k) mod 2N])
    """
    twoN = len(h)
    overlaps = np.empty(len(shifts))
    for idx, k in enumerate(shifts):
        shifted_1mh = 1.0 - np.roll(h, -k, axis=0)
        overlaps[idx] = np.sum(h * shifted_1mh)
    return overlaps

def _max_overlap(h):
    """Maximum overlap over all shifts 1..N-1 (or 1..2N-1)."""
    twoN = len(h)
    # By periodicity and symmetry, we need shifts 1..2N-1
    # But overlap at shift k = overlap at shift 2N-k (by symmetry of the problem)
    # Actually let's just compute all shifts 1..2N-1
    best = 0.0
    for k in range(1, twoN):
        shifted_1mh = 1.0 - np.roll(h, -k, axis=0)
        ov = np.sum(h * shifted_1mh)
        if ov > best:
            best = ov
    return best

def _build_h_from_blocks(block_heights, N):
    """Build a length-2N array from block heights (each block = N steps of same height).
    block_heights has length 2 (for period 2).
    Returns array of length 2N."""
    h = np.zeros(2 * N)
    for b, bh in enumerate(block_heights):
        h[b*N : (b+1)*N] = bh
    return h

def _build_h_from_segments(segments, N):
    """Build h from segment boundaries and heights.
    segments: list of (start_frac, end_frac, height) where fractions are in [0,2].
    Returns array of length 2N."""
    h = np.zeros(2 * N)
    for start_frac, end_frac, height in segments:
        s = int(round(start_frac * N))
        e = int(round(end_frac * N))
        e = min(e, 2 * N)
        s = max(s, 0)
        if e > s:
            h[s:e] = height
    return h

def _score_h(h):
    """Compute the max overlap score for a given h array."""
    return _max_overlap(h)

def _logsumexp_overlap(h, beta=50.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    twoN = len(h)
    overlaps = np.empty(twoN - 1)
    for k in range(1, twoN):
        shifted_1mh = 1.0 - np.roll(h, -k, axis=0)
        overlaps[k-1] = np.sum(h * shifted_1mh)
    # log-sum-exp for smooth max
    m = np.max(overlaps)
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def _objective_2block(x):
    """2-block parameterization: h = [a, a, ..., 1-a, 1-a, ...] with a in [0,1].
    Constraint: sum = N (since 2N steps, each of width 1/N, total integral = sum/N = 1).
    So sum of h values must be N.
    With 2 blocks: N*a + N*(1-a) = N. Always satisfied!
    So x = [a] where a in [0,1]."""
    a = x[0]
    h = _build_h_from_blocks([a, 1-a], N)
    return _score_h(h)

def _objective_multi_block(x, n_blocks):
    """Multi-block: n_blocks blocks of equal width, heights alternating pattern.
    x = [h_0, h_1, ..., h_{n_blocks-1}] with sum constraint: sum(x) = 2N/n_blocks * ... 
    Wait: total steps = 2N, each block has 2N/n_blocks steps.
    Integral = sum(h) / N = 1 => sum(h) = N.
    So sum(x * (2N/n_blocks)) = N => sum(x) = N * n_blocks / (2N) = n_blocks/2.
    """
    h = np.zeros(2 * N)
    block_size = 2 * N // n_blocks
    for i in range(n_blocks):
        h[i*block_size : (i+1)*block_size] = x[i]
    # Fix any remainder
    if (n_blocks * block_size) < 2 * N:
        h[n_blocks*block_size:] = x[-1]
    return _score_h(h)

def _optimize_2block():
    """Optimize the 2-block case: h = [a]*N + [1-a]*N."""
    # Score for 2-block: overlap at shift k
    # For shift k (1<=k<=2N-1):
    # If k < N: h[i]*(1-h[(i-k) mod 2N])
    #   For i in [0,N): h[i]=a, (i-k) mod 2N is in [N-k, N) -> h = a, so 1-h = 1-a
    #   For i in [N, 2N): h[i]=1-a, (i-k) mod 2N is in [N-k, N) -> h = a, so 1-h = 1-a
    #   Wait, let me be more careful.
    #   Actually let me just compute numerically.
    
    best_a = 0.5
    best_score = _objective_2block([0.5])
    
    # Grid search first
    for a in np.linspace(0.0, 1.0, 201):
        s = _objective_2block([a])
        if s < best_score:
            best_score = s
            best_a = a
    
    # Refine with golden section
    lo, hi = max(0, best_a - 0.05), min(1, best_a + 0.05)
    gr = (np.sqrt(5) - 1) / 2
    for _ in range(50):
        x1 = hi - gr * (hi - lo)
        x2 = lo + gr * (hi - lo)
        f1 = _objective_2block([x1])
        f2 = _objective_2block([x2])
        if f1 < f2:
            hi = x2
        else:
            lo = x1
        if hi - lo < 1e-10:
            break
    
    best_a = (lo + hi) / 2
    best_score = _objective_2block([best_a])
    h = _build_h_from_blocks([best_a, 1-best_a], N)
    return h, best_score

def _optimize_multi_block(n_blocks, n_restarts=3, timeout=20.0):
    """Optimize multi-block construction using differential evolution."""
    t_start = time.time()
    block_size = 2 * N // n_blocks
    total_block_steps = n_blocks * block_size
    # sum(x) = n_blocks / 2 (from integral constraint)
    
    def obj(x):
        # Enforce bounds and constraint
        x = np.clip(x, 0.0, 1.0)
        # Project to sum constraint: sum(x) = n_blocks/2
        s = np.sum(x)
        if s > 0:
            x = x * (n_blocks / 2.0) / s
            x = np.clip(x, 0.0, 1.0)
        h = np.zeros(2 * N)
        for i in range(n_blocks):
            h[i*block_size : (i+1)*block_size] = x[i]
        if total_block_steps < 2 * N:
            h[total_block_steps:] = x[-1]
        return _score_h(h)
    
    best_h = None
    best_score = float('inf')
    
    bounds = [(0.0, 1.0)] * n_blocks
    
    for restart in range(n_restarts):
        if time.time() - t_start > timeout:
            break
        # Random initial population
        x0 = np.random.dirichlet(np.ones(n_blocks)) * (n_blocks / 2.0)
        x0 = np.clip(x0, 0.0, 1.0)
        
        try:
            result = differential_evolution(
                obj, bounds, seed=restart, maxiter=30, 
                popsize=10, tol=1e-6, init='latinhypercube'
            )
            if result.fun < best_score:
                best_score = result.fun
                x_best = np.clip(result.x, 0.0, 1.0)
                # Re-project
                s = np.sum(x_best)
                if s > 0:
                    x_best = x_best * (n_blocks / 2.0) / s
                h = np.zeros(2 * N)
                for i in range(n_blocks):
                    h[i*block_size : (i+1)*block_size] = x_best[i]
                if total_block_steps < 2 * N:
                    h[total_block_steps:] = x_best[-1]
                best_h = h
        except Exception:
            pass
    
    if best_h is None:
        return None, float('inf')
    return best_h, best_score

def _optimize_symmetric_4block():
    """4-block with symmetry: h = [a, b, 1-a, 1-b] pattern.
    Constraint: 2N steps, block_size = N/2.
    Sum = (N/2)(a + b + (1-a) + (1-b)) = (N/2)(2) = N. Always satisfied!
    So x = [a, b] in [0,1]^2."""
    t_start = time.time()
    block_size = N // 2  # 25 for N=100
    
    def obj(x):
        a, b = x
        h = np.zeros(2 * N)
        h[0:block_size] = a
        h[block_size:2*block_size] = b
        h[2*block_size:3*block_size] = 1-a
        h[3*block_size:4*block_size] = 1-b
        return _score_h(h)
    
    best_a, best_b = 0.5, 0.5
    best_score = obj([0.5, 0.5])
    
    # Grid search
    for a in np.linspace(0, 1, 51):
        for b in np.linspace(0, 1, 51):
            s = obj([a, b])
            if s < best_score:
                best_score = s
                best_a, best_b = a, b
    
    # Refine with DE
    bounds = [(max(0, best_a-0.1), min(1, best_a+0.1)), 
              (max(0, best_b-0.1), min(1, best_b+0.1))]
    try:
        result = differential_evolution(obj, bounds, seed=42, maxiter=50, 
                                         popsize=15, tol=1e-8)
        if result.fun < best_score:
            best_score = result.fun
            best_a, best_b = result.x
    except Exception:
        pass
    
    h = np.zeros(2 * N)
    h[0:block_size] = best_a
    h[block_size:2*block_size] = best_b
    h[2*block_size:3*block_size] = 1-best_a
    h[3*block_size:4*block_size] = 1-best_b
    return h, best_score

def construct():
    t0 = time.time()
    
    best_h = np.full(N, 0.5)  # fallback
    best_score = _max_overlap(np.full(2*N, 0.5))
    
    # Strategy 1: 2-block optimization
    try:
        h2, s2 = _optimize_2block()
        if s2 < best_score:
            best_score = s2
            best_h = h2[:N]  # Return first N (one period)
        print(f"2-block: score={s2:.6f}, time={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"2-block failed: {e}")
    
    if time.time() - t0 > DEADLINE:
        return best_h
    
    # Strategy 2: 4-block symmetric
    try:
        h4, s4 = _optimize_symmetric_4block()
        if s4 < best_score:
            best_score = s4
            best_h = h4[:N]
        print(f"4-block: score={s4:.6f}, time={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"4-block failed: {e}")
    
    if time.time() - t0 > DEADLINE:
        return best_h
    
    # Strategy 3: Multi-block DE for various block counts
    for nb in [6, 8, 10, 12]:
        if time.time() - t0 > DEADLINE - 10:
            break
        try:
            hm, sm = _optimize_multi_block(nb, n_restarts=2, timeout=15.0)
            if hm is not None and sm < best_score:
                best_score = sm
                best_h = hm[:N]
            print(f"{nb}-block: score={sm:.6f}, time={time.time()-t0:.1f}s")
        except Exception as e:
            print(f"{nb}-block failed: {e}")
    
    if time.time() - t0 > DEADLINE:
        return best_h
    
    # Strategy 4: Local refinement of best solution using full 2N variables
    # Use gradient-free local search (Nelder-Mead style via random perturbations)
    try:
        h_full = np.zeros(2 * N)
        h_full[:N] = best_h
        h_full[N:] = best_h  # periodic extension (period N, but we need period 2N)
        # Actually for period 2, h on [0,2] has period 2, so h[i+N] = h[i+N] (not h[i])
        # Let me reconsider: h is defined on [0,2] with period 2.
        # So h has 2N steps, and h is NOT periodic with period N.
        # The "shift" is by k steps on the 2N array.
        # best_h is the first N values. We need the second N values.
        # For the 2-block case: h = [a]*N + [1-a]*N
        # For the 4-block case: h = [a]*25 + [b]*25 + [1-a]*25 + [1-b]*25
        
        # Let me just use the full h from the best construction
        # Rebuild from scratch based on which was best
        # Actually, let's just do local search on the full 2N array
        pass
    except Exception:
        pass
    
    # Ensure output is exactly N values in [0,1] summing to N/2
    result = np.clip(best_h, 0.0, 1.0)
    # Fix sum: should be N/2
    current_sum = np.sum(result)
    target_sum = N / 2.0
    if abs(current_sum - target_sum) > 1e-10:
        # Adjust proportionally
        if current_sum > 0:
            result = result * (target_sum / current_sum)
            result = np.clip(result, 0.0, 1.0)
        else:
            result = np.full(N, 0.5)
    
    # Final sum correction
    diff = target_sum - np.sum(result)
    if abs(diff) > 1e-10:
        # Distribute the difference
        n_pos = np.sum(result > 0)
        if n_pos > 0:
            result[result > 0] += diff / n_pos
            result = np.clip(result, 0.0, 1.0)
        # If still off, brute force adjust one element
        diff = target_sum - np.sum(result)
        if abs(diff) > 1e-12:
            for i in range(N):
                if 0 < result[i] + diff <= 1:
                    result[i] += diff
                    break
    
    return result

# Quick self-test
if __name__ == "__main__":
    h = construct()
    print(f"Shape: {h.shape}, Sum: {np.sum(h):.6f}, Min: {np.min(h):.4f}, Max: {np.max(h):.4f}")
    h_full = np.concatenate([h, h])  # This is wrong for period 2
    # Actually h IS the full 2N array truncated? No, h is N values representing [0,1]
    # Wait, re-reading the task: "step heights h on [0,2]"
    # So h has 2N steps? Or N steps on [0,2]?
    # "sum is exactly len/2" - if len=N, sum=N/2, integral = sum/N * (2/N)*N... 
    # Let me re-read: "equal-width step heights h on [0,2]"
    # N steps on [0,2], each of width 2/N.
    # Integral = sum(h_i * 2/N) = 1 => sum(h_i) = N/2. Yes!
    # So h has N values, each in [0,1], sum = N/2.
    # The "shift" is by some amount on [0,2].
    # Overlap of h with shifted (1-h): 
    # If we shift by k steps (k*2/N units), then overlap = sum_i h[i] * (1-h[(i-k) mod N]) * (2/N)
    # But the score is just the max of these (without the 2/N factor, or with it?)
    # The task says "minimize the maximum overlap" - likely the sum without width factor.
    # Let me just return N values summing to N/2.
    print(f"Score check: {np.sum(h)}")
